import argparse
import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


class AsyncLLM(LLM):
    def add_request(self, prompt: Union[str, List[int]], sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id


engine: Optional[AsyncLLM] = None
request_futures: Dict[int, asyncio.Future] = {}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "nanovllm"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


def create_engine(
    model_path: str,
    tp: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    enforce_eager: bool,
) -> AsyncLLM:
    return AsyncLLM(
        os.path.expanduser(model_path),
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=enforce_eager,
    )


def init_engine_from_env() -> AsyncLLM:
    model_path = os.environ.get("NANOVLLM_MODEL")
    if not model_path:
        raise RuntimeError("NANOVLLM_MODEL is required")
    tp = int(os.environ.get("NANOVLLM_TP", "1"))
    max_model_len = int(os.environ.get("NANOVLLM_MAX_MODEL_LEN", "4096"))
    max_num_batched_tokens = int(os.environ.get("NANOVLLM_MAX_BATCHED_TOKENS", str(max(16384, max_model_len))))
    enforce_eager = os.environ.get("NANOVLLM_ENFORCE_EAGER", "0") == "1"
    return create_engine(model_path, tp, max_model_len, max_num_batched_tokens, enforce_eager)


async def engine_loop():
    global engine
    while True:
        if engine is None or engine.is_finished():
            await asyncio.sleep(0.01)
            continue
        try:
            outputs, _ = engine.step()
        except Exception as e:
            for future in request_futures.values():
                if not future.done():
                    future.set_exception(e)
            request_futures.clear()
            await asyncio.sleep(0.1)
            continue
        for seq_id, token_ids in outputs:
            future = request_futures.pop(seq_id, None)
            if future and not future.done():
                future.set_result(token_ids)
        await asyncio.sleep(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    if engine is None:
        engine = init_engine_from_env()
    task = asyncio.create_task(engine_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Nano-vLLM Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    model_id = os.environ.get("NANOVLLM_SERVED_MODEL_NAME", "nanovllm")
    return ModelList(data=[ModelCard(id=model_id)])


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    if request.stream:
        raise HTTPException(status_code=400, detail="stream is not supported")
    if request.top_p is not None and request.top_p != 1.0:
        raise HTTPException(status_code=400, detail="top_p is not supported")

    try:
        prompt = engine.tokenizer.apply_chat_template(
            [m.model_dump() for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error applying chat template: {str(e)}")

    prompt_token_ids = engine.tokenizer.encode(prompt)
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    future = asyncio.get_running_loop().create_future()
    seq_id = engine.add_request(prompt_token_ids, sampling_params)
    request_futures[seq_id] = future

    try:
        token_ids = await future
    except asyncio.CancelledError:
        request_futures.pop(seq_id, None)
        raise
    except Exception as e:
        request_futures.pop(seq_id, None)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    text = engine.tokenizer.decode(token_ids)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": len(prompt_token_ids),
            "completion_tokens": len(token_ids),
            "total_tokens": len(prompt_token_ids) + len(token_ids),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--served-model-name", type=str, default="nanovllm")

    args = parser.parse_args()
    os.environ["NANOVLLM_SERVED_MODEL_NAME"] = args.served_model_name

    max_batched = args.max_num_batched_tokens or max(16384, args.max_model_len)
    engine = create_engine(args.model, args.tp, args.max_model_len, max_batched, args.enforce_eager)

    uvicorn.run(app, host=args.host, port=args.port)
