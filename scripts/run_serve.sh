#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

# ls -l /dev/shm | grep nanovllm
rm -f /dev/shm/nanovllm*

# export NANOVLLM_ENFORCE_EAGER=1
# export NANOVLLM_MODEL=/work/models/QwQ-32B/
# export NANOVLLM_TP=8
# export NANOVLLM_MAX_MODEL_LEN=32768
# export NANOVLLM_MAX_BATCHED_TOKENS=32768
# export NANOVLLM_SERVED_MODEL_NAME=QwQ-32B
# uvicorn serve:app --host 0.0.0.0 --port 8000

# --enforce-eager \
python3 ${SCRIPT_DIR}/../serve.py \
  --model /work/models/QwQ-32B/ \
  --tp 8 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --served-model-name QwQ-32B
