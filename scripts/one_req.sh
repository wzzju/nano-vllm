#!/usr/bin/env bash

curl http://localhost:8000/health
printf "\n"

curl http://localhost:8000/v1/models
printf "\n"

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "QwQ-32B",
    "messages": [{"role": "user", "content": "你好，简单自我介绍"}],
    "max_tokens": 256,
    "temperature": 0.6
  }'
printf "\n"

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "QwQ-32B",
    "prompt": "请解释一下量子纠缠",
    "max_tokens": 256,
    "temperature": 0.6
  }'
printf "\n"

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "QwQ-32B",
    "prompt": ["重庆是", "列出1到10的质数"],
    "max_tokens": 128,
    "temperature": 0.6
  }'
printf "\n"
