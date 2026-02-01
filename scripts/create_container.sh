#!/usr/bin/env bash

sudo sysctl fs.inotify.max_user_watches=99999

image=vllm/vllm-openai:v0.11.2
name=vllm-ref

docker run -dit --name $name \
    --entrypoint /bin/bash \
    --gpus all \
    --privileged \
    --security-opt=seccomp=unconfined \
    --ulimit core=-1 --ulimit memlock=-1 --ulimit stack=67108864  \
    --net=host --uts=host --ipc=host \
    -v /ssd1/:/ssd1 \
    -v /ssd2/:/ssd2 \
    -v /ssd3/:/ssd3 \
    -v /ssd4/:/ssd4 \
    -v /ssd3/wangzhen31/work:/work \
    -w /work \
    --restart=always \
    --shm-size=64g \
    --tmpfs /tmp:exec,size=64g \
    $image