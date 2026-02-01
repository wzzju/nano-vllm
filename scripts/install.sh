#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn==2.8.3 --no-build-isolation

pip install -e ${SCRIPT_DIR}/.. --no-build-isolation
