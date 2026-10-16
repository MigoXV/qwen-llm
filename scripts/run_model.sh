#!/usr/bin/env bash
set -eo pipefail
QWEN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_PROJECT_ROOT"
source "$QWEN_PROJECT_ROOT/scripts/env_ascend_291.sh"
export VLLM_CACHE_ROOT="$QWEN_RUNTIME_ROOT/vllm-cache/llm"
mkdir -p "$VLLM_CACHE_ROOT"
# 省略模型名时先补齐，避免CLI把--port等选项的值识别为模型名。
if [[ $# -eq 0 || "$1" == -* ]]; then
    set -- "${QWEN_MODEL_PROFILE:-qwen3.8-27b-mtp2}" "$@"
fi
exec poetry run qwen-llm serve-model "$@"
