#!/usr/bin/env bash
# 显式使用数据盘Torch 2.9.1环境；保留 QWEN_RUNTIME=legacy 回退入口。
set -eo pipefail
QWEN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_PROJECT_ROOT"
source "$QWEN_PROJECT_ROOT/scripts/env_ascend_291.sh"
export VLLM_CACHE_ROOT="$QWEN_RUNTIME_ROOT/vllm-cache/llm"
mkdir -p "$VLLM_CACHE_ROOT"
exec poetry run qwen-llm serve --port 8100 --graph-mode FULL --async-scheduling \
    --npugraph-ex --batch-tokens 2048 --kv-cache-gib 10.25 "$@"
