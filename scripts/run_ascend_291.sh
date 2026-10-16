#!/usr/bin/env bash
# 显式使用数据盘Torch 2.9.1环境；保留 QWEN_RUNTIME=legacy 回退入口。
set -eo pipefail
QWEN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_PROJECT_ROOT"
source "$QWEN_PROJECT_ROOT/scripts/env_ascend_291.sh"
export VLLM_CACHE_ROOT="$QWEN_RUNTIME_ROOT/vllm-cache/llm"
mkdir -p "$VLLM_CACHE_ROOT"
# 默认使用已测的27B MTP2配置；QWEN_MODEL_PROFILE可选择其他目录配置。
exec poetry run qwen-llm serve-model "${QWEN_MODEL_PROFILE:-qwen3.8-27b-mtp2}" --port 8100 "$@"
