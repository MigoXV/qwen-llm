#!/usr/bin/env bash
set -eo pipefail
QWEN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_PROJECT_ROOT"
source "$QWEN_PROJECT_ROOT/scripts/env_ascend_291.sh"
export VLLM_CACHE_ROOT="$QWEN_RUNTIME_ROOT/vllm-cache/llm"
mkdir -p "$VLLM_CACHE_ROOT"
exec poetry run qwen-llm serve-model "$@"
