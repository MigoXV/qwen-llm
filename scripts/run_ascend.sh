#!/usr/bin/env bash
# 本机验证使用的昇腾运行环境与 W8A8 配置；额外 Typer 参数放在最后覆盖默认值。
set -eo pipefail
QWEN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${QWEN_RUNTIME:-cann911}" in
    cann911) exec bash "$QWEN_PROJECT_ROOT/scripts/run_ascend_291.sh" "$@" ;;
    legacy) ;;
    *) echo 'QWEN_RUNTIME must be cann911 or legacy' >&2; exit 2 ;;
esac
QWEN_BUILD_ROOT="${QWEN_BUILD_ROOT:-/home/qwen-llm-build}"
QWEN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export POETRY_CACHE_DIR="$QWEN_BUILD_ROOT/poetry-cache"
export POETRY_VIRTUALENVS_PATH="$QWEN_BUILD_ROOT/virtualenvs"
# 不使用调用者可能激活的旧环境；只选择上述独立 Poetry 环境。
unset VIRTUAL_ENV
cd "$QWEN_PROJECT_ROOT"
QWEN_CXX_ABI="$(poetry run python -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh "--cxx_abi=$QWEN_CXX_ABI"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE="${VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE:-0}"
export VLLM_ASCEND_ENABLE_PREFETCH_MLP="${VLLM_ASCEND_ENABLE_PREFETCH_MLP:-0}"
export VLLM_CACHE_ROOT="$QWEN_BUILD_ROOT/vllm-cache"
export ASCEND_PROCESS_LOG_PATH="$QWEN_BUILD_ROOT/ascend-logs"
export TMPDIR="$QWEN_BUILD_ROOT/tmp"
mkdir -p "$VLLM_CACHE_ROOT" "$ASCEND_PROCESS_LOG_PATH" "$TMPDIR"
exec poetry run qwen-llm serve \
    --model-path "$QWEN_PROJECT_ROOT/model-bin/Qwen3-14B-W8A8-Ascend" \
    --port 8100 --dtype bfloat16 --batch-tokens 2048 --kv-cache-gib 10.25 --graph-mode PIECEWISE "$@"
