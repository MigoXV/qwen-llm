#!/usr/bin/env bash
# 供启动器 source；CANN 与缓存全部使用数据盘独立目录。
export QWEN_RUNTIME_ROOT="${QWEN_RUNTIME_ROOT:-/home/qwen-upgrade-20261015}"
export QWEN_MODELS_ROOT="${QWEN_MODELS_ROOT:-/home/qwen-models}"
export POETRY_CACHE_DIR="$QWEN_RUNTIME_ROOT/poetry-cache"
export POETRY_VIRTUALENVS_PATH="$QWEN_RUNTIME_ROOT/virtualenvs"
unset VIRTUAL_ENV PYTHONPATH ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
source "$QWEN_RUNTIME_ROOT/cann/cann-9.1.1/set_env.sh"
source "$QWEN_RUNTIME_ROOT/cann/nnal/atb/set_env.sh" --cxx_abi=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PIP_CACHE_DIR="$QWEN_RUNTIME_ROOT/pip-cache"
export TMPDIR="$QWEN_RUNTIME_ROOT/tmp"
export XDG_CACHE_HOME="$QWEN_RUNTIME_ROOT/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export ASCEND_CACHE_PATH="$QWEN_RUNTIME_ROOT/ascend-cache"
export ASCEND_WORK_PATH="$QWEN_RUNTIME_ROOT/ascend-work"
export ASCEND_PROCESS_LOG_PATH="$QWEN_RUNTIME_ROOT/ascend-logs"
export HF_HOME="$QWEN_MODELS_ROOT/huggingface-cache"
export MODELSCOPE_CACHE="$QWEN_MODELS_ROOT/modelscope-cache"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$ASCEND_CACHE_PATH" "$ASCEND_WORK_PATH" \
    "$ASCEND_PROCESS_LOG_PATH" "$HF_HOME" "$MODELSCOPE_CACHE"
