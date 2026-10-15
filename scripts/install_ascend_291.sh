#!/usr/bin/env bash
# 在独立 Poetry 环境重建候选栈；须已安装数据盘 CANN 9.1.1 + 910B ops + NNAL。
set -eo pipefail
QWEN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$QWEN_PROJECT_ROOT/scripts/env_ascend_291.sh"
QWEN_TARGET_PROJECT="${1:-$QWEN_PROJECT_ROOT}"
cd "$QWEN_TARGET_PROJECT"
poetry env use python3.10
poetry install
QWEN_CONSTRAINTS="$QWEN_PROJECT_ROOT/deploy/ascend-291/constraints.txt"
mkdir -p "$QWEN_RUNTIME_ROOT/src" "$QWEN_RUNTIME_ROOT/wheels"
for QWEN_SOURCE_NAME in vllm vllm-ascend; do
    if [[ "$QWEN_SOURCE_NAME" == vllm ]]; then
        QWEN_SOURCE_DIR="$QWEN_RUNTIME_ROOT/src/vllm018"
        QWEN_SOURCE_SHA=bcf2be96120005e9aea171927f85055a6a5c0cf6
    else
        QWEN_SOURCE_DIR="$QWEN_RUNTIME_ROOT/src/ascend018"
        QWEN_SOURCE_SHA=e18643f8a4d5bd9990727654318ad069ea0b56e2
    fi
    if [[ ! -d "$QWEN_SOURCE_DIR/.git" ]]; then
        git clone --depth 1 --branch v0.18.0 "https://github.com/vllm-project/$QWEN_SOURCE_NAME.git" "$QWEN_SOURCE_DIR"
    fi
    [[ "$(git -C "$QWEN_SOURCE_DIR" rev-parse HEAD)" == "$QWEN_SOURCE_SHA" ]]
    QWEN_PATCH="$QWEN_PROJECT_ROOT/deploy/ascend-291/patches/$QWEN_SOURCE_NAME.patch"
    if ! git -C "$QWEN_SOURCE_DIR" apply --reverse --check "$QWEN_PATCH" 2>/dev/null; then
        git -C "$QWEN_SOURCE_DIR" apply --check "$QWEN_PATCH"
        git -C "$QWEN_SOURCE_DIR" apply "$QWEN_PATCH"
    fi
done
# 上游 PR #14618：language_model_only 的预热与实际请求必须使用相同输入路径。
QWEN_MM_PATCH="$QWEN_PROJECT_ROOT/deploy/ascend-291/patches/language-model-only.patch"
if ! git -C "$QWEN_RUNTIME_ROOT/src/ascend018" apply --reverse --check "$QWEN_MM_PATCH" 2>/dev/null; then
    git -C "$QWEN_RUNTIME_ROOT/src/ascend018" apply --check "$QWEN_MM_PATCH"
    git -C "$QWEN_RUNTIME_ROOT/src/ascend018" apply "$QWEN_MM_PATCH"
fi
# 先安装基础 Triton，最后覆盖 Ascend 实现，避免共用 triton 目录被覆盖。
env -u PYTHONPATH poetry run python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
env -u PYTHONPATH poetry run python -m pip install --index-url https://pypi.org/simple \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    -c "$QWEN_CONSTRAINTS" torch-npu==2.9.1 triton-ascend==3.2.2 \
    setuptools==79.0.1 setuptools-scm cmake ninja wheel pybind11
env -u PYTHONPATH poetry run python -m pip install --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --force-reinstall --no-deps triton-ascend==3.2.2
SETUPTOOLS_SCM_PRETEND_VERSION=0.18.0 VLLM_TARGET_DEVICE=empty \
    poetry run python -m pip install --index-url https://pypi.org/simple \
    -c "$QWEN_CONSTRAINTS" --no-build-isolation -e "$QWEN_RUNTIME_ROOT/src/vllm018"
git -C "$QWEN_RUNTIME_ROOT/src/ascend018" submodule update --init --recursive
# CANN 的后处理会修改构建产物，不能复用已完成的 build 目录重新链接。
if [[ -d "$QWEN_RUNTIME_ROOT/src/ascend018/build" ]]; then
    mv "$QWEN_RUNTIME_ROOT/src/ascend018/build" "$QWEN_RUNTIME_ROOT/build-backup-$(date +%s)"
fi
MAX_JOBS="${MAX_JOBS:-8}" SETUPTOOLS_SCM_PRETEND_VERSION=0.18.0+torch291.cann911 \
    poetry run python -m pip wheel --index-url https://pypi.org/simple --no-deps --no-build-isolation \
    -c "$QWEN_CONSTRAINTS" "$QWEN_RUNTIME_ROOT/src/ascend018" --wheel-dir "$QWEN_RUNTIME_ROOT/wheels"
env -u PYTHONPATH poetry run python -m pip install -c "$QWEN_CONSTRAINTS" \
    --index-url https://pypi.org/simple --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    "$QWEN_RUNTIME_ROOT/wheels/vllm_ascend-0.18.0+torch291.cann911-cp310-cp310-linux_aarch64.whl"
env -u PYTHONPATH poetry run python -m pip check
