#!/usr/bin/env bash
# 固定的 CANN 8.3.RC1 诊断基线；不修改系统 CANN 或原有虚拟环境。
set -euo pipefail
BUILD_ROOT="${QWEN_BUILD_ROOT:-/home/qwen-llm-build}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$BUILD_ROOT" "$BUILD_ROOT/logs"
export POETRY_CACHE_DIR="$BUILD_ROOT/poetry-cache"
export POETRY_VIRTUALENVS_PATH="$BUILD_ROOT/virtualenvs"
export PIP_CACHE_DIR="$BUILD_ROOT/pip-cache"
export PIP_DEFAULT_TIMEOUT=30
export PIP_RETRIES=1
export TMPDIR="$BUILD_ROOT/tmp"
mkdir -p "$TMPDIR"
cd "$PROJECT_ROOT"
poetry env use python3.10
poetry install
record_result() {
    INSTALL_EXIT_CODE=$?
    trap - EXIT
    env -u PYTHONPATH poetry run python -m pip freeze > "$BUILD_ROOT/logs/freeze.txt" || true
    poetry run qwen-llm doctor --probe-runtime > "$BUILD_ROOT/logs/doctor.json" || true
    echo "$INSTALL_EXIT_CODE" > "$BUILD_ROOT/logs/install-exit-code.txt"
    exit "$INSTALL_EXIT_CODE"
}
trap record_result EXIT
if [[ ! -d "$BUILD_ROOT/vllm/.git" ]]; then
    git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm.git "$BUILD_ROOT/vllm"
fi
if [[ ! -d "$BUILD_ROOT/vllm-ascend/.git" ]]; then
    git clone --depth 1 --branch v0.11.0rc1 https://github.com/vllm-project/vllm-ascend.git "$BUILD_ROOT/vllm-ascend"
fi
[[ "$(git -C "$BUILD_ROOT/vllm" rev-parse HEAD)" == "b8b302cde434df8c9289a2b465406b47ebab1c2d" ]]
[[ "$(git -C "$BUILD_ROOT/vllm-ascend" rev-parse HEAD)" == "c5fe179cef7f91096c5090bdd3de846cab92b736" ]]
git -C "$BUILD_ROOT/vllm" rev-parse HEAD > "$BUILD_ROOT/logs/vllm-revision.txt"
git -C "$BUILD_ROOT/vllm-ascend" rev-parse HEAD > "$BUILD_ROOT/logs/ascend-revision.txt"
cp "$PROJECT_ROOT/scripts/ascend-constraints.txt" "$BUILD_ROOT/logs/constraints.txt"
CONSTRAINTS="$PROJECT_ROOT/scripts/ascend-constraints.txt"
# Build isolation uses upstream's own build-time Torch; runtime constraints apply below.
# pip is confined to this Poetry environment; upstream builds need their own build metadata.
VLLM_TARGET_DEVICE=empty poetry run python -m pip install -v -c "$CONSTRAINTS" -e "$BUILD_ROOT/vllm" 2>&1 | tee "$BUILD_ROOT/logs/vllm-install.log"
poetry run python -m pip install -v -c "$CONSTRAINTS" "vllm-ascend==0.11.0rc1" 2>&1 | tee "$BUILD_ROOT/logs/ascend-install.log"
env -u PYTHONPATH poetry run python -m pip check 2>&1 | tee "$BUILD_ROOT/logs/pip-check.log"
git -C "$BUILD_ROOT/vllm" rev-parse HEAD > "$BUILD_ROOT/logs/vllm-revision.txt"
git -C "$BUILD_ROOT/vllm-ascend" rev-parse HEAD > "$BUILD_ROOT/logs/ascend-revision.txt"
env -u PYTHONPATH poetry run python -m pip freeze > "$BUILD_ROOT/logs/freeze.txt"
poetry run qwen-llm doctor
