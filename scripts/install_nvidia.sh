#!/usr/bin/env bash
set -euo pipefail
QWEN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_PROJECT_ROOT"
source scripts/env_nvidia.sh
poetry env use "${QWEN_PYTHON:-python3.11}"
poetry install --only main
poetry run python -m pip install --index-url https://pypi.org/simple uv
poetry run uv pip install --index-url https://pypi.org/simple \
    -r "deploy/nvidia/$QWEN_NVIDIA_ENV.txt"
if [[ "$QWEN_NVIDIA_ENV" == "runtime" ]]; then
    poetry run python scripts/patch_nvidia.py
fi
poetry run python -m pip check
poetry run python -m pip freeze > "$QWEN_NVIDIA_ROOT/$QWEN_NVIDIA_ENV/installed.txt"
