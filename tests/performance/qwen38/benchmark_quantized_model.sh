#!/usr/bin/env bash
# 单模型独占测试：真实服务、固定能力题、8K检索、短/长输入TPS，异常同样保留。
set -eo pipefail
QWEN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$QWEN_PROJECT_ROOT"
if [[ $# -lt 3 ]]; then
    echo 'Usage: benchmark_quantized_model.sh MODEL_NAME MODEL_PATH OUTPUT_DIR [serve options...]' >&2
    exit 2
fi
QWEN_SUITE_MODEL="$1"
QWEN_SUITE_MODEL_PATH="$(realpath "$2")"
QWEN_SUITE_OUTPUT="$(realpath -m "$3")"
shift 3
source "$QWEN_PROJECT_ROOT/scripts/env_ascend_291.sh"
QWEN_SUITE_UNIT=qwen-llm-model-benchmark
if [[ -e "$QWEN_SUITE_OUTPUT/health.json" ]]; then
    echo 'Output already contains a run; use a fresh directory.' >&2
    exit 2
fi
poetry run python - <<'PY'
from tests.performance.qwen38.benchmark import inventory
assert not inventory()['owners'], 'NPU is already owned by another process'
PY
mkdir -p "$QWEN_SUITE_OUTPUT/logs"
poetry run qwen-llm doctor --model-path "$QWEN_SUITE_MODEL_PATH" > "$QWEN_SUITE_OUTPUT/doctor.json"
poetry run python - "$QWEN_SUITE_OUTPUT/doctor.json" <<'PY'
import json, sys
with open(sys.argv[1]) as source:
    report = json.load(source)
assert report['quantization'] in ('ascend', 'compressed-tensors'), 'This suite only loads quantized weights'
PY
QWEN_SUITE_STARTED="$(date '+%Y-%m-%d %H:%M:%S')"
QWEN_SUITE_OWNED=false
cleanup() {
    QWEN_SUITE_EXIT=$?
    trap - EXIT
    set +e
    if [[ "$QWEN_SUITE_OWNED" == true ]]; then
        journalctl -u "$QWEN_SUITE_UNIT" --since "$QWEN_SUITE_STARTED" --no-pager > "$QWEN_SUITE_OUTPUT/service.log"
        systemctl stop "$QWEN_SUITE_UNIT" > "$QWEN_SUITE_OUTPUT/stop.log" 2>&1
        QWEN_SUITE_STOP_CODE=$?
        if [[ "$QWEN_SUITE_STOP_CODE" -ne 0 && "$QWEN_SUITE_EXIT" -eq 0 ]]; then
            QWEN_SUITE_EXIT="$QWEN_SUITE_STOP_CODE"
        fi
    fi
    echo "$QWEN_SUITE_EXIT" > "$QWEN_SUITE_OUTPUT/suite-exit-code.txt"
    exit "$QWEN_SUITE_EXIT"
}
trap cleanup EXIT
systemctl reset-failed "$QWEN_SUITE_UNIT" 2>/dev/null || true
# 固定原有基准起点，避免服务默认MTP2参数渗入其他模型；本轮参数仍在最后覆盖。
systemd-run --unit="$QWEN_SUITE_UNIT" --property="WorkingDirectory=$QWEN_PROJECT_ROOT" \
    --property=KillMode=control-group --property=TimeoutStopSec=45 \
    /bin/bash "$QWEN_PROJECT_ROOT/scripts/run_ascend_291.sh" \
    --model-name "$QWEN_SUITE_MODEL" --model-path "$QWEN_SUITE_MODEL_PATH" \
    --graph-mode FULL --async-scheduling --npugraph-ex --batch-tokens 2048 \
    --kv-cache-gib 10.25 --speculative-tokens 0 "$@"
QWEN_SUITE_OWNED=true
QWEN_SUITE_READY=false
for QWEN_WAIT_INDEX in $(seq 1 450); do
    if curl --silent --fail --max-time 3 http://127.0.0.1:8100/health/ready > "$QWEN_SUITE_OUTPUT/health.partial"; then
        mv "$QWEN_SUITE_OUTPUT/health.partial" "$QWEN_SUITE_OUTPUT/health.json"
        QWEN_SUITE_READY=true
        break
    fi
    if systemctl is-failed --quiet "$QWEN_SUITE_UNIT"; then
        echo "Startup failed: $QWEN_SUITE_MODEL" >&2
        exit 1
    fi
    sleep 2
done
[[ "$QWEN_SUITE_READY" == true ]]
export QWEN_MODEL_PATH="$QWEN_SUITE_MODEL_PATH" QWEN_MODEL_NAME="$QWEN_SUITE_MODEL"
QWEN_SUITE_FAILURES=0
QWEN_SUITE_STAGES="${QWEN_SUITE_STAGES:-quality,retrieval,short,long}"
run_stage() {
    QWEN_STAGE_NAME="$1"
    shift
    echo "Stage $QWEN_STAGE_NAME: $QWEN_SUITE_MODEL"
    if "$@" > "$QWEN_SUITE_OUTPUT/logs/$QWEN_STAGE_NAME.log" 2>&1; then
        echo 0 > "$QWEN_SUITE_OUTPUT/$QWEN_STAGE_NAME-exit-code.txt"
    else
        QWEN_STAGE_CODE=$?
        echo "$QWEN_STAGE_CODE" > "$QWEN_SUITE_OUTPUT/$QWEN_STAGE_NAME-exit-code.txt"
        QWEN_SUITE_FAILURES=$((QWEN_SUITE_FAILURES+1))
    fi
    tail -n 1 "$QWEN_SUITE_OUTPUT/logs/$QWEN_STAGE_NAME.log"
}
if [[ ",$QWEN_SUITE_STAGES," == *",quality,"* ]]; then
    run_stage quality poetry run python -m tests.performance.qwen38.cli evaluate --model-name "$QWEN_SUITE_MODEL" \
        --math-dataset "$QWEN_MODELS_ROOT/evaluation/gsm8k-test.jsonl" --output "$QWEN_SUITE_OUTPUT/quality.json"
fi
# 先用四路不同8K材料做检索，兼作长输入形状预热；测速材料不同，不会命中这些前缀。
if [[ ",$QWEN_SUITE_STAGES," == *",retrieval,"* ]]; then
    run_stage retrieval poetry run python -m tests.performance.qwen38.evaluate_long_context --output "$QWEN_SUITE_OUTPUT/retrieval.json"
    run_stage retrieval_mixed poetry run python -m tests.performance.qwen38.evaluate_long_context --material mixed --output "$QWEN_SUITE_OUTPUT/retrieval_mixed.json"
fi
if [[ ",$QWEN_SUITE_STAGES," == *",short,"* ]]; then
    run_stage short poetry run python -m tests.performance.qwen38.cli benchmark --model-name "$QWEN_SUITE_MODEL" \
        --prompt '请连续列出至少100个常见汉语成语，每个成语附一句简短解释，不要省略，不要提前结束。' \
        --output "$QWEN_SUITE_OUTPUT/short.json"
fi
if [[ ",$QWEN_SUITE_STAGES," == *",long,"* ]]; then
    QWEN_BUILD_ROOT="$QWEN_SUITE_OUTPUT" QWEN_LONG_TAG=steady QWEN_LONG_ROUNDS="${QWEN_SUITE_LONG_ROUNDS:-7}" \
        run_stage long poetry run python -m tests.performance.qwen38.benchmark_long_context
fi
exit "$QWEN_SUITE_FAILURES"
