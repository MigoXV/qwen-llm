#!/usr/bin/env bash
# 基线完成后才运行独占算子实验；保存退出状态并恢复正式服务。
set -eo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
report_root="${1:?报告根目录}"
baseline_unit="${2:?基线systemd单元}"
library="${3:?编译后的动态库绝对路径}"
cd "$project_root"
source scripts/env_ascend_291.sh
while systemctl is-active --quiet "$baseline_unit"; do sleep 5; done
poetry run python - "$report_root" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);m=json.loads((root/'baseline/manifest.json').read_text())
if not m.get('valid') or not m.get('released') or len(m.get('cases',[]))!=23:
    raise SystemExit('基线未完整通过，拒绝把算子结果当成正式对照')
PY
restore() {
    result=$?
    trap - EXIT
    systemctl start qwen-llm.service || result=1
    ready=0
    for ((i=0; i<900; i++)); do
        if curl --fail --silent --max-time 3 http://127.0.0.1:8100/health/ready > "$report_root/after-micro-ready.json"; then ready=1; break; fi
        sleep 1
    done
    [[ "$ready" == 1 ]] || result=1
    printf '%s\n' "$result" > "$report_root/after-micro-exit.txt"
    exit "$result"
}
trap restore EXIT
systemctl stop qwen-llm.service
sleep 5
npu-smi info > "$report_root/before-micro-npu.txt"
poetry run python -m tests.performance.qwen38.cli profile \
    --catalog configs/models.json --model-name qwen3.8-27b-optimized-v1 \
    --kv-cache-gib 22.5 --suite trace --record-shapes \
    --cases short128-c1,chinese8192-c1-cold --output-dir "$report_root/current-trace" \
    > "$report_root/current-trace.log" 2>&1
sleep 5
set +e
TASK_QUEUE_ENABLE=0 poetry run python -m tests.performance.qwen38.verify_ascendc_w4a16 \
    --library "$library" --output "$report_root/micro" > "$report_root/micro.log" 2>&1
micro_status=$?
set -e
printf '%s\n' "$micro_status" > "$report_root/micro-exit.txt"
npu-smi info > "$report_root/after-micro-npu.txt"
# 退出1既可能是正常的门槛拒绝，也可能是异常；必须核对结果文件，不能吞掉异常。
[[ -f "$report_root/micro/result.json" ]] || exit "$micro_status"
poetry run python -m tests.performance.qwen38.summarize_ascendc "$report_root" > "$report_root/comparison.log" 2>&1
