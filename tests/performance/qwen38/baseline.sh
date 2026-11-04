#!/usr/bin/env bash
# 独占设备重测当前生产优化版；无论成功失败都恢复正式服务。
set -euo pipefail
report_root="${1:?用法：baseline.sh 新的绝对输出目录}"
# 将完整脚本作为一次性的命令字符串执行，避免长任务期间源码修改影响Bash后续读取。
if [[ "${2:-}" != '--immutable' ]]; then
    project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    exec /bin/bash -c "$(cat -- "${BASH_SOURCE[0]}")" "$0" "$report_root" --immutable "$project_root"
fi
project_root="${3:?缺少项目目录}"
[[ "$report_root" == /* && ! -e "$report_root" ]] || { echo '输出目录必须是不存在的绝对路径' >&2; exit 2; }
mkdir -p "$report_root"
printf '%s\n' "$BASH_EXECUTION_STRING" > "$report_root/launcher.sh"
cd "$project_root"
# CANN的官方环境脚本会读取未设置的可选变量，不能在nounset下加载。
set +u
source scripts/env_ascend_291.sh
set -u
restore_service() {
    result=$?
    trap - EXIT
    systemctl start qwen-llm.service || result=1
    ready=0
    for ((attempt=0; attempt<900; attempt++)); do
        if curl --fail --silent --max-time 3 http://127.0.0.1:8100/health/ready > "$report_root/restored-ready.json"; then
            ready=1
            break
        fi
        sleep 1
    done
    [[ "$ready" == 1 ]] || result=1
    printf '%s\n' "$result" > "$report_root/exit-code.txt"
    exit "$result"
}
git rev-parse HEAD > "$report_root/revision.txt"
git diff --binary > "$report_root/implementation.patch"
systemctl is-active qwen-llm.service > "$report_root/original-service-state.txt" || true
trap restore_service EXIT
systemctl stop qwen-llm.service
# 驱动内存回收可能比进程退出晚，设备状态由runner再次核验。
sleep 5
npu-smi info > "$report_root/npu-before.txt"
poetry run python -m tests.performance.qwen38.cli profile \
  --catalog configs/models.json --model-name qwen3.8-27b-optimized-v1 \
  --kv-cache-gib 22.5 --suite full --rounds 5 --no-capture \
  --quality-samples 1319 --long-validation --cache-boundary-validation \
  --prefix-growth-validation --stability-seconds 3600 \
  --output-dir "$report_root/baseline" > "$report_root/baseline.log" 2>&1
