#!/usr/bin/env bash
# 同权重、同64K上下文和KV预算的独占热点诊断；保留所有原始结果。
set -eo pipefail
QWEN_PROFILE_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$QWEN_PROFILE_PROJECT"
source scripts/env_ascend_291.sh
QWEN_PROFILE_ROOT="${1:?Usage: profile_qwen38.sh NEW_OUTPUT_ROOT}"
if [[ -e "$QWEN_PROFILE_ROOT" ]]; then
    echo 'Output root already exists; use a new directory.' >&2
    exit 2
fi
mkdir -p "$QWEN_PROFILE_ROOT"
run_profile_case() {
    QWEN_PROFILE_VARIANT="$1"
    shift
    poetry run python -m tests.performance.qwen38.cli profile --output-dir "$QWEN_PROFILE_ROOT/$QWEN_PROFILE_VARIANT" \
        --kv-cache-gib 22.5 "$@" > "$QWEN_PROFILE_ROOT/$QWEN_PROFILE_VARIANT.log" 2>&1
}
# 当前12.5GiB预算仅能容纳2.24路64K上下文；诊断统一用22.5GiB。
run_profile_case mtp2-full --suite full
run_profile_case mtp0-graph --suite comparison --speculative-tokens 0
run_profile_case mtp0-eager --suite comparison --speculative-tokens 0 --runner eager
run_profile_case mtp0-gmm --suite comparison --speculative-tokens 0 --w4a8-linear-backend gmm
run_profile_case module-map --suite trace --cases short128-c1 --speculative-tokens 0 \
    --runner eager --module-ranges --record-shapes
run_profile_case memory-counters --suite trace --cases short128-c1,chinese8192-c1-cold \
    --aic-metrics Memory
# 原报告16K配置复现，使用其原有预算，不和64K结果混做单变量归因。
run_profile_case historical-mtp2 --suite historical --max-input-tokens 8192 --kv-cache-gib 9.375
run_profile_case historical-graph --suite historical --model-name qwen3.8-27b --kv-cache-gib 6.5
# 最大输入热缓存出现失配后，补充相同预算下的分组驱逐与算子审计。
run_profile_case cache-long-audit --suite trace --cases chinese57344-c4-hot
# Memory的主存带宽字段与权重体积量级不一致，改用独立传输字节计数交叉核对。
run_profile_case memory-access-audit --suite trace --cases short128-c1,chinese8192-c1-cold \
    --aic-metrics MemoryAccess
