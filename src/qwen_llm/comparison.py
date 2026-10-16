"""按同负载原始数据比较升级环境，20% 是用户指定的回退门槛。"""
import json
from pathlib import Path
import statistics


def metric(component, case, name, before, after, higher_is_better=False):
    if before <= 0 or after <= 0:
        raise ValueError("比较指标必须为正数")
    degradation = (1 - after / before) if higher_is_better else (after / before - 1)
    return {"component": component, "case": case, "metric": name, "before": before,
            "after": after, "degradation_percent": 100 * degradation,
            "over_threshold": degradation > .20 + 1e-9}


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * q + .5))]


def compare(root: Path):
    rows = []
    read = lambda p: json.loads(p.read_text())
    a = read(root / "baseline/llm-short.json")
    b = read(root / "candidate/llm-short.json")
    if not a["valid"] or not b["valid"] or a["prompt"] != b["prompt"] or a["max_output_tokens"] != b["max_output_tokens"]:
        raise ValueError("LLM基准未通过独占验证或负载不一致")
    for concurrency in (1, 2, 4):
        old = [r for r in a["rounds"] if r["concurrency"] == concurrency]
        new = [r for r in b["rounds"] if r["concurrency"] == concurrency]
        for collection in (old, new):
            if len(collection) < 3 or any(s.get("http_status") != 200 for r in collection for s in r["samples"]):
                raise ValueError("LLM有效测量轮次不足或存在生成失败")
        label = f"short/concurrency={concurrency}"
        rows.append(metric("llm", label, "tokens/s", statistics.median(r["tokens_per_second"] for r in old),
                           statistics.median(r["tokens_per_second"] for r in new), True))
        for key in ("seconds", "ttft_seconds"):
            rows.append(metric("llm", label, key + "_p95",
                               percentile([s[key] for r in old for s in r["samples"]], .95),
                               percentile([s[key] for r in new for s in r["samples"]], .95)))
    long_data = [read(root / f"logs/long-{tag}-128.json") for tag in ("baseline", "candidate")]
    hot_rounds = []
    for data in long_data:
        rounds = [r for r in data["rounds"] if all(
            s.get("response", {}).get("usage", {}).get("input_tokens_details", {}).get("cached_tokens", 0) > 0
            for s in r["samples"])]
        if len(rounds) < 3 or data["input_tokens"] != 8192:
            raise ValueError("长输入需要至少三轮已命中缓存的同负载结果")
        hot_rounds.append(rounds)
    rows.append(metric("llm", "8192-input/hot/concurrency=4", "tokens/s",
                       statistics.median(r["tokens_per_second"] for r in hot_rounds[0]),
                       statistics.median(r["tokens_per_second"] for r in hot_rounds[1]), True))
    rows.append(metric("llm", "8192-input/hot/concurrency=4", "ttft_seconds_p95",
                       percentile([s["ttft_seconds"] for r in hot_rounds[0] for s in r["samples"]], .95),
                       percentile([s["ttft_seconds"] for r in hot_rounds[1] for s in r["samples"]], .95)))
    old_meta = read(root / "baseline/asr/metadata.json")
    new_meta = read(root / "candidate/asr/metadata.json")
    for key in ("workers", "inputs", "rounds", "clients", "requests_per_client", "min_requests", "warmup_rounds", "dtype"):
        if old_meta[key] != new_meta[key]:
            raise ValueError(f"ASR负载配置不一致：{key}")
    old_asr = read(root / "baseline/asr/summary.json")
    new_asr = {(r["duration_s"], r["clients"]): r for r in read(root / "candidate/asr/summary.json")}
    for old in old_asr:
        new = new_asr[(old["duration_s"], old["clients"])]
        if old["errors"] or new["errors"] or old["success"] != new["success"]:
            raise ValueError("ASR出现失败或完成请求数不一致")
        label = f"{old['duration_s']:.3f}s/clients={old['clients']}"
        for key in ("throughput_x_median", "first_text_s_p95", "latency_s_p95"):
            rows.append(metric("asr", label, key, old[key], new[key], key == "throughput_x_median"))
    result = {"threshold_percent": 20, "rows": rows,
              "requires_confirmation_or_rollback": any(r["over_threshold"] for r in rows)}
    (root / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ["# CANN 9.1.1 / Torch 2.9.1 环境对比", "",
             "吞吐下降超过20%或P95延迟增加超过20%的负载需额外复测；确认后回退。负值代表改善。", "",
             "| 项目 | 负载 | 指标 | 旧环境 | 新环境 | 退化幅度 | 超过20% |",
             "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    lines.extend(f"| {r['component']} | {r['case']} | {r['metric']} | {r['before']:.4f} | {r['after']:.4f} | {r['degradation_percent']:.1f}% | {'是' if r['over_threshold'] else '否'} |" for r in rows)
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return result
