"""离线解析 Ascend trace；统计区分累计设备工作量和并行区间并集。"""
from collections import Counter, defaultdict
from bisect import bisect_right
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from statistics import median


def interval_union(intervals):
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def module_group(name):
    value = name.lower()
    compact = value.replace("_", "").replace(" ", "")
    if "weightquantbatchmatmul" in compact:
        return "FFN INT4 antiquant matmul"
    if "groupedmatmul" in compact:
        return "FFN INT4 GMM"
    if any(key in value for key in ("gated_delta", "delta_rule", "chunk_", "solve_tril", "fused_recurrent", "causal_conv", "sigmoid_gating", "recompute_w_u", "l2norm_fwd")) or "causalconv" in compact:
        return "GDN core / convolution"
    if any(key in compact for key in ("flashattention", "pagedattention", "increflash", "fusedinferattention")):
        return "full attention"
    if "dynamicquant" in compact:
        return "dynamic activation quantization"
    if "quantbatchmatmul" in compact:
        return "INT8 projections"
    if any(key in value for key in ("rmsnorm", "rms_norm", "layernorm", "layer_norm")):
        return "normalization / gate"
    if "matmul" in compact or "gemm" in compact:
        return "floating matmul (projection / lm_head / draft)"
    if any(key in value for key in ("memcpy", "memset", "transdata", "contiguous", "cast", "index")):
        return "copy / layout / indexing"
    return "other / unassigned"


class ScopeIndex:
    """按线程查询包含某时刻的最内层区间，支持嵌套 scope。"""
    def __init__(self, events):
        self.threads = {}
        groups = defaultdict(list)
        for event in events:
            groups[event.get("tid")].append(event)
        for tid, rows in groups.items():
            rows.sort(key=lambda e: (e["ts"], -e["dur"]))
            ends, maximum = [], float("-inf")
            for row in rows:
                maximum = max(maximum, row["ts"] + row["dur"])
                ends.append(maximum)
            self.threads[tid] = (rows, [e["ts"] for e in rows], ends)

    def find(self, tid, timestamp):
        if tid not in self.threads:
            return None
        rows, starts, ends = self.threads[tid]
        index = bisect_right(starts, timestamp) - 1
        while index >= 0 and ends[index] > timestamp:
            row = rows[index]
            if row["ts"] <= timestamp < row["ts"] + row["dur"]:
                return row
            index -= 1
        return None


def attribute_kernels(events, kernels, scopes, *, decode_only_graph=False):
    phase_index = ScopeIndex(e for e in scopes if e["name"].startswith(("qwen_step/", "qwen_sample/")))
    stage_index = ScopeIndex(e for e in scopes if not e["name"].startswith(("qwen_", "module/")))
    module_index = ScopeIndex(e for e in scopes if e["name"].startswith("module/"))
    dequeue_index = ScopeIndex(e for e in events if e.get("cat") == "dequeue" and e.get("ph") == "X")

    def host_label(tid, timestamp):
        phase = phase_index.find(tid, timestamp)
        stage = stage_index.find(tid, timestamp)
        module = module_index.find(tid, timestamp)
        return {"phase": phase["name"].split("/")[2] if phase else "unassigned",
                "stage": stage["name"] if stage else "unassigned",
                "module": module["name"] if module else None}

    queue_labels = {e["args"]["correlation_id"]: host_label(e.get("tid"), e["ts"])
                    for e in events if e.get("cat") == "enqueue" and e.get("ph") == "X"}
    async_starts = {e["id"]: host_label(e.get("tid"), e["ts"])
                    for e in events if e.get("cat") == "async_npu" and e.get("ph") == "s"}
    device_labels = {(e["pid"], e.get("tid"), e["ts"]): async_starts[e["id"]]
                     for e in events if e.get("cat") == "async_npu" and e.get("ph") == "f" and e["id"] in async_starts}
    connection_labels = {}
    for event in events:
        if event.get("cat") != "HostToDevice" or event.get("ph") != "s":
            continue
        dequeue = dequeue_index.find(event.get("tid"), event["ts"])
        label = queue_labels.get(dequeue["args"]["correlation_id"]) if dequeue else None
        connection_labels[int(event["id"]) >> 32] = label or host_label(event.get("tid"), event["ts"])

    by_phase, by_stage, by_module = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for kernel in kernels:
        args = kernel.get("args", {})
        model_id = args.get("Model Id")
        if decode_only_graph and model_id is not None and model_id != 4294967295:
            # 本诊断固定 FULL_DECODE_ONLY，草稿 enforce_eager=True；图内工作属于 target decode。
            label = {"phase": "decode", "stage": "target_graph", "module": None}
        else:
            label = device_labels.get((kernel["pid"], kernel.get("tid"), kernel["ts"]))
            if label is None or label["phase"] == "unassigned":
                label = connection_labels.get(args.get("connection_id"), label)
            label = label or {"phase": "unassigned", "stage": "unassigned", "module": None}
        kernel["attribution"] = label
        by_phase[label["phase"]][kernel["name"]] += kernel["dur"]
        by_stage[label["phase"] + "/" + label["stage"]][kernel["name"]] += kernel["dur"]
        if label["module"]:
            by_module[label["module"]][kernel["name"]] += kernel["dur"]
    phase_timing = {}
    for phase in by_phase:
        selected = [k for k in kernels if k["attribution"]["phase"] == phase]
        span = max(k["ts"] + k["dur"] for k in selected) - min(k["ts"] for k in selected)
        phase_timing[phase] = {"span_us": span,
            "busy_union_us": interval_union((k["ts"], k["ts"] + k["dur"]) for k in selected)}
    return {"phase_operators_us": {k: dict(v.most_common()) for k, v in by_phase.items()},
            "phase_timing": phase_timing,
            "stage_operators_us": {k: dict(v.most_common()) for k, v in by_stage.items()},
            "module_operators_us": {k: dict(v.most_common()) for k, v in by_module.items()}}


def kernel_shape_details(path, origin, kernels):
    csv_path = path.with_name("kernel_details.csv")
    if not csv_path.exists():
        return []
    details = {}
    with csv_path.open() as source:
        for row in csv.DictReader(source):
            at = float(Decimal(row["Start Time(us)"].strip()) - origin)
            details[(row["Task ID"], row["Stream ID"], round(at, 3))] = row
    buckets = defaultdict(lambda: {"calls": 0, "total_us": 0.0, "counter_sums": Counter()})
    for kernel in kernels:
        args = kernel.get("args", {})
        row = details.get((str(args.get("Task Id")), str(args.get("Physic Stream Id")), round(kernel["ts"], 3)))
        if row is None:
            continue
        label = kernel.get("attribution", {})
        key = (label.get("phase"), label.get("stage"), kernel["name"], row["Input Shapes"], row["Input Data Types"])
        bucket = buckets[key]
        bucket["calls"] += 1
        bucket["total_us"] += kernel["dur"]
        for name, value in row.items():
            if (name.endswith(("_ratio", "_rate", "(GB/s)", "(KB)", "bytes"))
                    or "bandwidth" in name.lower() or name == "cube_utilization(%)"):
                try:
                    bucket["counter_sums"][name] += float(value)
                except ValueError:
                    pass
    result = []
    for (phase, stage, name, shape, dtype), bucket in buckets.items():
        result.append({"phase": phase, "stage": stage, "name": name, "input_shapes": shape,
                       "input_dtypes": dtype, "calls": bucket["calls"], "total_us": bucket["total_us"],
                       "counter_means": {k: v / bucket["calls"] for k, v in bucket["counter_sums"].items()}})
    result.sort(key=lambda r: r["total_us"], reverse=True)
    return result


def host_costs(events, scopes, processes):
    phase_index = ScopeIndex(e for e in scopes if e["name"].startswith(("qwen_step/", "qwen_sample/")))
    cpu = [e for e in events if e.get("cat") == "cpu_op" and e.get("ph") == "X"]
    by_thread = defaultdict(list)
    for event in cpu:
        by_thread[event.get("tid")].append(event)
    totals = defaultdict(lambda: defaultdict(lambda: {"calls": 0, "self_host_us": 0.0, "inclusive_host_us": 0.0}))
    for rows in by_thread.values():
        rows.sort(key=lambda e: (e["ts"], -e["dur"]))
        stack, children = [], defaultdict(list)
        for index, event in enumerate(rows):
            while stack and rows[stack[-1]]["ts"] + rows[stack[-1]]["dur"] <= event["ts"]:
                stack.pop()
            if stack and event["ts"] + event["dur"] <= rows[stack[-1]]["ts"] + rows[stack[-1]]["dur"] + 0.01:
                children[stack[-1]].append((event["ts"], event["ts"] + event["dur"]))
            stack.append(index)
        for index, event in enumerate(rows):
            scope = phase_index.find(event.get("tid"), event["ts"])
            phase = scope["name"].split("/")[2] if scope else "unassigned"
            result = totals[phase][event["name"]]
            result["calls"] += 1
            result["inclusive_host_us"] += event["dur"]
            result["self_host_us"] += max(0, event["dur"] - interval_union(children[index]))
    apis = defaultdict(Counter)
    for event in events:
        if event.get("ph") != "X" or processes.get(event.get("pid")) != "CANN":
            continue
        scope = phase_index.find(event.get("tid"), event["ts"])
        phase = scope["name"].split("/")[2] if scope else "unassigned"
        apis[phase][event["name"]] += event["dur"]
    return {"cpu_ops": {phase: sorted([{"name": name, **data} for name, data in rows.items()],
                                     key=lambda row: row["self_host_us"], reverse=True)
                        for phase, rows in totals.items()},
            "cann_api_host_us": {phase: dict(values.most_common()) for phase, values in apis.items()}}


def analyze_trace(path: Path, *, decode_only_graph=False):
    data = json.loads(path.read_text())
    events = data["traceEvents"] if isinstance(data, dict) else data
    origin = min((Decimal(str(e["ts"])) for e in events if "ts" in e), default=Decimal(0))
    for event in events:
        if "ts" in event:
            event["ts"] = float(Decimal(str(event["ts"])) - origin)
        if "dur" in event:
            event["dur"] = float(event["dur"])
    processes = {e["pid"]: e.get("args", {}).get("name", "") for e in events
                 if e.get("name") == "process_name"}
    kernels, controls, scopes, categories = [], [], [], Counter()
    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        category = event.get("cat", "")
        categories[category] += 1
        process = str(processes.get(event.get("pid"), "")).lower().replace("_", " ")
        if category.lower() in {"kernel", "kernels", "aicore", "aicpu"} or "ascend hardware" in process:
            task_type = event.get("args", {}).get("Task Type", "")
            if task_type and task_type not in {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV", "AI_CPU",
                                              "SDMA_SQE", "PCIE_DMA_SQE"}:
                controls.append(event)
            else:
                kernels.append(event)
        elif event["name"].startswith(("qwen_step/", "qwen_sample/", "module/")) or event["name"] in {
            "prepare input", "forward", "post process", "sample_token", "draft_token", "async_state_update"}:
            scopes.append(event)
    by_name = defaultdict(lambda: {"calls": 0, "total_us": 0.0, "min_us": float("inf"), "max_us": 0.0})
    groups = Counter()
    for event in kernels:
        row = by_name[event["name"]]
        row["calls"] += 1
        row["total_us"] += event["dur"]
        row["min_us"] = min(row["min_us"], event["dur"])
        row["max_us"] = max(row["max_us"], event["dur"])
        groups[module_group(event["name"])] += event["dur"]
    work = sum(k["dur"] for k in kernels)
    start = min((k["ts"] for k in kernels), default=0)
    end = max((k["ts"] + k["dur"] for k in kernels), default=0)
    busy = interval_union((k["ts"], k["ts"] + k["dur"]) for k in kernels)
    top = [{"name": name, **row, "device_work_percent": 100 * row["total_us"] / work if work else 0,
            "group": module_group(name)} for name, row in by_name.items()]
    top.sort(key=lambda r: r["total_us"], reverse=True)
    scope_totals = defaultdict(lambda: {"calls": 0, "host_wall_us": 0.0})
    for scope in scopes:
        name = scope["name"]
        if name.startswith(("qwen_step/", "qwen_sample/")):
            parts = name.split("/")
            name = parts[0] + "/" + parts[2]
        scope_totals[name]["calls"] += 1
        scope_totals[name]["host_wall_us"] += scope["dur"]
    attribution = attribute_kernels(events, kernels, scopes, decode_only_graph=decode_only_graph)
    return {"path": str(path), "time_origin_us": str(origin), "device_event_count": len(kernels), "categories": dict(categories),
            "processes": list(processes.values()), "device_work_us": work,
            "device_span_us": end - start, "device_busy_union_us": busy,
            "device_idle_within_span_us": end - start - busy,
            "device_control_tasks": dict(Counter(e["name"] for e in controls)),
            "device_control_accumulated_us": sum(e["dur"] for e in controls),
            "groups_us": dict(groups), "top_operators": top, "host_scopes": dict(scope_totals),
            "attribution": attribution, "kernel_shapes": kernel_shape_details(path, origin, kernels),
            "host_costs": host_costs(events, scopes, processes),
            "valid_device_trace": bool(kernels)}


def analyze_round(http, frontend, worker):
    requests = []
    for sample in http["samples"]:
        if "response" not in sample:
            continue
        request_id = sample["response"]["id"]
        deliveries = [d for d in frontend["deliveries"] if d["request_id"] == request_id and d["new_tokens"] > 0]
        if not deliveries:
            continue
        first, last = deliveries[0], deliveries[-1]
        http_id = first.get("http_id")
        prepared = next((p for p in frontend["preparations"] if p.get("http_id") == http_id), None)
        server_http = next((p for p in frontend.get("http", []) if p["http_id"] == http_id), None)
        schedules = [s for s in worker["schedules"]
                     if any(rid == request_id or rid.startswith(request_id + "-") for rid in s["requests"])]
        row = {"request_id": request_id, "engine_delivery_batches": len(deliveries),
               "batch_tokens": dict(Counter(d["new_tokens"] for d in deliveries)),
               "engine_to_first_delivery_ms": (first["at_ns"] - first["submitted_ns"]) / 1e6,
               "client_finish_after_last_delivery_ms": (sample["end_ns"] - last["at_ns"]) / 1e6}
        if prepared:
            row["template_tokenize_ms"] = (prepared["end_ns"] - prepared["start_ns"]) / 1e6
        if server_http and prepared:
            row["http_before_template_ms"] = (prepared["start_ns"] - server_http["start_ns"]) / 1e6
            row["http_send_host_ms"] = sum(s["end_ns"] - s["start_ns"] for s in server_http["sends"]) / 1e6
        if schedules:
            row["engine_to_first_schedule_ms"] = (schedules[0]["start_ns"] - first["submitted_ns"]) / 1e6
            row["scheduler_host_ms"] = sum(s["end_ns"] - s["start_ns"] for s in schedules) / 1e6
        later_tokens = last["total_tokens"] - first["total_tokens"]
        if later_tokens:
            row["decode_ms_per_token"] = (last["at_ns"] - first["at_ns"]) / 1e6 / later_tokens
        gaps = [(right["at_ns"] - left["at_ns"]) / 1e6 for left, right in zip(deliveries, deliveries[1:])]
        if gaps:
            row["engine_batch_gap_ms_median"] = median(gaps)
            row["engine_batch_gap_ms_max"] = max(gaps)
        requests.append(row)
    device_phases = defaultdict(lambda: {"steps": 0, "forward_ms": 0.0, "sample_and_draft_ms": 0.0})
    execution = {}
    for step in worker["steps"]:
        group = execution.setdefault(step["phase"], {"steps": 0, "graph_modes": Counter(),
            "scheduled_tokens": Counter(), "padded_tokens": Counter(), "active_requests": Counter()})
        group["steps"] += 1
        group["graph_modes"][step.get("graph_mode", "unknown")] += 1
        group["scheduled_tokens"][sum(step.get("scheduled_tokens", {}).values())] += 1
        group["padded_tokens"][step.get("padded_tokens", 0)] += 1
        group["active_requests"][len(step.get("scheduled_tokens", {}))] += 1
        if "device_forward_ms" in step:
            group = device_phases[step["phase"]]
            group["steps"] += 1
            group["forward_ms"] += step["device_forward_ms"]
            group["sample_and_draft_ms"] += step.get("device_sample_and_draft_ms", 0)
    errors = [{"index": sample["index"], "error": sample["error"]}
              for sample in http["samples"] if "error" in sample]
    return {"valid": not errors, "errors": errors, "requests": requests,
            "cache_evictions_by_group": worker.get("cache_evictions_by_group"),
            "cache_eviction_examples": worker.get("cache_eviction_examples"),
            "cache_pools": worker.get("cache_pools"),
            "device_phases": dict(device_phases), "execution": execution}


def analyze_directory(root: Path):
    result = {"root": str(root), "traces": [], "cases": [],
              "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    decode_only_graph = bool(manifest and manifest["config"].get("runner") == "aclgraph"
                             and manifest["config"].get("graph_mode") == "FULL_DECODE_ONLY")
    modules_path = root / "modules.json"
    modules = json.loads(modules_path.read_text())[0] if modules_path.exists() else []
    w4_layers = sum((row.get("role", "target") == "target"
        and ".mlp." in row["name"] and row["parameters"].get("weight", {}).get("dtype") == "torch.int32")
        for row in modules)
    for path in sorted(root.rglob("trace_view.json")):
        trace = analyze_trace(path, decode_only_graph=decode_only_graph)
        backend = manifest["config"].get("w4a8_linear_backend") if manifest else None
        if w4_layers and backend in {"upstream", "gmm"}:
            case = next((c for c in manifest["cases"] if path.parents[1].name.startswith(c["name"] + "-trace")), None)
            worker_path = root / case["name"] / "trace-worker.json" if case else None
            if worker_path and worker_path.exists():
                steps = json.loads(worker_path.read_text())[0]["steps"]
                forwards = sum(bool(sum(s.get("scheduled_tokens", {}).values())) for s in steps)
                expected = w4_layers * forwards
                operator = "groupedmatmul" if backend == "gmm" else "weightquantbatchmatmul"
                observed = sum(op["calls"] for op in trace["top_operators"]
                               if operator in op["name"].lower().replace("_", ""))
                trace["coverage"] = {"target_forwards": forwards, "w4_layers": w4_layers,
                    "expected_matmuls": expected, "observed_matmuls": observed, "complete": expected == observed}
        result["traces"].append(trace)
    if manifest is not None:
        for case in manifest["cases"]:
            rounds = case.get("rounds", [])
            row = {"name": case["name"], "rounds": len(rounds)}
            if rounds:
                row.update(tps_median=median(r["tokens_per_second"] for r in rounds),
                    tps_min=min(r["tokens_per_second"] for r in rounds),
                    tps_max=max(r["tokens_per_second"] for r in rounds),
                    ttft_ms_median=median(t for r in rounds for t in r["ttft_ms"]),
                    cached_tokens=[r["cached_tokens"] for r in rounds])
                drafted = sum(r["draft_tokens"] for r in rounds)
                row["acceptance_rate"] = sum(r["accepted_tokens"] for r in rounds) / drafted if drafted else None
            result["cases"].append(row)
            details = {}
            for path in sorted((root / case["name"]).glob("*-http.json")):
                label = path.name.removesuffix("-http.json")
                worker_path, frontend_path = (path.with_name(label + suffix) for suffix in ("-worker.json", "-frontend.json"))
                try:
                    workers = json.loads(worker_path.read_text())
                    frontend = json.loads(frontend_path.read_text())
                    details[label] = analyze_round(json.loads(path.read_text()), frontend, workers[0])
                except (OSError, json.JSONDecodeError) as exc:
                    details[label] = {"valid": False, "incomplete_artifacts": str(exc)}
            row["details"] = details
    (root / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def summarize_profiles(root: Path):
    """重新解析旧采集并汇总多组独立实例；保留失败记录及原始 manifest。"""
    runs = []
    lines = ["# Qwen3.8 热点诊断原始结果汇总", "",
        "吞吐来自关闭 profiler 的三轮测量；设备百分比按累计计算与搬运工作量统计，等待任务单列。", ""]
    for manifest_path in sorted(root.rglob("manifest.json")):
        path = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        analysis = analyze_directory(path)
        traces = []
        for trace in analysis["traces"]:
            work = trace["device_work_us"]
            phases = trace["attribution"]["phase_operators_us"]
            traces.append({"path": trace["path"], "device_work_us": work,
                "coverage": trace.get("coverage"),
                "device_span_us": trace["device_span_us"], "device_busy_union_us": trace["device_busy_union_us"],
                "phase_coverage_percent": 100 * (1 - sum(phases.get("unassigned", {}).values()) / work) if work else 0,
                "phases": {phase: {"work_us": sum(ops.values()),
                    "top10": [{"name": name, "total_us": value, "percent": 100 * value / sum(ops.values())}
                              for name, value in list(ops.items())[:10]]}
                    for phase, ops in phases.items()}, "phase_timing": trace["attribution"]["phase_timing"],
                "stages_us": {key: sum(value.values()) for key, value in trace["attribution"]["stage_operators_us"].items()},
                "kernel_shapes": trace["kernel_shapes"]})
        runs.append({"name": str(path.relative_to(root)), "path": str(path),
                     "valid": manifest["valid"], "released": manifest.get("released"),
                     "error": manifest.get("error"), "config": manifest["config"],
                     "cases": analysis["cases"], "traces": traces})
        lines.extend(["## " + str(path.relative_to(root)), "",
            f"valid={manifest['valid']}，设备已释放={manifest.get('released')}。", "",
            "| 场景 | TPS 中位数 | TPS 范围 | TTFT 中位数/ms | MTP 接受率 |",
            "| --- | ---: | --- | ---: | ---: |"])
        for case in analysis["cases"]:
            if "tps_median" in case:
                rate = case["acceptance_rate"]
                lines.append(f"| {case['name']} | {case['tps_median']:.2f} | "
                    f"{case['tps_min']:.2f}–{case['tps_max']:.2f} | {case['ttft_ms_median']:.2f} | "
                    + (f"{rate:.1%}" if rate is not None else "—") + " |")
        lines.append("")
        for trace in traces:
            lines.extend(["### " + Path(trace["path"]).parents[1].name, "",
                          f"阶段归因覆盖设备工作量 {trace['phase_coverage_percent']:.2f}%。", ""])
            for phase, values in trace["phases"].items():
                lines.extend([f"{phase}：累计 {values['work_us'] / 1000:.2f} ms。", "",
                              "| 算子 | 累计设备/ms | 阶段占比 |", "| --- | ---: | ---: |"])
                for op in values["top10"]:
                    lines.append(f"| {op['name']} | {op['total_us'] / 1000:.3f} | {op['percent']:.2f}% |")
                lines.append("")
    result = {"root": str(root), "runs": runs}
    (root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    return result
