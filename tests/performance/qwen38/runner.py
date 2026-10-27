"""可复现的独立热点诊断；不安装依赖、不改权重或服务配置。"""
import asyncio
from collections import Counter
from contextvars import ContextVar
from dataclasses import asdict
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from tests.performance.qwen38.benchmark import inventory
from .config import ExperimentConfig as ServingConfig, ProfileConfig
from .engine import ExperimentEngine as V1Engine

logger = logging.getLogger(__name__)
http_context = ContextVar("qwen_profile_http", default=None)

SHORT = "请连续列出至少100个常见汉语成语，每个成语附一句简短解释，不要省略，不要提前结束。"
QA = "解释数据库事务隔离级别，逐一给出脏读、不可重复读、幻读的并发时序，并讨论工程中的选择。"
CODE = "用Python实现一个有容量上限和TTL的LRU缓存，包含清晰的接口、完整的类实现及使用示例，讨论边界情况。"
LONG_SUFFIX = "\n材料结束。请分条梳理材料中的项目进度、风险、负责人和后续行动，解释依据，尽量详细。"


class ProfileSuite(str, Enum):
    FULL = "full"
    COMPARISON = "comparison"
    HISTORICAL = "historical"
    SMOKE = "smoke"
    TRACE = "trace"


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def process_resources(pid):
    try:
        root = Path(f"/proc/{pid}")
        status = (root / "status").read_text()
        stat = (root / "stat").read_text().split(") ", 1)[1].split()
        rss = re.search(r"^VmRSS:\s+(\d+)", status, re.MULTILINE)
        return {"pid": int(pid), "cpu_seconds": (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK"),
                "rss_bytes": int(rss.group(1)) * 1024 if rss else None,
                "threads": int(stat[17]), "open_fds": len(list((root / "fd").iterdir()))}
    except (OSError, ValueError) as exc:
        return {"pid": int(pid), "error": str(exc)}


def config_from_catalog(catalog: Path, name: str, **overrides) -> ServingConfig:
    row = next((r for r in json.loads(catalog.read_text()) if r["name"].casefold() == name.casefold()), None)
    if row is None:
        raise ValueError(f"模型配置不存在: {name}")
    if row.get("max_concurrency", 4) != 4 or row.get("max_output_tokens", 8192) != 8192:
        raise ValueError("诊断沿用已验证的四路、8192输出容量")
    fields = ServingConfig.__dataclass_fields__
    values = {k: v for k, v in row.items() if k in fields}
    values.update(model_path=Path(row["path"]).resolve(), model_name=row["name"],
                  kv_cache_bytes=int(row["kv_cache_gib"] * 1024**3), timeout=1800)
    values.update({k: v for k, v in overrides.items() if v is not None})
    return ServingConfig(**values)


def make_long_prompt(tokenizer, length: int, index: int, namespace: str, *, repeated=False, historical=False) -> str:
    prefix = (f"独立测试材料编号 {index + 1}。以下是占位材料：\n" if historical
              else f"独立材料 {namespace} 线路{index}。请阅读以下工作记录：\n")
    suffix = LONG_SUFFIX
    count = lambda s: len(tokenizer.apply_chat_template([{"role": "user", "content": s}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False))
    if repeated:
        suffix = "\n材料结束。请从1开始连续逐行输出整数直到20000，每行一个数字，不解释，不省略，不要提前结束。"
        for n in range(max(0, length - count(prefix + suffix) - 4), length):
            candidate = prefix + " data" * n + suffix
            actual = count(candidate)
            if actual == length:
                return candidate
            if actual > length:
                break
        raise ValueError("无法构造精确长度的重复输入")
    # 不重复整段填充；每条记录包含不同编号、数值、负责人和工作事项。
    topics = ["交付验收", "仓储容量", "数据库迁移", "客户回访", "服务延迟", "回归测试", "采购预算", "数据核对"]
    corpus = "".join(
        f"记录{i:05d}：{topics[i % len(topics)]}由负责人{chr(0x4e00 + i % 200)}推进，"
        f"预算{(i * 7919 + index * 17) % 100003}元，完成{(i * 37) % 101}项，"
        f"校验号{hashlib.sha256(f'{namespace}/{index}/{i}'.encode()).hexdigest()[:12]}。"
        f"本轮检查发现第{(i * 13) % 97}批数据需要复核，计划在第{i % 28 + 1}日交付，"
        "风险是上游资料延迟，后续先核对样本再组织验收。\n"
        for i in range(length // 35 + 30))
    body = tokenizer.encode(corpus, add_special_tokens=False)
    cut = length - count(prefix + suffix)
    for _ in range(30):
        candidate = prefix + tokenizer.decode(body[:cut]) + suffix
        difference = count(candidate) - length
        if not difference:
            return candidate
        cut -= difference
    raise ValueError(f"无法构造精确 {length} token 输入")


def scenarios(suite: ProfileSuite, max_input: int) -> list[dict]:
    cases = [{"name": f"short128-c{c}", "concurrency": c, "output_tokens": 128,
              "kind": "short", "cache": "cold"}
             for c in ((1, 4) if suite is ProfileSuite.COMPARISON else (1, 2, 4))]
    if suite is ProfileSuite.SMOKE:
        return cases[:1]
    if suite is ProfileSuite.FULL:
        cases += [{"name": f"generation512-c{c}", "concurrency": c, "output_tokens": 512,
                   "kind": "short", "cache": "cold"} for c in (1, 2, 4)]
        cases += [{"name": kind + "-c1", "concurrency": 1, "output_tokens": 512,
                   "kind": kind, "cache": "cold"} for kind in ("qa", "code")]
    lengths = (8192, 32768, 57344) if suite in {ProfileSuite.FULL, ProfileSuite.TRACE} else (8192,)
    if suite is ProfileSuite.HISTORICAL:
        lengths = ()  # 原报告锚点使用原始四路data材料，不混入新的中文负载。
    for length in lengths:
        if length > max_input:
            continue
        for c in (1, 4):
            for cache in (("cold",) if suite is ProfileSuite.COMPARISON else ("cold", "hot")):
                cases.append({"name": f"chinese{length}-c{c}-{cache}", "concurrency": c,
                              "input_tokens": length, "output_tokens": 128,
                              "kind": "chinese", "cache": cache})
    if suite in (ProfileSuite.FULL, ProfileSuite.HISTORICAL):
        cases += [{"name": "repeated8192-c4-" + cache, "concurrency": 4,
                   "input_tokens": 8192, "output_tokens": 128,
                   "kind": "repeated", "cache": cache} for cache in ("cold", "hot")]
    if suite is ProfileSuite.FULL and max_input >= 32768:
        cases.append({"name": "mixed32k-c4", "concurrency": 4, "output_tokens": 512,
                      "kind": "mixed", "cache": "cold", "delays": [0, 0, 0, 1.5]})
    return cases


def summarize_round(client: dict, worker: dict) -> dict:
    samples = client["samples"]
    errors = [s for s in samples if "error" in s]
    elapsed = (max(s["end_ns"] for s in samples) - min(s["start_ns"] for s in samples)) / 1e9
    successful = [s for s in samples if "response" in s]
    return {"valid": not errors, "makespan_seconds": elapsed,
        "tokens_per_second": sum(s["response"]["usage"]["output_tokens"] for s in successful) / elapsed,
        "ttft_ms": [(s["first_text_ns"] - s["start_ns"]) / 1e6 for s in successful if s["first_text_ns"]],
        "cached_tokens": [s["response"]["usage"]["input_tokens_details"]["cached_tokens"] for s in successful],
        "input_tokens": [s["response"]["usage"]["input_tokens"] for s in successful],
        "output_tokens": [s["response"]["usage"]["output_tokens"] for s in successful],
        "graph_modes": dict(Counter(s.get("graph_mode", "unknown") for s in worker.get("steps", []))),
        "phases": dict(Counter(s["phase"] for s in worker.get("steps", []))),
        "draft_tokens": sum(s["num_draft_tokens"] for s in worker.get("spec_decoding", [])),
        "accepted_tokens": sum(s["num_accepted_tokens"] for s in worker.get("spec_decoding", []))}


class ObservedEngine(V1Engine):
    """仅记录 CPU 时间和引擎交付批次；不把 SSE chunk 当作单个 token。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_observations()

    def reset_observations(self):
        self.preparations, self.deliveries, self.http_events = [], [], []
        self.observing = True

    def pause_observations(self):
        self.preparations, self.deliveries, self.http_events = [], [], []
        self.observing = False

    async def prepare(self, messages, effort):
        start = time.perf_counter_ns()
        result = await super().prepare(messages, effort)
        if self.observing:
            self.preparations.append({"http_id": http_context.get(), "start_ns": start,
                                      "end_ns": time.perf_counter_ns(), "tokens": len(result)})
        return result

    async def generate(self, prompt, request_id, **sampling):
        observed = self.observing and request_id != "warmup"
        start, previous = time.perf_counter_ns(), 0
        async for output in super().generate(prompt, request_id, **sampling):
            if observed:
                count = len(output.token_ids)
                self.deliveries.append({"http_id": http_context.get(), "request_id": request_id, "submitted_ns": start,
                    "at_ns": time.perf_counter_ns(), "new_tokens": count - previous,
                    "total_tokens": count, "finished": output.finish_reason})
                previous = count
            yield output


class ProfileHTTPMiddleware:
    def __init__(self, app, engine):
        self.app, self.engine = app, engine

    async def __call__(self, scope, receive, send):
        if not self.engine.observing or scope["type"] != "http" or not scope["path"].endswith("/responses"):
            return await self.app(scope, receive, send)
        record = {"http_id": len(self.engine.http_events), "start_ns": time.perf_counter_ns(), "sends": []}
        self.engine.http_events.append(record)
        token = http_context.set(record["http_id"])

        async def observed_send(message):
            before = time.perf_counter_ns()
            await send(message)
            record["sends"].append({"type": message["type"], "start_ns": before,
                                    "end_ns": time.perf_counter_ns(), "bytes": len(message.get("body", b""))})
        try:
            await self.app(scope, receive, observed_send)
        finally:
            record["end_ns"] = time.perf_counter_ns()
            http_context.reset(token)


async def run_profile(config: ServingConfig, profile: ProfileConfig, output: Path, *,
                      suite=ProfileSuite.FULL, rounds=3, port=18100, capture=True, only_cases=None, quality_samples=None, quality_reference=None, long_validation=False, cache_boundary_validation=False, prefix_growth_validation=False, stability_seconds=0):
    import uvicorn
    from qwen_llm.servicer.app import create_app
    from tests.performance.qwen38.analysis import analyze_directory

    output = output.resolve()
    cases = scenarios(suite, config.max_input_tokens)
    if only_cases:
        selected = set(only_cases.split(","))
        cases = [c for c in cases if c["name"] in selected]
        if {c["name"] for c in cases} != selected:
            raise ValueError("存在未知 case 名称")
    if output.exists():
        raise ValueError("输出目录已存在，请使用新的目录以保留原始记录")
    before = inventory()
    if before["owners"]:
        raise RuntimeError("NPU已有拥有进程，拒绝运行独占诊断")
    with socket.socket() as check:
        check.bind(("127.0.0.1", port))
    output.mkdir(parents=True)
    manifest = {"config": asdict(config), "profile": asdict(profile), "suite": suite,
                "rounds": 0 if suite is ProfileSuite.TRACE else rounds,
                "before": before, "cases": [], "valid": False}
    manifest['validation_options'] = {'quality_samples':quality_samples,
        'quality_reference':str(quality_reference) if quality_reference else None,
        'long_validation':long_validation,'cache_boundary_validation':cache_boundary_validation,
        'prefix_growth_validation':prefix_growth_validation or long_validation,
        'stability_seconds':stability_seconds}
    source_root = Path(__file__).resolve().parents[3]
    sources = [*Path(__file__).parent.glob('*.py'), *source_root.joinpath('src/qwen_llm').rglob('*.py')]
    manifest['implementation_sha256'] = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                          for path in sources}
    for relative in manifest['implementation_sha256']:
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
    write_json(output / "manifest.json", manifest)
    engine = ObservedEngine(config, profiling=profile)
    application = create_app(config, engine=engine)
    application.add_middleware(ProfileHTTPMiddleware, engine=engine)

    class DiagnosticServer(uvicorn.Server):
        async def startup(self, sockets=None):
            try:
                return await super().startup(sockets=sockets)
            except SystemExit as exc:
                raise RuntimeError(f"诊断服务启动退出 {exc.code}，详见启动日志") from exc

    server = DiagnosticServer(uvicorn.Config(application, host="127.0.0.1",
        port=port, log_level="warning", timeout_graceful_shutdown=30))
    task = asyncio.create_task(server.serve())
    async def sample_resources():
        with (output / "npu-samples.jsonl").open("w") as destination:
            while True:
                try:
                    raw = await asyncio.to_thread(subprocess.check_output, ["npu-smi", "info"], text=True, timeout=10)
                    owners = re.findall(r"\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|", raw)
                    destination.write(json.dumps({"at_ns": time.perf_counter_ns(), "npu": raw,
                        "processes": [process_resources(pid) for pid in [os.getpid(), *(p for p, _ in owners)]]}) + "\n")
                    destination.flush()
                except (OSError, subprocess.SubprocessError) as exc:
                    destination.write(json.dumps({"error": str(exc)}) + "\n")
                await asyncio.sleep(2)
    sampler = asyncio.create_task(sample_resources())
    client_process = None
    namespace = "qwen38-hotspots-v1"
    try:
        for _ in range(900):
            if task.done():
                await task
                raise RuntimeError("诊断服务启动失败")
            if server.started:
                break
            await asyncio.sleep(1)
        else:
            raise TimeoutError("诊断服务启动超时")
        owned = inventory()
        if not owned["isolated"]:
            raise RuntimeError("未确认独占的真实 vLLM Worker")
        manifest["ready"] = {"runtime": engine.runtime_identity, "inventory": owned}
        manifest["ready"]["cache_layout"] = await engine.llm.collective_rpc("diagnostic_cache_layout")
        write_json(output / "modules.json", await engine.llm.collective_rpc("diagnostic_modules"))
        engine.pause_observations()
        if quality_samples is not None:
            from .validation import quality
            manifest['quality'] = await quality(f'http://127.0.0.1:{port}', config.model_name, output/'quality.json', math_samples=quality_samples)
            write_json(output/'manifest.json', manifest)
            if manifest['quality']['execution_errors'] or not manifest['quality']['instance_unchanged']:
                raise RuntimeError('质量检查出现执行错误或实例变更')
            if quality_reference:
                from .compare import quality_gate
                manifest['quality_gate'] = quality_gate(json.loads(Path(quality_reference).read_text()),
                    json.loads((output/'quality.json').read_text()), require_full=quality_samples==1319)
                write_json(output/'manifest.json', manifest)
                if not manifest['quality_gate']['passed']:
                    raise RuntimeError('质量相对基线未通过，停止候选后续测量')
        if long_validation:
            from .validation import long_quality
            manifest['long_quality'] = await long_quality(engine, f'http://127.0.0.1:{port}', config.model_name, output/'long-quality.json')
            write_json(output/'manifest.json', manifest)
            if quality_reference and manifest['long_quality']['passed'] != manifest['long_quality']['total']:
                raise RuntimeError('长文检索或并发隔离未全部通过')
        if cache_boundary_validation:
            from .validation import long_quality
            manifest['boundary_quality'] = await long_quality(engine, f'http://127.0.0.1:{port}', config.model_name,
                output/'boundary-quality.json', lengths=(1535,1536,1537,3071,3072,3073,6143,6144,6145), positions=('middle',))
            write_json(output/'manifest.json', manifest)
            if quality_reference and manifest['boundary_quality']['passed'] != manifest['boundary_quality']['total']:
                raise RuntimeError('缓存块边界检索未全部通过')
        if prefix_growth_validation or long_validation:
            from .validation import prefix_growth
            manifest['prefix_growth_quality'] = await prefix_growth(engine, f'http://127.0.0.1:{port}', config.model_name, output/'prefix-growth.json')
            write_json(output/'manifest.json', manifest)
            if quality_reference and manifest['prefix_growth_quality']['passed'] != manifest['prefix_growth_quality']['total']:
                raise RuntimeError('前缀增长检索未全部通过')
        if config.speculative_tokens:
            manifest['head_equivalence'] = await engine.llm.collective_rpc('diagnostic_head_equivalence')
            if 'head' in config.optimizations and not all(x['shared_module'] for x in manifest['head_equivalence']):
                raise RuntimeError('请求输出头共享但实际模块未共享')
            if not all(x['equal'] for x in manifest['head_equivalence']):
                raise RuntimeError('主模型和草稿相同隐藏态的输出 logits 不一致')
            write_json(output/'manifest.json', manifest)
        async def execute(case, label, *, trace=False, tokens=None):
            nonlocal client_process
            case_dir = output / case["name"]
            case_dir.mkdir(exist_ok=True)
            prompts = case["prompts"]
            spec = {"url": f"http://127.0.0.1:{port}", "model": config.model_name,
                    "prompts": prompts, "output_tokens": tokens or case["output_tokens"],
                    "delays": case.get("delays", [0] * len(prompts))}
            spec_path, result_path = case_dir / (label + "-request.json"), case_dir / (label + "-http.json")
            write_json(spec_path, spec)
            engine.reset_observations()
            await engine.llm.collective_rpc("diagnostic_begin_case", args=(case["name"] + "-" + label, trace))
            profiling_started = False
            try:
                if trace:
                    await engine.llm.start_profile(case["name"] + "-" + label)
                    profiling_started = True
                client_process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.performance.qwen38.client",
                    str(spec_path), str(result_path))
                code = await client_process.wait()
                client_process = None
                if code:
                    raise RuntimeError(f"HTTP 负载进程退出 {code}")
            finally:
                if profiling_started:
                    await engine.llm.stop_profile()
                workers = await engine.llm.collective_rpc("diagnostic_end_case")
            write_json(case_dir / (label + "-worker.json"), workers)
            write_json(case_dir / (label + "-frontend.json"),
                       {"preparations": engine.preparations, "deliveries": engine.deliveries,
                        "http": engine.http_events})
            if any(not p['state_end']['uncached_index_matches'] for w in workers for p in w.get('cache_pools', [])):
                raise RuntimeError('KV分配索引与实际空闲块不一致')
            client = json.loads(result_path.read_text())
            summary = summarize_round(client, workers[0])
            resources = inventory()
            if not resources["isolated"]:
                raise RuntimeError("诊断期间设备被其他实例占用")
            summary["inventory"] = resources
            if not summary["valid"] or len(summary["output_tokens"]) != case["concurrency"]:
                raise RuntimeError(f"请求失败: {result_path}")
            if case["cache"] == "cold" and any(summary["cached_tokens"]):
                raise RuntimeError("冷缓存测试意外命中前缀缓存")
            if "input_tokens" in case and any(n != case["input_tokens"] for n in summary["input_tokens"]):
                raise RuntimeError("服务端实际输入长度与 case 不一致")
            logger.info("%s %s: %.2f TPS, TTFT=%s, cache=%s", case["name"], label,
                        summary["tokens_per_second"], summary["ttft_ms"], summary["cached_tokens"])
            return summary

        trace_names = {"short128-c1", "short128-c4", "chinese8192-c1-cold", "chinese8192-c4-cold",
                       "chinese32768-c4-cold", "mixed32k-c4"}
        for case in cases:
            name, kind = case["name"], case["kind"]
            case["prompts"] = []
            for index in range(case["concurrency"]):
                if kind in {"chinese", "repeated"} or kind == "mixed" and index == 3:
                    case["prompts"].append(make_long_prompt(engine.tokenizer,
                        case.get("input_tokens", 32768), index, namespace,
                        repeated=kind == "repeated", historical=suite is ProfileSuite.HISTORICAL and kind == "repeated"))
                else:
                    case["prompts"].append({"qa": QA, "code": CODE}.get(kind, SHORT))
            entry = {k: v for k, v in case.items() if k != "prompts"}
            entry["prompt_sha256"] = [hashlib.sha256(p.encode()).hexdigest() for p in case["prompts"]]
            entry["rounds"] = []
            manifest["cases"].append(entry)
            if not await engine.llm.reset_prefix_cache():
                raise RuntimeError("无法清理前缀缓存")
            entry["warmup"] = await execute(case, "warmup")
            for iteration in range(rounds if suite is not ProfileSuite.TRACE else 0):
                if case["cache"] == "cold":
                    if not await engine.llm.reset_prefix_cache():
                        raise RuntimeError("无法清理前缀缓存")
                entry["rounds"].append(await execute(case, f"round{iteration}"))
                write_json(output / "manifest.json", manifest)
            if capture and (name in trace_names or suite in {ProfileSuite.SMOKE, ProfileSuite.TRACE}
                            or suite is ProfileSuite.HISTORICAL and name == "repeated8192-c4-cold"):
                if case["cache"] == "cold":
                    if not await engine.llm.reset_prefix_cache():
                        raise RuntimeError("无法清理前缀缓存")
                entry["capture"] = await execute(case, "trace", trace=True,
                                                 tokens=64 if kind == "mixed" else 16)
            write_json(output / "manifest.json", manifest)
        if stability_seconds:
            engine.pause_observations()
            from .validation import stability
            manifest['stability'] = await stability(engine, f'http://127.0.0.1:{port}', config.model_name, output/'stability.json', stability_seconds)
        manifest["valid"] = True
    except BaseException as exc:
        manifest["error"] = repr(exc)
        raise
    finally:
        if client_process is not None and client_process.returncode is None:
            client_process.terminate()
            await client_process.wait()
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=60)
        finally:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
            manifest["after"] = inventory()
            manifest["last_runtime"] = engine.runtime_identity
            with socket.socket() as check:
                check.settimeout(1)
                manifest["port_released"] = check.connect_ex(("127.0.0.1", port)) != 0
            manifest["released"] = not manifest["after"]["owners"]
            manifest["valid"] &= manifest["released"] and manifest["port_released"]
            write_json(output / "manifest.json", manifest)
    analysis = analyze_directory(output)
    if capture and any("capture" in case for case in manifest["cases"]):
        if not analysis["traces"] or not all(t["valid_device_trace"]
                and t.get("coverage", {}).get("complete", True) for t in analysis["traces"]):
            manifest["valid"] = False
            manifest["error"] = "采集缺少真实设备内核或目标FFN调用不完整"
            write_json(output / "manifest.json", manifest)
            raise RuntimeError(manifest["error"])
    return manifest
