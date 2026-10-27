import asyncio
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlsplit

import httpx


def inventory() -> dict:
    result = {}
    for name, command in {"npu": ["npu-smi", "info"], "revision": ["git", "rev-parse", "HEAD"],
                          "worktree": ["git", "status", "--short"]}.items():
        try:
            result[name] = subprocess.check_output(command, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            result[name] = str(exc)
    result["owners"] = re.findall(r"\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|", result["npu"])
    result["isolated"] = len(result["owners"]) == 1 and "VLLM" in result["owners"][0][1].upper()
    return result


async def run_benchmark(base_url: str, model: str, prompt: str, tokens: int, output: Path):
    record = {"base_url": base_url, "model": model, "prompt": prompt, "max_output_tokens": tokens,
              "before": await asyncio.to_thread(inventory), "rounds": [], "valid": False}
    try:
        if urlsplit(base_url).hostname not in ("127.0.0.1", "localhost", "::1"):
            raise RuntimeError("本地基准只支持本机地址，以保证设备盘点与目标实例一致")
        if not record["before"]["isolated"]:
            raise RuntimeError("未确认独占 V1 设备进程，拒绝产生性能结论")
        async with httpx.AsyncClient(timeout=650) as client:
            origin = urlsplit(base_url)
            health_url = f"{origin.scheme}://{origin.netloc}/health/ready"
            health = await client.get(health_url)
            health.raise_for_status()
            record["service"] = health.json()
            if record["service"].get("engine") != "v1" or record["service"].get("model") != model:
                raise RuntimeError("服务引擎/模型身份不匹配")

            async def one():
                started = time.perf_counter()
                first_token = None
                final = None
                try:
                    async with client.stream("POST", base_url.rstrip("/") + "/responses", json={
                        "model": model, "input": prompt, "max_output_tokens": tokens,
                        "stream": True, "temperature": 0}) as response:
                        if response.status_code != 200:
                            await response.aread()
                            return {"seconds": time.perf_counter() - started,
                                    "http_status": response.status_code, "response": response.json()}
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            event = json.loads(line[6:])
                            if event["type"] in ("response.output_text.delta", "response.reasoning_text.delta"):
                                if event.get("delta") and first_token is None:
                                    first_token = time.perf_counter()
                            if event["type"] in ("response.completed", "response.incomplete", "response.failed"):
                                final = event["response"]
                    ended = time.perf_counter()
                    if final is None or final["status"] == "failed":
                        raise RuntimeError(f"生成未正常结束：{final}")
                    out_tokens = final["usage"]["output_tokens"]
                    return {"seconds": ended - started, "http_status": 200,
                            "ttft_seconds": first_token - started if first_token else None,
                            "decode_tokens_per_second": (out_tokens - 1) / (ended - first_token)
                                if first_token and out_tokens > 1 else None,
                            "response": final}
                except Exception as exc:
                    return {"seconds": time.perf_counter() - started, "error": str(exc)}

            record["warmup"] = await one()
            if record["warmup"].get("http_status") != 200:
                raise RuntimeError("预热失败，禁止报告性能")
            for concurrency in (1, 2, 4, 8):
                for iteration in range(3):
                    started = time.perf_counter()
                    samples = await asyncio.gather(*(one() for _ in range(concurrency)))
                    elapsed = time.perf_counter() - started
                    successful = [x for x in samples if x.get("http_status") == 200]
                    durations = sorted(x["seconds"] for x in samples)
                    def percentile(q):
                        return durations[min(len(durations)-1, int((len(durations)-1)*q + 0.5))]
                    resources = await asyncio.to_thread(inventory)
                    record["rounds"].append({"concurrency": concurrency, "round": iteration,
                        "makespan_seconds": elapsed, "samples": samples,
                        "p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99),
                        "tokens_per_second": sum(x["response"]["usage"]["output_tokens"] for x in successful)/elapsed,
                        "resources": resources})
                    if not resources["isolated"]:
                        raise RuntimeError("设备盘点发现实例污染，此轮结果无效")
            final_health = await client.get(health_url)
            final_health.raise_for_status()
            if final_health.json().get("instance_id") != record["service"].get("instance_id"):
                raise RuntimeError("测量期间服务实例已更换")
            record["valid"] = True
    except Exception as exc:
        record["error"] = str(exc)
        raise
    finally:
        record["after"] = await asyncio.to_thread(inventory)
        if not record["after"]["isolated"]:
            record["valid"] = False
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2))
