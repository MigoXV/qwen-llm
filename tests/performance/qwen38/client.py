"""诊断专用独立 HTTP 负载进程，防止客户端与服务竞争同一事件循环。"""
import asyncio
import json
from pathlib import Path
import sys
import time

import httpx


async def run(spec: dict) -> dict:
    started = time.perf_counter_ns()
    async with httpx.AsyncClient(timeout=1800, limits=httpx.Limits(max_connections=8)) as client:
        async def one(index, prompt):
            await asyncio.sleep(spec.get("delays", [0] * len(spec["prompts"]))[index])
            begin = time.perf_counter_ns()
            chunks, final, first = [], None, None
            try:
                async with client.stream("POST", spec["url"] + "/v1/responses", json={
                    "model": spec["model"], "input": prompt, "stream": True,
                    "temperature": 0, "seed": 42, "max_output_tokens": spec["output_tokens"]}) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode(errors="replace")
                        raise RuntimeError(f"HTTP {response.status_code}: {body}")
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        now = time.perf_counter_ns()
                        if event["type"].endswith("text.delta") and event.get("delta"):
                            first = first or now
                            chunks.append({"at_ns": now, "characters": len(event["delta"])})
                        if event["type"] in {"response.completed", "response.incomplete", "response.failed"}:
                            final = event["response"]
                if final is None or final["status"] == "failed":
                    raise RuntimeError(f"请求未正常完成: {final}")
                return {"index": index, "start_ns": begin, "end_ns": time.perf_counter_ns(),
                        "first_text_ns": first, "chunks": chunks, "response": final}
            except Exception as exc:
                return {"index": index, "start_ns": begin, "end_ns": time.perf_counter_ns(),
                        "error": repr(exc)}
        samples = await asyncio.gather(*(one(i, prompt) for i, prompt in enumerate(spec["prompts"])))
    return {"start_ns": started, "end_ns": time.perf_counter_ns(), "samples": samples}


if __name__ == "__main__":
    specification, output = map(Path, sys.argv[1:])
    output.write_text(json.dumps(asyncio.run(run(json.loads(specification.read_text()))),
                                 ensure_ascii=False, indent=2))
