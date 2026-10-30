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
    from tests.performance.benchmark import run_benchmark as shared_benchmark
    return await shared_benchmark(base_url, model, prompt, tokens, output, inventory=inventory)
