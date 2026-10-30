"""两个实例每轮通过同一屏障启动，固定对端负载后比较 TPS。"""
import asyncio
from functools import partial
import json
from pathlib import Path
import time

import httpx
import typer

from tests.performance.benchmark import run_benchmark
from tests.performance.nvidia.inventory import inventory
from tests.performance.qwen38.runner import SHORT


async def run(first, second, output, tokens, rounds, seconds):
    output.mkdir(parents=True, exist_ok=False)
    urls = [first, second]
    async with httpx.AsyncClient(timeout=10) as client:
        replies = await asyncio.gather(*(client.get(url+'/health/ready') for url in urls))
        for reply in replies:reply.raise_for_status()
        services = [r.json() for r in replies]
        metrics = await asyncio.gather(*(client.get(url+'/metrics') for url in urls))
        for reply in metrics:reply.raise_for_status()
        metrics_before = [reply.text for reply in metrics]
    devices = [{int(x) for x in s['runtime']['cuda_visible_devices'].split(',')} for s in services]
    if devices[0] & devices[1]:raise ValueError('两个实例不能共享 GPU')
    checks = [partial(inventory, sorted(d), [w['pid'] for w in s['runtime']['workers']])
              for d, s in zip(devices, services)]
    started = time.monotonic()
    record = {'services': services, 'paired_round_starts': True, 'requested_seconds': seconds,
              'metrics_before': metrics_before, 'completed': False, 'batches': []}
    iteration = 0
    while iteration == 0 or time.monotonic()-started < seconds:
        barrier = asyncio.Barrier(2)
        paths = [output/f'batch-{iteration}-instance-{i}.json' for i in range(2)]
        async with asyncio.TaskGroup() as group:
            for url, service, check, path in zip(urls, services, checks, paths):
                group.create_task(run_benchmark(url+'/v1', service['model'], SHORT, tokens, path,
                    inventory=check, concurrencies=(1, 2, 3, 4), rounds=rounds, before_round=barrier.wait))
        values = [json.loads(path.read_text()) for path in paths]
        if any(not value['valid'] or value['service']['instance_id'] != service['instance_id']
               for value, service in zip(values, services)):
            raise RuntimeError('测量无效或实例发生切换')
        record['batches'].append([str(path) for path in paths])
        record['elapsed_seconds'] = time.monotonic()-started
        (output/'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(json.dumps({'batch': iteration, 'elapsed_seconds': record['elapsed_seconds']}), flush=True)
        iteration += 1
    # 给 vLLM 周期统计留出刷新时间；不计入任何请求 TPS 或稳定性时长。
    await asyncio.sleep(5.1)
    async with httpx.AsyncClient(timeout=10) as client:
        metrics = await asyncio.gather(*(client.get(url+'/metrics') for url in urls))
        for reply in metrics:reply.raise_for_status()
    record['metrics_after'] = [reply.text for reply in metrics]
    record['completed'] = True
    (output/'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))


def main(output: Path = typer.Option(...), first: str = typer.Option('http://127.0.0.1:8120'),
         second: str = typer.Option('http://127.0.0.1:8121'), tokens: int = typer.Option(128, min=1),
         rounds: int = typer.Option(3, min=1), seconds: int = typer.Option(0, min=0)):
    asyncio.run(run(first, second, output, tokens, rounds, seconds))


if __name__ == '__main__':typer.run(main)
