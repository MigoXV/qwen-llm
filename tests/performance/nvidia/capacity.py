"""对每档并发探测无抢占的完整上下文上限；失败边界保留原始证据。"""
import asyncio
import json
from pathlib import Path
import re

import httpx
import typer

app = typer.Typer()


def passed(record):
    rows = record.get('rounds', [])
    return bool(rows) and record.get('instance_unchanged', False) and record.get('pool_can_hold_all',False) and all(
        row.get('all_full_output') and row.get('all_exact_input') and row.get('all_uncached')
        and row.get('isolated') and row.get('preemptions_delta') == 0 for row in rows)


async def sweep(url, output, tokens, concurrencies, step, start_context=16384, from_allocated_pool=False):
    from tests.performance.nvidia.long_context import run
    output.mkdir(parents=True, exist_ok=False)
    async with httpx.AsyncClient(timeout=10) as client:
        reply = await client.get(url+'/health/ready')
        reply.raise_for_status()
        service = reply.json()
    if not service['runtime'].get('force_length'):
        raise ValueError('容量测试要求 force_length=True；每轮使用唯一前缀并验证零命中')
    limit = service['max_input_tokens'] + tokens
    limit = limit // step * step
    record = {'service': service, 'output_tokens': tokens, 'resolution_tokens': step,
              'context_includes_output': True, 'start_context':start_context,
              'search_mode': 'allocated_pool' if from_allocated_pool else 'coarse_binary',
              'results': {}, 'trials': []}

    async def trial(context, concurrency):
        case = output/f'context-{context}-c{concurrency}'
        worker_capacities = [w['capacity_by_context'][str(context)] for w in service['runtime']['workers']]
        resident_capacity = min(worker_capacities)
        try:
            result = await run(url, case, [context-tokens], [concurrency], 1, tokens,
                               'cold', False, stop_on_preempt=True)
            error = None
        except RuntimeError as exc:
            result = json.loads((case/'manifest.json').read_text())
            error = str(exc)
        result['allocated_pool_capacity'] = resident_capacity
        result['pool_can_hold_all'] = resident_capacity >= concurrency
        (case/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        okay = passed(result)
        record['trials'].append({'context': context, 'concurrency': concurrency,
                                'passed': okay, 'path': str(case), 'error': error,
                                'allocated_pool_capacity':resident_capacity,
                                'rounds': result['rounds']})
        (output/'capacity.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(json.dumps(record['trials'][-1], ensure_ascii=False), flush=True)
        if not okay:
            # 只将真实分配池不足或抢占用作上界；协议错误或实例污染不能作为容量结论。
            if result['pool_can_hold_all'] and not any(r.get('preemptions_delta',0)>0 for r in result['rounds']):
                raise RuntimeError('轮次未通过且没有分配池不足或抢占证据，停止容量推断')
            if not all(r.get('isolated') for r in result['rounds']):
                raise RuntimeError('设备归属发生变化，停止容量推断')
            async with httpx.AsyncClient(timeout=10) as client:
                current = await client.get(url+'/health/ready');current.raise_for_status()
                if current.json()['instance_id'] != service['instance_id']:
                    raise RuntimeError('实例发生变化，停止容量推断')
                for _ in range(90):
                    metrics=(await client.get(url+'/metrics')).text
                    running=[float(m[1]) for m in re.finditer(
                        r'^(?:responses_inflight|vllm:num_requests_(?:running|waiting))(?:\{[^\n]*\})? ([\d.]+)',metrics,re.M)]
                    if len(running)>=3 and not any(running):break
                    await asyncio.sleep(1)
                else:raise RuntimeError('抢占轮次取消后未排空，停止容量推断')
        return okay

    for concurrency in concurrencies:
        low = 0
        high = None
        if from_allocated_pool:
            # 真实分配池给出驻留的必要条件；候选点仍须实际完成完整生成。
            fits = [x for x in range(step, limit+1, step) if x > tokens and all(
                w['capacity_by_context'][str(x)] >= concurrency for w in service['runtime']['workers'])]
            if not fits:
                raise RuntimeError('实际分配池没有可测的完整上下文候选点')
            context = max(fits)
            while context > tokens:
                if await trial(context, concurrency):
                    low = context
                    break
                high = context
                context -= step
            # 下一档也实际运行并记录，不能只把公式当成实测失败。
            points = [low+step] if high is None and low < limit else []
        else:
            points = sorted({x for x in (16384, 32768, 65536, 131072, 196608, 262144, limit)
                             if tokens < x <= limit and (x >= start_context or x == limit)})
        for context in points:
            if await trial(context, concurrency):
                low = context
            else:
                high = context
                break
        if high is not None and not from_allocated_pool:
            while high-low > step:
                middle = ((low+high)//2)//step*step
                if middle <= tokens:break
                if await trial(middle, concurrency):low = middle
                else:high = middle
        record['results'][str(concurrency)] = {
            'largest_passed_context': low, 'smallest_failed_context': high,
            'reached_configured_limit': low == limit, 'configured_limit': limit}
        (output/'capacity.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


@app.command()
def main(output: Path = typer.Option(...), base_url: str = typer.Option('http://127.0.0.1:8121'),
         tokens: int = typer.Option(8192, min=1), concurrencies: str = typer.Option('1,2,3,4'),
         step: int = typer.Option(8192, min=1024),
         start_context: int = typer.Option(16384, min=1024),
         from_allocated_pool: bool = typer.Option(False)):
    if step % 1024:
        raise typer.BadParameter('step 必须是 1024 的倍数，以匹配实际池观测长度')
    asyncio.run(sweep(base_url, output, tokens, [int(x) for x in concurrencies.split(',')],
                      step, start_context, from_allocated_pool))


if __name__ == '__main__':app()
