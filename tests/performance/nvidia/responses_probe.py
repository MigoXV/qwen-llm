"""隔离实例的 Responses 流式时序、token 对账和前缀缓存对照。"""
import asyncio
import hashlib
import json
from pathlib import Path
import statistics
import time

import httpx
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer
import typer

from tests.performance.nvidia.inventory import inventory
from tests.performance.nvidia.long_context import prompt_for

app = typer.Typer()
TOOLS = [{'type': 'function', 'name': 'record', 'description': '记录文本',
          'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}},
                         'required': ['text'], 'additionalProperties': False}}]


def counters(text):
    result = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            result[sample.name] = result.get(sample.name, 0) + sample.value
    return result


async def stream(client, url, body):
    start = time.perf_counter()
    stamps, events, final = [], [], None
    async with client.stream('POST', url + '/v1/responses', json={**body, 'stream': True}) as reply:
        reply.raise_for_status()
        async for line in reply.aiter_lines():
            if not line.startswith('data: '):
                continue
            event = json.loads(line[6:])
            now = time.perf_counter() - start
            if event['type'].endswith('.delta') and event.get('delta'):
                stamps.append(now)
            events.append({'seconds': now, **event})
            if event['type'] in ('response.completed', 'response.incomplete', 'response.failed'):
                final = event['response']
    elapsed = time.perf_counter() - start
    if final is None or final['status'] == 'failed':
        raise RuntimeError(f'响应失败: {final}')
    n = final['usage']['output_tokens']
    return {'seconds': elapsed, 'ttft_seconds': stamps[0] if stamps else None,
            'delta_events': len(stamps), 'output_tokens': n,
            'cached_tokens': final['usage']['input_tokens_details']['cached_tokens'],
            'dsh_tokens_per_second': n/(elapsed-stamps[0]) if stamps and elapsed > stamps[0] else None,
            'end_to_end_tokens_per_second': n/elapsed,
            'input_sha256': hashlib.sha256(json.dumps(body['input'], ensure_ascii=False).encode()).hexdigest(),
            'events': events, 'response': final}


async def run(url, output, rounds, input_tokens):
    output.mkdir(parents=True, exist_ok=False)
    async with httpx.AsyncClient(timeout=7200) as client:
        reply = await client.get(url+'/health/ready'); reply.raise_for_status()
        service = reply.json()
        runtime = service['runtime']
        devices = [int(x) for x in runtime['cuda_visible_devices'].split(',')]
        owners = [w['pid'] for w in runtime['workers']]
        def check():
            value = inventory(devices, owners)
            if not value['isolated']:
                raise RuntimeError('被测设备存在其他任务')
            return value
        tokenizer = AutoTokenizer.from_pretrained(runtime['tokenizer_path'], local_files_only=True)
        namespace = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
        record = {'service': service, 'rounds': [], 'input_tokens': input_tokens,
                  'namespace': namespace, 'inventory_before': check()}
        def save():
            (output/'report.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        async def measure(label, bodies):
            check()
            before = counters((await client.get(url+'/metrics')).text)
            start = time.perf_counter()
            samples = await asyncio.gather(*(stream(client, url, b) for b in bodies))
            elapsed = time.perf_counter() - start
            await asyncio.sleep(5.2)  # vLLM 指标异步刷新，不计入请求时长。
            after = counters((await client.get(url+'/metrics')).text)
            delta = {k: v-before.get(k, 0) for k, v in after.items()
                     if k.startswith('vllm:') and (k.endswith('_total') or k.endswith('_sum') or k.endswith('_count'))}
            decode = delta['vllm:inter_token_latency_seconds_sum']
            intervals = delta['vllm:inter_token_latency_seconds_count']
            row = {'case': label, 'concurrency': len(bodies), 'makespan_seconds': elapsed,
                   'aggregate_end_to_end_tps': sum(s['output_tokens'] for s in samples)/elapsed,
                   'engine_decode_tps_per_request': intervals/decode if decode else None,
                   'median_dsh_tps': statistics.median(s['dsh_tokens_per_second'] for s in samples),
                   'median_ttft_seconds': statistics.median(s['ttft_seconds'] for s in samples),
                   'cached_tokens': [s['cached_tokens'] for s in samples],
                   'metrics_delta': delta, 'samples': samples, 'inventory_after': check()}
            record['rounds'].append(row); save()
            assert sum(s['output_tokens'] for s in samples) == delta['vllm:generation_tokens_total'], 'token 对账失败，检查是否混入其他请求'
            assert delta.get('vllm:num_preemptions_total', 0) == 0
            print(json.dumps({k: v for k, v in row.items() if k not in ('samples','metrics_delta','inventory_after')}, ensure_ascii=False), flush=True)
        common = {'model': service['model'], 'temperature': 0, 'seed': 42, 'max_output_tokens': 256}
        text = '不需要调用任何工具。请从1开始逐行列出整数，直到120，不要解释或省略。'
        for concurrency in (1, 2):
            for mode in ('text', 'tools_auto'):
                bodies = [{**common, 'input': text, **({'tools': TOOLS} if mode == 'tools_auto' else {})}
                          for _ in range(concurrency)]
                await measure(f'{mode}-c{concurrency}', bodies)
            for i in range(rounds):
                bodies = [{**common, 'max_output_tokens': 128, 'prompt_cache_key': 'probe-'+namespace,
                           'input': prompt_for(tokenizer, input_tokens, j, f'{namespace}-c{concurrency}')[0]}
                          for j in range(concurrency)]
                await measure(f'prefix-repeat-{i}-c{concurrency}', bodies)
            # 相同 cache key、开头不同的内容，不能据 key 返回旧缓存命中。
            bodies = [{**common, 'max_output_tokens': 128, 'prompt_cache_key': 'probe-'+namespace,
                       'input': prompt_for(tokenizer, input_tokens, j, f'changed-{namespace}-c{concurrency}')[0]}
                      for j in range(concurrency)]
            await measure(f'prefix-changed-c{concurrency}', bodies)
        await measure('function-stream', [{**common, 'max_output_tokens': 512, 'tools': TOOLS,
            'tool_choice': 'required', 'input': '调用record，把下面这段话原样写入text参数：'+'实时显示工具参数，不能等全部生成后才发送。'*12}])
        assert service['instance_id'] == (await client.get(url+'/health/ready')).json()['instance_id']
        record['inventory_after'] = check(); save()
    return record


@app.command()
def main(output: Path = typer.Option(...), url: str = typer.Option('http://127.0.0.1:8133'),
         rounds: int = typer.Option(3), input_tokens: int = typer.Option(16384)):
    asyncio.run(run(url, output, rounds, input_tokens))


if __name__ == '__main__':
    app()
