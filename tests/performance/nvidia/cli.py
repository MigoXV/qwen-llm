"""显式硬件实验入口；FP8 KV 与满长度压力仅在这里开放。"""
import asyncio
from functools import partial
import json
import logging
from pathlib import Path

import httpx
import typer

from qwen_llm.config import Runner
from qwen_llm.engine.nvidia import NvidiaConfig, NvidiaEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
app = typer.Typer(no_args_is_help=True)


@app.callback()
def main():
    pass


@app.command()
def serve(
    model_path: Path = typer.Option(..., exists=True, envvar='QWEN_MODEL_PATH'),
    model_name: str = typer.Option('qwen38-nvidia', envvar='QWEN_MODEL_NAME'),
    host: str = typer.Option('0.0.0.0', envvar='QWEN_HOST'),
    port: int = typer.Option(8121, envvar='QWEN_PORT'),
    tp: int = typer.Option(1, envvar='QWEN_TP'),
    context: int = typer.Option(262144, envvar='QWEN_MAX_MODEL_LEN'),
    max_output_tokens: int = typer.Option(8192, min=1, envvar='QWEN_MAX_OUTPUT_TOKENS'),
    concurrency: int = typer.Option(4, envvar='QWEN_MAX_CONCURRENCY'),
    batch_tokens: int = typer.Option(2048, envvar='QWEN_BATCH_TOKENS'),
    gpu_memory_utilization: float = typer.Option(0.9, min=0.1, max=0.98, envvar='QWEN_GPU_MEMORY_UTILIZATION'),
    kv_cache_gib: float | None = typer.Option(None, min=0.1, envvar='QWEN_KV_CACHE_GIB'),
    kv_dtype: str = typer.Option('fp8_e4m3', envvar='QWEN_KV_DTYPE'),
    attention_backend: str | None = typer.Option(None, envvar='QWEN_ATTENTION_BACKEND'),
    cuda_graph: bool = typer.Option(False, envvar='QWEN_CUDA_GRAPH'),
    prefix_caching: bool = typer.Option(False, envvar='QWEN_PREFIX_CACHING'),
    speculative_tokens: int = typer.Option(0, envvar='QWEN_SPECULATIVE_TOKENS'),
    force_length: bool = typer.Option(False, envvar='QWEN_FORCE_LENGTH'),
):
    import uvicorn
    from qwen_llm.servicer.app import create_app
    if context <= max_output_tokens:
        raise typer.BadParameter('context 必须大于 max-output-tokens')
    config = NvidiaConfig(model_path=model_path, model_name=model_name,
        max_input_tokens=context - max_output_tokens, max_output_tokens=max_output_tokens, max_concurrency=concurrency,
        tensor_parallel_size=tp, kv_cache_dtype=kv_dtype, attention_backend=attention_backend,
        experimental=True, batch_tokens=batch_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_bytes=int(kv_cache_gib * 1024**3) if kv_cache_gib is not None else None,
        runner=Runner.CUDAGRAPH if cuda_graph else Runner.EAGER,
        prefix_caching=prefix_caching, speculative_tokens=speculative_tokens, timeout=7200)
    class CapacityEngine(NvidiaEngine):
        async def start(self):
            await super().start()
            self.runtime_identity['force_length'] = force_length

        async def generate(self, prompt, request_id, **sampling):
            if force_length and request_id != 'warmup':
                sampling['ignore_eos'] = True
                sampling['min_tokens'] = sampling['max_tokens']
            async for result in super().generate(prompt, request_id, **sampling):
                yield result
    engine = CapacityEngine(config)
    service = create_app(config, engine, max_body_bytes=16*1024**2)
    uvicorn.run(service,
                host=host, port=port, workers=1, timeout_graceful_shutdown=30)


@app.command()
def benchmark(
    output: Path = typer.Option(...),
    base_url: str = typer.Option('http://127.0.0.1:8120'),
    tokens: int = typer.Option(128),
    rounds: int = typer.Option(3),
):
    from tests.performance.benchmark import run_benchmark
    from tests.performance.nvidia.inventory import inventory
    from tests.performance.qwen38.runner import SHORT
    async def run():
        async with httpx.AsyncClient() as client:
            r = await client.get(base_url + '/health/ready'); r.raise_for_status(); health = r.json()
        runtime = health['runtime']
        check = partial(inventory, [int(x) for x in runtime['cuda_visible_devices'].split(',')],
                        [w['pid'] for w in runtime['workers']])
        await run_benchmark(base_url + '/v1', health['model'], SHORT, tokens, output,
                            inventory=check, concurrencies=(1, 2, 3, 4), rounds=rounds)
    asyncio.run(run())


@app.command()
def protocol(
    output: Path = typer.Option(...),
    base_url: str = typer.Option('http://127.0.0.1:8120'),
):
    """复用原生产路径探针，验证真实推理的格式约束与断连回收。"""
    from tests.performance.qwen38.production_verify import probes
    async def run():
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.get(base_url+'/health/ready');r.raise_for_status();service=r.json()
            result = await probes(base_url, service['model'])
            tool = {'type':'function','function':{'name':'add','description':'计算两个整数之和',
                'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},
                              'required':['a','b'],'additionalProperties':False}}}
            r = await client.post(base_url+'/v1/chat/completions',json={
                'model':service['model'],'messages':[{'role':'user','content':'调用add工具，参数a=3，b=4。'}],
                'temperature':0,'max_tokens':128,'tools':[tool],'tool_choice':'required'})
            r.raise_for_status();result['tool_call']=r.json()
            call = result['tool_call']['choices'][0]['message']['tool_calls'][0]['function']
            assert call['name']=='add' and json.loads(call['arguments'])=={'a':3,'b':4}
            final = (await client.get(base_url+'/health/ready')).json()
            result['service']=service
            result['instance_unchanged']=service['instance_id']==final['instance_id']
            assert result['instance_unchanged']
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    asyncio.run(run())


@app.command()
def evaluate(
    output: Path = typer.Option(...),
    dataset: Path = typer.Option(Path('outputs/nvidia-20261028/gsm8k-test.jsonl')),
    base_url: str = typer.Option('http://127.0.0.1:8120'),
    math_samples: int = typer.Option(1319),
):
    from tests.performance.qwen38.validation import quality
    async def run():
        async with httpx.AsyncClient() as client:
            r = await client.get(base_url + '/health/ready'); r.raise_for_status(); model = r.json()['model']
        output.parent.mkdir(parents=True, exist_ok=True)
        result = await quality(base_url, model, output, math_samples=math_samples, dataset=dataset)
        typer.echo(json.dumps(result, ensure_ascii=False))
        if result['execution_errors'] or not result['instance_unchanged']:
            raise typer.Exit(1)
    asyncio.run(run())


if __name__ == '__main__':
    app()
