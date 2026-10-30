"""指定 GPU 的显存预算监督器；持续服务，日志轮转，stop.json 请求退出。"""
import asyncio
import hashlib
import json
import os
import signal
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import subprocess
import sys
import time

import httpx
import pynvml
from tests.performance.nvidia.inventory import inventory

config = json.loads(Path(sys.argv[1]).read_text())
GPU = config['gpu']
URL = f'http://127.0.0.1:{8130+GPU}'
ROOT = Path(config['run_dir'])
ROOT.mkdir(exist_ok=False, parents=True)
def save(name, value):
    temporary = ROOT/(name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(ROOT/name)

async def main():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(GPU)
    assert not pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    env = os.environ.copy()
    for key in ('PYTHONPATH','RSS_DIAGNOSTICS','QWEN_RSS_NATIVE_DIR','QWEN_RSS_SHM_VARIANT'):
        env.pop(key, None)
    env['CUDA_VISIBLE_DEVICES'] = str(GPU)
    env['VLLM_CACHE_ROOT'] = f'/workspace/qwen-nvidia/cache/{config["cache"]}/vllm'
    env['TRITON_CACHE_DIR'] = f'/workspace/qwen-nvidia/cache/{config["cache"]}/triton'
    args = ['poetry','run','python','-m','tests.performance.nvidia.cli','serve',
        '--host',config.get('host','0.0.0.0'),
        '--model-path',config.get('model_path','/workspace/model-bin/windowsxp811203/Qwen3.8-27B-Abliterated-W4A16-AWQ-GPTQ-NVIDIA'),
        '--concurrency',str(config.get('concurrency',4)),'--model-name','qwen38-24g','--port',str(8130+GPU),'--tp','1','--context',str(config['context']),
        '--max-output-tokens',str(config.get('max_output_tokens',8192)),
        '--kv-dtype','fp8_e4m3','--attention-backend','FLASHINFER','--cuda-graph',
        '--gpu-memory-utilization',str(config['utilization']), '--batch-tokens',str(config['batch_tokens'])]
    if config.get('kv_gib') is not None:
        args += ['--kv-cache-gib',str(config['kv_gib'])]
    if config.get('prefix_caching', False):
        args.append('--prefix-caching')
    if config['force_length']:
        args.append('--force-length')
    with (ROOT/'service.log').open('w') as log:
        proc = subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save('process.json',{'pid':proc.pid,'config':config,'command':args,'limit_bytes':24*1024**3,
                         'cache_path':env['VLLM_CACHE_ROOT'],'gpu':GPU,
                         'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'worker_source_sha256':hashlib.sha256(Path('src/qwen_llm/engine/nvidia_worker.py').read_bytes()).hexdigest()})
    done = asyncio.Event()
    sample_count = 0
    peak_used = 0
    serving_peak = 0
    stop_requested = False
    def request_stop():
        nonlocal stop_requested
        stop_requested = True
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, request_stop)
    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, request_stop)
    exceeded = False
    pids = [proc.pid]
    state = 'startup'
    async def monitor():
        nonlocal exceeded, sample_count, peak_used, serving_peak
        handler = RotatingFileHandler(ROOT/'memory.jsonl', maxBytes=10*1024**2, backupCount=2)
        logger = logging.getLogger('gpu_budget')
        logger.setLevel(logging.INFO); logger.addHandler(handler); logger.propagate = False
        try:
            while not done.is_set():
                m = pynvml.nvmlDeviceGetMemoryInfo(handle)
                row = {'time':time.time(),'used_bytes':m.used,'stage':state}
                logger.info(json.dumps(row)); sample_count += 1
                peak_used = max(peak_used,m.used)
                if state == 'serving': serving_peak = max(serving_peak,m.used)
                if sample_count % 50 == 0:
                    save('budget-status.json',dict(row,peak_used_bytes=peak_used,samples=sample_count))
                if m.used > 24*1024**3 and not exceeded:
                    exceeded = True
                    if proc.poll() is None: proc.terminate()
                    save('over-budget.json',row)
                await asyncio.sleep(0.1)
        finally:
            handler.close(); logger.removeHandler(handler)
    task = asyncio.create_task(monitor())
    reason = None
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            for _ in range(480):
                if task.done():
                    reason = 'monitor_failed'; break
                if stop_requested or (ROOT/'stop.json').exists():
                    reason = 'requested_stop'; break
                if proc.poll() is not None:
                    reason = 'over_budget' if exceeded else 'startup_failed'
                    break
                try:
                    response = await client.get(URL+'/health/ready')
                    response.raise_for_status()
                    ready = response.json()
                    assert ready['runtime']['cuda_visible_devices']==str(GPU)
                    inv = inventory([GPU],[w['pid'] for w in ready['runtime']['workers']])
                    assert inv['isolated']
                    pids = [p['pid'] for p in inv['processes']]
                    save('ready.json',ready);save('ready-inventory.json',inv)
                    state = 'serving'
                    print('READY',str(ROOT),flush=True)
                    break
                except httpx.HTTPError:
                    await asyncio.sleep(2)
            else:
                reason = 'startup_timeout'
            if state == 'serving':
                while True:
                    if task.done():
                        reason = 'monitor_failed'; break
                    if stop_requested or (ROOT/'stop.json').exists():
                        reason = 'requested_stop';break
                    if proc.poll() is not None:
                        reason = 'over_budget' if exceeded else 'runtime_failed';break
                    await asyncio.sleep(1)
    finally:
        state = 'shutdown'
        if proc.poll() is None:
            proc.terminate()
        for _ in range(90):
            proc.poll()
            if not any(Path(f'/proc/{pid}').exists() for pid in pids):
                break
            await asyncio.sleep(1)
        else:
            raise RuntimeError('owned processes did not exit')
        done.set();await task
        owners = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        save('complete.json',{'reason':reason,'returncode':proc.returncode,'over_budget':exceeded,
            'peak_used_bytes':peak_used,
            'serving_peak_used_bytes':serving_peak,
            'samples':sample_count,'sample_interval_seconds':0.1,
            'owned_pids_released':pids,'gpu_compute_processes_after':[p.pid for p in owners]})
        print('COMPLETE',reason,flush=True)

asyncio.run(main())
