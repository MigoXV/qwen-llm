"""使用普通生产CLI核对HTTP行为及新进程性能；没有测试Worker或引擎注入。"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import httpx
from transformers import AutoTokenizer

from .benchmark import inventory
from .client import run as stream_run
from .runner import SHORT, make_long_prompt, summarize_round


def response_text(value):
    return ''.join(p.get('text','') for x in value.get('output',[]) if x.get('type')=='message'
                   for p in x.get('content',[]))


async def probes(url,model):
    async with httpx.AsyncClient(timeout=1800) as client:
        r=await client.post(url+'/v1/responses',json={'model':model,'input':'只输出ABC','temperature':0,'max_output_tokens':32})
        r.raise_for_status();plain=r.json()
        assert plain['status']=='completed' and response_text(plain).strip()=='ABC',plain
        schema={'type':'object','properties':{'result':{'type':'integer','const':42}},
                'required':['result'],'additionalProperties':False}
        r=await client.post(url+'/v1/responses',json={'model':model,'input':'计算6乘7。','temperature':0,
            'max_output_tokens':128,'text':{'format':{'type':'json_schema','name':'product','schema':schema,'strict':True}}})
        r.raise_for_status();structured=r.json()
        assert structured['status']=='completed' and json.loads(response_text(structured))=={'result':42},structured
        cancelled=False
        async with client.stream('POST',url+'/v1/responses',json={'model':model,'input':SHORT,
            'stream':True,'temperature':0,'max_output_tokens':8192}) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith('data: '):
                    event=json.loads(line[6:])
                    if event['type'].endswith('text.delta') and event.get('delta'):
                        cancelled=True;break
        for _ in range(100):
            metrics=(await client.get(url+'/metrics')).text
            if 'responses_inflight 0.0' in metrics and re.search(r'responses_requests_total\{outcome="cancelled"\} [1-9]',metrics):break
            await asyncio.sleep(.1)
        else:raise RuntimeError('断开流后没有及时取消请求并回收并发槽')
        return {'plain':plain,'structured':structured,'disconnected_after_text':cancelled,'metrics_after_cancel':metrics}


async def verify(args):
    concurrencies=tuple(map(int,args.concurrencies.split(',')))
    if args.rounds<=0 or not concurrencies or not set(concurrencies)<={1,2,4}:raise ValueError('需要正轮数及1/2/4并发')
    if args.throughput_only and args.capacity_validation:raise ValueError('短吞吐模式不运行满容量长测')
    output=args.output.resolve()
    if output.exists():raise ValueError('拒绝覆盖生产路径验证结果')
    before=inventory()
    if before['owners']:raise RuntimeError('设备尚未释放')
    with socket.socket() as sock:sock.bind(('127.0.0.1',args.port))
    output.mkdir(parents=True)
    command=([ 'taskset','-c',args.cpus ] if args.cpus!='none' else [])+[sys.executable,'-m','qwen_llm.commands.app','serve-model',args.model,
             '--catalog',str(args.catalog.resolve()),'--port',str(args.port),'--kv-cache-gib','22.5',
             '--qwen38-fast-w4' if args.fast_w4 else '--no-qwen38-fast-w4']
    manifest={'command':command,'before':before,'cases':[],'valid':False,
              'path':'ordinary production CLI; no diagnostic Worker, profiler or injected engine options',
              'fast_w4_requested':args.fast_w4,'rounds_requested':args.rounds,'throughput_only':args.throughput_only}
    source_root=Path(__file__).resolve().parents[3]
    sources=[*source_root.joinpath('src/qwen_llm').rglob('*.py'),args.catalog.resolve()]
    manifest['implementation_sha256']={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sources}
    for relative in manifest['implementation_sha256']:
        destination=output/'source'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((source_root/relative).read_bytes())
    def save():(output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    save();url=f'http://127.0.0.1:{args.port}'
    started=time.monotonic()
    with (output/'server.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                for _ in range(900):
                    if process.poll() is not None:raise RuntimeError('普通生产CLI启动失败，见server.log')
                    try:
                        r=await client.get(url+'/health/ready')
                        if r.status_code==200:
                            manifest['ready']=r.json();break
                    except httpx.HTTPError:pass
                    manifest['startup_elapsed_seconds']=time.monotonic()-started;save()
                    await asyncio.sleep(1)
                else:raise TimeoutError('生产路径启动超时')
            workers=manifest['ready']['runtime']['workers']
            assert workers and all(w['worker'].startswith('qwen_llm.') and 'Profiling' not in w['worker'] for w in workers)
            modes=[w['model_adaptation']['w4_decode'] for w in workers]
            assert all(m['backend']==('native-tuned' if args.fast_w4 else 'upstream') for m in modes)
            assert all(m['layers']==128 and m['fast_M']==([6,12] if args.fast_w4 else []) for m in modes)
            if args.fast_w4:
                assert all(m.get('capture_dispatch',{}).get('6:1',0)>=128 and m.get('capture_dispatch',{}).get('12:1',0)>=128 for m in modes),modes
            manifest['w4_decode']=modes;save()
            if not args.throughput_only:manifest['probes']=await probes(url,args.model);save()
            cases=[{'name':f'generation512-c{n}','prompts':[SHORT]*n,'output_tokens':512,'cache':'cold'} for n in map(int,args.concurrencies.split(','))]
            if not args.throughput_only:
                catalog=json.loads(args.catalog.read_text());row=next(r for r in catalog if r['name']==args.model)
                tokenizer=AutoTokenizer.from_pretrained(row['path'],local_files_only=True)
                cases.append({'name':'chinese57344-c4-hot','prompts':[
                    make_long_prompt(tokenizer,57344,i,'qwen38-hotspots-v1') for i in range(4)],'output_tokens':128,'cache':'hot'})
            for case in cases:
                entry={k:v for k,v in case.items() if k!='prompts'};entry['rounds']=[]
                manifest['cases'].append(entry)
                directory=output/case['name'];directory.mkdir()
                for label in ['warmup',*[f'round{i}' for i in range(args.rounds)]]:
                    raw=await stream_run({'url':url,'model':args.model,'prompts':case['prompts'],'output_tokens':case['output_tokens']})
                    (directory/(label+'-http.json')).write_text(json.dumps(raw,ensure_ascii=False,indent=2))
                    summary=summarize_round(raw,{})
                    if not summary['valid'] or summary['output_tokens']!=[case['output_tokens']]*len(case['prompts']):raise RuntimeError('生产路径HTTP失败或输出长度不符')
                    if case['cache']=='cold' and any(summary['cached_tokens']):raise RuntimeError('短请求意外命中1536前缀块')
                    summary['inventory']=inventory()
                    if not summary['inventory']['isolated']:raise RuntimeError('设备失去独占')
                    if label=='warmup':entry['warmup']=summary
                    else:entry['rounds'].append(summary)
                    save()
            if args.throughput_only:
                manifest['probes']=await probes(url,args.model);save()
            if args.capacity_validation:
                # 普通Responses接口不能忽略EOS；实际达到8192才算满容量证据。
                prompts=[make_long_prompt(tokenizer,57344,i,'qwen38-capacity',repeated=True) for i in range(4)]
                samples=[]
                async def sample_capacity():
                    while True:
                        samples.append(await asyncio.to_thread(inventory))
                        await asyncio.sleep(10)
                sampler=asyncio.create_task(sample_capacity())
                try:
                    raw=await stream_run({'url':url,'model':args.model,'prompts':prompts,'output_tokens':8192})
                finally:
                    sampler.cancel();await asyncio.gather(sampler,return_exceptions=True)
                (output/'capacity-http.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2))
                summary=summarize_round(raw,{})
                if not summary['valid'] or summary['input_tokens']!=[57344]*4:raise RuntimeError('实际容量请求失败')
                manifest['capacity']={'summary':summary,'full_64k_all_four_verified':summary['output_tokens']==[8192]*4,
                                      'inventory_samples':samples}
                save()
            async with httpx.AsyncClient(timeout=10) as client:
                final=(await client.get(url+'/health/ready')).json()
                assert final['instance_id']==manifest['ready']['instance_id']
            manifest['valid']=True
        except BaseException as error:
            manifest['error']=repr(error);raise
        finally:
            descendants=[]
            def remember_children(pid):
                try:children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
                except FileNotFoundError:return
                for child in children:
                    remember_children(int(child));descendants.append(int(child))
            remember_children(process.pid)
            if process.poll() is None:process.terminate()
            try:await asyncio.to_thread(process.wait,60)
            except subprocess.TimeoutExpired:
                for pid in [*descendants,process.pid]:
                    try:
                        if os.getpgid(pid)==os.getpgrp():os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                await asyncio.to_thread(process.wait,10)
            manifest['exit_code']=process.returncode
            manifest['after']=inventory()
            with socket.socket() as sock:manifest['port_released']=sock.connect_ex(('127.0.0.1',args.port))!=0
            manifest['released']=not manifest['after']['owners']
            save()
            if not manifest['released'] or not manifest['port_released']:raise RuntimeError('生产路径退出后仍占用设备或端口')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,default=Path('configs/models.json'));parser.add_argument('--port',type=int,default=18100)
    parser.add_argument('--capacity-validation',action='store_true')
    parser.add_argument('--fast-w4',action='store_true');parser.add_argument('--throughput-only',action='store_true')
    parser.add_argument('--rounds',type=int,default=5);parser.add_argument('--concurrencies',default='1,4')
    parser.add_argument('--cpus',default='62,63')
    def stopped(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stopped)
    asyncio.run(verify(parser.parse_args()))


if __name__=='__main__':main()
