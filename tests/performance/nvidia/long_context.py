"""精确长度、多并发 CUDA HTTP 测量；逐轮落盘，显式记录 preemption。"""
import asyncio
from functools import partial
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

import httpx
import typer
from transformers import AutoTokenizer
from tests.performance.nvidia.inventory import inventory

app = typer.Typer()
SUFFIX = '\n材料结束。请从1开始连续逐行输出整数直到20000，每行一个数字，不解释，不省略，不要提前结束。'


def count(tokenizer, text):
    return len(tokenizer.apply_chat_template([{'role':'user','content':text}],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False))


def prompt_for(tokenizer, length, index, namespace, *, retrieval=False, position='middle'):
    prefix = f'{namespace} 独立材料编号 {index}。\n'
    expected = f'QZ{hashlib.sha256(prefix.encode()).hexdigest()[:10].upper()}'
    suffix = '\n只输出本材料的唯一校验码，不添加其他文字。' if retrieval else SUFFIX
    if retrieval:
        topics = [
            '档案馆按年份整理资料，工作人员核对日期和记录。',
            '学校根据教学计划组织活动，老师汇总学生的反馈。',
            '团队讨论软件版本、负责人、风险和后续测试安排。',
            '气象站记录温度、风速和降雨量，每晚检查观测表。']
        text = '\n'.join(f'第{i}条：{topics[i%4]}' for i in range(length//8+100))
        ids = tokenizer.encode(text, add_special_tokens=False)
        marker = '\n本材料的唯一校验码是'+expected+'。\n'
        def render(n):
            at = {'begin':0,'middle':n//2,'end':n}[position]
            return prefix+tokenizer.decode(ids[:at])+marker+tokenizer.decode(ids[at:n])+suffix
    else:
        def render(n): return prefix + ' data'*n + suffix
    n = length-count(tokenizer, render(0))
    for _ in range(12):
        text = render(n); delta = length-count(tokenizer,text)
        if delta == 0 and '\ufffd' not in text:return text, expected
        n += delta
    for cut in range(n, max(0,n-32), -1):
        base = render(cut)
        if '\ufffd' in base:continue
        delta = length-count(tokenizer,base)
        for pad in range(max(0,delta-3), max(0,delta)+4):
            text = base+' data'*pad
            if count(tokenizer,text)==length:return text,expected
    raise ValueError(f'无法构造精确长度{length}')


def preemptions(text):
    values=[float(m[1]) for m in re.finditer(r'^vllm:num_preemptions_total(?:\{[^\n]*\})? ([\d.e+]+)',text,re.M)]
    if not values:raise RuntimeError('未取得 vLLM 抢占计数，不能判定容量')
    return sum(values)


async def stream_one(client, url, model, prompt, tokens, expected=None):
    start=time.perf_counter();first=None;final=None;events=[]
    try:
        async with client.stream('POST',url+'/v1/responses',json={
            'model':model,'input':prompt,'stream':True,'temperature':0,'seed':42,'max_output_tokens':tokens}) as r:
            if r.status_code!=200:
                return {'http_status':r.status_code,'body':(await r.aread()).decode(), 'seconds':time.perf_counter()-start}
            async for line in r.aiter_lines():
                if not line.startswith('data: '):continue
                event=json.loads(line[6:]);now=time.perf_counter()
                if event['type'].endswith('text.delta') and event.get('delta'):
                    first=first or now;events.append({'seconds':now-start,'text':event['delta']})
                if event['type'] in ('response.completed','response.incomplete','response.failed'):
                    final=event['response']
        end=time.perf_counter()
        if final is None or final['status']=='failed':raise RuntimeError(f'生成未正常结束{final}')
        n=final['usage']['output_tokens']
        answer=''.join(p.get('text','') for x in final.get('output',[]) if x.get('type')=='message' for p in x.get('content',[]))
        return {'http_status':200,'seconds':end-start,'ttft_seconds':first-start if first else None,
                'decode_tokens_per_second':(n-1)/(end-first) if first and n>1 else None,
                'response':final,'events':events,'expected':expected,
                'retrieval_passed':answer.strip()==expected if expected else None,
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
    except Exception as e:
        return {'error':repr(e),'seconds':time.perf_counter()-start}


async def run(url, output, lengths, concurrencies, rounds, tokens, cache, retrieval, *, stop_on_preempt=False):
    output.mkdir(parents=True,exist_ok=False)
    async with httpx.AsyncClient(timeout=7500,limits=httpx.Limits(max_connections=16)) as client:
        r=await client.get(url+'/health/ready');r.raise_for_status();service=r.json()
        rt=service['runtime'];tokenizer=AutoTokenizer.from_pretrained(rt['tokenizer_path'],local_files_only=True)
        if retrieval and rt.get('force_length'):
            raise ValueError('检索质量测试要求自然停止，不能使用 force_length 容量服务')
        if cache not in ('cold','hot') or (cache=='hot' and not service['prefix_caching']):
            raise ValueError('cache 必须为 cold/hot，hot 要求服务启用前缀缓存')
        inv=partial(inventory,[int(x) for x in rt['cuda_visible_devices'].split(',')], [w['pid'] for w in rt['workers']])
        record={'service':service,'rounds':[],'cache':cache,'output_tokens':tokens,'retrieval':retrieval}
        for length in lengths:
            for concurrency in concurrencies:
                for position in (('begin','middle','end') if retrieval else ('none',)):
                    for iteration in range(rounds):
                        if length>service['max_input_tokens']:
                            record['rounds'].append({'input_tokens':length,'concurrency':concurrency,'error':'exceeds_service_input_limit'})
                            break
                        namespace=hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
                        ns=f'{namespace}-{length}-{concurrency}-{position}-{iteration if cache=="cold" else 0}'
                        material=[prompt_for(tokenizer,length,i,ns,retrieval=retrieval,position=position) for i in range(concurrency)]
                        before=inv()
                        if not before['isolated']:raise RuntimeError(f'设备归属发生变化{before}')
                        m0=(await client.get(url+'/metrics')).text
                        resources=[];done=asyncio.Event();requests=[];observed_preemptions=0
                        async def monitor():
                            nonlocal observed_preemptions
                            while not done.is_set():
                                resources.append(await asyncio.to_thread(inv))
                                if stop_on_preempt and len(resources)%5==0:
                                    current=(await client.get(url+'/metrics')).text
                                    observed_preemptions=max(observed_preemptions,preemptions(current)-preemptions(m0))
                                    if observed_preemptions>0:
                                        for request in requests:request.cancel()
                                        return
                                try:await asyncio.wait_for(done.wait(),timeout=1)
                                except asyncio.TimeoutError:pass
                        task=asyncio.create_task(monitor());begin=time.perf_counter()
                        try:
                            requests=[asyncio.create_task(stream_one(client,url,service['model'],p,tokens,e if retrieval else None)) for p,e in material]
                            values=await asyncio.gather(*requests,return_exceptions=True)
                            rows=[{'error':repr(x)} if isinstance(x,BaseException) else x for x in values]
                        finally:
                            done.set();await task
                        elapsed=time.perf_counter()-begin
                        # vLLM 的 Prometheus 计数由周期统计刷新；不计入请求计时。
                        await asyncio.sleep(5.1)
                        m1=(await client.get(url+'/metrics')).text
                        successful=[x for x in rows if x.get('http_status')==200]
                        result={'input_tokens':length,'concurrency':concurrency,'position':position,'iteration':iteration,
                            'samples':rows,'makespan_seconds':elapsed,
                            'tokens_per_second':sum(x['response']['usage']['output_tokens'] for x in successful)/elapsed,
                            'preemptions_delta':max(observed_preemptions,preemptions(m1)-preemptions(m0)),
                            'preemptions_before':preemptions(m0), 'preemptions_after':preemptions(m1),
                            'all_full_output':len(successful)==concurrency and all(x['response']['usage']['output_tokens']==tokens for x in successful),
                            'all_exact_input':len(successful)==concurrency and all(x['response']['usage']['input_tokens']==length for x in successful),
                            'all_uncached':len(successful)==concurrency and all(x['response']['usage']['input_tokens_details']['cached_tokens']==0 for x in successful),
                            'isolated':all(x['isolated'] for x in resources),
                            'resources':resources}
                        filename=f'{length}-c{concurrency}-{position}-r{iteration}.json'
                        (output/filename).write_text(json.dumps(result,ensure_ascii=False,indent=2))
                        (output/(filename+'.metrics')).write_text(m1)
                        record['rounds'].append({k:v for k,v in result.items() if k not in ('samples','resources')})
                        record['rounds'][-1]['file']=filename
                        (output/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
                        print(json.dumps(record['rounds'][-1]),flush=True)
                        if len(successful)!=concurrency:raise RuntimeError('存在失败请求；保留结果，停止本次矩阵')
        final=(await client.get(url+'/health/ready')).json()
        record['instance_unchanged']=final['instance_id']==service['instance_id']
        (output/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
        return record


@app.command()
def main(output:Path=typer.Option(...),base_url:str=typer.Option('http://127.0.0.1:8120'),
         lengths:str=typer.Option('8192'),concurrencies:str=typer.Option('1,2,3,4'),
         rounds:int=typer.Option(3),tokens:int=typer.Option(128),cache:str=typer.Option('cold'),
         retrieval:bool=typer.Option(False)):
    asyncio.run(run(base_url,output,[int(x) for x in lengths.split(',')],
                    [int(x) for x in concurrencies.split(',')],rounds,tokens,cache,retrieval))


if __name__=='__main__':app()
