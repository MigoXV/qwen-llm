"""四路精确 8192-token 输入基准；QWEN_LONG_TAG/OUTPUT/ROUNDS 配置输出。"""
import asyncio
import json
import os
from pathlib import Path
import time

import httpx
from transformers import AutoTokenizer

from qwen_llm.benchmark import inventory

root=Path(os.getenv('QWEN_BUILD_ROOT','/home/qwen-llm-build'))
base_url=os.getenv('QWEN_BASE_URL','http://127.0.0.1:8100/v1').removesuffix('/v1').rstrip('/')
tag=os.getenv('QWEN_LONG_TAG','eager')
model_name=os.getenv('QWEN_MODEL_NAME','qwen3-14b')
output_tokens=int(os.getenv('QWEN_LONG_OUTPUT','128'))
rounds=int(os.getenv('QWEN_LONG_ROUNDS','3'))
cold=os.getenv('QWEN_LONG_COLD','false').lower() in ('1','true')
cold_namespace=os.getenv('QWEN_LONG_COLD_NAMESPACE','cold-20261015')
tokenizer=AutoTokenizer.from_pretrained(os.getenv('QWEN_MODEL_PATH','model-bin/Qwen3-14B-W8A8-Ascend'),local_files_only=True)

def tokens(content):
    return len(tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=True,add_generation_prompt=True,enable_thinking=False))

def make_prompts(iteration=0):
    prompts=[]
    for request_index in range(4):
        prefix=f'独立测试材料编号 {request_index+1}。以下是占位材料：\n'
        if cold:
            prefix=f'{cold_namespace}轮次{iteration}线路{request_index+1}。以下是占位材料：\n'
        suffix='\n材料结束。请从1开始连续逐行输出整数直到20000，每行一个数字，不解释，不省略，不要提前结束。'
        count=8192-tokens(prefix+suffix)
        for count in range(max(0,count-4),count+5):
            candidate=prefix+' data'*count+suffix
            if tokens(candidate)==8192:
                prompts.append(candidate)
                break
        else:raise RuntimeError('未构造出精确的8192-token输入')
    return prompts

async def main():
    from urllib.parse import urlsplit
    if urlsplit(base_url).hostname not in ('127.0.0.1','localhost','::1'):
        raise RuntimeError('仅支持本机基准，以保证设备盘点匹配')
    before=await asyncio.to_thread(inventory)
    if not before['isolated']:raise RuntimeError('非独占设备')
    record={'tag':tag,'input_tokens':8192,'max_output_tokens':output_tokens,'rounds':[], 'before':before,
            'cache_workload': 'cold' if cold else 'cold_then_hot', 'cold_namespace': cold_namespace if cold else None}
    async with httpx.AsyncClient(timeout=1500) as client:
        health=await client.get(base_url+'/health/ready')
        health.raise_for_status()
        record['service']=health.json()
        if record['service'].get('model') != model_name:raise RuntimeError('服务模型身份不匹配')
        async def one(index):
            started=time.perf_counter();first=None;final=None
            try:
                async with client.stream('POST',base_url+'/v1/responses',json={
                    'model':model_name,'input':prompts[index],'temperature':0,'stream':True,'max_output_tokens':output_tokens}) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data: '):continue
                        event=json.loads(line[6:])
                        if event['type']=='response.output_text.delta' and event.get('delta') and first is None:first=time.perf_counter()
                        if event['type'] in ('response.completed','response.incomplete','response.failed'):final=event['response']
                ended=time.perf_counter()
                return {'request_index':index,'seconds':ended-started,'ttft_seconds':first-started if first else None,
                        'response':final,'decode_tokens_per_second':(final['usage']['output_tokens']-1)/(ended-first)
                        if first and final and final.get('usage') else None}
            except Exception as exc:return {'request_index':index,'error':str(exc),'seconds':time.perf_counter()-started}
        (root/'run-phase').write_text(f'{tag}-long-{output_tokens}')
        for iteration in range(rounds):
            prompts=make_prompts(iteration)
            started=time.perf_counter()
            samples=await asyncio.gather(*(one(i) for i in range(4)))
            elapsed=time.perf_counter()-started
            success=[s for s in samples if s.get('response') and s['response']['status'] in ('completed','incomplete')]
            result={'round':iteration,'makespan_seconds':elapsed,'samples':samples,
                    'tokens_per_second':sum(s['response']['usage']['output_tokens'] for s in success)/elapsed,
                    'resources':await asyncio.to_thread(inventory)}
            record['rounds'].append(result)
            path=root/f'logs/long-{tag}-{output_tokens}.json'
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2))
            print('round',iteration,'seconds',elapsed,'throughput',result['tokens_per_second'],
                  'usages',[s.get('response',{}).get('usage') if s.get('response') else s for s in samples],flush=True)
            if len(success)!=4:raise RuntimeError('四路请求未全部成功')
            if cold and any(s['response']['usage']['input_tokens_details']['cached_tokens'] for s in success):
                raise RuntimeError('冷缓存测量意外命中缓存，请更换QWEN_LONG_COLD_NAMESPACE或重启服务')
    async with httpx.AsyncClient() as client:
        final_health=await client.get(base_url+'/health/ready')
        final_health.raise_for_status()
        if final_health.json()['instance_id'] != record['service']['instance_id']:
            raise RuntimeError('测量期间服务实例已更换')
    record['after']=await asyncio.to_thread(inventory)
    if not record['after']['isolated']:raise RuntimeError('测量结束设备不再独占')
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2))

if __name__=='__main__':asyncio.run(main())
