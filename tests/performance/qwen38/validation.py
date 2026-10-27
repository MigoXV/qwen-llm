"""独立 HTTP 质量、长文隔离及长稳态验证；逐请求保存原始结果。"""
import ast
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time

import httpx
from .evaluation import cases as smoke_cases, check_answer
from .client import run as stream_run


async def allocator_state(engine):
    states = await engine.llm.collective_rpc('diagnostic_allocator_state')
    if not states or not all(states):
        raise RuntimeError('未取得实际KV分配池状态')
    for worker in states:
        for state in worker:
            for name in ('uncached_index_matches','metadata_bounded',
                         'checkpoint_owners_coherent','old_state_index_matches'):
                if not state[name]:raise RuntimeError(f'KV分配池不变量失败：{name}')
    return states


def code_cases():
    items=[]
    for k in range(1,9):
        definitions=[
            (f'返回列表xs中所有元素加{k}后构成的新列表。', lambda xs,k=k:[x+k for x in xs]),
            (f'返回列表xs中所有能被{k+1}整除的元素之和。', lambda xs,k=k:sum(x for x in xs if x%(k+1)==0)),
            (f'返回列表xs中严格大于{k}的元素个数。', lambda xs,k=k:sum(x>k for x in xs)),
            (f'将xs去重后升序排列，再返回前{k}个元素。', lambda xs,k=k:sorted(set(xs))[:k]),
            (f'返回列表xs中每个元素乘{k}后的绝对值列表。',lambda xs,k=k:[abs(x*k) for x in xs]),
            (f'保留xs中绝对值不大于{k}的元素并反转顺序。',lambda xs,k=k:[x for x in xs if abs(x)<=k][::-1]),
            (f'返回xs中偶数元素的平方之和，再加{k}。',lambda xs,k=k:sum(x*x for x in xs if x%2==0)+k),
            (f'返回列表xs的每个长度为{k}的连续窗口之和；不足一个窗口时返回空列表。',lambda xs,k=k:[sum(xs[i:i+k]) for i in range(len(xs)-k+1)])]
        for j,(desc,oracle) in enumerate(definitions):
            inputs=[[],[0],[-3,2,2,8,-1,0,5],[1,2,3,4,5,6,7,8,9]]
            items.append({'id':f'code-{k}-{j}','category':'code','prompt':desc+'\n使用Python编写函数solve(xs)。只输出函数代码，不导入模块，不添加解释。',
                          'check':'code','tests':[{'input':xs,'expected':oracle(xs)} for xs in inputs]})
    return items


async def check_code_isolated(text, tests):
    import sys
    proc = await asyncio.create_subprocess_exec(sys.executable, '-I', str(Path(__file__).with_name('code_check.py')),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(json.dumps({'text':text,'tests':tests}).encode()), timeout=5)
        return proc.returncode == 0 and json.loads(stdout)['passed']
    except (asyncio.TimeoutError, ValueError):
        if proc.returncode is None: proc.kill()
        await proc.wait()
        return False


async def quality(url,model,output,*,math_samples=128,include_code=True):
    items=smoke_cases()
    dataset=Path('/home/qwen-models/evaluation/gsm8k-test.jsonl')
    raw=dataset.read_bytes()
    for i,line in enumerate(raw.decode().splitlines()[:math_samples]):
        r=json.loads(line)
        items.append({'id':f'gsm8k-{i}','category':'gsm8k','check':'final_number',
                      'expected':r['answer'].rsplit('####',1)[1].strip().replace(',',''),
                      'prompt':r['question']+'\nSolve the problem. End with a separate line FINAL: <answer number>.'})
    if include_code:items+=code_cases()
    sem=asyncio.Semaphore(4)
    output=Path(output);progress=output.with_suffix('.jsonl');started=time.monotonic()
    async with httpx.AsyncClient(timeout=1800) as client:
        identity=(await client.get(url+'/health/ready')).json()
        async def one(case):
            async with sem:
                begin=time.monotonic()
                try:
                    r=await client.post(url+'/v1/responses',json={'model':model,'input':case['prompt'],
                        'temperature':0,'seed':42,'reasoning':{'effort':'none'},'max_output_tokens':2048 if case['category']=='gsm8k' else 512 if case['category']=='code' else 256})
                    r.raise_for_status();value=r.json()
                    text=''.join(p.get('text','') for x in value.get('output',[]) if x.get('type')=='message' for p in x.get('content',[]))
                    okay=await check_code_isolated(text,case['tests']) if case['check']=='code' else check_answer(case,text)
                    result={**case,'response':value,'text':text,'passed':value.get('status')=='completed' and okay,'seconds':time.monotonic()-begin}
                except Exception as exc:
                    result={**case,'error':repr(exc),'passed':False,'seconds':time.monotonic()-begin}
                with progress.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                return result
        results=await asyncio.gather(*(one(c) for c in items))
        final=(await client.get(url+'/health/ready')).json()
    record={'samples':results,'service':identity,'instance_unchanged':identity['instance_id']==final['instance_id'],
            'dataset_sha256':hashlib.sha256(raw).hexdigest(),
            'code_checker_sha256':hashlib.sha256(Path(__file__).with_name('code_check.py').read_bytes()).hexdigest(),'math_samples':math_samples,
            'prompt_policy':'greedy nonthinking; solve and FINAL answer; math 2048 cap, smoke 256, code 512',
            'seconds':time.monotonic()-started,'execution_errors':sum('error' in x for x in results),
            'categories':{cat:{'passed':sum(x['passed'] for x in results if x['category']==cat),'total':sum(x['category']==cat for x in results)} for cat in sorted({x['category'] for x in results})}}
    output.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    return {k:v for k,v in record.items() if k!='samples'}


def retrieval_prompt(tokenizer,length,index,position):
    from .runner import make_long_prompt,LONG_SUFFIX
    # 使用已验证可构造的材料长度，再独立调整检索提示，覆盖块边界相邻长度。
    source_length = max(8192, length)
    body=make_long_prompt(tokenizer,source_length,index,'qwen38-hotspots-v1').removesuffix(LONG_SUFFIX)
    ids=tokenizer.encode(body,add_special_tokens=False)
    code=f'QZ{index:02d}F{length//1024:02d}X'
    marker=f'\n本材料的唯一校验码是{code}。\n'
    suffix='\n只输出本材料的唯一校验码，不添加其他文字。'
    def render(n,pad=0):
        at={'begin':0,'middle':n//2,'end':n}[position]
        return tokenizer.decode(ids[:at])+marker+tokenizer.decode(ids[at:n])+(' data'*pad)+suffix
    def count(text):
        return len(tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,
            add_generation_prompt=True,enable_thinking=False))
    n=min(len(ids),length)-len(tokenizer.encode(marker+suffix))
    for _ in range(12):
        text=render(n);actual=count(text)
        if actual==length and '\ufffd' not in text:return text,code
        n=max(0,min(len(ids),n+length-actual))
    # BPE切分可能在相邻长度间振荡，检查附近有效UTF-8切点，最多用64个token补齐。
    for cut in range(n,max(-1,n-65),-1):
        text=render(cut)
        if '\ufffd' in text:continue
        missing=length-count(text)
        if 0<=missing<=64:
            for pad in range(max(0,missing-3),missing+4):
                candidate=render(cut,pad)
                if count(candidate)==length:return candidate,code
    raise ValueError(f'无法构造精确长度的检索材料：{length}/{index}/{position}')


async def long_quality(engine,url,model,output,*,lengths=(8192,32768,57344),positions=('begin','middle','end')):
    rows=[]
    async with httpx.AsyncClient(timeout=1800) as client:
        for length in lengths:
            for position in positions:
                material=[retrieval_prompt(engine.tokenizer,length,i,position) for i in range(4)]
                await engine.llm.reset_prefix_cache()
                for cache in ('cold','hot'):
                    async def one(i):
                        prompt,expected=material[i];t=time.monotonic()
                        r=await client.post(url+'/v1/responses',json={'model':model,'input':prompt,'temperature':0,'seed':42,'max_output_tokens':32})
                        r.raise_for_status();v=r.json()
                        if v['usage']['input_tokens'] != length: raise ValueError('服务端检索输入长度不一致')
                        if cache == 'cold' and v['usage']['input_tokens_details']['cached_tokens']: raise ValueError('冷检索意外命中缓存')
                        text=''.join(p.get('text','') for x in v.get('output',[]) if x.get('type')=='message' for p in x.get('content',[]))
                        return {'length':length,'position':position,'cache':cache,'index':i,'expected':expected,'text':text,
                                'passed':v['status']=='completed' and check_answer({'check':'exact','expected':expected},text),'response':v,'seconds':time.monotonic()-t}
                    rows+=await asyncio.gather(*(one(i) for i in range(4)))
                    Path(output).write_text(json.dumps({'samples':rows,'passed':sum(x['passed'] for x in rows),'total':len(rows)},ensure_ascii=False,indent=2))
    return {'passed':sum(x['passed'] for x in rows),'total':len(rows)}


async def stability(engine,url,model,output,seconds):
    from .runner import SHORT,make_long_prompt
    long=make_long_prompt(engine.tokenizer,32768,0,'qwen38-stability')
    start=time.monotonic();rounds=[]
    while time.monotonic()-start<seconds:
        prompts=[SHORT]*3+[long]
        if len(rounds)%2==0:await engine.llm.reset_prefix_cache()
        result=await stream_run({'url':url,'model':model,'prompts':prompts,'output_tokens':128,'delays':[0,0,0,1.5]})
        from .benchmark import inventory
        rounds.append({'elapsed_seconds':time.monotonic()-start,'http':result,'inventory':inventory(),
                       'runtime':await engine.llm.collective_rpc('runtime_status'),
                       'allocator':await allocator_state(engine)})
        Path(output).write_text(json.dumps({'requested_seconds':seconds,'elapsed_seconds':time.monotonic()-start,'rounds':rounds},ensure_ascii=False,indent=2))
        if any('error' in x for x in result['samples']):raise RuntimeError('长稳态请求失败')
    return {'rounds':len(rounds),'elapsed_seconds':time.monotonic()-start}


async def prefix_growth(engine,url,model,output):
    """四路前缀连续增长16次，验证检查点元数据有界及跨请求隔离。"""
    prompts=[];codes=[];rows=[];states=[]
    tokenizer=engine.tokenizer
    for i in range(4):
        text,code=retrieval_prompt(tokenizer,8192,i,'middle')
        prompts.append(text);codes.append(code)
    await engine.llm.reset_prefix_cache()
    def count(text):
        return len(tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,
            add_generation_prompt=True,enable_thinking=False))
    async with httpx.AsyncClient(timeout=1800) as client:
        for step in range(17):
            target=8192+1536*step
            if step:
                for i in range(4):
                    prefix=prompts[i]+f'\n追加材料第{step}轮线路{i}：'
                    suffix='\n请仍然只输出最初材料的唯一校验码。'
                    guess=target-count(prefix+suffix)
                    for n in range(max(0,guess-8),guess+9):
                        text=prefix+' data'*n+suffix
                        if count(text)==target:
                            prompts[i]=text;break
                    else:raise ValueError('无法构造精确长度的增长前缀')
            async def one(i):
                started=time.monotonic()
                r=await client.post(url+'/v1/responses',json={'model':model,'input':prompts[i],
                    'temperature':0,'seed':42,'max_output_tokens':32})
                r.raise_for_status();v=r.json()
                if v['usage']['input_tokens'] != target: raise ValueError('服务端增长前缀长度不一致')
                text=''.join(p.get('text','') for x in v.get('output',[]) if x.get('type')=='message' for p in x.get('content',[]))
                return {'step':step,'index':i,'input_tokens':target,'prompt_sha256':hashlib.sha256(prompts[i].encode()).hexdigest(),
                    'expected':codes[i],'text':text,'passed':v['status']=='completed' and check_answer({'check':'exact','expected':codes[i]},text),
                    'seconds':time.monotonic()-started,'response':v}
            rows+=await asyncio.gather(*(one(i) for i in range(4)))
            states.append({'step':step,'workers':await allocator_state(engine)})
            Path(output).write_text(json.dumps({'samples':rows,'allocator_states':states,'passed':sum(r['passed'] for r in rows),'total':len(rows)},ensure_ascii=False,indent=2))
    return {'passed':sum(r['passed'] for r in rows),'total':len(rows),'allocator_checks':len(states)}
