"""固定22.5GiB预算的BF16/C8校准、实际缓存池观测及四路边界测试。"""
import argparse
import gc
import json
import os
import re
import signal
import subprocess
from pathlib import Path
import threading
import time
import uuid

from .benchmark import inventory

MODEL='/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend-GDNHeadW8-v1'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--mode',choices=['bf16','int8'],required=True)
    p.add_argument('--scales',type=Path);p.add_argument('--long',action='store_true')
    p.add_argument('--align-c8-pages',action='store_true')
    p.add_argument('--model',default=MODEL);p.add_argument('--vocab4',action='store_true')
    p.add_argument('--kv-gib',type=float,default=22.5);p.add_argument('--speed',action='store_true')
    p.add_argument('--hbm-limit-mib',type=int)
    p.add_argument('--speed-tokens',type=int,default=64)
    p.add_argument('--speed-concurrencies',default='1,4')
    p.add_argument('--skip-calibration',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--share-c8-tables',action='store_true')
    p.add_argument('--long-output-tokens',type=int,default=256)
    p.add_argument('--mtp',type=int,choices=[0,2],default=0)
    a=p.parse_args()
    if a.align_c8_pages and a.mode!='int8':raise ValueError('C8页对齐只用于INT8实验')
    if a.output.exists() or inventory()['owners']:raise RuntimeError('需要新目录与空闲NPU')
    a.output.mkdir(parents=True)
    manifest={'mode':a.mode,'model':a.model,'vocab4':a.vocab4,'align_c8_pages':a.align_c8_pages,'kv_budget_bytes':int(a.kv_gib*1024**3),'mtp':a.mtp,'valid':False,'phase':'startup'}
    stop=threading.Event();started=time.monotonic();llm=None
    def save():
        manifest['elapsed_seconds']=time.monotonic()-started
        temporary=a.output/'progress.tmp';temporary.write_text(json.dumps(dict(manifest),ensure_ascii=False,indent=2))
        temporary.replace(a.output/'manifest.json')
    def heartbeat():
        while not stop.wait(2):
            try:
                raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=5)
                line=next(s for s in raw.splitlines() if '/ 65536' in s)
                used=int(re.findall(r'(\d+)\s*/\s*65536',line)[-1])
                manifest.setdefault('hbm_samples',[]).append({'seconds':time.monotonic()-started,'used_mib':used,'phase':manifest['phase']})
                manifest['max_sampled_hbm_mib']=max(used,manifest.get('max_sampled_hbm_mib',0))
                if llm is not None:
                    metrics={m.name:m.value for m in llm.get_metrics() if hasattr(m,'value')
                        and any(k in m.name for k in ('cache_usage','num_requests_running','num_requests_waiting'))}
                    manifest.setdefault('engine_samples',[]).append({'seconds':time.monotonic()-started,**metrics})
                if a.hbm_limit_mib and used>a.hbm_limit_mib and not manifest.get('budget_exceeded'):
                    manifest['budget_exceeded']=True
                    os.kill(os.getpid(),signal.SIGINT)
            except Exception as error:manifest['monitor_error']=repr(error)
            save()
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    from vllm import LLM,SamplingParams
    from vllm.config import ProfilerConfig
    from transformers import AutoTokenizer
    if a.align_c8_pages:
        # Scope: this experiment process only. Let the installed Ascend layout
        # rule calculate pages using the actual C8 item size. The public cache
        # dtype remains auto; the worker selects INT8 for attention only.
        import vllm_ascend.patch.platform
        from vllm.model_executor.models.config import HybridAttentionMambaModelConfig
        original=HybridAttentionMambaModelConfig.verify_and_update_config
        @classmethod
        def c8_layout(cls,config):
            saved=config.cache_config.cache_dtype
            config.cache_config.cache_dtype='int8'
            try:original(config)
            finally:config.cache_config.cache_dtype=saved
        HybridAttentionMambaModelConfig.verify_and_update_config=c8_layout
    llm=None
    try:
        llm=LLM(model=a.model,tokenizer=a.model,dtype='bfloat16',quantization='ascend',
            worker_cls='tests.performance.qwen38.kv_capacity_worker.KVCapacityWorker',
            max_model_len=262144,max_num_seqs=4,max_num_batched_tokens=8192,
            kv_cache_memory_bytes=int(a.kv_gib*1024**3),gpu_memory_utilization=.82,
            tensor_parallel_size=1,enforce_eager=False,language_model_only=True,
            compilation_config={'mode':3,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[n*(a.mtp+1) for n in (1,2,4)]},
            speculative_config={'method':'qwen3_5_mtp','num_speculative_tokens':2,'enforce_eager':False} if a.mtp else None,
            disable_log_stats=False,
            profiler_config=ProfilerConfig(profiler='torch',torch_profiler_dir=str(a.output/'profile'),ignore_frontend=True) if a.profile else None,
            enable_prefix_caching=True,mamba_cache_mode='align',enable_chunked_prefill=True,
            async_scheduling=not bool(a.mtp),additional_config={'kv_capacity_mode':a.mode,'kv_scales':str(a.scales),'vocab4':a.vocab4,
                'share_c8_tables':a.share_c8_tables,
                'ascend_compilation_config':{'enable_npugraph_ex':True}})
        manifest['pool']=llm.collective_rpc('kv_status')[0];manifest['phase']='short_checks'
        assert manifest['pool']['graph']['graphs_captured']>=3 and not manifest['pool']['graph']['enforce_eager']
        if a.mtp:assert manifest['pool']['graph']['draft_graph_enabled'] and not manifest['pool']['graph']['draft_enforce_eager']
        tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
        def chat(text):return tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        def generate(ids,count,force=False):
            before={m.name:m.value for m in llm.get_metrics() if 'preempt' in m.name}
            t=time.monotonic()
            outputs=llm.generate([{'prompt_token_ids':x} for x in ids],
                SamplingParams(temperature=0,seed=42,max_tokens=count,ignore_eos=force),use_tqdm=False)
            after={m.name:m.value for m in llm.get_metrics() if 'preempt' in m.name}
            return {'seconds':time.monotonic()-t,'preemptions_delta':{n:v-before.get(n,0) for n,v in after.items()},'responses':[{'input_tokens':len(o.prompt_token_ids),
                'output_tokens':len(o.outputs[0].token_ids),'text':o.outputs[0].text,
                'finish_reason':o.outputs[0].finish_reason,'cached_tokens':getattr(o,'num_cached_tokens',None)} for o in outputs]}
        checks=generate([chat(s) for s in ('只输出ABC','只输出数字：1+1等于几？','只输出数字：6乘7等于几？')],32)
        manifest['short_checks']=checks
        expected=['ABC','2','42'];manifest['short_checks_passed']=all(r['text'].strip()==e for r,e in zip(checks['responses'],expected))
        if not manifest['short_checks_passed']:raise RuntimeError('短请求答案未对齐')
        if a.mode=='bf16' and not a.skip_calibration:
            manifest['phase']='calibration'
            llm.collective_rpc('begin_calibration')
            material=('项目组讨论了发布计划、接口性能、数据库事务、缓存容量及下一阶段安排。负责人确认每项任务的时间和风险。\n'*180)
            manifest['calibration']=generate([chat(material),chat('请总结以下技术讨论。\n'+material)],16)
            scales=llm.collective_rpc('export_scales')[0]
            (a.output/'scales.json').write_text(json.dumps(scales)+'\n')
        manifest['pool']=llm.collective_rpc('kv_status')[0]
        if a.speed:
            from .runner import SHORT
            manifest['phase']='speed';manifest['speed']=[]
            speculative_before={m.name:m.value for m in llm.get_metrics()
                if hasattr(m,'value') and 'spec_decode' in m.name}
            for concurrency in map(int,a.speed_concurrencies.split(',')):
                if concurrency not in (1,2,4):raise ValueError('速度测试只支持1/2/4并发')
                for i in range(4):
                    params=SamplingParams(temperature=0,seed=42,max_tokens=a.speed_tokens,ignore_eos=True)
                    first={};finished={};t=time.monotonic()
                    for j in range(concurrency):llm.llm_engine.add_request(f'speed-{concurrency}-{i}-{j}',{'prompt_token_ids':chat(SHORT)},params)
                    while llm.llm_engine.has_unfinished_requests():
                        for o in llm.llm_engine.step():
                            if o.outputs and o.outputs[0].token_ids:first.setdefault(o.request_id,time.monotonic()-t)
                            if o.finished:finished[o.request_id]=len(o.outputs[0].token_ids)
                    elapsed=time.monotonic()-t
                    if len(finished)!=concurrency or any(n!=a.speed_tokens for n in finished.values()):raise RuntimeError('速度测试长度不足')
                    if i:manifest['speed'].append({'concurrency':concurrency,'round':i,'seconds':elapsed,
                        'output_tokens_each':a.speed_tokens,'tokens_per_second':sum(finished.values())/elapsed,'ttft_ms':[x*1000 for x in first.values()]})
            manifest['speed_speculative_metrics_delta']={m.name:m.value-speculative_before.get(m.name,0)
                for m in llm.get_metrics() if hasattr(m,'value') and 'spec_decode' in m.name}
        if a.profile:
            manifest['phase']='profile'
            from .runner import SHORT
            llm.start_profile()
            try:manifest['profile_request']=generate([chat(SHORT) for _ in range(4)],16,True)
            finally:llm.stop_profile()
        if a.long:
            manifest['phase']='long_c4'
            capacity=manifest['pool']['capacity_by_context']
            blocks=manifest['pool']['num_blocks']
            context=max(int(n) for n,c in capacity.items() if c>=4 and 4*round(blocks/c)<=blocks-1)
            output=a.long_output_tokens;manifest['long_output_tokens']=output;ids=[];needle='739251'
            filler=tokenizer.encode('项目进度正常，技术团队正在核对接口、部署记录和测试结果。\n',add_special_tokens=False)
            for i in range(4):
                text=f'独立材料{uuid.uuid4().hex}。请在最后回答文中的检索码。\nPLACEHOLDER\n只输出检索码，禁止解释。'
                template=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                left,right=template.split('PLACEHOLDER');prefix=tokenizer.encode(left,add_special_tokens=False);suffix=tokenizer.encode(right,add_special_tokens=False)
                mark=tokenizer.encode('\n检索码为'+needle+'。\n',add_special_tokens=False)
                n=context-output-len(prefix)-len(suffix)-len(mark)
                padding=(filler*(n//len(filler)+1))[:n];middle=n//2
                ids.append(prefix+padding[:middle]+mark+padding[middle:]+suffix)
            manifest['long_context']=context
            result=generate(ids,output,True);manifest['long_result']=result
            manifest['long_full_lengths']=all(r['input_tokens']==context-output and r['output_tokens']==output for r in result['responses'])
            manifest['long_retrieval_hits']=sum(needle in r['text'] for r in result['responses'])
            if not manifest['long_full_lengths']:raise RuntimeError('长请求没有达到目标长度')
        manifest['final_runtime']=llm.collective_rpc('kv_status')[0]
        manifest['valid']=True;manifest['phase']='complete'
    except BaseException as error:
        manifest['error']=repr(error);raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
            del llm;gc.collect()
        stop.set();thread.join();save()


if __name__=='__main__':main()
