"""区分实际缓存池计算、完整长度推理及生产部署状态。"""
import argparse
import json
from pathlib import Path
import statistics


def context_bounds(pool):
    return {str(c):max((int(n) for n,v in pool['capacity_by_context'].items() if v>=c),default=0)
            for c in (1,2,4)}


def verified_long(m):
    result=m.get('long_result',{});rows=result.get('responses',[])
    output=m.get('long_output_tokens',64)
    preemptions=result.get('preemptions_delta',{})
    return bool(m.get('valid') and m.get('long_full_lengths') and len(rows)==4
        and preemptions and all(v==0 for v in preemptions.values())
        and all(r['input_tokens']==m['long_context']-output and r['output_tokens']==output
                and r.get('cached_tokens')==0 for r in rows))


def graph_verified(m):
    graph=m.get('final_runtime',{}).get('graph',{})
    return bool(m.get('valid') and graph.get('mode')=='FULL_DECODE_ONLY'
        and graph.get('enforce_eager') is False and graph.get('graphs_captured',0)>=3
        and (not m.get('mtp') or (graph.get('draft_enforce_eager') is False
                                 and graph.get('draft_graph_enabled') is True
                                 and graph.get('graphs_captured',0)>=6)))


def residency_verified(m):
    sample=m.get('final_runtime',{}).get('four_full_prompt_residency') or {}
    prompts=sample.get('prompt_tokens',[]);computed=sample.get('computed_tokens',[])
    expected=m.get('long_context',0)-m.get('long_output_tokens',0)
    return bool(len(sample.get('request_ids',[]))==4 and len(prompts)==len(computed)==4
        and all(p==expected and c>=p for p,c in zip(prompts,computed)))


def profile(m):
    speeds={}
    for concurrency in sorted({s['concurrency'] for s in m.get('speed',[])}):
        rows=[s for s in m.get('speed',[]) if s['concurrency']==concurrency]
        if rows:
            speeds[str(concurrency)]={'rounds':len(rows),
                'output_tokens_each':rows[0]['output_tokens_each'],
                'median_tokens_per_second':statistics.median(s['tokens_per_second'] for s in rows),
                'median_ttft_ms':statistics.median(t for s in rows for t in s['ttft_ms']),
                'raw':rows}
    pool=m.get('pool',{});runtime=m.get('final_runtime',{})
    return {'valid':m.get('valid',False),'mode':m['mode'],'mtp':m['mtp'],'vocab4':m.get('vocab4',False),
        'model':m.get('model'),'graph_verified':graph_verified(m),'graph':runtime.get('graph'),
        'kv_budget_bytes':m['kv_budget_bytes'],'cache_storage_bytes':pool.get('cache_storage_bytes'),
        'ready_npu_allocated_bytes':pool.get('npu_allocated_bytes'),
        'max_sampled_hbm_mib':m.get('max_sampled_hbm_mib'),'speed':speeds,
        'speed_speculative_metrics_delta':m.get('speed_speculative_metrics_delta'),
        'calculated_bounds_tokens':context_bounds(pool) if pool else None,
        'long_verified':graph_verified(m) and verified_long(m) and m.get('long_retrieval_hits')==4
            and (not m['mtp'] or residency_verified(m)),
        'long_context_tokens':m.get('long_context'),'long_output_tokens':m.get('long_output_tokens'),
        'long_seconds':m.get('long_result',{}).get('seconds'),
        'long_retrieval_hits':m.get('long_retrieval_hits'),
        'four_full_prompt_residency':runtime.get('four_full_prompt_residency'),
        'error':m.get('error')}


def summarize_graph(root):
    rows={p.parent.name:profile(json.loads(p.read_text())) for p in sorted(root.glob('*/manifest.json'))}
    result={'schema_version':2,'scope':'Graph decode only. Target and MTP draft enforce_eager=False. Legacy eager results excluded.',
        'hbm_limit_mib':50307,'experiments':rows,
        'full_quality_tested':False,'int8_default_enabled':False,'vocab4_default_enabled':False,
        'formal_service':json.loads((root/'controller-final.json').read_text())
            if (root/'controller-final.json').exists() else {'passed':False,'reason':'pending restore'},
        'artifacts_root':str(root)}
    (root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def summarize(root):
    read=lambda p:json.loads((root/p).read_text())
    runs={name:read(name+'/manifest.json') for name in ('bf16-no-mtp','int8-no-mtp','int8-aligned-no-mtp')}
    rows={}
    for name,m in runs.items():
        rows[name]={'valid':m.get('valid',False),'mtp':m['mtp'],'cache_budget_bytes':m['kv_budget_bytes'],
            'pool':m.get('pool'),'calculated_bounds_tokens':context_bounds(m['pool']) if m.get('pool') else None,
            'short_checks_passed':m.get('short_checks_passed',False),
            'long_context_tokens':m.get('long_context'),'long_result':m.get('long_result'),
            'long_retrieval_hits':m.get('long_retrieval_hits'),'long_fully_verified':verified_long(m),
            'error':m.get('error')}
    result={'schema_version':1,'scope':'Fixed 22.5GiB KV budget, isolated eager INT8 experiment; no MTP, no full quality acceptance.',
        'operator_probe':read('operator-probe/result.json'),'production_before':read('production-before.json'),
        'summary_service':read('summary-context-audit.json'),'experiments':rows,
        'production_context_changed':False,'int8_default_enabled':False,
        'formal_service':read('controller-final.json') if (root/'controller-final.json').exists() else {'passed':False,'reason':'pending restore'},
        'artifacts_root':str(root)}
    (root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--graph',action='store_true');a=p.parse_args()
    result=summarize_graph(a.root) if a.graph else summarize(a.root)
    print(json.dumps({n:{k:v[k] for k in ('valid','calculated_bounds_tokens','long_context_tokens')}
                      for n,v in result['experiments'].items()},indent=2))
