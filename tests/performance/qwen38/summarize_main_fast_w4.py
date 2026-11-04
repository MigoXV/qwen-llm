"""主程序跨启动复核；严格区分有收益、短测稳定和完整质量验收。"""
import argparse
import json
import math
from pathlib import Path
import signal
import statistics

RUNS=('A1-native','B1-fast','B2-fast','A2-native')
CASES=('generation512-c2','generation512-c4')


def summarize(root):
    plan=root/'acceptance-plan.json'
    order=json.loads(plan.read_text())['order'] if plan.exists() else list(RUNS)
    if len(order)!=4 or set(order)!=set(RUNS):raise ValueError('必须包含两次原生及两次优化启动')
    manifests={name:json.loads((root/name/'manifest.json').read_text()) for name in order}
    reference=manifests[order[0]]['implementation_sha256']
    complete=True;identities={};measurements={}
    for name,m in manifests.items():
        enabled=name.startswith('B')
        # Uvicorn re-raises a handled SIGTERM on some starts after completing
        # lifespan shutdown. Keep the raw exit code and require its completion log.
        log=root/name/'server.log'
        clean_exit=m.get('exit_code')==0 or (m.get('exit_code')==-signal.SIGTERM
            and log.exists() and 'Application shutdown complete.' in log.read_text())
        complete &= (m.get('valid') and m.get('released') and m.get('port_released')
            and clean_exit and m.get('fast_w4_requested')==enabled
            and m['implementation_sha256']==reference and {c['name'] for c in m['cases']}==set(CASES))
        workers=m['ready']['runtime']['workers'];identities[name]=workers
        complete &= bool(workers) and all(w['worker']=='qwen_llm.engine.qwen38_worker.Qwen38NPUWorker'
            and w['full_context_capacity']>=4 and w['speculative_tokens']==2
            and w['graph_mode']=='FULL_DECODE_ONLY' and w['graphs_captured']>=3
            and w['model_adaptation']['w4_decode']['backend']==('native-tuned' if enabled else 'upstream')
            and w['model_adaptation']['w4_decode']['layers']==128 for w in workers)
        if enabled:
            complete &= all(w['model_adaptation']['w4_decode'].get('capture_dispatch',{}).get(f'{m}:1',0)>=128
                            for w in workers for m in (6,12))
        for c in m['cases']:
            rounds=c['rounds']
            complete &= len(rounds)==3 and all(r['valid'] and r['inventory']['isolated']
                and r['output_tokens']==[512]*int(c['name'][-1]) for r in rounds)
            values=[r['tokens_per_second'] for r in rounds]
            if not all(math.isfinite(v) and v>0 for v in values):raise ValueError('非法吞吐数据')
            measurements[(name,c['name'])]={'median_tps':statistics.median(values),'round_tps':values,
                'ttft_ms':statistics.median(t for r in rounds for t in r['ttft_ms'])}
    rows=[]
    for case in CASES:
        native=[measurements[(n,case)] for n in ('A1-native','A2-native')]
        fast=[measurements[(n,case)] for n in ('B1-fast','B2-fast')]
        old=[v for row in native for v in row['round_tps']];new=[v for row in fast for v in row['round_tps']]
        cv=lambda v:statistics.pstdev(v)/statistics.mean(v)
        conservative=min(r['median_tps'] for r in fast)/max(r['median_tps'] for r in native)
        rows.append({'case':case,'native_median_tps':statistics.median(old),'fast_median_tps':statistics.median(new),
            'median_tps_ratio':statistics.median(new)/statistics.median(old),
            'conservative_process_median_ratio':conservative,'native_cv':cv(old),'fast_cv':cv(new),
            'stable_positive':conservative>=1.01 and max(cv(old),cv(new))<=.05,
            'runs':{n:measurements[(n,case)] for n in order}})
    allocated={name:max(w['npu_allocated_bytes'] for w in workers) for name,workers in identities.items()}
    extra=max(allocated[n] for n in ('B1-fast','B2-fast'))-min(allocated[n] for n in ('A1-native','A2-native'))
    stable=bool(complete and all(r['stable_positive'] for r in rows) and extra<=1024**3)
    final=root/'controller-final.json'
    deployment=root/'deployment.json'
    result={'schema_version':1,'order':order,'production_path_complete':bool(complete),
        'stable_positive':stable,'cases':rows,'extra_allocated_hbm_bytes_at_ready':extra,
        'runtime':identities,'full_quality_tested':False,
        'shutdown_policy':'0 or handled SIGTERM with Application shutdown complete; all probes and resource release must pass',
        'policy':'min optimized process median / max native process median >=1.01 for both c2/c4; round CV <=5%; 3 rounds per case per start; two independent starts per mode',
        'formal_service':json.loads(final.read_text()) if final.exists() else {'passed':False,'reason':'pending final restore'},
        'deployment':json.loads(deployment.read_text()) if deployment.exists() else {'reason':'pending decision'},
        'artifacts_root':str(root)}
    (root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(summarize(p.parse_args().root),ensure_ascii=False,indent=2))
