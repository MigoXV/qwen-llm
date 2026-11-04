"""从短算子复核和三档真实服务吞吐生成报告，不把算子收益冒充TPS。"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from .compare import metrics


def summarize(root):
    read=lambda p:json.loads((root/p).read_text())
    runs=[read(f'fastpath-stable-{i:02d}/result.json') for i in (1,2)]
    expected={(l,p,m) for l in (0,30,63) for p in ('gate_up_proj','down_proj') for m in (6,12)}
    checks=[]
    for i,run in enumerate(runs,1):
        samples=run['samples']
        if len(samples)!=12 or {(r['layer'],r['projection'],r['M']) for r in samples}!=expected or run['calls_per_replay_small']!=512:
            raise ValueError('算子复核覆盖或计时不完整')
        for row in samples:
            errors=[row['relative_l2'],row['graph_relative_l2'],*row['patterns']]
            if not row['passed'] or len(row['candidate_ms'])!=5 or len(row['native_ms'])!=5 or not all(math.isfinite(x) and 0<=x<=.01 for x in errors):
                raise ValueError('算子精度或计时不合格')
            checks.append({'run':i,'layer':row['layer'],'projection':row['projection'],'M':row['M'],
                'time_ratio':row['time_ratio'],'relative_l2':max(errors),
                'native_ms':statistics.median(row['native_ms']),'candidate_ms':statistics.median(row['candidate_ms'])})
    old=read('service-native/manifest.json');new=read('service-tuned/manifest.json');a,b=metrics(old),metrics(new)
    names=[f'generation512-c{c}' for c in (1,2,4)]
    complete=all(m.get('valid') and m.get('released') and m.get('port_released')
        and {c['name'] for c in m['cases']}==set(names) and all(len(c['rounds'])==3 and all(r['valid'] for r in c['rounds']) for c in m['cases']) for m in (old,new))
    rows=[{'case':name,'native_tps':a[name]['tps'],'tuned_tps':b[name]['tps'],'tps_ratio':b[name]['tps']/a[name]['tps'],
        'native_tps_rounds':a[name]['tps_rounds'],'tuned_tps_rounds':b[name]['tps_rounds'],
        'native_acceptance':a[name]['accepted_fraction'],'tuned_acceptance':b[name]['accepted_fraction']} for name in names]
    workers=[m['ready']['runtime']['workers'][0] for m in (old,new)]
    backend=workers[1]['experiment'].get('w4_backend')
    capacity=all(w['full_context_capacity']>=4 and w['speculative_tokens']==2 and w['graph_mode']=='FULL_DECODE_ONLY' for w in workers)
    delta=workers[1]['npu_allocated_bytes']-workers[0]['npu_allocated_bytes']
    lifecycle=read('controller-final.json') if (root/'controller-final.json').exists() else {'passed':False}
    diagnosis=root/'folded-branch-evidence/diagnosis.json'
    dispatch=read('folded-branch-evidence/diagnosis.json') if diagnosis.exists() else {
        'service_acceleration_evidence_valid':False,'reason':'Actual compiled dispatch was not verified.'}
    measured_source=root/'jobs/008-service-tuned/source/native_tuned.py'
    result={'scope':'short operator verification and three-round generation throughput; not full model quality acceptance',
        'operator_checks':checks,'operator_passed':all(r['time_ratio']<(.98 if r['M']==6 else .9) for r in checks),
        'max_relative_l2':max(c['relative_l2'] for c in checks),'service_complete':complete,'service':rows,
        'generation_geomean_ratio':math.prod(r['tps_ratio'] for r in rows)**(1/3),
        'capacity_preserved':capacity,'extra_allocated_hbm_bytes_at_ready':delta,
        'backend_identity':backend,'candidate_source_sha256':sha256(measured_source.read_bytes()).hexdigest() if measured_source.exists() else None,
        'candidate_source_snapshot':str(measured_source),'compiled_dispatch':dispatch,
        'custom_panel':read('panel-01/result.json'),'full_quality_tested':False,'default_promoted':False,
        'lifecycle':lifecycle,'artifacts_root':str(root),
        'discarded_run':{'path':'fastpath-verify-02','reason':'Highly variable short event spans while host extension compilation overlapped. Retained as invalid timing evidence; replaced by two independent 512-call checks with no concurrent compilation.'}}
    result['short_screen_valid']=(dispatch['service_acceleration_evidence_valid'] and result['operator_passed'] and complete and capacity and delta<=1024**3
        and backend and backend['fast_M']==[6,12] and all(r['tps_ratio']>=.95 for r in rows)
)
    result['original_generation_target_met']=result['generation_geomean_ratio']>=1.05
    result['short_screen_passed']=bool(result['short_screen_valid'] and result['original_generation_target_met'])
    result['observed_service_improvement']=bool(dispatch['service_acceleration_evidence_valid'] and any(r['tps_ratio']>=1.02 for r in rows))
    for name,path in (('rolling_nd','rolling-nd-01/result.json'),('w8_scale','w8-scale/result.json'),('larger_m','fastpath-larger-m/result.json')):
        if (root/path).exists():result[name]=read(path)
    (root/'native-tuned-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(summarize(p.parse_args().root),ensure_ascii=False,indent=2))
