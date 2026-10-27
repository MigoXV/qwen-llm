"""从原始文件产生最终验收证据；不以阶段性吞吐代替完整验收。"""
import argparse
import json
from pathlib import Path
import statistics

from .compare import quality_gate, directory_metrics


def read(directory, name):
    return json.loads((Path(directory)/name).read_text())


def stability_summary(directory):
    directory=Path(directory)
    data=read(directory,'stability.json')
    rounds=data['rounds']
    errors=[{'round':i,'sample':s} for i,r in enumerate(rounds) for s in r['http']['samples'] if 'error' in s]
    states=[s for r in rounds for w in r['allocator'] for s in w]
    invariant_names=['uncached_index_matches','metadata_bounded','checkpoint_owners_coherent','old_state_index_matches']
    invariants=bool(states) and all(s[k] for s in states for k in invariant_names)
    start=rounds[0]['http']['start_ns'];end=rounds[-1]['http']['end_ns']
    resources=[]
    with (directory/'npu-samples.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if start<=row.get('at_ns',0)<=end:resources.append(row)
    pids={p['pid'] for r in resources for p in r['processes']}
    process_growth=[]
    for pid in sorted(pids):
        samples=[p for r in resources for p in r['processes'] if p['pid']==pid]
        window=max(1,len(samples)//5)
        rss=statistics.median(p['rss_bytes'] for p in samples[-window:])-statistics.median(p['rss_bytes'] for p in samples[:window])
        fds=statistics.median(p['open_fds'] for p in samples[-window:])-statistics.median(p['open_fds'] for p in samples[:window])
        process_growth.append({'pid':pid,'samples':len(samples),'rss_growth_bytes':rss,'fd_growth':fds,
                               'passed':len(samples)>=30 and rss<=128*1024**2 and fds<=4})
    workers=[w for r in rounds for w in r['runtime']]
    allocated=max(w['npu_allocated_bytes'] for w in workers)-min(w['npu_allocated_bytes'] for w in workers)
    reserved=max(w['npu_reserved_bytes'] for w in workers)-min(w['npu_reserved_bytes'] for w in workers)
    enough=data['elapsed_seconds']>=3600 and data['requested_seconds']>=3600
    passed=(enough and not errors and invariants and bool(process_growth)
        and all(p['passed'] for p in process_growth) and allocated<=16*1024**2 and reserved<=128*1024**2
        and all(r['inventory']['isolated'] for r in rounds))
    return {'passed':passed,'elapsed_seconds':data['elapsed_seconds'],'rounds':len(rounds),
            'requests':sum(len(r['http']['samples']) for r in rounds),'errors':len(errors),
            'allocator_invariants':invariants,'process_growth':process_growth,
            'npu_allocated_range_bytes':allocated,'npu_reserved_range_bytes':reserved,
            'method':'RSS/FD compare medians of first/last 20% within stability interval; CPU RSS allowance128MiB, FD4; NPU allocated16MiB/reserved128MiB'}


def full_acceptance(baseline,candidate):
    base=read(baseline,'manifest.json');new=read(candidate,'manifest.json')
    quality=quality_gate(read(baseline,'quality.json'),read(candidate,'quality.json'),require_full=True)
    long={}
    for name,expected in [('long-quality.json',72),('boundary-quality.json',72),('prefix-growth.json',68)]:
        r=read(candidate,name)
        long[name]={'passed':r['passed'],'total':r['total'],'expected':expected,
                    'all_correct':r['passed']==r['total']==expected}
        if name=='prefix-growth.json':
            long[name]['allocator_steps']=len(r.get('allocator_states',[]))
            long[name]['all_correct'] &= len(r.get('allocator_states',[]))==17
    bm,cm=directory_metrics(baseline),directory_metrics(candidate)
    performance=[]
    for name in sorted(bm.keys() & cm.keys()):
        ratios={'tps':cm[name]['tps']/bm[name]['tps'],'ttft':cm[name]['ttft_ms']/bm[name]['ttft_ms']}
        long_input=name.startswith(('chinese','repeated')) or name=='mixed32k-c4'
        if name=='mixed32k-c4':
            ratios['ttft']=cm[name]['long_ttft_ms']/bm[name]['long_ttft_ms']
        performance.append({'case':name,'baseline':bm[name],'candidate':cm[name],'ratios':ratios,
                            'ttft_gated':long_input,
                            'within_regression_limit':ratios['tps']>=.95 and (not long_input or ratios['ttft']<=1.05)})
    complete=(len(performance)==23 and all(len(c['rounds'])>=5 for c in new['cases']))
    mixed_required=(new['config'].get('mixed_prefill_tokens') is not None
                    or new['config']['batch_tokens']!=base['config']['batch_tokens'])
    gap_ratio=cm['mixed32k-c4']['short_max_gap_ms']/bm['mixed32k-c4']['short_max_gap_ms']
    mixed={'required':mixed_required,'short_max_delivery_gap_ratio':gap_ratio,
           'passed':not mixed_required or gap_ratio<=.5}
    stable=stability_summary(candidate)
    lifecycle=all(m.get('valid') and m.get('released') and m.get('port_released') for m in [base,new])
    # 全部并发和缓存场景都列出，未通过的场景必须进一步消融，不能由整体收益掩盖。
    passing=(quality['passed'] and all(v['all_correct'] for v in long.values())
             and stable['passed'] and lifecycle and complete and mixed['passed']
             and all(p['within_regression_limit'] for p in performance))
    meaningful=any(p['ratios']['tps']>=1.05 for p in performance) or any(
        p['ratios']['ttft']<=.8 and p['case'].endswith('-hot') for p in performance) or gap_ratio<=.5
    return {'passed':passing and meaningful,'quality':quality,'long_correctness':long,
            'stability':stable,'lifecycle':lifecycle,'full_performance_coverage':complete,
            'meaningful_gain':meaningful,'mixed_stall':mixed,'performance':performance}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('baseline',type=Path);parser.add_argument('candidate',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=full_acceptance(args.baseline,args.candidate)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='performance'},ensure_ascii=False,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
