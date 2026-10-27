"""按共同原始请求对比质量和逐轮性能；只读取结果，不手写加速结论。"""
import argparse
import json
from pathlib import Path
import statistics
import math


def quality_gate(baseline,candidate, *, require_full=False):
    old={str(s['id']):s for s in baseline['samples']};new={str(s['id']):s for s in candidate['samples']}
    regressions=[k for k in new if k in old and old[k]['passed'] and not new[k]['passed']]
    fixed=[k for k in regressions if new[k]['category'] not in {'gsm8k','code'}]
    math_keys=[k for k in new if k in old and new[k]['category']=='gsm8k']
    code_keys=[k for k in new if k in old and new[k]['category']=='code']
    delta=lambda keys:(sum(new[k]['passed'] for k in keys)-sum(old[k]['passed'] for k in keys))/len(keys) if keys else 0
    math_delta=delta(math_keys);code_delta=delta(code_keys)
    full_coverage = len(math_keys) == 1319
    allowed_drop = .01 if full_coverage else 4/128
    matched = (baseline['prompt_policy'] == candidate['prompt_policy']
        and baseline.get('dataset_sha256') == candidate.get('dataset_sha256')
        and baseline.get('code_checker_sha256') == candidate.get('code_checker_sha256')
        and all(old[k]['prompt'] == new[k]['prompt'] for k in new if k in old))
    coverage = (len(new) == len(candidate['samples']) and set(new) <= set(old)
        and len(math_keys) == candidate['math_samples'] and len(code_keys) == 64
        and sum(s['category'] not in {'gsm8k','code'} for s in new.values()) == 40
        and (not require_full or full_coverage))
    okay=(matched and coverage and not fixed and math_delta>=-allowed_drop and code_delta>=0 and not candidate['execution_errors'] and candidate['instance_unchanged'])
    return {'passed':okay,'matched_prompt_policy':matched,'coverage_complete':coverage,'new_fixed_answer_failures':fixed,'all_regression_ids':regressions,
            'gsm8k_delta_percentage_points':math_delta*100,'code_delta_percentage_points':code_delta*100,
            'full_gsm8k_evaluated':full_coverage,'math_allowed_drop_percentage_points':100*allowed_drop,
            'compared_math_questions':len(math_keys),'compared_code_questions':len(code_keys),
            'candidate_categories':candidate['categories']}


def delivery_metrics(frontend, http_ids=None):
    """CPU侧相邻非空引擎交付批次间隔；一个交付批次可能含多个token。"""
    groups = {}
    for row in frontend['deliveries']:
        if row['new_tokens'] > 0 and (http_ids is None or row['http_id'] in http_ids):
            groups.setdefault(row['request_id'], []).append(row['at_ns'])
    gaps = [(b-a)/1e6 for times in groups.values()
            for a,b in zip(sorted(times), sorted(times)[1:])]
    ordered = sorted(gaps)
    return {'gap_count': len(gaps), 'max_ms': max(gaps, default=0),
            'median_ms': statistics.median(gaps) if gaps else 0,
            'p95_ms': ordered[max(0, math.ceil(.95*len(ordered))-1)] if ordered else 0}


def directory_metrics(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    result = metrics(manifest)
    if 'mixed32k-c4' in result:
        rounds = next(c['rounds'] for c in manifest['cases'] if c['name']=='mixed32k-c4')
        gaps = [delivery_metrics(json.loads((directory/'mixed32k-c4'/f'round{i}-frontend.json').read_text()), {0,1,2})
                for i in range(len(rounds))]
        result['mixed32k-c4']['short_delivery_gaps'] = gaps
        result['mixed32k-c4']['short_max_gap_ms'] = max(g['max_ms'] for g in gaps)
    return result


def select_batch(baseline, candidates):
    """四个预算同负载筛选，未通过约束时保留8192。输入均为directory_metrics。"""
    required = ['generation512-c4','chinese8192-c4-cold','mixed32k-c4']
    rows = []
    for budget, candidate in candidates.items():
        if any(name not in candidate or name not in baseline for name in required):
            raise ValueError('分块预算筛选缺少必需场景')
        ratios = {name:candidate[name]['tps']/baseline[name]['tps'] for name in required}
        ttft = candidate['chinese8192-c4-cold']['ttft_ms']/baseline['chinese8192-c4-cold']['ttft_ms']
        mixed_ttft = candidate['mixed32k-c4']['long_ttft_ms']/baseline['mixed32k-c4']['long_ttft_ms']
        gap = candidate['mixed32k-c4']['short_max_gap_ms']/baseline['mixed32k-c4']['short_max_gap_ms']
        passed = gap <= .5 and min(ratios.values()) >= .95 and ttft <= 1.05 and mixed_ttft <= 1.05
        rows.append({'batch_tokens':int(budget),'passed':passed,'tps_ratios':ratios,
                     'long_cold_ttft_ratio':ttft,'mixed_long_ttft_ratio':mixed_ttft,'short_max_delivery_gap_ratio':gap,
                     'geometric_mean_tps_ratio':math.prod(ratios.values())**(1/3)})
    eligible = [r for r in rows if r['passed']]
    best = max(eligible,key=lambda r:r['geometric_mean_tps_ratio']) if eligible else None
    return {'selected_batch_tokens':best['batch_tokens'] if best else 8192,'candidates':rows}


def metrics(manifest):
    result={}
    for case in manifest['cases']:
        rounds=case['rounds']
        if not rounds:continue
        result[case['name']]={'tps':statistics.median(r['tokens_per_second'] for r in rounds),
            'ttft_ms':statistics.median(t for r in rounds for t in r['ttft_ms']),
            'tps_rounds':[r['tokens_per_second'] for r in rounds],
            'cache_rounds':[r['cached_tokens'] for r in rounds],
            'accepted_fraction':sum(r['accepted_tokens'] for r in rounds)/max(1,sum(r['draft_tokens'] for r in rounds))}
        if case['name']=='mixed32k-c4':
            result[case['name']]['long_ttft_ms']=statistics.median(r['ttft_ms'][3] for r in rounds)
            result[case['name']]['short_ttft_ms']=statistics.median(t for r in rounds for t in r['ttft_ms'][:3])
    return result


def compare(baseline,candidate):
    b,c=metrics(baseline),metrics(candidate)
    return [{'case':name,'baseline':b[name],'candidate':c[name],
             'tps_ratio':c[name]['tps']/b[name]['tps'],'ttft_ratio':c[name]['ttft_ms']/b[name]['ttft_ms']}
            for name in b.keys() & c.keys()]


def main():
    p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    load=lambda d,f:json.loads((d/f).read_text())
    result={'performance':compare(load(a.baseline,'manifest.json'),load(a.candidate,'manifest.json'))}
    if (a.baseline/'quality.json').exists() and (a.candidate/'quality.json').exists():
        result['quality']=quality_gate(load(a.baseline,'quality.json'),load(a.candidate,'quality.json'))
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
