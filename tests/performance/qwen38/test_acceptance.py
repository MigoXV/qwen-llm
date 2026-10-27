"""验收必须覆盖同一题集；批交付间隔不能按SSE片段或token推算。"""
from copy import deepcopy
from .compare import delivery_metrics, quality_gate, select_batch


def quality_result(math_count=128):
    samples = [{'id':f'{category}-{i}','category':category,'prompt':f'{category} {i}','passed':True}
               for category,count in [('arithmetic',40),('code',64),('gsm8k',math_count)]
               for i in range(count)]
    return {'samples':samples,'math_samples':math_count,'prompt_policy':'fixed',
            'dataset_sha256':'same-dataset','code_checker_sha256':'same-checker',
            'execution_errors':0,'instance_unchanged':True,'categories':{}}


def test_acceptance_rejects_missing_changed_or_duplicate_questions():
    baseline=quality_result(1319);candidate=quality_result()
    assert quality_gate(baseline,candidate)['passed']
    assert not quality_gate(baseline,candidate,require_full=True)['passed']
    for mutate in [lambda q:q['samples'].pop(),
                   lambda q:q['samples'].append(q['samples'][0]),
                   lambda q:q['samples'][0].update(prompt='different'),
                   lambda q:q.update(dataset_sha256='other')]:
        changed=deepcopy(candidate);mutate(changed)
        assert not quality_gate(baseline,changed)['passed']


def test_full_math_gate_uses_one_percentage_point_and_keeps_code_and_fixed_answers():
    baseline=quality_result(1319);candidate=deepcopy(baseline)
    for s in candidate['samples'][-13:]:s['passed']=False
    assert quality_gate(baseline,candidate,require_full=True)['passed']
    candidate['samples'][-14]['passed']=False
    assert not quality_gate(baseline,candidate,require_full=True)['passed']
    for category in ['code','arithmetic']:
        c=deepcopy(baseline)
        next(s for s in c['samples'] if s['category']==category)['passed']=False
        assert not quality_gate(baseline,c)['passed']


def test_delivery_gap_counts_only_consecutive_nonempty_batches_per_short_request():
    def row(request,at,new=3,http_id=0):
        return {'request_id':request,'at_ns':at*1_000_000,'new_tokens':new,'http_id':http_id}
    data={'deliveries':[row('a',100),row('b',90,http_id=1),row('a',200,new=0),
                        row('a',1100),row('b',100,http_id=1),row('long',5000,http_id=3)]}
    result=delivery_metrics(data,{0,1,2})
    assert result['gap_count']==2
    assert result['max_ms']==1000 and result['median_ms']==505


def test_batch_selection_rejects_throughput_regression_even_with_shorter_stall():
    def metric(tps,gap=3000,ttft=10000):
        return {'tps':tps,'short_max_gap_ms':gap,'ttft_ms':ttft,'long_ttft_ms':ttft}
    baseline={n:metric(100) for n in ['generation512-c4','chinese8192-c4-cold','mixed32k-c4']}
    good={n:metric(100,gap=1200) for n in baseline}
    bad=deepcopy(good);bad['chinese8192-c4-cold']['tps']=94
    assert select_batch(baseline,{1548:bad})['selected_batch_tokens']==8192
    assert select_batch(baseline,{1548:bad,3084:good})['selected_batch_tokens']==3084
    bad_mixed=deepcopy(good);bad_mixed['mixed32k-c4']['tps']=94
    assert select_batch(baseline,{3084:bad_mixed})['selected_batch_tokens']==8192
