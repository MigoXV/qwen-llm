import copy
import pytest
from .ascendc_v2_candidate import MS,SHAPES
from .summarize_ascendc_v2 import micro_summary


def records():
    samples=[]
    for layer in (0,30,63):
        for m in MS:
            for n,k in SHAPES:
                for layout in (0,1):
                    samples.append({'layer':layer,'projection':'gate_up_proj' if n==34816 else 'down_proj',
                        'M':m,'N':n,'K':k,'layout':layout,'fallback_correct':True,'fallback_relative_l2':0,
                        'v1_ms':[1.5]*5,'dequant_component_ms':[.6]*5,'gemm_component_ms':[.5]*5,
                        'fallback_ms':[1.1]*5,'variants':[{'configuration':[64,128,16,2,1,1],
                            'passed':True,'candidate_ms':[1.2]*5,'native_ms':[1.]*5}]})
    a={'phase':'validate','correct':True,'run_id':'one','samples':samples,'workspace_bytes':600_000_000}
    b=copy.deepcopy(a);b['run_id']='two'
    return a,b


def test_negative_summary_requires_two_complete_shape_layout_matrices():
    a,b=records();result=micro_summary(a,b)
    assert result['variants']['correct']==264
    assert len(result['operator_comparison'])==44
    assert result['operator_comparison'][0]['median_best_time_ratio']==1.2
    broken=copy.deepcopy(b);broken['samples'][-1]=broken['samples'][0]
    with pytest.raises(ValueError):micro_summary(a,broken)
    broken=copy.deepcopy(b);broken['samples'][0]['variants']=[]
    with pytest.raises(ValueError):micro_summary(a,broken)
    broken=copy.deepcopy(b);broken['run_id']='one'
    with pytest.raises(ValueError):micro_summary(a,broken)


def test_negative_tuning_summary_requires_all_three_stages():
    from .summarize_ascendc_v2 import tuning_summary
    a,_=records()
    rows=[r for r in a['samples'] if r['layer']==0 and r['M'] in (3,6,12,38,117,1536,8192)]
    for r in rows:
        r['variants']=[]
        for family,split in [(0,1),(1,1),*((1,s) for s in (2,4,8) if r['M']<=117)]:
            r['variants'].append({'configuration':[64,128,16,2,split,family],
                'passed':True,'candidate_ms':[1.2]*5,'native_ms':[1.]*5})
    tune={'phase':'tune','correct':True,'samples':rows}
    assert tuning_summary(tune)['shape_layout_cases']==28
    tune['samples'][0]['variants'].pop()
    with pytest.raises(ValueError):tuning_summary(tune)
