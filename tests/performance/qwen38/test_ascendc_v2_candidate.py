import copy
import pytest
from .ascendc_v2_candidate import validate_selection,MS,SHAPES
from .ascendc_v2 import LAYOUT_VERSION


def test_install_rejects_python_changed_after_validation():
    from hashlib import sha256
    from pathlib import Path
    from .ascendc_v2 import SOURCE_FILES
    from .ascendc_v2_candidate import verify_sources
    record={'python_source_sha256':{name:sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCE_FILES}}
    verify_sources(record)
    record['python_source_sha256']['ascendc_v2.py']='stale'
    with pytest.raises(ValueError):verify_sources(record)


def selection():
    samples=[]
    for layer in (0,30,63):
        for m in MS:
            for n,k in SHAPES:
                row={'layer':layer,'M':m,'N':n,'K':k,'layout':1,'fallback_correct':True,'fallback_relative_l2':0,'variants':[]}
                if m==3 and n==34816:
                    row['variants']=[{'configuration':[128,256,16,2,1,1],'passed':True,'graph_compatible':True,
                        'calls_per_replay':16,'candidate_ms':[.8]*5,'native_ms':[1.]*5,
                        'relative_l2':.001,'graph_relative_l2':.001,
                        'patterns':[{'amplitude':a,'passed':True,'relative_l2':0} for a in (0,64)]}]
                samples.append(row)
    run={'phase':'validate','correct':True,'raw_library_sha256':'raw','host_sha256':'host','model_config_sha256':'model','run_id':'first',
         'environment':{},'timing_schema':'grouped-16-4-v2','samples':samples}
    second=copy.deepcopy(run);second['run_id']='second'
    return {'schema_version':2,'layout_version':LAYOUT_VERSION,'layout':1,'model_config_sha256':'model',
            'passed_micro_gate':True,'raw_library_sha256':'raw','host_sha256':'host','environment':{},
            'validation_runs':[run,second],
            'profiles':[{'M':3,'N':34816,'K':5120,'configuration':[128,256,16,2,1,1]}]}


def test_two_runs_require_both_native_fallback_and_real_three_layer_evidence():
    record=selection()
    assert validate_selection(record,'model')==record['profiles']
    broken=copy.deepcopy(record);broken['validation_runs'][1]['samples'][2]['variants'][0]['candidate_ms']=[float('nan')]*5
    with pytest.raises(ValueError):validate_selection(broken,'model')
    broken=copy.deepcopy(record);broken['validation_runs'][1]['samples'][-1]['fallback_relative_l2']=float('nan')
    with pytest.raises(ValueError):validate_selection(broken,'model')
    broken=copy.deepcopy(record);broken['validation_runs'][1]['samples'][2]['variants'][0]['calls_per_replay']=1
    with pytest.raises(ValueError):validate_selection(broken,'model')
    broken=copy.deepcopy(record);broken['validation_runs'][1]['samples'][-1]=broken['validation_runs'][1]['samples'][0]
    with pytest.raises(ValueError):validate_selection(broken,'model')
