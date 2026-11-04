import copy
import pytest
from .ascendc_candidate import selection_rows
from .config import ExperimentConfig,W4A8LinearBackend


def record():
    value={'correct':True,'passed_micro_gate':True,'model_config_sha256':'known','samples':[],
           'nz_selection':[{'M':1,'N':34816,'K':5120,'tiling':[64,256,1],'kind':'nz'}]}
    for layer in (0,30,63):
        for m in (1,3,6,12,38,117,1536,2048,6144,7680,8192):
            for n,k in ((34816,5120),(5120,17408)):
                sample={'layer':layer,'M':m,'N':n,'K':k,'correct':True,'upstream_ms':[1.0]*5,'variants':[]}
                if m==1 and n==34816:
                    sample['variants']=[{'kind':'nz','tiling':[64,256,1],'correct':True,'graph_compatible':True,
                        'candidate_ms':[.8]*5,'fp32_oracle_relative_l2':.002,
                        'patterns':[{'amplitude':a,'relative_l2':0,'passed':True} for a in (0,64)]}]
                value['samples'].append(sample)
    return value


def test_only_common_three_layer_nz_profiles_can_be_selected():
    value=record()
    assert selection_rows(value,'known')==[[1,34816,5120,64,256]]
    with pytest.raises(ValueError,match='模型配置'):
        selection_rows(value,'different')
    invalid=copy.deepcopy(value)
    invalid['samples'][0]['variants'][0]['candidate_ms']=[float('nan')]*5
    with pytest.raises(ValueError):selection_rows(invalid,'known')
    invalid=copy.deepcopy(value)
    invalid['samples'][22]['variants'][0]['graph_compatible']=False
    with pytest.raises(ValueError):selection_rows(invalid,'known')
    invalid=copy.deepcopy(value)
    invalid['samples'][-1]=invalid['samples'][0]
    with pytest.raises(ValueError,match='覆盖'):selection_rows(invalid,'known')


def test_experimental_backend_requires_explicit_verified_inputs():
    with pytest.raises(ValueError,match='筛选记录'):
        ExperimentConfig(w4a8_linear_backend=W4A8LinearBackend.ASCENDC)
