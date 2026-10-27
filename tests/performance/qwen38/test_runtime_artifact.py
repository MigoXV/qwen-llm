"""专项适配不能误用于其他模型、未经验证的依赖或被改动的量化配方。"""
from dataclasses import replace
import json

import pytest

from qwen_llm.config import ServingConfig
from qwen_llm.engine.qwen38 import (ADAPTATION, MANIFEST, MODEL_FILES, PACKAGES,
                                  TEXT_SHAPE, file_sha256, serving_options)


@pytest.fixture
def verified(tmp_path):
    text={**TEXT_SHAPE,'layer_types':['full_attention' if i%4==3 else 'linear_attention' for i in range(64)]}
    config={'model_type':'qwen3_5','text_config':text}
    description={'lm_head.weight':'FLOAT'}
    for i in range(64):
        prefix=f'model.language_model.layers.{i}.'
        for name in ['gate_proj','up_proj','down_proj']:
            description[prefix+f'mlp.{name}.weight']='W4A8_DYNAMIC'
        if i%4!=3:description[prefix+'linear_attn.out_proj.weight']='W8A8_DYNAMIC'
    for name,value in [('config.json',config),('quant_model_description.json',description),
                       ('quant_model_weights.safetensors.index.json',{'weight_map':{}})]:
        (tmp_path/name).write_text(json.dumps(value))
    packages={k:'tested-version' for k in PACKAGES}
    record={'schema_version':1,'adaptation':ADAPTATION,'validated_for_inference':True,
            'recipe':'gdn','software':packages,
            'model_files':{k:file_sha256(tmp_path/k) for k in MODEL_FILES},
            'serving':{'batch_tokens':[3084],'kv_cache_bytes':[int(22.5*1024**3)],'gpu_memory_utilization':.82}}
    (tmp_path/MANIFEST).write_text(json.dumps(record))
    settings=ServingConfig(model_path=tmp_path,max_input_tokens=57344,batch_tokens=3084,
        graph_mode='FULL_DECODE_ONLY',speculative_tokens=2,async_scheduling=True,npugraph_ex=True,
        kv_cache_bytes=int(22.5*1024**3))
    return settings,packages


def test_verified_artifact_selects_only_production_worker_options(verified):
    config,packages=verified
    options,identity=serving_options(config,packages)
    assert identity['recipe']=='gdn'
    assert options['scheduler_cls'].startswith('qwen_llm.')
    assert 'tests.' not in json.dumps(options)


def test_rejects_unverified_changed_or_incomplete_artifacts(verified):
    config,packages=verified;path=config.model_path/MANIFEST
    original=path.read_text();record=json.loads(original)
    record['validated_for_inference']=False;path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='验收'):serving_options(config,packages)
    path.write_text(original)
    (config.model_path/'.quantization.incomplete').touch()
    with pytest.raises(ValueError,match='导出'):serving_options(config,packages)
    (config.model_path/'.quantization.incomplete').unlink()
    with (config.model_path/'quant_model_description.json').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='不一致'):serving_options(config,packages)


def test_rejects_other_models_and_untested_dependency_builds(verified):
    config,packages=verified
    changed={**packages,'vllm-ascend':'another-build'}
    with pytest.raises(ValueError,match='依赖版本'):serving_options(config,changed)
    path=config.model_path/'config.json';value=json.loads(path.read_text())
    value['text_config']['hidden_size']=4096;path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='只支持'):serving_options(config,packages)


@pytest.mark.parametrize('change',[{'speculative_tokens':0},{'prefix_caching':False},
    {'async_scheduling':False},{'npugraph_ex':False},{'max_input_tokens':32768},
    {'batch_tokens':8192},{'kv_cache_bytes':int(24.5*1024**3)}])
def test_rejects_untested_serving_combinations(verified,change):
    config,packages=verified
    with pytest.raises(ValueError):serving_options(replace(config,**change),packages)
