import pytest
from types import SimpleNamespace
from .config import ExperimentConfig,W4A8LinearBackend
from .engine import diagnostic_worker


def test_native_tuned_is_explicit_and_requires_the_qwen38_bf16_path():
    c=ExperimentConfig(model_path='/tmp/model',model_name='test',model_adaptation='qwen38-910b-v1',dtype='bfloat16',w4a8_linear_backend='native-tuned')
    assert c.w4a8_linear_backend is W4A8LinearBackend.NATIVE_TUNED
    assert diagnostic_worker(c).endswith('ProfilingQwen38NPUWorker')
    assert ExperimentConfig(model_path='/tmp/model',model_name='test').w4a8_linear_backend is W4A8LinearBackend.UPSTREAM
    with pytest.raises(ValueError):ExperimentConfig(model_path='/tmp/model',model_name='test',w4a8_linear_backend='native-tuned')


@pytest.mark.parametrize('already_enabled',[False,True])
def test_legacy_entry_reuses_main_installation_and_updates_runtime(monkeypatch,already_enabled):
    torch=pytest.importorskip('torch');pytest.importorskip('torch_npu')
    from qwen_llm.engine import qwen38_worker
    from .native_tuned import install
    identity={'backend':'native-tuned','layers':128}
    calls=[]
    monkeypatch.setattr(qwen38_worker,'install_fast_decode',lambda model:calls.append(model) or identity)
    worker=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16),
        model_runner=SimpleNamespace(model=object()),
        _qwen38_runtime={'w4_decode':identity if already_enabled else {'backend':'upstream'}})
    install(worker)
    assert len(calls)==(0 if already_enabled else 1)
    assert worker._native_tuned_identity is worker._qwen38_runtime['w4_decode'] is identity


def test_profile_can_explicitly_disable_a_catalog_enabled_fast_path(tmp_path):
    import json
    from .runner import config_from_catalog
    p=tmp_path/'catalog.json'
    p.write_text(json.dumps([{'name':'model','path':'/models/test','kv_cache_gib':22.5,
        'model_adaptation':'qwen38-910b-v1','dtype':'bfloat16','qwen38_fast_w4':True}]))
    assert config_from_catalog(p,'model').qwen38_fast_w4 is True
    assert config_from_catalog(p,'model',qwen38_fast_w4=False).qwen38_fast_w4 is False
    assert config_from_catalog(p,'model',w4a8_linear_backend='native-tuned').qwen38_fast_w4 is True
