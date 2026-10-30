import asyncio
import json
from collections import UserDict
from types import SimpleNamespace

import pytest

from qwen_llm.config import Runner, ServingConfig
from qwen_llm.engine.nvidia import NvidiaConfig, inspect_model


def test_platform_options_do_not_leak():
    old = ServingConfig()
    cuda = NvidiaConfig()
    assert old.runner is Runner.ACLGRAPH
    assert old.kv_cache_bytes == int(10.25 * 1024**3)
    assert cuda.runner is Runner.EAGER and cuda.kv_cache_bytes is None
    assert not hasattr(cuda, 'npugraph_ex')
    with pytest.raises(ValueError):
        NvidiaConfig(runner='aclgraph')
    with pytest.raises(ValueError):
        ServingConfig(runner='cudagraph')


def test_fp8_requires_explicit_experiment():
    with pytest.raises(ValueError, match='实验'):
        NvidiaConfig(kv_cache_dtype='fp8_e4m3')
    assert NvidiaConfig(kv_cache_dtype='fp8_e4m3', experimental=True).experimental
    with pytest.raises(ValueError):
        NvidiaConfig(tensor_parallel_size=3)
    with pytest.raises(ValueError, match='对照实验'):
        NvidiaConfig(attention_backend='FLASHINFER')
    assert NvidiaConfig(attention_backend='FLASHINFER', experimental=True).attention_backend == 'FLASHINFER'


def test_artifact_validation(tmp_path):
    (tmp_path / '.quantization.incomplete').touch()
    with pytest.raises(ValueError, match='未完成'):
        inspect_model(tmp_path)
    (tmp_path / '.quantization.incomplete').unlink()
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': '../escape'}}))
    with pytest.raises(ValueError, match='非法'):
        inspect_model(tmp_path)


@pytest.mark.parametrize('encoded', [[1, 2], UserDict(input_ids=[1, 2])])
def test_shared_template_accepts_legacy_list_and_transformers5_mapping(encoded):
    from qwen_llm.engine.generation import AsyncGenerationEngine
    engine = AsyncGenerationEngine(NvidiaConfig())
    engine.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: encoded)
    assert asyncio.run(engine.prepare([{'role': 'user', 'content': 'test'}], 'none')) == [1, 2]


