import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qwen_llm.commands.app import app
from qwen_llm.config import ServingConfig
from qwen_llm.engine.diagnostics import diagnose
from qwen_llm.engine.vllm import V1Engine


def test_doctor_missing_model(tmp_path):
    result = diagnose(tmp_path)
    assert not result['ready']
    assert any('模型文件不完整' in x for x in result['blockers'])
    cli = CliRunner().invoke(app, ['doctor', '--model-path', str(tmp_path)])
    assert cli.exit_code == 1
    assert json.loads(cli.stdout)['ready'] is False


def test_awq_preflight_without_loading_engine(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'quantization_config': {'quant_method':'awq'}}))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a':'shard.safetensors'}}))
    report = diagnose(tmp_path)
    assert any('AWQ' in x for x in report['blockers'])
    assert any('分片' in x for x in report['blockers'])
    async def run():
        engine = V1Engine(ServingConfig(model_path=tmp_path))
        with pytest.raises(RuntimeError, match='AWQ'):
            await engine.start()
        assert engine.llm is None
        await engine.close()
    asyncio.run(run())


def test_config_rejects_zero():
    with pytest.raises(ValueError): ServingConfig(max_concurrency=0)


def test_ascend_quantized_index_is_recognized(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'architectures':['Qwen3ForCausalLM']}))
    (tmp_path/'quant_model_description.json').write_text(json.dumps({'model_quant_type':'W8A8'}))
    (tmp_path/'quant_model_weight_w8a8.safetensors.index.json').write_text(
        json.dumps({'weight_map':{'model.weight':'quant_model_weight_w8a8-00001-of-00001.safetensors'}}))
    (tmp_path/'quant_model_weight_w8a8-00001-of-00001.safetensors').touch()
    report = diagnose(tmp_path)
    assert report['quantization']=='ascend'
    assert report['model_quant_type']=='W8A8'
    assert not any('缺少权重' in x for x in report['blockers'])
