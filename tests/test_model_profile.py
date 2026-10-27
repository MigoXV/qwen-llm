import json
import os

import pytest
from typer.testing import CliRunner

from qwen_llm.commands.app import app
def test_profile_selects_matching_model_and_preserves_explicit_override(tmp_path, monkeypatch):
    catalog = tmp_path / 'models.json'
    catalog.write_text(json.dumps([{
        'name': 'qwen3.5-9b', 'path': '/models/verified-W8A8',
        'kv_cache_gib': 8.5, 'graph_mode': 'FULL_DECODE_ONLY', 'speculative_tokens': 2,
        'runner': 'eager',
    }]))
    invoked = []
    monkeypatch.setattr(os, 'execv', lambda executable, args: invoked.extend(args))
    result = CliRunner().invoke(app, ['serve-model', 'Qwen3.5-9B', '--catalog', str(catalog),
                                      '--kv-cache-gib', '2.5'])
    assert result.exit_code == 0, result.output
    assert invoked[invoked.index('--model-path')+1] == '/models/verified-W8A8'
    assert invoked[invoked.index('--model-name')+1] == 'qwen3.5-9b'
    assert invoked[invoked.index('--speculative-tokens')+1] == '2'
    assert '--w4a8-linear-backend' not in invoked
    assert '--model-adaptation' not in invoked
    assert invoked[-2:] == ['--kv-cache-gib', '2.5']


def test_profile_rejects_capacity_values_that_would_be_ignored(tmp_path, monkeypatch):
    catalog = tmp_path / 'models.json'
    catalog.write_text(json.dumps([{'name': 'invalid', 'path': '/models/weights',
                                    'kv_cache_gib': 3, 'max_concurrency': 8}]))
    invoked = []
    monkeypatch.setattr(os, 'execv', lambda *args: invoked.append(args))
    result = CliRunner().invoke(app, ['serve-model', 'invalid', '--catalog', str(catalog)])
    assert result.exit_code != 0
    assert 'max_concurrency' in result.output
    assert invoked == []


def test_profile_forwards_wider_input_capacity_and_explicit_override(tmp_path, monkeypatch):
    catalog = tmp_path / 'models.json'
    catalog.write_text(json.dumps([{'name': 'long', 'path': '/models/weights',
                                    'kv_cache_gib': 5.75, 'max_input_tokens': 24576}]))
    invoked = []
    monkeypatch.setattr(os, 'execv', lambda executable, args: invoked.extend(args))
    result = CliRunner().invoke(app, ['serve-model', 'long', '--catalog', str(catalog),
                                     '--max-input-tokens', '16384'])
    assert result.exit_code == 0, result.output
    assert invoked[invoked.index('--max-input-tokens') + 1] == '24576'
    assert invoked[-2:] == ['--max-input-tokens', '16384']


@pytest.mark.parametrize('capacity', [0, -1, True, '24576'])
def test_profile_rejects_invalid_input_capacity(tmp_path, monkeypatch, capacity):
    catalog = tmp_path / 'models.json'
    catalog.write_text(json.dumps([{'name': 'bad', 'path': '/models/weights',
                                    'kv_cache_gib': 3, 'max_input_tokens': capacity}]))
    invoked = []
    monkeypatch.setattr(os, 'execv', lambda *args: invoked.append(args))
    result = CliRunner().invoke(app, ['serve-model', 'bad', '--catalog', str(catalog)])
    assert result.exit_code != 0
    assert 'max_input_tokens' in result.output
    assert invoked == []


def test_serve_uses_configured_input_capacity_without_loading_model(monkeypatch):
    import uvicorn
    import qwen_llm.servicer.app as service

    configs = []
    monkeypatch.setattr(service, 'create_app', lambda config: configs.append(config) or object())
    monkeypatch.setattr(uvicorn, 'run', lambda *args, **kwargs: None)
    result = CliRunner().invoke(app, ['serve', '--max-input-tokens', '24576'])
    assert result.exit_code == 0, result.output
    assert configs[0].max_input_tokens == 24576
    assert configs[0].max_output_tokens == 8192


def test_model_adaptation_is_opt_in_and_config_rejects_unknown_values():
    from qwen_llm.config import ModelAdaptation, ServingConfig
    assert ServingConfig().model_adaptation is ModelAdaptation.NONE
    assert ServingConfig(model_adaptation='qwen38-910b-v1').model_adaptation is ModelAdaptation.QWEN38_910B_V1
    with pytest.raises(ValueError):
        ServingConfig(model_adaptation='arbitrary')


def test_catalog_forwards_model_adaptation_and_keeps_overrides(tmp_path, monkeypatch):
    catalog = tmp_path / 'models.json'
    catalog.write_text(json.dumps([{'name': 'specialized', 'path': '/models/independent',
                                    'kv_cache_gib': 22.5, 'model_adaptation': 'qwen38-910b-v1'}]))
    invoked = []
    monkeypatch.setattr(os, 'execv', lambda executable, args: invoked.extend(args))
    result = CliRunner().invoke(app, ['serve-model', 'specialized', '--catalog', str(catalog),
                                     '--model-adaptation', 'none'])
    assert result.exit_code == 0, result.output
    assert invoked[invoked.index('--model-adaptation') + 1] == 'qwen38-910b-v1'
    assert invoked[-2:] == ['--model-adaptation', 'none']


def test_serve_passes_typed_model_adaptation_without_loading_model(monkeypatch):
    import uvicorn
    import qwen_llm.servicer.app as service
    from qwen_llm.config import ModelAdaptation
    configs = []
    monkeypatch.setattr(service, 'create_app', lambda config: configs.append(config) or object())
    monkeypatch.setattr(uvicorn, 'run', lambda *args, **kwargs: None)
    result = CliRunner().invoke(app, ['serve', '--model-adaptation', 'qwen38-910b-v1'])
    assert result.exit_code == 0, result.output
    assert configs[0].model_adaptation is ModelAdaptation.QWEN38_910B_V1


def test_production_commands_exclude_experiment_entry_points():
    result = CliRunner().invoke(app, ['--help'], color=False)
    assert result.exit_code == 0
    for name in ('benchmark', 'evaluate', 'profile-report', 'quantize-w4a8', 'smoke'):
        assert name not in result.output
