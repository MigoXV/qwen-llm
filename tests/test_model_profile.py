import json
import os

import pytest
from typer.testing import CliRunner

from qwen_llm.commands.app import app
from qwen_llm.config import ServingConfig


def test_native_w4_backend_rejects_unsupported_dtype_before_worker_start():
    with pytest.raises(ValueError, match='BF16'):
        ServingConfig(w4a8_linear_backend='gmm', dtype='float16')


def test_profile_selects_matching_model_and_preserves_explicit_override(tmp_path, monkeypatch):
    catalog = tmp_path / 'models.json'
    catalog.write_text(json.dumps([{
        'name': 'qwen3.5-9b', 'path': '/models/verified-W8A8',
        'kv_cache_gib': 8.5, 'graph_mode': 'FULL_DECODE_ONLY', 'speculative_tokens': 2,
        'w4a8_linear_backend': 'gmm',
    }]))
    invoked = []
    monkeypatch.setattr(os, 'execv', lambda executable, args: invoked.extend(args))
    result = CliRunner().invoke(app, ['serve-model', 'Qwen3.5-9B', '--catalog', str(catalog),
                                      '--kv-cache-gib', '2.5'])
    assert result.exit_code == 0, result.output
    assert invoked[invoked.index('--model-path')+1] == '/models/verified-W8A8'
    assert invoked[invoked.index('--model-name')+1] == 'qwen3.5-9b'
    assert invoked[invoked.index('--speculative-tokens')+1] == '2'
    assert invoked[invoked.index('--w4a8-linear-backend')+1] == 'gmm'
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
