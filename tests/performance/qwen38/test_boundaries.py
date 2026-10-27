import ast
from pathlib import Path
import pytest
from typer.testing import CliRunner
from qwen_llm.commands.app import app


def test_native_w4_backend_rejects_unsupported_dtype_before_worker_start():
    from .config import ExperimentConfig
    with pytest.raises(ValueError, match='BF16'):
        ExperimentConfig(w4a8_linear_backend='gmm', dtype='float16')


def test_production_has_no_experiment_imports_or_cli_commands():
    root=Path(__file__).resolve().parents[3]/'src/qwen_llm'
    for path in root.rglob('*.py'):
        tree=ast.parse(path.read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom):assert not (n.module or '').startswith('tests')
            if isinstance(n,ast.Import):assert not any(x.name.startswith('tests') for x in n.names)
    result=CliRunner().invoke(app,['--help'])
    for name in ['profile-report','benchmark','evaluate','quantize-w4a8','smoke']:
        assert name not in result.output
