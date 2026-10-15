"""为 Transformers 5 导出的 Qwen tokenizer 提供独立、可追溯的兼容目录。"""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def tokenizer_path(model: Path) -> Path:
    model = model.resolve()
    config_file = model / 'tokenizer_config.json'
    if not config_file.exists():
        return model
    original = config_file.read_bytes()
    config = json.loads(original)
    if config.get('tokenizer_class') != 'TokenizersBackend':
        return model
    architecture = json.loads((model / 'config.json').read_text()).get('model_type')
    if architecture not in {'qwen3_5', 'qwen3_5_moe'}:
        raise ValueError('TokenizersBackend 兼容转换仅验证了 Qwen3.5 tokenizer')
    tokenizer_file = model / 'tokenizer.json'
    tokenizer_bytes = tokenizer_file.read_bytes()
    backend = json.loads(tokenizer_bytes)
    if backend['model']['type'] != 'BPE':
        raise ValueError('Qwen tokenizer 必须为原始 BPE backend')
    source_files = [tokenizer_file]
    for name in ('chat_template.jinja', 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt'):
        if (model / name).exists():
            source_files.append(model / name)
    digest = hashlib.sha256(original)
    for path in source_files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    root = Path(os.getenv('QWEN_MODELS_ROOT', '/home/qwen-models')) / 'tokenizer-adapters'
    destination = root / f'{model.name}-{fingerprint[:16]}'
    if destination.exists():
        return destination
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.building-', dir=root) as work:
        stage = Path(work) / 'tokenizer'
        stage.mkdir()
        for path in source_files:
            (stage / path.name).symlink_to(path)
        config['tokenizer_class'] = 'Qwen2Tokenizer'
        config['extra_special_tokens'] = config.pop('model_specific_special_tokens', {})
        config['additional_special_tokens'] = [token['content'] for token in backend.get('added_tokens', [])
            if token.get('special') and token['content'] not in (config.get('eos_token'), config.get('pad_token'))]
        (stage / 'tokenizer_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        (stage / 'adapter.json').write_text(json.dumps({
            'source': str(model), 'source_sha256': fingerprint,
            'change': 'TokenizersBackend -> Qwen2TokenizerFast; original BPE and chat template unchanged',
        }, ensure_ascii=False, indent=2) + '\n')
        try:
            stage.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
    return destination
