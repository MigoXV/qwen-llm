import json

from qwen_llm.engine.tokenizer import tokenizer_path


def test_adapter_preserves_original_files_and_token_ids(tmp_path, monkeypatch):
    model = tmp_path / 'original'
    model.mkdir()
    config = {'tokenizer_class': 'TokenizersBackend', 'eos_token': '<end>', 'pad_token': '<pad>',
              'model_specific_special_tokens': {'image_token': '<image>'}}
    (model / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
    source = model / 'tokenizer_config.json'
    source.write_text(json.dumps(config))
    vocabulary = {'model': {'type': 'BPE', 'vocab': {'hello': 0}}, 'added_tokens': [
        {'id': 1, 'content': '<end>', 'special': True},
        {'id': 2, 'content': '<image>', 'special': True}]}
    (model / 'tokenizer.json').write_text(json.dumps(vocabulary))
    (model / 'chat_template.jinja').write_text('original template')
    before = {p.name: p.read_bytes() for p in model.iterdir()}
    monkeypatch.setenv('QWEN_MODELS_ROOT', str(tmp_path / 'data'))
    adapter = tokenizer_path(model)
    assert adapter != model
    assert {p.name: p.read_bytes() for p in model.iterdir()} == before
    assert (adapter / 'tokenizer.json').is_symlink()
    assert json.loads((adapter / 'tokenizer.json').read_text()) == vocabulary
    assert (adapter / 'chat_template.jinja').read_text() == 'original template'
    assert json.loads((adapter / 'tokenizer_config.json').read_text())['tokenizer_class'] == 'Qwen2Tokenizer'
    assert tokenizer_path(model) == adapter
    (model / 'chat_template.jinja').write_text('revised template')
    assert tokenizer_path(model) != adapter


def test_legacy_tokenizer_needs_no_adapter(tmp_path):
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({'tokenizer_class': 'Qwen2Tokenizer'}))
    assert tokenizer_path(tmp_path) == tmp_path
