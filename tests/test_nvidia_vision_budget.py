import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from qwen_llm.engine.generation import AsyncGenerationEngine
from qwen_llm.engine.nvidia import NvidiaConfig, NvidiaEngine, NvidiaVisionInputError


def engine(monkeypatch, **limits):
    preprocess = pytest.importorskip('vllm.renderers.inputs.preprocess')
    monkeypatch.setattr(preprocess, 'parse_model_prompt', lambda config, prompt: prompt)
    raw = {'prompt_token_ids': [1, 99, 2], 'multi_modal_data': {'image': []}}
    monkeypatch.setattr(AsyncGenerationEngine, 'prepare', AsyncMock(return_value=raw))
    expanded = {'type': 'multimodal', 'prompt_token_ids': list(range(100)),
                'mm_placeholders': {'image': [SimpleNamespace(length=80)]}}
    e = NvidiaEngine(NvidiaConfig(vision=True, experimental=True, **limits))
    render = AsyncMock(return_value=[expanded])
    e.llm = SimpleNamespace(model_config=object(), renderer=SimpleNamespace(render_cmpl_async=render))
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': SimpleNamespace(width=10, height=10)}]}]
    return e, messages, expanded, render


def test_actual_expanded_tokens_are_forwarded_once(monkeypatch):
    e, messages, expanded, render = engine(monkeypatch)
    assert asyncio.run(e.prepare(messages, 'none')) is expanded
    assert len(expanded['prompt_token_ids']) == 100
    render.assert_awaited_once()


def test_context_checked_after_visual_expansion(monkeypatch):
    e, messages, _, _ = engine(monkeypatch, max_input_tokens=50)
    with pytest.raises(NvidiaVisionInputError) as error:
        asyncio.run(e.prepare(messages, 'none'))
    assert error.value.code == 'context_length_exceeded'


def test_pixel_budget_rejects_before_processing(monkeypatch):
    e, messages, _, render = engine(monkeypatch, max_image_pixels_per_prompt=99)
    with pytest.raises(NvidiaVisionInputError, match='像素'):
        asyncio.run(e.prepare(messages, 'none'))
    render.assert_not_awaited()


def test_visual_budget_independent_of_context(monkeypatch):
    e, messages, _, _ = engine(monkeypatch, max_visual_tokens_per_prompt=79)
    with pytest.raises(NvidiaVisionInputError, match='视觉 token'):
        asyncio.run(e.prepare(messages, 'none'))


def test_text_mode_preserves_existing_path(monkeypatch):
    e, messages, _, render = engine(monkeypatch)
    object.__setattr__(e.config, 'vision', False)
    result = asyncio.run(e.prepare(messages, 'none'))
    assert result['prompt_token_ids'] == [1, 99, 2]
    render.assert_not_awaited()


def test_nvidia_rejection_returns_400_and_releases_slot():
    from tests.test_vision_inputs import _client, _image_request
    with _client(vision=True) as client:
        engine = client.app.state.test_engine
        original = engine.prepare
        engine.prepare = AsyncMock(side_effect=NvidiaVisionInputError('预算不足', 'context_length_exceeded'))
        reply = client.post('/v1/chat/completions', json=_image_request())
        assert reply.status_code == 400
        assert reply.json()['error']['code'] == 'context_length_exceeded'
        engine.prepare = original
        assert client.post('/v1/chat/completions', json=_image_request()).status_code == 200
