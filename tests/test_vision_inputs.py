"""视觉输入改造的验收测试：媒体解析、请求归一化、引擎多模态 prompt 接线、配置门控与工件守卫。"""
import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from qwen_llm.engine.base import Generation
from qwen_llm.engine.generation import AsyncGenerationEngine
from qwen_llm.engine.nvidia import NvidiaConfig, has_visual_weights
from qwen_llm.servicer.chat import ChatRequest
from qwen_llm.servicer.media import count_image_parts, parse_image_part, trim_image_history

# 1x1 红色 PNG。
PNG_1PX = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==')


def data_uri(raw: bytes = PNG_1PX, mime: str = 'image/png') -> str:
    return f'data:{mime};base64,{base64.b64encode(raw).decode()}'


def test_media_accepts_png_data_uri():
    image = parse_image_part({'type': 'image_url', 'image_url': {'url': data_uri()}})
    assert (image.mode, image.width, image.height) == ('RGB', 1, 1)


def test_media_rejects_bad_parts():
    with pytest.raises(ValueError, match='内联'):
        parse_image_part({'type': 'image_url', 'image_url': {'url': 'https://example.com/a.png'}})
    with pytest.raises(ValueError, match='不受支持'):
        parse_image_part({'type': 'image_url', 'image_url': {'url': 'data:image/gif;base64,QUJD'}})
    with pytest.raises(ValueError, match='base64'):
        parse_image_part({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,QU#B$'}})
    with pytest.raises(ValueError, match='MB'):
        parse_image_part({'type': 'image_url', 'image_url': {'url': data_uri(b'\x00' * 17 * 1024 * 1024)}})
    with pytest.raises(ValueError, match='image_url'):
        parse_image_part({'type': 'image_url', 'image_url': {'url': data_uri()}, 'detail': 'low'})


def _chat(content):
    return ChatRequest(model='m', messages=[{'role': 'user', 'content': content}]).normalized_messages()


def test_chat_request_keeps_text_only_flattening():
    messages = _chat([{'type': 'text', 'text': '甲'}, {'type': 'text', 'text': '乙'}])
    assert messages == [{'role': 'user', 'content': '甲乙'}]


def test_chat_request_normalizes_mixed_parts_in_order():
    messages = _chat([{'type': 'text', 'text': '看'}, {'type': 'image_url', 'image_url': {'url': data_uri()}}])
    content = messages[0]['content']
    assert isinstance(content, list) and count_image_parts(messages) == 1
    assert [part['type'] for part in content] == ['text', 'image']
    assert content[1]['image'].mode == 'RGB'
    with pytest.raises(ValueError, match='content parts'):
        _chat([{'type': 'audio_url', 'audio_url': {'url': data_uri()}}])


def _vision_engine():
    engine = AsyncGenerationEngine(NvidiaConfig())
    placeholder = '<|vision_start|><|image_pad|><|vision_end|>'
    engine.image_placeholder = placeholder
    engine.image_pad_token_id = 99

    def apply_chat_template(messages, **kwargs):
        content = messages[0]['content']
        if not kwargs.get('tokenize', True):
            return content
        return [1, 99, 2] if placeholder in content else [1, 2]

    engine.tokenizer = SimpleNamespace(
        apply_chat_template=apply_chat_template, convert_tokens_to_ids=lambda token: [],
        encode=lambda text, **kwargs: [1, 99, 2] if placeholder in text else [1, 2])
    return engine, placeholder


def test_engine_prepare_builds_multimodal_prompt():
    engine, placeholder = _vision_engine()
    prompt = asyncio.run(engine.prepare(
        [{'role': 'user', 'content': [{'type': 'text', 'text': '看'},
                                      {'type': 'image', 'image': parse_image_part(
                                          {'type': 'image_url', 'image_url': {'url': data_uri()}})}]}], 'none'))
    assert isinstance(prompt, dict)
    assert prompt['prompt_token_ids'] == [1, 99, 2]
    assert len(prompt['multi_modal_data']['image']) == 1


def test_engine_prepare_keeps_plain_list_for_text():
    engine, _ = _vision_engine()
    prompt = asyncio.run(engine.prepare(
        [{'role': 'user', 'content': [{'type': 'text', 'text': '纯文本'}]}], 'none'))
    assert prompt == [1, 2]


def test_engine_prepare_rejects_image_without_vision_wiring():
    engine, _ = _vision_engine()
    engine.image_placeholder = None
    with pytest.raises(ValueError, match='视觉'):
        asyncio.run(engine.prepare(
            [{'role': 'user', 'content': [{'type': 'image', 'image': object()}]}], 'none'))


def test_engine_generate_passes_structured_prompt_through():
    engine, _ = _vision_engine()

    class StubLLM:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt_arg, sampling, request_id):
            self.prompts.append(prompt_arg)

            async def stream():
                yield SimpleNamespace(outputs=[SimpleNamespace(
                    text='ok', token_ids=[5, 6], finish_reason='stop')], num_cached_tokens=0)
            return stream()

    engine.llm = StubLLM()
    image = parse_image_part({'type': 'image_url', 'image_url': {'url': data_uri()}})
    async def collect():
        return [chunk async for chunk in engine.generate(
            {'prompt_token_ids': [1, 99, 2], 'multi_modal_data': {'image': [image]}}, 'rid', temperature=0.1)]
    chunks = asyncio.run(collect())
    assert len(chunks) == 1 and chunks[0].text == 'ok'
    sent = engine.llm.prompts[0]
    assert sent['prompt_token_ids'] == [1, 99, 2] and sent['multi_modal_data']['image'] == [image]


def test_vision_config_gating():
    with pytest.raises(ValueError, match='实验'):
        NvidiaConfig(vision=True)
    assert NvidiaConfig(vision=True, experimental=True).vision
    with pytest.raises(ValueError, match='至少为 1'):
        NvidiaConfig(vision=True, experimental=True, max_images_per_prompt=0)
    assert NvidiaConfig(vision=True, experimental=True, max_images_per_prompt=64).max_images_per_prompt == 64


def _artifact(tmp_path, weight_map):
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))


def test_has_visual_weights_detection(tmp_path):
    _artifact(tmp_path, {'model.language.0.q_proj': 'f1'})
    assert has_visual_weights(tmp_path) is False
    _artifact(tmp_path, {'model.language.0.q_proj': 'f1', 'model.visual.0.q_proj': 'f1'})
    assert has_visual_weights(tmp_path) is True


def _client(vision: bool, max_images: int = 1, trim_history: bool = False):
    from fastapi.testclient import TestClient
    from qwen_llm.servicer.app import create_app

    class StubEngine:
        supports_thinking = True

        async def start(self):
            pass

        async def close(self):
            pass

        async def health(self):
            pass

        async def prepare(self, messages, effort, **kwargs):
            self.prepared_messages = messages
            return [1, 2]

        async def generate(self, prompt, request_id, **sampling):
            yield Generation('ok', (5,), 'stop')

    config = NvidiaConfig(model_name='m', experimental=True, vision=vision, max_images_per_prompt=max_images, trim_history_images=trim_history)
    engine = StubEngine()
    app = create_app(config=config, engine=engine)
    app.state.test_engine = engine
    return TestClient(app)


def _image_request():
    return {'model': 'm', 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': '看'}, {'type': 'image_url', 'image_url': {'url': data_uri()}}]}]}


def test_api_rejects_image_when_vision_disabled():
    with _client(vision=False) as client:
        response = client.post('/v1/chat/completions', json=_image_request())
    assert response.status_code == 400 and '视觉' in response.text


def test_api_rejects_image_over_limit():
    request = _image_request()
    request['messages'][0]['content'].append({'type': 'image_url', 'image_url': {'url': data_uri()}})
    with _client(vision=True, max_images=1) as client:
        response = client.post('/v1/chat/completions', json=request)
    assert response.status_code == 400 and '每请求最多' in response.text


def test_api_trims_history_and_preserves_current_images():
    request = _image_request()
    old = request['messages'][0]
    request['messages'] = [old, {'role': 'assistant', 'content': '旧图回答'},
                           old, {'role': 'assistant', 'content': '上一轮回答'},
                           old]
    with _client(vision=True, max_images=2, trim_history=True) as client:
        response = client.post('/v1/chat/completions', json=request)
        sent = client.app.state.test_engine.prepared_messages
    assert response.status_code == 200, response.text
    assert count_image_parts(sent) == 2
    assert count_image_parts(sent[:1]) == 0
    assert count_image_parts(sent[2:3]) == 1
    assert count_image_parts(sent[4:]) == 1
    assert sent[1]['content'] == '旧图回答'
    assert '本轮未发送' in sent[0]['content'][1]['text']


def test_history_trim_keeps_latest_images_without_mutating_inputs():
    images = [object() for _ in range(5)]
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': f'文字{i}'}, {'type': 'image', 'image': image}]}
        for i, image in enumerate(images)]
    result = trim_image_history(messages, 2)
    assert count_image_parts(result) == 2
    assert count_image_parts(messages) == 5
    assert result[-1]['content'][1]['image'] is images[-1]
    assert result[-2]['content'][1]['image'] is images[-2]
    assert [m['content'][0]['text'] for m in result] == [f'文字{i}' for i in range(5)]


def test_history_trim_protects_whole_current_turn_and_tool_results():
    old_image, new_image, tool_image = object(), object(), object()
    messages = [
        {'role': 'user', 'content': [{'type': 'image', 'image': old_image}]},
        {'role': 'assistant', 'content': '旧回答'},
        {'role': 'user', 'content': [{'type': 'image', 'image': new_image}]},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'read'}}]},
        {'role': 'tool', 'content': [{'type': 'image', 'image': tool_image}]}]
    result = trim_image_history(messages, 2)
    assert count_image_parts(result) == 2
    assert result[2:] == messages[2:]
    assert trim_image_history(messages, 1) is messages


def test_text_followup_keeps_recent_history_and_under_limit_is_unchanged():
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': object()}]},
                {'role': 'user', 'content': [{'type': 'image', 'image': object()}]},
                {'role': 'user', 'content': '继续说明'}]
    assert trim_image_history(messages, 2) is messages
    result = trim_image_history(messages, 1)
    assert count_image_parts(result) == 1
    assert result[-2:] == messages[-2:]


def test_api_preserves_large_image_history_by_default():
    request = _image_request()
    request['messages'] = request['messages'] * 12
    with _client(vision=True, max_images=64) as client:
        response = client.post('/v1/chat/completions', json=request)
        sent = client.app.state.test_engine.prepared_messages
    assert response.status_code == 200, response.text
    assert count_image_parts(sent) == 12


def test_api_default_does_not_silently_remove_history():
    request = _image_request()
    request['messages'] = request['messages'] * 3
    with _client(vision=True, max_images=2) as client:
        response = client.post('/v1/chat/completions', json=request)
    assert response.status_code == 400
