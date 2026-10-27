"""思考档位在 API、模板及实际 tokenizer 之间保持一致。"""
import asyncio
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from qwen_llm.config import ReasoningEffort, ServingConfig
from qwen_llm.engine.vllm import V1Engine
from qwen_llm.servicer.app import create_app
from tests.test_responses import FakeEngine, parse_sse, payload


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('requested,actual', [
    ('none', 'none'), ('low', 'low'), ('medium', 'medium'),
    ('xhigh', 'xhigh'), ('high', 'xhigh'),
])
def test_effort_reaches_engine_and_response(requested, actual, stream):
    engine = FakeEngine(['答案' if actual == 'none' else '思考</think>答案'])
    with TestClient(create_app(engine=engine)) as client:
        reply = client.post('/responses', json=payload(
            reasoning={'effort': requested}, stream=stream))
        assert reply.status_code == 200
        response = parse_sse(reply.text)[-1]['response'] if stream else reply.json()
        assert engine.prepared[0][1] is ReasoningEffort(actual)
        assert response['reasoning']['effort'] == actual
        assert response['output'][-1]['content'][0]['text'] == '答案'
        assert response['usage']['output_tokens_details']['reasoning_tokens'] == (0 if actual == 'none' else 2)


@pytest.mark.parametrize('extra,actual', [
    ({}, 'none'), ({'reasoning': {}}, 'none'),
    ({'chat_template_kwargs': {}}, 'none'),
    ({'enable_thinking': False}, 'none'),
    ({'enable_thinking': True}, 'xhigh'),
    ({'chat_template_kwargs': {'enable_thinking': True}}, 'xhigh'),
    ({'chat_template_kwargs': {'enable_thinking': False}}, 'none'),
    ({'reasoning': {}, 'enable_thinking': True}, 'xhigh'),
    ({'reasoning': {'effort': 'low'}, 'enable_thinking': True}, 'low'),
    ({'reasoning': {'effort': 'medium'}, 'chat_template_kwargs': {'enable_thinking': True}}, 'medium'),
    ({'chat_template_kwargs': {'reasoning_effort': 'low'}}, 'low'),
    ({'chat_template_kwargs': {'reasoning_effort': 'medium', 'enable_thinking': True}}, 'medium'),
    ({'reasoning': {'effort': 'xhigh'}, 'chat_template_kwargs': {'reasoning_effort': 'high'}, 'enable_thinking': True}, 'xhigh'),
    ({'reasoning': {'effort': 'none'}, 'enable_thinking': False,
      'chat_template_kwargs': {'enable_thinking': False, 'reasoning_effort': 'none'}}, 'none'),
])
def test_defaults_aliases_and_consistent_options(extra, actual):
    engine = FakeEngine()
    with TestClient(create_app(engine=engine)) as client:
        response = client.post('/responses', json=payload(**extra))
        assert response.status_code == 200
        assert response.json()['reasoning']['effort'] == actual
        assert engine.prepared[0][1] is ReasoningEffort(actual)


@pytest.mark.parametrize('extra', [
    {'reasoning': {'effort': 'invalid'}},
    {'reasoning': {'effort': True}},
    {'reasoning': {'effort': None}},
    {'reasoning': {'effort': 'low'}, 'chat_template_kwargs': {'reasoning_effort': 'medium'}},
    {'reasoning': {'effort': 'none'}, 'enable_thinking': True},
    {'enable_thinking': False, 'chat_template_kwargs': {'reasoning_effort': 'low'}},
    {'enable_thinking': True, 'chat_template_kwargs': {'enable_thinking': False}},
    {'chat_template_kwargs': {'enable_thinking': 'true'}},
    {'chat_template_kwargs': {'enable_thinking': None}},
    {'chat_template_kwargs': {'reasoning_effort': 'invalid'}},
    {'chat_template_kwargs': {'reasoning_effort': None}},
    {'chat_template_kwargs': {'unknown': True}},
    *[{'reasoning': {'effort': effort}, 'text': {'format': {'type': 'json_object'}}}
      for effort in ('low', 'medium', 'xhigh', 'high')],
])
def test_bad_options_rejected_before_preparation(extra):
    engine = FakeEngine()
    with TestClient(create_app(engine=engine)) as client:
        assert client.post('/responses', json=payload(**extra)).status_code == 400
        assert not engine.prepared
        assert 'responses_inflight 0.0' in client.get('/metrics').text
        assert client.get('/health/ready').status_code == 200


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('effort', list(ReasoningEffort))
def test_runtime_only_passes_effort_to_native_templates(native, effort):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return [1, 2]

    engine = V1Engine(ServingConfig())
    engine.tokenizer = Tokenizer()
    engine.supports_reasoning_effort = native
    assert asyncio.run(engine.prepare([{'role': 'user', 'content': '你好'}], effort)) == [1, 2]
    expected = {'tokenize': True, 'add_generation_prompt': True,
                'enable_thinking': effort is not ReasoningEffort.NONE}
    if native and effort is not ReasoningEffort.NONE:
        expected['reasoning_effort'] = effort.value
    assert engine.tokenizer.kwargs == expected


@pytest.fixture(scope='module')
def original_tokenizer():
    model = Path('/home/model-bin/Qwen/Qwen3.8-27B')
    if not (model / 'tokenizer.json').exists():
        pytest.skip('本地未安装原始 Qwen3.8 tokenizer；不自动下载')
    transformers = pytest.importorskip('transformers')
    return transformers.AutoTokenizer.from_pretrained(str(model), local_files_only=True)


@pytest.mark.parametrize('name', ['Qwen3.8-27B', 'Qwen3.8-27B-W4A8-Ascend',
                                  'Qwen3.8-27B-W4A8-Ascend-GDNHeadW8-v1'])
def test_actual_runtime_tokens_match_original_template(original_tokenizer, name):
    from transformers import AutoTokenizer

    model = Path('/home/model-bin/Qwen') / name
    if not (model / 'tokenizer.json').exists():
        pytest.skip('本地未安装此量化模型的 tokenizer')
    engine = V1Engine(ServingConfig())
    engine.tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    engine.supports_reasoning_effort = 'reasoning_effort' in engine.tokenizer.chat_template
    assert engine.supports_reasoning_effort
    for messages in (
        [{'role': 'user', 'content': '计算 17 × 23。'}],
        [{'role': 'system', 'content': '请用中文作答。'},
         {'role': 'user', 'content': '17 × 23 是多少？'},
         {'role': 'assistant', 'content': '391'},
         {'role': 'user', 'content': '请检查上一个结果。'}],
    ):
        prompts = []
        for effort in ReasoningEffort:
            kwargs = {'enable_thinking': effort is not ReasoningEffort.NONE}
            if effort is not ReasoningEffort.NONE:
                kwargs['reasoning_effort'] = effort.value
            expected = original_tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, **kwargs)
            actual = asyncio.run(engine.prepare(messages, effort))
            assert actual == expected
            prompts.append(tuple(actual))
        assert len(set(prompts)) == 4
