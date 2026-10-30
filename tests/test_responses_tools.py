"""Responses 客户端组合请求、函数事件与无状态工具往返回归。"""
import json

from fastapi.testclient import TestClient
import httpx
from openai import OpenAI
import pytest

from qwen_llm.config import ServingConfig
from qwen_llm.engine.base import Generation
from qwen_llm.engine.nvidia import NvidiaConfig
from qwen_llm.servicer.app import create_app
from tests.test_chat_completions import CALL, TOOLS, ToolEngine
from tests.test_responses import FakeEngine, parse_sse, payload


FUNCTIONS = [{'type': 'function', **tool['function']} for tool in TOOLS]


def test_reported_client_request_32k_cache_key_and_tools():
    engine = ToolEngine([CALL])
    sampling = {}
    original = engine.generate

    async def generate(*args, **kwargs):
        sampling.update(kwargs)
        async for item in original(*args, **kwargs):
            yield item

    engine.generate = generate
    config = NvidiaConfig(max_input_tokens=98304, max_output_tokens=32768, max_concurrency=2)
    with TestClient(create_app(config, engine)) as client:
        body = payload(max_output_tokens=32768, prompt_cache_key='session-a0b8f071',
                       tools=FUNCTIONS, stream=True)
        response = client.post('/v1/responses', json=body)
        assert response.status_code == 200, response.text
        events = parse_sse(response.text)
        assert events[-1]['type'] == 'response.completed'
        result = events[-1]['response']
        assert result['prompt_cache_key'] == body['prompt_cache_key']
        assert result['tools'][0]['name'] == 'weather'
        assert result['tools'][0]['strict'] is False
        assert result['tool_choice'] == 'auto'
        assert sampling['max_tokens'] == 32768
        assert not any('<tool_call' in str(event) for event in events)
        assert client.post('/v1/responses', json={**body, 'max_output_tokens': 32769}).status_code == 400
        assert 'responses_inflight 0.0' in client.get('/metrics').text
        # 两种协议服从同一个运行实例上限。
        assert client.post('/v1/chat/completions', json={'model': config.model_name,
            'messages': [{'role': 'user', 'content': 'hi'}], 'max_completion_tokens': 32768}).status_code == 200


@pytest.mark.parametrize('stream', [False, True])
def test_function_events_and_roundtrip(stream):
    engine = ToolEngine([CALL[:18], CALL])
    with TestClient(create_app(engine=engine)) as client:
        body = payload(tools=FUNCTIONS, stream=stream, tool_choice='required')
        reply = client.post('/v1/responses', json=body)
        assert reply.status_code == 200, reply.text
        if stream:
            events = parse_sse(reply.text)
            assert [e['sequence_number'] for e in events] == list(range(len(events)))
            assert events[0]['response']['output'] == []
            result = events[-1]['response']
            added = next(e for e in events if e['type'] == 'response.output_item.added')
            assert added['item']['arguments'] == '' and added['item']['status'] == 'in_progress'
            delta = next(e for e in events if e['type'] == 'response.function_call_arguments.delta')
            done = next(e for e in events if e['type'] == 'response.function_call_arguments.done')
            assert delta['item_id'] == done['item_id'] == added['item']['id']
            assert delta['delta'] == done['arguments'] == result['output'][0]['arguments']
            assert '[DONE]' not in reply.text
        else:
            result = reply.json()
        call = result['output'][0]
        assert call['type'] == 'function_call' and call['status'] == 'completed'
        assert call['call_id'] != call['id']
        assert json.loads(call['arguments']) == {'city': '北京', 'days': 2}
        engine.chunks = ['北京晴天']
        history = [{'role': 'user', 'content': '北京天气'}] + result['output'] + [
            {'type': 'function_call_output', 'call_id': call['call_id'], 'output': '北京晴天'}]
        followup = client.post('/v1/responses', json=payload(input=history, tools=FUNCTIONS)).json()
        assert followup['status'] == 'completed'
        assert followup['output'][0]['content'][0]['text'] == '北京晴天'
        assert engine.prepared[-1][0][-1] == {'role': 'tool', 'content': '北京晴天'}
        assert engine.prepared[-1][0][-2]['tool_calls'][0]['function']['arguments']['city'] == '北京'


def test_multiple_calls_named_choice_none_and_parallel_policy():
    second = {'type': 'function', 'name': 'time', 'parameters': {'type': 'object'}}
    engine = ToolEngine([CALL + '<tool_call><function=time></function></tool_call>'])
    with TestClient(create_app(engine=engine)) as client:
        body = payload(tools=FUNCTIONS + [second])
        response = client.post('/v1/responses', json=body).json()
        assert [item['name'] for item in response['output']] == ['weather', 'time']
        assert len({item['call_id'] for item in response['output']}) == 2
        history = [{'role': 'user', 'content': '天气和时间'}] + response['output'] + [
            {'type': 'function_call_output', 'call_id': item['call_id'],
             'output': [{'type': 'input_text', 'text': '完成'}]} for item in response['output']]
        engine.chunks = ['完成']
        assert client.post('/v1/responses', json=payload(input=history)).status_code == 200
        engine.chunks = [CALL + '<tool_call><function=time></function></tool_call>']
        assert client.post('/v1/responses', json={**body, 'parallel_tool_calls': False}).status_code == 502
        engine.chunks = [CALL]
        named = client.post('/v1/responses', json={**body, 'tool_choice': {'type': 'function', 'name': 'weather'}})
        assert named.status_code == 200
        assert named.json()['tool_choice'] == {'type': 'function', 'name': 'weather'}
        engine.chunks = ['普通回答']
        assert client.post('/v1/responses', json={**body, 'tool_choice': 'none'}).status_code == 200
        assert engine.prepared[-1][2] is None


@pytest.mark.parametrize('extra', [
    {'tools': [{'type': 'function', 'name': 'bad name'}]},
    {'tools': [{'type': 'function', 'name': 'f', 'parameters': {'type': 'bad'}}]},
    {'tools': FUNCTIONS * 2},
    {'tools': FUNCTIONS, 'tool_choice': {'type': 'function', 'name': 'missing'}},
    {'tools': FUNCTIONS, 'tool_choice': {'type': 'function', 'function': {'name': 'weather'}}},
    {'tool_choice': 'required'},
    {'tools': FUNCTIONS, 'text': {'format': {'type': 'json_object'}}},
    {'input': [{'type': 'function_call_output', 'call_id': 'unknown', 'output': 'result'}]},
    {'input': [{'type': 'function_call', 'call_id': 'a', 'name': 'f', 'arguments': '{}'}]},
    {'input': [{'type': 'function_call', 'call_id': 'a', 'name': 'f', 'arguments': 'bad'},
               {'type': 'function_call_output', 'call_id': 'a', 'output': 'result'}]},
    {'input': [{'type': 'reasoning', 'encrypted_content': 'opaque'}]},
    {'unrecognized_field': True},
])
def test_invalid_protocol_rejected_before_generation(extra):
    engine = ToolEngine()
    with TestClient(create_app(engine=engine)) as client:
        assert client.post('/v1/responses', json=payload(**extra)).status_code == 400
        assert not engine.prepared
        assert 'responses_inflight 0.0' in client.get('/metrics').text


@pytest.mark.parametrize('stream', [False, True])
def test_invalid_tool_is_request_local(stream):
    engine = ToolEngine(['<tool_call><function=weather>'])
    with TestClient(create_app(engine=engine)) as client:
        result = client.post('/v1/responses', json=payload(tools=FUNCTIONS, stream=stream))
        response = parse_sse(result.text)[-1]['response'] if stream else result.json()
        assert result.status_code == (200 if stream else 502)
        assert response['status'] == 'failed' and response['error']['code'] == 'invalid_tool_output'
        assert response['output'] == []
        assert client.get('/health/ready').status_code == 200
        engine.chunks = ['正常']
        assert client.post('/v1/responses', json=payload()).status_code == 200


def test_truncated_function_never_becomes_executable():
    class Truncated(ToolEngine):
        async def generate(self, *args, **kwargs):
            yield Generation('<tool_call><function=weather>', (1, 2), 'length')

    with TestClient(create_app(engine=Truncated())) as client:
        reply = client.post('/v1/responses', json=payload(tools=FUNCTIONS, stream=True, tool_choice='required'))
        result = parse_sse(reply.text)[-1]['response']
        assert result['status'] == 'incomplete'
        assert result['output'] == [] and '<tool_call' not in reply.text


def test_reasoning_history_and_plain_tools_response():
    engine = ToolEngine(['<think>检查天气</think>' + CALL])
    with TestClient(create_app(engine=engine)) as client:
        result = client.post('/v1/responses', json=payload(tools=FUNCTIONS, reasoning={'effort': 'low'})).json()
        assert [item['type'] for item in result['output']] == ['reasoning', 'function_call']
        engine.chunks = ['北京晴']
        history = [{'role': 'user', 'content': '天气'}] + result['output'] + [
            {'type': 'function_call_output', 'call_id': result['output'][-1]['call_id'], 'output': '晴'}]
        reply = client.post('/v1/responses', json=payload(tools=FUNCTIONS, input=history, stream=True))
        assert reply.status_code == 200, reply.text
        assert engine.prepared[-1][0][-2]['reasoning_content'] == '检查天气'
        assert ''.join(e['delta'] for e in parse_sse(reply.text) if e['type'] == 'response.output_text.delta') == '北京晴'


def test_legacy_limit_and_empty_tools():
    with TestClient(create_app(ServingConfig(), engine=FakeEngine())) as client:
        assert client.post('/v1/responses', json=payload(tools=[], prompt_cache_key='same')).status_code == 200
        assert client.post('/v1/responses', json=payload(max_output_tokens=32768)).status_code == 400
        assert client.post('/v1/responses', json=payload(tools=FUNCTIONS)).status_code == 400


def test_official_sdk_accumulates_function_stream_and_replays_history():
    engine = ToolEngine([CALL[:12], CALL])
    with TestClient(create_app(engine=engine)) as service:
        def handle(request):
            response = service.request(request.method, request.url.path, content=request.content,
                                       headers={'content-type': 'application/json'})
            return httpx.Response(response.status_code, headers=dict(response.headers), content=response.content)
        transport = httpx.Client(transport=httpx.MockTransport(handle))
        with OpenAI(api_key='local-test', base_url='http://testserver/v1', http_client=transport,
                    _strict_response_validation=True) as client:
            args = dict(model='qwen3-14b', input='天气', tools=FUNCTIONS, prompt_cache_key='session-test')
            response = client.responses.create(**args)
            assert response.output[0].type == 'function_call'
            with client.responses.stream(**args) as stream:
                events = list(stream)
                final = stream.get_final_response()
            assert events[-1].type == 'response.completed'
            call = final.output[0]
            assert json.loads(call.arguments)['city'] == '北京'
            engine.chunks = ['北京晴天']
            response = client.responses.create(model='qwen3-14b', tools=FUNCTIONS, input=[
                {'role': 'user', 'content': '天气'}, *[item.model_dump(exclude_none=True) for item in final.output],
                {'type': 'function_call_output', 'call_id': call.call_id, 'output': '晴天'}])
            assert response.output_text == '北京晴天'
