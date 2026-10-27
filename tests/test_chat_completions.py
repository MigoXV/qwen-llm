import json

from fastapi.testclient import TestClient
from openai import OpenAI
import pytest

from qwen_llm.config import ServingConfig
from qwen_llm.engine.base import Generation
from qwen_llm.servicer.app import create_app
from tests.test_network import live_service
from tests.test_responses import FakeEngine


TOOLS = [{'type': 'function', 'function': {'name': 'weather', 'description': '查询天气', 'parameters': {
    'type': 'object', 'properties': {'city': {'type': 'string'}, 'days': {'type': 'integer'}},
    'required': ['city'], 'additionalProperties': False}}}]
CALL = '<tool_call>\n<function=weather>\n<parameter=city>\n北京\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>'


class ToolEngine(FakeEngine):
    supports_tools = True

    async def prepare(self, messages, effort, *, tools=None):
        self.prepared.append((messages, effort, tools))
        return [1] * self.input_length


def payload(**kwargs):
    return {'model': 'qwen3-14b', 'messages': [{'role': 'system', 'content': '指令'},
                                                 {'role': 'user', 'content': '你好'}], **kwargs}


def data(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']


def test_compat_defaults_and_history():
    engine = ToolEngine()
    original = engine.generate
    sampling = {}
    async def generate(prompt, request_id, **kwargs):
        sampling.update(kwargs)
        async for item in original(prompt, request_id, **kwargs):
            yield item
    engine.generate = generate
    with TestClient(create_app(engine=engine)) as client:
        reply = client.post('/v1/chat/completions', json=payload()).json()
        assert reply['choices'][0]['message']['content'] == '你好'
        assert reply['choices'][0]['finish_reason'] == 'stop'
        assert reply['usage'] == {'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6}
        assert sampling['temperature'] == 0 and sampling['max_tokens'] == 2048
        assert engine.prepared[0][0] == payload()['messages']
        assert client.get('/v1/models').json()['data'][0]['id'] == 'qwen3-14b'
        client.post('/v1/chat/completions', json=payload(max_tokens=5, max_completion_tokens=7))
        assert sampling['max_tokens'] == 7


def test_length_reason():
    class LengthEngine(ToolEngine):
        async def generate(self, prompt, request_id, **kwargs):
            yield Generation('截断', (1, 2), 'length')
    with TestClient(create_app(engine=LengthEngine())) as client:
        r = client.post('/v1/chat/completions', json=payload())
        assert r.status_code == 200 and r.json()['choices'][0]['finish_reason'] == 'length'
        r = client.post('/v1/chat/completions', json=payload(stream=True))
        assert data(r)[-1]['choices'][0]['finish_reason'] == 'length'


def test_tools_multi_turn_and_stream():
    engine = ToolEngine([CALL[:15], CALL])
    with TestClient(create_app(engine=engine)) as client:
        body = payload(tools=TOOLS, tool_choice='required')
        response = client.post('/v1/chat/completions', json=body)
        assert response.status_code == 200, response.text
        choice = response.json()['choices'][0]
        call = choice['message']['tool_calls'][0]
        assert choice['finish_reason'] == 'tool_calls'
        assert json.loads(call['function']['arguments']) == {'city': '北京', 'days': 2}
        assert engine.prepared[0][2] == TOOLS
        body['messages'] += [choice['message'], {'role': 'tool', 'tool_call_id': call['id'], 'content': '晴'}]
        response = client.post('/v1/chat/completions', json=body)
        assert response.status_code == 200, response.text
        assert engine.prepared[-1][0][-2]['tool_calls'][0]['function']['arguments']['city'] == '北京'
        streamed = client.post('/v1/chat/completions', json={**body, 'stream': True, 'stream_options': {'include_usage': True}})
        events = data(streamed)
        assert '[DONE]' in streamed.text
        assert not any('<tool_call>' in str(event) for event in events)
        tool = next(e['choices'][0]['delta']['tool_calls'][0] for e in events if e['choices'] and 'tool_calls' in e['choices'][0]['delta'])
        assert json.loads(tool['function']['arguments']) == {'city': '北京', 'days': 2}
        assert events[-1]['choices'] == [] and events[-1]['usage']['prompt_tokens'] == 4


@pytest.mark.parametrize('extra', [
    {'messages': []}, {'tools': TOOLS, 'tool_choice': {'type': 'function', 'function': {'name': 'missing'}}},
    {'tool_choice': 'required'}, {'unknown': 1}, {'stream_options': {'include_usage': True}},
    {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': 'x'}]}]},
    {'messages': [{'role': 'tool', 'tool_call_id': 'missing', 'content': 'result'}]},
])
def test_reject_invalid(extra):
    with TestClient(create_app(engine=ToolEngine())) as client:
        assert client.post('/v1/chat/completions', json=payload(**extra)).status_code == 400


def test_tool_errors_are_request_local():
    engine = ToolEngine(['文本'])
    with TestClient(create_app(engine=engine)) as client:
        assert client.post('/v1/chat/completions', json=payload(tools=TOOLS, tool_choice='required')).status_code == 502
        assert client.post('/v1/chat/completions', json=payload()).status_code == 200
        assert client.get('/health/ready').status_code == 200
    with TestClient(create_app(engine=FakeEngine())) as client:
        assert client.post('/v1/chat/completions', json=payload(tools=TOOLS)).status_code == 400


def test_sdk_chat_and_stream(live_service):
    url, _, _ = live_service
    with OpenAI(api_key='local-test', base_url=url+'/v1', _strict_response_validation=True) as client:
        reply = client.chat.completions.create(model='qwen3-14b', messages=[{'role': 'user', 'content': 'hi'}])
        assert reply.choices[0].message.content == '你好'
        chunks = list(client.chat.completions.create(model='qwen3-14b', messages=[{'role': 'user', 'content': 'hi'}], stream=True, stream_options={'include_usage': True}))
        assert ''.join(chunk.choices[0].delta.content or '' for chunk in chunks if chunk.choices) == '你好'
        assert chunks[-1].usage.total_tokens == 6


def test_sdk_tools_strict_response_validation():
    import httpx

    with TestClient(create_app(engine=ToolEngine([CALL]))) as service:
        def handle(request):
            response = service.request(request.method, request.url.path, content=request.content,
                                       headers={'content-type': request.headers.get('content-type', 'application/json')})
            return httpx.Response(response.status_code, headers=dict(response.headers), content=response.content)
        transport = httpx.Client(transport=httpx.MockTransport(handle))
        with OpenAI(api_key='local-test', base_url='http://testserver/v1', http_client=transport,
                    _strict_response_validation=True) as client:
            args = {'model': 'qwen3-14b', 'messages': [{'role': 'user', 'content': '天气'}], 'tools': TOOLS}
            reply = client.chat.completions.create(**args)
            assert json.loads(reply.choices[0].message.tool_calls[0].function.arguments)['city'] == '北京'
            chunks = list(client.chat.completions.create(**args, stream=True))
            tool = next(c.choices[0].delta.tool_calls[0] for c in chunks if c.choices and c.choices[0].delta.tool_calls)
            assert json.loads(tool.function.arguments)['city'] == '北京'


def test_shared_capacity():
    import threading

    engine = ToolEngine(delay=0.5)
    with TestClient(create_app(ServingConfig(max_concurrency=1), engine=engine)) as client:
        thread = threading.Thread(target=lambda: client.post('/v1/responses', json={'model': 'qwen3-14b', 'input': 'hi'}))
        thread.start()
        assert engine.entered.wait(2)
        assert client.post('/v1/chat/completions', json=payload()).status_code == 429
        thread.join()
        assert client.post('/v1/chat/completions', json=payload()).status_code == 200


def test_chat_network_disconnect(live_service):
    import httpx
    import time

    url, _, _ = live_service
    with httpx.Client(base_url=url, timeout=3) as client:
        with client.stream('POST', '/v1/chat/completions', json=payload(stream=True)) as response:
            for line in response.iter_lines():
                if line.startswith('data: '):
                    break
        for _ in range(100):
            metrics = client.get('/metrics').text
            if 'responses_requests_total{outcome="cancelled"} 1.0' in metrics and 'responses_inflight 0.0' in metrics:
                break
            time.sleep(.01)
        else:
            pytest.fail('chat disconnect did not abort and release the request')


def test_multiple_tools_named_choice_and_none():
    second = {'type': 'function', 'function': {'name': 'time', 'parameters': {'type': 'object', 'properties': {}}}}
    call2 = '<tool_call><function=time></function></tool_call>'
    engine = ToolEngine([CALL + call2])
    with TestClient(create_app(engine=engine)) as client:
        result = client.post('/v1/chat/completions', json=payload(tools=TOOLS + [second]))
        assert len(result.json()['choices'][0]['message']['tool_calls']) == 2
        result = client.post('/v1/chat/completions', json=payload(tools=TOOLS + [second], parallel_tool_calls=False))
        assert result.status_code == 502
        result = client.post('/v1/chat/completions', json=payload(tools=TOOLS + [second], tool_choice={'type': 'function', 'function': {'name': 'weather'}}))
        assert result.status_code == 502
        result = client.post('/v1/chat/completions', json=payload(tools=TOOLS, tool_choice='none'))
        assert result.status_code == 200 and engine.prepared[-1][2] is None


def test_partial_invalid_tool_stream_and_length():
    partial = '<tool_call>\n<function=weather>\n<parameter=city>\n北京'
    with TestClient(create_app(engine=ToolEngine([partial]))) as client:
        r = client.post('/v1/chat/completions', json=payload(tools=TOOLS, stream=True))
        assert data(r)[-1]['error']['code'] == 'invalid_tool_output'
        assert '[DONE]' not in r.text
        assert partial not in r.text
    class Truncated(ToolEngine):
        async def generate(self, prompt, request_id, **kwargs):
            yield Generation(partial, (1, 2), 'length')
    with TestClient(create_app(engine=Truncated())) as client:
        r = client.post('/v1/chat/completions', json=payload(tools=TOOLS, tool_choice='required'))
        assert r.json()['choices'][0]['finish_reason'] == 'length'
        assert '<tool_call' not in r.text


def test_reasoning_and_structured_output():
    engine = ToolEngine(['<think>思考</think>答案'])
    with TestClient(create_app(engine=engine)) as client:
        result = client.post('/v1/chat/completions', json=payload(reasoning_effort='low'))
        assert result.json()['choices'][0]['message']['reasoning_content'] == '思考'
        stream = data(client.post('/v1/chat/completions', json=payload(reasoning_effort='low', stream=True)))
        assert ''.join(x['choices'][0]['delta'].get('reasoning_content', '') for x in stream) == '思考'
    with TestClient(create_app(engine=ToolEngine(['{"ok":true}']))) as client:
        result = client.post('/v1/chat/completions', json=payload(response_format={'type': 'json_schema', 'json_schema': {
            'name': 'result', 'schema': {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}}}))
        assert result.status_code == 200, result.text
        assert json.loads(result.json()['choices'][0]['message']['content']) == {'ok': True}


def test_chat_timeout_and_unknown_model():
    engine = ToolEngine(delay=0.1)
    with TestClient(create_app(ServingConfig(timeout=.02), engine=engine)) as client:
        assert client.post('/v1/chat/completions', json=payload(model='missing')).status_code == 404
        assert client.post('/v1/chat/completions', json=payload()).status_code == 504
        assert engine.aborted
        assert 'responses_inflight 0.0' in client.get('/metrics').text


def test_tool_schema_union_and_untyped_json_arguments():
    tools = [{'type': 'function', 'function': {'name': 'save', 'parameters': {
        'type': 'object', 'properties': {
            'label': {'type': 'string'},
            'optional': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
            'data': {'type': 'object'},
        }, 'required': ['label', 'optional', 'data']}}}]
    call = ('<tool_call><function=save><parameter=label>123</parameter>'
            '<parameter=optional>hello</parameter><parameter=data>{"count":2}</parameter>'
            '</function></tool_call>')
    with TestClient(create_app(engine=ToolEngine([call]))) as client:
        response = client.post('/v1/chat/completions', json=payload(tools=tools))
        assert response.status_code == 200, response.text
        arguments = response.json()['choices'][0]['message']['tool_calls'][0]['function']['arguments']
        assert json.loads(arguments) == {'label': '123', 'optional': 'hello', 'data': {'count': 2}}
