"""真实 tokenizer + 原生 parser；ASGI 在生成完成前必须送达增量。"""
import asyncio
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from qwen_llm.engine.base import Generation
from qwen_llm.servicer.app import create_app
from tests.test_chat_completions import ToolEngine, TOOLS, CALL
from tests.test_responses import payload, parse_sse
from tests.test_responses_tools import FUNCTIONS


@pytest.fixture(scope='module')
def tokenizer():
    pytest.importorskip('vllm.parser.qwen3')
    from transformers import AutoTokenizer
    path = Path(os.environ.get('QWEN_NVIDIA_TEST_TOKENIZER',
        '/workspace/model-bin/windowsxp811203/Qwen3.8-27B-Abliterated-W4A16-AWQ-GPTQ-VOCAB4-NVIDIA'))
    if not path.exists():
        pytest.skip('需要本地 Qwen tokenizer')
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


class NativeEngine(ToolEngine):
    runtime_identity = {'platform': 'cuda'}

    def __init__(self, tokenizer, text=CALL, *, finish='stop', gate=None):
        super().__init__()
        self.tokenizer, self.text, self.finish, self.gate = tokenizer, text, finish, gate
        self.finished = False

    async def generate(self, *args, **kwargs):
        ids = self.tokenizer.encode(self.text, add_special_tokens=False)
        for i in range(1, len(ids) + 1):
            text = self.tokenizer.decode(ids[:i], skip_special_tokens=False)
            if text.endswith('\ufffd'):
                continue
            yield Generation(text, tuple(ids[:i]), None, cached_tokens=3)
            if self.gate and i >= len(ids)//2:
                await self.gate.wait()
        self.finished = True
        yield Generation(self.text, tuple(ids), self.finish, cached_tokens=3)


@pytest.mark.parametrize('route', ['responses', 'chat/completions'])
@pytest.mark.parametrize('text', ['这是不需要使用工具的普通回答。' * 8,
    '<tool_call><function=weather><parameter=city>' + '北京' * 40 +
    '</parameter><parameter=days>2</parameter></function></tool_call>'])
def test_delta_arrives_before_generation_finishes(tokenizer, route, text):
    async def run():
        gate = asyncio.Event()
        engine = NativeEngine(tokenizer, text, gate=gate)
        app = create_app(engine=engine)
        body = payload(stream=True, tools=FUNCTIONS) if route == 'responses' else {
            'model': 'qwen3-14b', 'messages': [{'role': 'user', 'content': '天气'}],
            'tools': TOOLS, 'stream': True}
        incoming = True
        async def receive():
            nonlocal incoming
            if incoming:
                incoming = False
                return {'type': 'http.request', 'body': json.dumps(body).encode(), 'more_body': False}
            await asyncio.Event().wait()
        messages = []
        async def send(message):
            messages.append(message)
            if message['type'] != 'http.response.body':
                return
            for line in message.get('body', b'').decode().splitlines():
                if not line.startswith('data: ') or line.endswith('[DONE]'):
                    continue
                event = json.loads(line[6:])
                if route == 'responses':
                    visible = event.get('type', '').endswith('.delta') and event.get('delta')
                else:
                    delta = event['choices'][0]['delta']
                    visible = delta.get('content') or any(
                        c.get('function', {}).get('arguments') for c in delta.get('tool_calls', []))
                if visible and not gate.is_set():
                    assert not engine.finished
                    gate.set()
        path = '/v1/' + route
        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'},
            'http_version': '1.1', 'method': 'POST', 'scheme': 'http', 'path': path,
            'raw_path': path.encode(), 'query_string': b'', 'root_path': '',
            'headers': [(b'content-type', b'application/json')],
            'client': ('127.0.0.1', 12345), 'server': ('test', 80)}
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(app(scope, receive, send), timeout=5)
        assert gate.is_set() and engine.finished and len(engine.aborted) == 1
        wire = b''.join(m.get('body', b'') for m in messages).decode()
        assert '<tool_call>' not in wire
        if route == 'responses':
            result = parse_sse(wire)[-1]['response']
            assert result['usage']['output_tokens'] == len(tokenizer.encode(text, add_special_tokens=False))
            assert result['usage']['input_tokens_details']['cached_tokens'] == 3
    asyncio.run(run())


def test_nested_arguments_reasoning_and_no_repeated_final_delta(tokenizer):
    value = [{'label': '甲', 'value': 2}, {'label': '乙', 'value': 3}]
    raw = '<think>检查一下</think><tool_call><function=ask><parameter=options>' + json.dumps(value, ensure_ascii=False) + '</parameter></function></tool_call>'
    tool = {'type': 'function', 'name': 'ask', 'parameters': {'type': 'object',
        'properties': {'options': {'type': 'array', 'items': {'type': 'object'}}}, 'required': ['options']}}
    with TestClient(create_app(engine=NativeEngine(tokenizer, raw))) as client:
        response = client.post('/v1/responses', json=payload(stream=True, tools=[tool], reasoning={'effort': 'low'}))
        events = parse_sse(response.text)
        final = events[-1]['response']
        assert final['status'] == 'completed'
        assert [e['sequence_number'] for e in events] == list(range(len(events)))
        assert ''.join(e['delta'] for e in events if e['type'] == 'response.reasoning_text.delta') == '检查一下'
        args = ''.join(e['delta'] for e in events if e['type'] == 'response.function_call_arguments.delta')
        assert args == final['output'][-1]['arguments']
        assert json.loads(args) == {'options': value}
        assert sum(e['type'] == 'response.function_call_arguments.done' for e in events) == 1


@pytest.mark.parametrize('text,finish,status', [
    ('<tool_call><function=weather><parameter=city>北京', 'length', 'incomplete'),
    ('<tool_call><function=weather><parameter=city>北京', 'stop', 'failed'),
    (CALL.replace('\n2\n', '\ninvalid\n'), 'stop', 'failed'),
    (CALL + CALL, 'stop', 'failed'),
])
def test_partial_or_invalid_calls_never_complete(tokenizer, text, finish, status):
    with TestClient(create_app(engine=NativeEngine(tokenizer, text, finish=finish))) as client:
        events = parse_sse(client.post('/v1/responses', json=payload(
            stream=True, tools=FUNCTIONS, parallel_tool_calls=False)).text)
        assert events[-1]['response']['status'] == status
        assert not any(e['type'] == 'response.function_call_arguments.done' for e in events)
        assert all(i['status'] != 'completed' for i in events[-1]['response']['output'] if i['type'] == 'function_call')
        assert client.get('/health/ready').status_code == 200


def test_disconnect_during_native_tool_stream_aborts(tokenizer):
    async def run():
        disconnected = asyncio.Event()
        engine = NativeEngine(tokenizer, CALL, gate=asyncio.Event())
        app = create_app(engine=engine)
        incoming = True
        async def receive():
            nonlocal incoming
            if incoming:
                incoming = False
                return {'type': 'http.request', 'body': json.dumps(payload(stream=True, tools=FUNCTIONS)).encode(), 'more_body': False}
            await disconnected.wait()
            return {'type': 'http.disconnect'}
        async def send(message):
            if b'response.function_call_arguments.delta' in message.get('body', b''):
                disconnected.set()
        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'},
            'http_version': '1.1', 'method': 'POST', 'scheme': 'http', 'path': '/v1/responses',
            'raw_path': b'/v1/responses', 'query_string': b'', 'root_path': '',
            'headers': [(b'content-type', b'application/json')],
            'client': ('127.0.0.1', 12345), 'server': ('test', 80)}
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(app(scope, receive, send), timeout=5)
            assert disconnected.is_set() and len(engine.aborted) == 1
            assert not engine.finished
    asyncio.run(run())
