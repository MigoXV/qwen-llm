import json
import pytest
from fastapi.testclient import TestClient
from qwen_llm.servicer.app import create_app
from tests.test_responses import FakeEngine, payload, parse_sse

SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}},
          "required": ["text"], "additionalProperties": False}
FORMAT = {"type": "json_schema", "name": "MeetingProduct", "schema": SCHEMA, "strict": True}


class StructuredEngine(FakeEngine):
    async def generate(self, prompt, request_id, **kwargs):
        self.sampling = kwargs
        async for item in super().generate(prompt, request_id, **kwargs):
            yield item


@pytest.mark.parametrize('stream', [False, True])
def test_meetnote_request_honors_schema_and_thinking_fields(stream):
    engine = StructuredEngine(['{"text":"纪要"}'])
    with TestClient(create_app(engine=engine)) as client:
        reply = client.post('/v1/responses', json=payload(
            stream=stream, enable_thinking=False, text={'format': FORMAT}))
        assert reply.status_code == 200
        result = parse_sse(reply.text)[-1]['response'] if stream else reply.json()
        assert result['status'] == 'completed'
        assert result['text']['format'] == FORMAT
        assert json.loads(result['output'][0]['content'][0]['text']) == {'text': '纪要'}
        assert engine.sampling['json_schema'] == SCHEMA
        assert engine.prepared[0][1] == 'none'
        assert json.dumps(SCHEMA, ensure_ascii=False, separators=(',', ':')) in engine.prepared[0][0][0]['content']
        assert engine.prepared[0][0][-1] == {'role':'user','content':'你好'}


@pytest.mark.parametrize('extra', [
    {'enable_thinking': False, 'reasoning': {'effort': 'high'}},
    {'enable_thinking': False, 'chat_template_kwargs': {'enable_thinking': True}},
    {'enable_thinking': 'false'},
    {'enable_thinking': True, 'text': {'format': FORMAT}},
    {'text': {'format': {**FORMAT, 'schema': {'type': 'invalid'}}}},
])
def test_invalid_format_or_conflicting_flags_rejected_before_engine(extra):
    engine = StructuredEngine()
    with TestClient(create_app(engine=engine)) as client:
        assert client.post('/responses', json=payload(**extra)).status_code == 400
        assert not engine.prepared


def test_invalid_structured_completion_does_not_disable_engine():
    engine = StructuredEngine(['{"text":42}'])
    with TestClient(create_app(engine=engine)) as client:
        result = client.post('/responses', json=payload(text={'format': FORMAT}))
        assert result.status_code == 500
        assert result.json()['error']['code'] == 'structured_output_invalid'
        assert client.get('/health/ready').status_code == 200
        assert 'responses_inflight 0.0' in client.get('/metrics').text
        engine.chunks = ['{"text":"恢复"}']
        assert client.post('/responses', json=payload(text={'format': FORMAT})).json()['status'] == 'completed'


def test_legacy_thinking_flag_and_json_object_remain_supported():
    engine = StructuredEngine(['{"text":"ok"}'])
    with TestClient(create_app(engine=engine)) as client:
        result = client.post('/responses', json=payload(
            chat_template_kwargs={'enable_thinking': False}, text={'format': {'type': 'json_object'}}))
        assert result.json()['status'] == 'completed'
        assert engine.sampling['json_object'] is True
