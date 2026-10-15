import asyncio
import json
import threading

from fastapi.testclient import TestClient
import pytest

from qwen_llm.config import ServingConfig
from qwen_llm.engine.base import Generation
from qwen_llm.inferencers.text import split_thinking
from qwen_llm.servicer.app import create_app


class FakeEngine:
    def __init__(self, chunks=None, delay=0, fail=False, input_length=4):
        self.chunks = chunks or ["你好"]
        self.delay, self.fail, self.input_length = delay, fail, input_length
        self.started = False
        self.aborted = []
        self.prepared = []
        self.closed = False
        self.entered = threading.Event()

    async def start(self): self.started = True
    async def close(self): self.closed = True
    async def health(self):
        if not self.started: raise RuntimeError("not ready")
    async def prepare(self, messages, thinking):
        self.prepared.append((messages, thinking))
        return [1] * self.input_length
    def count_tokens(self, text): return len(text)
    async def abort(self, request_id): self.aborted.append(request_id)
    async def generate(self, prompt, request_id, **kwargs):
        self.entered.set()
        await asyncio.sleep(self.delay)
        if self.fail: raise RuntimeError("worker died")
        for idx, text in enumerate(self.chunks):
            yield Generation(text, tuple(range(len(text))), "stop" if idx == len(self.chunks)-1 else None,
                             reasoning_tokens=2 if kwargs.get("thinking") else 0)


def payload(**extra):
    return {"model": "qwen3-14b", "input": "你好", **extra}


def parse_sse(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_roundtrip_and_lifecycle():
    engine = FakeEngine()
    with TestClient(create_app(engine=engine)) as client:
        assert client.get('/health/ready').status_code == 200
        assert client.get('/v1/models').json()['data'][0]['id'] == 'qwen3-14b'
        response = client.post('/v1/responses', json=payload()).json()
        assert response['status'] == 'completed'
        assert response['store'] is False
        assert response['output'][0]['content'][0]['text'] == '你好'
        assert response['usage']['input_tokens'] == 4
        assert response['usage']['output_tokens'] == 2
        assert 'responses_inflight 0.0' in client.get('/metrics').text
    assert engine.closed and engine.aborted


def test_sse_reasoning_split_and_sequence():
    engine = FakeEngine(['<thi', '<think>思考</thi', '<think>思考</think>答', '<think>思考</think>答案'])
    with TestClient(create_app(engine=engine)) as client:
        result = client.post('/responses', json=payload(stream=True, reasoning={'effort':'high'}))
        events = parse_sse(result.text)
        assert [e['sequence_number'] for e in events] == list(range(len(events)))
        assert events[0]['type'] == 'response.created'
        assert events[0]['response']['output'] == []
        assert events[-1]['type'] == 'response.completed'
        assert '[DONE]' not in result.text
        assert ''.join(e['delta'] for e in events if e['type']=='response.output_text.delta') == '答案'
        assert ''.join(e['delta'] for e in events if e['type']=='response.reasoning_text.delta') == '思考'
        assert events[-1]['response']['usage']['output_tokens_details']['reasoning_tokens'] == 2


def test_stream_and_json_equal_content():
    engine = FakeEngine(['你', '你好'])
    with TestClient(create_app(engine=engine)) as client:
        plain = client.post('/responses', json=payload()).json()
        streamed = parse_sse(client.post('/responses', json=payload(stream=True)).text)[-1]['response']
        assert plain['usage'] == streamed['usage']
        assert plain['output'][0]['content'] == streamed['output'][0]['content']


def test_instruct_model_rejects_reasoning_without_consuming_a_slot():
    engine = FakeEngine()
    engine.supports_thinking = False
    with TestClient(create_app(engine=engine)) as client:
        rejected = client.post('/responses', json=payload(reasoning={'effort': 'medium'}))
        assert rejected.status_code == 400
        assert rejected.json()['error']['code'] == 'unsupported_parameter'
        assert not engine.prepared
        assert client.post('/responses', json=payload()).json()['status'] == 'completed'


@pytest.mark.parametrize('extra', [
    {'previous_response_id':'r1'}, {'store':True}, {'tools':[]},
    {'input':[{'role':'user','content':[{'type':'input_image','image_url':'x'}]}]},
    {'input':[{'type':'function_call','name':'test'}]},
    {'input':None}, {'max_output_tokens':8193},
])
def test_reject_unsupported(extra):
    with TestClient(create_app(engine=FakeEngine())) as client:
        result = client.post('/responses', json=payload(**extra))
        assert result.status_code == 400
        assert 'error' in result.json()


def test_context_and_history():
    engine = FakeEngine(input_length=8193)
    with TestClient(create_app(engine=engine)) as client:
        result = client.post('/responses', json=payload(input=[
            {'role':'developer','content':'规则'}, {'role':'user','content':[{'type':'input_text','text':'你好'}]},
            {'role':'assistant','content':[{'type':'output_text','text':'您好'}]}], instructions='系统'))
        assert result.status_code == 400
        assert result.json()['error']['code'] == 'context_length_exceeded'
        assert engine.prepared[0][0][0]['content'] == '系统'
        assert engine.prepared[0][0][1]['role'] == 'system'
        assert 'responses_inflight 0.0' in client.get('/metrics').text


def test_four_slots_and_cancel():
    engine = FakeEngine(delay=.5)
    with TestClient(create_app(engine=engine)) as client:
        results = []
        threads = [threading.Thread(target=lambda: results.append(client.post('/responses',json=payload()))) for _ in range(4)]
        for thread in threads: thread.start()
        assert engine.entered.wait(2)
        # Wait until all four requests have reached the engine.
        for _ in range(100):
            if 'responses_inflight 4.0' in client.get('/metrics').text: break
            threading.Event().wait(.002)
        assert client.post('/responses', json=payload()).status_code == 429
        for thread in threads: thread.join()
        assert all(x.status_code==200 for x in results)
        assert len(engine.aborted)==4


def test_timeout_and_engine_failure():
    engine = FakeEngine(delay=.1)
    with TestClient(create_app(ServingConfig(timeout=.02), engine)) as client:
        result = client.post('/responses', json=payload())
        assert result.status_code == 504
        assert result.json()['error']['code'] == 'request_timeout'
        assert engine.aborted
        assert 'responses_inflight 0.0' in client.get('/metrics').text
    with TestClient(create_app(engine=FakeEngine(fail=True))) as client:
        events = parse_sse(client.post('/responses',json=payload(stream=True)).text)
        assert events[-1]['type']=='response.failed'
        assert client.get('/health/ready').status_code==503
        assert client.post('/responses',json=payload()).status_code==503


def test_length_finish():
    class LengthEngine(FakeEngine):
        async def generate(self, *args, **kwargs):
            yield Generation('截断', (1, 2), 'length')
    with TestClient(create_app(engine=LengthEngine())) as client:
        result = client.post('/responses',json=payload()).json()
        assert result['status']=='incomplete'
        assert result['incomplete_details']['reason']=='max_output_tokens'


@pytest.mark.parametrize('prefix', ['<', '<t', '<thi', '<think>思考</', '<think>思考</thi'])
def test_partial_thinking_labels(prefix):
    reasoning, answer = split_thinking(prefix, True)
    assert '<' not in reasoning
    assert not answer


def test_large_body_rejected_before_generation():
    engine = FakeEngine()
    with TestClient(create_app(engine=engine)) as client:
        result = client.post('/responses',json=payload(input='a'*(1024**2)))
        assert result.status_code == 413
        assert not engine.prepared


def test_bad_content_part_is_parameter_error():
    with TestClient(create_app(engine=FakeEngine())) as client:
        assert client.post('/responses',json=payload(input=[{'role':'user','content':[1]}])).status_code==400


def test_boundary_8192_tokens():
    with TestClient(create_app(engine=FakeEngine(input_length=8192))) as client:
        assert client.post('/responses',json=payload()).status_code==200


def test_cancelled_handler_aborts_and_frees_slot():
    import httpx
    async def run():
        engine = FakeEngine(delay=10)
        application = create_app(engine=engine)
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://test') as client:
                task = asyncio.create_task(client.post('/responses',json=payload()))
                while not engine.entered.is_set(): await asyncio.sleep(0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError): await task
                assert len(engine.aborted)==1
                assert 'responses_inflight 0.0' in (await client.get('/metrics')).text
    asyncio.run(run())
