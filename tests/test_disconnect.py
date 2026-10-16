"""直接驱动 ASGI disconnect，验证 SSE 客户端断开后取消设备请求。"""
import asyncio
import json

from qwen_llm.servicer.app import create_app
from tests.test_responses import FakeEngine, payload


def test_sse_disconnect_releases_slot():
    async def run():
        engine = FakeEngine(delay=10)
        app = create_app(engine=engine)
        disconnected = asyncio.Event()
        first = True
        async def receive():
            nonlocal first
            if first:
                first = False
                return {'type':'http.request', 'body': json.dumps(payload(stream=True)).encode(), 'more_body':False}
            await disconnected.wait()
            return {'type':'http.disconnect'}
        messages = []
        async def send(message):
            messages.append(message)
            if message['type']=='http.response.body' and b'response.in_progress' in message.get('body',b''):
                disconnected.set()
        scope = {'type':'http','asgi':{'version':'3.0','spec_version':'2.3'},'http_version':'1.1',
                 'method':'POST','scheme':'http','path':'/responses','raw_path':b'/responses',
                 'query_string':b'','root_path':'','headers':[(b'content-type',b'application/json')],
                 'client':('127.0.0.1',12345),'server':('test',80)}
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(app(scope,receive,send),timeout=2)
            assert len(engine.aborted)==1
        assert engine.closed
    asyncio.run(run())
