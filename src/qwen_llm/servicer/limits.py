"""在 JSON 解析前限制请求体；覆盖缺失或伪造 Content-Length 的分块请求。"""
import json


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            body.extend(message.get('body', b''))
            if len(body) > self.max_bytes:
                data = json.dumps({'error': {'type': 'invalid_request_error',
                    'code': 'request_too_large', 'message': '请求体超过 1MiB', 'param': None}}).encode()
                await send({'type': 'http.response.start', 'status': 413,
                            'headers': [(b'content-type', b'application/json')]})
                return await send({'type': 'http.response.body', 'body': data})
            if not message.get('more_body', False):
                break
        sent = False
        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()
        await self.app(scope, replay, send)
