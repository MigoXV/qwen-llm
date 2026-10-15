"""仅供网络与信号测试的假引擎服务，不能作为生产入口。"""
import os
from pathlib import Path

import uvicorn

from qwen_llm.servicer.app import create_app
from tests.test_responses import FakeEngine


class ClosingEngine(FakeEngine):
    async def close(self):
        await super().close()
        Path(os.environ['QWEN_TEST_CLOSED']).write_text('closed')


if __name__ == '__main__':
    uvicorn.run(create_app(engine=ClosingEngine(delay=.2)), host='127.0.0.1',
                port=int(os.environ['QWEN_TEST_PORT']), timeout_graceful_shutdown=2)
