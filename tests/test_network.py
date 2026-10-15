import os
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest


@pytest.fixture
def live_service(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    closed = tmp_path/'closed'
    env = {**os.environ, 'QWEN_TEST_PORT':str(port), 'QWEN_TEST_CLOSED':str(closed)}
    with (tmp_path/'server.log').open('w') as log:
        process = subprocess.Popen([sys.executable,'-m','tests.live_server'],env=env,stdout=log,stderr=log)
        url = f'http://127.0.0.1:{port}'
        try:
            for _ in range(100):
                if process.poll() is not None:
                    pytest.fail((tmp_path/'server.log').read_text())
                try:
                    if httpx.get(url+'/health/ready',timeout=.2).status_code==200: break
                except httpx.HTTPError: pass
                time.sleep(.05)
            else: pytest.fail('server startup timeout')
            yield url, process, closed
        finally:
            if process.poll() is None: process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                pytest.fail('server failed graceful shutdown')


@pytest.mark.parametrize('sig',[signal.SIGINT,signal.SIGTERM])
def test_real_http_disconnect_and_shutdown(live_service,sig):
    url, process, closed = live_service
    with httpx.Client(base_url=url,timeout=3) as client:
        with client.stream('POST','/responses',json={'model':'qwen3-14b','input':'hi','stream':True}) as response:
            assert response.status_code==200
            for line in response.iter_lines():
                if line.startswith('event: response.in_progress'): break
        for _ in range(100):
            metrics = client.get('/metrics').text
            if ('responses_requests_total{outcome="cancelled"} 1.0' in metrics
                    and 'responses_inflight 0.0' in metrics): break
            time.sleep(.01)
        else: pytest.fail('disconnected generation was not cancelled')
        assert 'responses_inflight 0.0' in metrics
    process.send_signal(sig)
    process.wait(timeout=5)
    assert closed.read_text()=='closed'
    with pytest.raises(OSError):
        socket.create_connection(('127.0.0.1',int(url.rsplit(':',1)[1])),timeout=.2)
