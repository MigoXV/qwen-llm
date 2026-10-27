import asyncio
import json
from .validation import check_code_isolated,code_cases


def test_code_regression_runs_in_bounded_child_and_rejects_io():
    tests=[{'input':[1,2,3],'expected':6}]
    assert asyncio.run(check_code_isolated('def solve(xs):\n    return sum(xs)',tests))
    assert not asyncio.run(check_code_isolated('import os\ndef solve(xs):\n    return 6',tests))
    assert not asyncio.run(check_code_isolated('def solve(xs):\n    return open("/etc/passwd").read()',tests))
    assert len(code_cases())==64


def test_broken_code_does_not_become_an_http_execution_failure():
    assert not asyncio.run(check_code_isolated('def solve(:\n invalid',[]))


def test_quality_http_can_arrive_before_first_measured_case(tmp_path):
    from .runner import ObservedEngine,ProfileHTTPMiddleware
    from .config import ExperimentConfig,ProfileConfig
    engine=ObservedEngine(ExperimentConfig(),profiling=ProfileConfig(tmp_path))
    sent=[]
    async def inner(scope,receive,send):
        await send({'type':'http.response.start','status':200,'headers':[]})
        await send({'type':'http.response.body','body':b'{}'})
    async def receive():return {'type':'http.request','body':b''}
    async def send(value):sent.append(value)
    asyncio.run(ProfileHTTPMiddleware(inner,engine)({'type':'http','path':'/v1/responses'},receive,send))
    assert len(engine.http_events)==1 and len(sent)==2
    engine.reset_observations();assert not engine.http_events
