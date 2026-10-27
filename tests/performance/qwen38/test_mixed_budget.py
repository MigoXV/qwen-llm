from types import SimpleNamespace
import pytest

pytest.importorskip('vllm')
from .runtime_cache_candidate import ShortDecodeBudget


def request(prompt,computed):
    return SimpleNamespace(num_prompt_tokens=prompt,num_computed_tokens=computed)


class SchedulerParent:
    def __init__(self,running,waiting,skipped=(),cap=3084):
        self.running=running;self.waiting=waiting;self.skipped_waiting=skipped
        self.max_num_scheduled_tokens=8192
        self.vllm_config=SimpleNamespace(additional_config={'qwen38_adaptation':{'mixed_prefill_tokens':cap}})
        self.raise_error=False

    def schedule(self):
        if self.raise_error:raise RuntimeError('parent scheduler failed')
        return self.max_num_scheduled_tokens


class Scheduler(ShortDecodeBudget,SchedulerParent):
    pass


@pytest.mark.parametrize('running,waiting,expected',[
    ([request(38,128)],[request(32768,0)],3084),
    ([request(38,128),request(32768,8192)],[],3084),
    ([request(38,0)],[request(32768,0)],8192),
    ([request(8192,8192),request(32768,8192)],[],8192),
    ([request(38,128)]*4,[],8192),
    ([request(38,128),request(32768,33000)],[],8192),
])
def test_limits_only_active_short_decode_with_pending_long_prefill(running,waiting,expected):
    scheduler=Scheduler(running,waiting)
    assert scheduler.schedule()==expected
    assert scheduler.max_num_scheduled_tokens==8192


def test_budget_is_restored_after_mixed_case_and_after_parent_failure():
    scheduler=Scheduler([request(38,128)],[request(32768,0)])
    assert scheduler.schedule()==3084
    scheduler.waiting=[]
    assert scheduler.schedule()==8192
    scheduler.waiting=[request(32768,0)];scheduler.raise_error=True
    with pytest.raises(RuntimeError):scheduler.schedule()
    assert scheduler.max_num_scheduled_tokens==8192


@pytest.mark.parametrize('cap',[0,1536,3072,8193,True,3084.0])
def test_budget_reserves_four_times_three_validation_tokens(cap):
    with pytest.raises(ValueError):Scheduler([],[],cap=cap)
    assert 1548-4*(2+1)>=1536
