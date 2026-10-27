"""只改变混合分块策略，不改变BlockPool，用于独立单变量对照。"""
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from .runtime_cache_candidate import ShortDecodeBudget


class Qwen38MixedScheduler(ShortDecodeBudget, AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .adaptations import check_model
        check_model(self.vllm_config)
        if self.num_spec_tokens != 2 or not self.use_eagle:
            raise ValueError('混合分块候选只验证MTP2')
