"""复用正式缓存核心；条件缩块仅保留为未采用的测试消融。"""
from qwen_llm.engine.qwen38_cache import Qwen38BlockPool
from qwen_llm.engine.qwen38_cache import Qwen38CheckpointScheduler as ValidatedCheckpointScheduler


def has_short_decode_and_long_prefill(running, waiting, skipped=()):
    """1536是本模型的状态对齐块；保护已在生成的短提示请求。"""
    short_decode = any(r.num_prompt_tokens <= 1536 and r.num_computed_tokens >= r.num_prompt_tokens
                       for r in running)
    if not short_decode:
        return False
    return any(r.num_prompt_tokens > 1536 and r.num_computed_tokens < r.num_prompt_tokens
               for queue in (running, waiting, skipped) for r in queue)


class ShortDecodeBudget:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        policy = (self.vllm_config.additional_config or {}).get('qwen38_adaptation', {})
        self.mixed_prefill_tokens = policy.get('mixed_prefill_tokens')
        if self.mixed_prefill_tokens is not None and (
                type(self.mixed_prefill_tokens) is not int
                or self.mixed_prefill_tokens not in {1548, 3084, 4620, 6156, 8192}
                or self.mixed_prefill_tokens > self.max_num_scheduled_tokens):
            raise ValueError('混合分块预算须为1536倍数加12，并在全局预算内')

    def schedule(self):
        original = self.max_num_scheduled_tokens
        if self.mixed_prefill_tokens is None or not has_short_decode_and_long_prefill(
                self.running, self.waiting, self.skipped_waiting):
            return super().schedule()
        self.max_num_scheduled_tokens = min(original, self.mixed_prefill_tokens)
        try:
            return super().schedule()
        finally:
            self.max_num_scheduled_tokens = original


class Qwen38CheckpointScheduler(ShortDecodeBudget, ValidatedCheckpointScheduler):
    pass
