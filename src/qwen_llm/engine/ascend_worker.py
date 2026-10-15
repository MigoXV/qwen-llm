"""保留上游 NPU worker，只补齐 0.11.0rc1 遗漏的 KV 字节上限。"""
import logging

from vllm_ascend.worker.worker_v1 import NPUWorker

from qwen_llm.engine.memory import bounded_kv_memory

logger = logging.getLogger(__name__)


class BudgetedNPUWorker(NPUWorker):
    def determine_available_memory(self) -> int:
        available = super().determine_available_memory()
        requested = self.cache_config.kv_cache_memory_bytes
        budget = bounded_kv_memory(available, requested)
        logger.info('KV cache budget: %.3f GiB; profiled available %.3f GiB',
                    budget / 1024**3, available / 1024**3)
        return budget
