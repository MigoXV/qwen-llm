"""保留上游 NPU worker，只补齐 0.11.0rc1 遗漏的 KV 字节上限。"""
import logging

from importlib.metadata import version

if version("vllm-ascend").startswith("0.11."):
    from vllm_ascend.worker.worker_v1 import NPUWorker
else:
    from vllm_ascend.worker.worker import NPUWorker

from qwen_llm.engine.memory import bounded_kv_memory

logger = logging.getLogger(__name__)


class BudgetedNPUWorker(NPUWorker):
    def runtime_status(self) -> dict:
        """从真实 worker 返回图捕获、缓存容量与 NPU 内存，供启动验收。"""
        import torch
        from vllm.compilation.counter import compilation_counter
        from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config

        cache = self.model_runner.kv_cache_config
        drafter = getattr(self.model_runner, "drafter", None)
        speculative = self.vllm_config.speculative_config
        return {
            "worker": f"{type(self).__module__}.{type(self).__name__}",
            "device": str(self.device),
            "graph_mode": str(self.vllm_config.compilation_config.cudagraph_mode),
            "graphs_captured": compilation_counter.num_cudagraph_captured,
            "drafter": type(drafter).__name__ if drafter is not None else None,
            "speculative_tokens": speculative.num_speculative_tokens if speculative else 0,
            "kv_cache_allocated_bytes": sum(t.size for t in cache.kv_cache_tensors),
            "cache_block_sizes": sorted({g.kv_cache_spec.block_size for g in cache.kv_cache_groups}),
            "full_context_capacity": get_max_concurrency_for_kv_cache_config(self.vllm_config, cache),
            "npu_allocated_bytes": torch.npu.memory_allocated(),
            "npu_reserved_bytes": torch.npu.memory_reserved(),
        }

    def determine_available_memory(self) -> int:
        available = super().determine_available_memory()
        requested = self.cache_config.kv_cache_memory_bytes
        budget = bounded_kv_memory(available, requested)
        logger.info('KV cache budget: %.3f GiB; profiled available %.3f GiB',
                    budget / 1024**3, available / 1024**3)
        return budget
