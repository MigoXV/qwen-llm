from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from qwen_llm.config import ServingConfig


class W4A8LinearBackend(str, Enum):
    UPSTREAM = "upstream"
    GMM = "gmm"


@dataclass(frozen=True)
class ProfileConfig:
    """仅供独立诊断实例使用，不改变部署配置。"""

    output_dir: Path
    record_shapes: bool = False
    aic_metrics: str = "PipeUtilization"
    module_ranges: bool = False

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        if self.aic_metrics not in {"PipeUtilization", "Memory", "MemoryAccess"}:
            raise ValueError("不支持的 NPU profiling 指标")


@dataclass(frozen=True)
class ExperimentConfig(ServingConfig):
    w4a8_linear_backend: W4A8LinearBackend = W4A8LinearBackend.UPSTREAM
    optimizations: tuple[str, ...] = ()
    mixed_prefill_tokens: int | None = None

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "w4a8_linear_backend", W4A8LinearBackend(self.w4a8_linear_backend))
        object.__setattr__(self, "optimizations", tuple(self.optimizations))
        if not set(self.optimizations) <= {'scale', 'head', 'cache', 'checkpoint', 'head-quant'}:
            raise ValueError('未知专项优化开关')
        if 'cache' in self.optimizations and 'checkpoint' in self.optimizations:
            raise ValueError('缓存策略只可选择一种')
        if 'head' in self.optimizations and not self.speculative_tokens:
            raise ValueError('输出头共享候选要求MTP')
        if set(self.optimizations) & {'cache', 'checkpoint'} and self.async_scheduling is False:
            raise ValueError('缓存候选要求异步调度')
        if self.w4a8_linear_backend is W4A8LinearBackend.GMM and self.dtype.value != "bfloat16":
            raise ValueError("原生Dense W4A8后端只验证了BF16残差计算类型")
        if self.mixed_prefill_tokens is not None and (
                type(self.mixed_prefill_tokens) is not int
                or self.mixed_prefill_tokens not in {1548,3084,4620,6156,8192}
                or self.mixed_prefill_tokens > self.batch_tokens):
            raise ValueError('混合分块预算必须为1536倍数加12，并在全局预算内')
