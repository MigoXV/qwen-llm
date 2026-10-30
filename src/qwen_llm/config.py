from dataclasses import dataclass
from pathlib import Path
from enum import Enum


class ReasoningEffort(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    XHIGH = "xhigh"


class Runner(str, Enum):
    EAGER = "eager"
    ACLGRAPH = "aclgraph"
    CUDAGRAPH = "cudagraph"


class ModelDType(str, Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class GraphMode(str, Enum):
    PIECEWISE = "PIECEWISE"
    FULL_DECODE_ONLY = "FULL_DECODE_ONLY"
    FULL = "FULL"


class ModelAdaptation(str, Enum):
    NONE = "none"
    QWEN38_910B_V1 = "qwen38-910b-v1"


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path = Path("model-bin/Qwen3-14B-W8A8-Ascend")
    model_name: str = "qwen3-14b"
    dtype: ModelDType = ModelDType.BFLOAT16
    max_input_tokens: int = 8192
    max_output_tokens: int = 8192
    max_concurrency: int = 4
    batch_tokens: int = 2048
    runner: Runner = Runner.ACLGRAPH
    prefix_caching: bool = True
    timeout: float = 600
    shutdown_grace: float = 30
    kv_cache_bytes: int | None = int(10.25 * 1024**3)
    gpu_memory_utilization: float = 0.82
    graph_mode: GraphMode = GraphMode.PIECEWISE
    speculative_tokens: int = 0
    async_scheduling: bool | None = None

    def __post_init__(self):
        object.__setattr__(self, "runner", Runner(self.runner))
        object.__setattr__(self, "dtype", ModelDType(self.dtype))
        object.__setattr__(self, "graph_mode", GraphMode(self.graph_mode))
        if self.speculative_tokens not in (0, 1, 2, 3):
            raise ValueError("MTP token 数须为 0 到 3")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("设备内存预算比例须大于零且不超过一")
        if min(self.max_input_tokens, self.max_output_tokens, self.max_concurrency,
               self.batch_tokens, self.timeout, self.shutdown_grace) <= 0:
            raise ValueError("容量、时间与 token 配置必须大于零")
        if self.kv_cache_bytes is not None and self.kv_cache_bytes <= 0:
            raise ValueError("KV 预算必须大于零，None 表示使用实际 profile 预算")


@dataclass(frozen=True)
class ServingConfig(InferenceConfig):
    """兼容现有 Ascend CLI 和实验的配置；不向 CUDA 传播专属字段。"""
    npugraph_ex: bool | None = None
    overlap_shared_expert: bool = False
    model_adaptation: ModelAdaptation = ModelAdaptation.NONE

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "model_adaptation", ModelAdaptation(self.model_adaptation))
        if self.kv_cache_bytes is None or self.runner is Runner.CUDAGRAPH:
            raise ValueError("Ascend 配置必须显式指定 KV 预算且不能使用 CUDA Graph")
