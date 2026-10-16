from dataclasses import dataclass
from pathlib import Path
from enum import Enum


class Runner(str, Enum):
    EAGER = "eager"
    ACLGRAPH = "aclgraph"


class ModelDType(str, Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class GraphMode(str, Enum):
    PIECEWISE = "PIECEWISE"
    FULL_DECODE_ONLY = "FULL_DECODE_ONLY"
    FULL = "FULL"


class W4A8LinearBackend(str, Enum):
    UPSTREAM = "upstream"
    GMM = "gmm"


@dataclass(frozen=True)
class ServingConfig:
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
    kv_cache_bytes: int = int(10.25 * 1024**3)
    gpu_memory_utilization: float = 0.82
    graph_mode: GraphMode = GraphMode.PIECEWISE
    speculative_tokens: int = 0
    async_scheduling: bool | None = None
    npugraph_ex: bool | None = None
    overlap_shared_expert: bool = False
    w4a8_linear_backend: W4A8LinearBackend = W4A8LinearBackend.UPSTREAM

    def __post_init__(self):
        object.__setattr__(self, "runner", Runner(self.runner))
        object.__setattr__(self, "dtype", ModelDType(self.dtype))
        object.__setattr__(self, "graph_mode", GraphMode(self.graph_mode))
        object.__setattr__(self, "w4a8_linear_backend", W4A8LinearBackend(self.w4a8_linear_backend))
        if self.w4a8_linear_backend is W4A8LinearBackend.GMM and self.dtype is not ModelDType.BFLOAT16:
            raise ValueError('原生Dense W4A8后端只验证了BF16残差计算类型')
        if self.speculative_tokens not in (0, 1, 2, 3):
            raise ValueError("MTP token 数须为 0 到 3")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("设备内存预算比例须大于零且不超过一")
        if min(self.max_input_tokens, self.max_output_tokens, self.max_concurrency,
               self.batch_tokens, self.timeout, self.shutdown_grace, self.kv_cache_bytes) <= 0:
            raise ValueError("容量、时间与 token 配置必须大于零")
