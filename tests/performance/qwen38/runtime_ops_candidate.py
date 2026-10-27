"""消融入口复用正式执行方法；不保留另一份生产实现。"""
from qwen_llm.engine.qwen38_worker import (
    CachedScaleLinearMethod, Qwen38NPUWorker, install_head_sharing,
    install_quantized_head, install_scale_cache,
)
