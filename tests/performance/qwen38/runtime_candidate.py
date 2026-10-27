"""测试入口复用正式运行时的模型产物约束。"""
from qwen_llm.engine.qwen38 import (
    ADAPTATION, MANIFEST, MODEL_FILES, PACKAGES, TEXT_SHAPE,
    artifact, file_sha256, model_shape, serving_options,
)
