"""测试开关与必要执行适配分开；后者完整验收后才能移入生产。"""
from importlib.metadata import version
import torch

from .runtime_candidate import model_shape
from .runtime_ops_candidate import (CachedScaleLinearMethod, install_head_sharing,
    install_quantized_head, install_scale_cache)


FEATURES = {'scale', 'head', 'cache', 'checkpoint', 'head-quant'}


def check_model(vllm_config):
    model_shape(vllm_config.model_config.model)
    if (vllm_config.parallel_config.tensor_parallel_size != 1
            or vllm_config.model_config.dtype != torch.bfloat16):
        raise ValueError('专项适配要求TP1、BF16残差计算')
    if version('vllm').split('+')[0] != '0.18.0' or not version('vllm-ascend').startswith('0.18.0'):
        raise ValueError('专项适配要求本项目vLLM/Ascend0.18.0环境')


def install(vllm_config):
    features = set((vllm_config.additional_config or {}).get('qwen_experiment', []))
    if not features <= FEATURES:
        raise ValueError(f'未知专项优化：{features - FEATURES}')
    if features:
        check_model(vllm_config)
    if 'head-quant' in features:
        install_quantized_head()
    if 'scale' in features:
        install_scale_cache()
    if 'head' in features:
        install_head_sharing()
