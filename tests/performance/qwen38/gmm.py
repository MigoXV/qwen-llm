"""显式可选的Dense W4A8：复用CANN单专家GMM与上游MoE的尺度/偏置约定。"""
import logging

import torch
import torch_npu
from vllm_ascend.quantization.methods.w4a8 import AscendW4A8DynamicLinearMethod
from vllm_ascend.utils import maybe_trans_nz

from tests.performance.qwen38.quantization import encode_w4a8_scales

logger = logging.getLogger(__name__)


class NativeW4A8LinearMethod(AscendW4A8DynamicLinearMethod):
    def get_pergroup_param(self, input_size, output_size, params_dtype, layer_type=None):
        if params_dtype != torch.bfloat16 or not self.new_quant_version or self.tp_size != 1:
            raise ValueError('原生Dense W4A8仅验证BF16残差、版本1.0.0打包权重和TP=1')
        return super().get_pergroup_param(input_size, output_size, torch.float32, layer_type)

    def process_weights_after_loading(self, layer):
        if torch.any(layer.weight_offset.data) or torch.any(layer.weight_offset_second.data):
            raise ValueError('原生Dense W4A8仅支持对称权重量化')
        weight = layer.weight.data.t().contiguous().unsqueeze(0)
        scale = encode_w4a8_scales(layer.weight_scale.data, layer.weight_scale_second.data)
        self.effective_group_size = weight.shape[1] // scale.shape[1]
        layer.weight.data = maybe_trans_nz(weight).view(torch.int32).contiguous()
        layer.weight_scale_second.data = scale.to(weight.device)
        layer.scale_bias.data = layer.scale_bias.data.float().sum(-1).unsqueeze(0).contiguous()
        for name in ('weight_scale', 'weight_offset', 'weight_offset_second'):
            delattr(layer, name)

    def apply(self, layer, x, bias=None, tp_rank=None):
        if x.ndim != 2 or x.dtype != torch.bfloat16:
            raise ValueError('原生Dense W4A8要求二维BF16输入')
        quantized, per_token_scale = torch_npu.npu_dynamic_quant(x)
        group_list = torch.full((1,), x.shape[0], dtype=torch.int64, device=x.device)
        output = torch_npu.npu_grouped_matmul(
            x=[quantized], weight=[layer.weight], scale=[layer.weight_scale_second],
            bias=[layer.scale_bias], per_token_scale=[per_token_scale],
            split_item=2, group_list_type=1, group_type=0, group_list=group_list,
            output_dtype=torch.bfloat16)[0]
        return output if bias is None else output + bias


def install_native_w4a8():
    """仅在明确选择此后端的独立Worker中替换线性层scheme。"""
    from vllm_ascend.quantization.methods.registry import _SCHEME_REGISTRY

    key = ('W4A8_DYNAMIC', 'linear')
    current = _SCHEME_REGISTRY.get(key)
    if current not in (AscendW4A8DynamicLinearMethod, NativeW4A8LinearMethod):
        raise RuntimeError(f'当前插件的W4A8 scheme不匹配：{current}')
    _SCHEME_REGISTRY[key] = NativeW4A8LinearMethod
    logger.info('Dense W4A8 backend: dynamic INT8 activations + packed INT4 single-expert GMM')
