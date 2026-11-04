"""Qwen3.8-27B的尺度、输出头加载适配及专用NPU Worker。"""
from functools import wraps
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.quantization.methods.w4a8 import AscendW4A8DynamicLinearMethod

from qwen_llm.engine.ascend_worker import BudgetedNPUWorker
from .qwen38 import ADAPTATION, MANIFEST, PACKAGES, artifact, file_sha256


class CachedScaleLinearMethod(AscendW4A8DynamicLinearMethod):
    def process_weights_after_loading(self, layer):
        if self.group_size != 128 or not self.new_quant_version or self.tp_size != 1:
            raise ValueError('尺度前移只验证版本1.0.0、group128和TP1')
        super().process_weights_after_loading(layer)
        # 保留上游FP32合并顺序，只将原apply()中的最终舍入前移到加载阶段。
        layer.weight_scale_second.data = layer.weight_scale_second.data.to(torch.bfloat16)


_fast_w4_dispatch = Counter()


def _fast_w4_impl(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # This executes with concrete tensors during eager execution / graph capture.
    # A Python shape branch in apply() gets specialized by vLLM's dynamic graph
    # compiler, which deliberately does not evaluate guards on subsequent sizes.
    mode = 1 if x.shape[0] in (6, 12) else 0
    _fast_w4_dispatch[f'{x.shape[0]}:{mode}'] += 1
    return torch_npu.npu_weight_quant_batchmatmul(
        x, weight, antiquant_scale=scale, antiquant_group_size=128, inner_precise=mode)


fast_w4_op = torch.library.custom_op(
    'qwen_llm::fast_w4', mutates_args=(), device_types='npu')(_fast_w4_impl)


@fast_w4_op.register_fake
def _fast_w4_fake(x, weight, scale):
    return x.new_empty((x.shape[0], scale.shape[1]))


class FastDecodeLinearMethod(CachedScaleLinearMethod):
    """原权重和尺度复用；仅对已复核的解码形状选择高性能分支。"""

    def apply(self, layer, x, bias=None, tp_rank=None):
        return fast_w4_op(x, layer.weight, layer.weight_scale_second)


def install_fast_decode(model):
    _fast_w4_dispatch.clear()
    target = model.get_language_model() if hasattr(model, 'get_language_model') else model
    count = 0
    for name, layer in target.named_modules():
        if not name.endswith(('.mlp.gate_up_proj', '.mlp.down_proj')):
            continue
        adapter = layer.quant_method
        original = getattr(adapter, 'quant_method', None)
        if (not isinstance(original, CachedScaleLinearMethod)
                or layer.weight_scale_second.dtype != torch.bfloat16
                or torch_npu.get_npu_format(layer.weight) != 29):
            raise ValueError('快速W4解码要求BF16尺度缓存与原生NZ权重')
        method = FastDecodeLinearMethod.__new__(FastDecodeLinearMethod)
        method.__dict__.update(original.__dict__)
        adapter.quant_method = method
        count += 1
    if count != 128:
        raise ValueError('快速W4解码必须覆盖全部128个目标FFN投影')
    return {'backend': 'native-tuned', 'layers': count, 'fast_M': [6, 12],
            'inner_precise': 1, 'fallback_inner_precise': 0,
            'weight_layout': 'original NZ', 'extra_weight_bytes': 0,
            'dispatch': 'opaque custom operator; native arithmetic kernel',
            'counter_semantics': 'eager/capture dispatches only; graph replays are not counted'}


def install_scale_cache():
    from vllm_ascend.quantization.methods.registry import _SCHEME_REGISTRY
    key = ('W4A8_DYNAMIC', 'linear')
    if _SCHEME_REGISTRY[key] not in (AscendW4A8DynamicLinearMethod, CachedScaleLinearMethod):
        raise RuntimeError('W4线性方法已经被其他扩展替换')
    _SCHEME_REGISTRY[key] = CachedScaleLinearMethod


def install_head_sharing():
    from vllm_ascend.spec_decode.eagle_proposer import SpecDecodeBaseProposer
    original = SpecDecodeBaseProposer._maybe_share_lm_head
    if getattr(original, '_qwen38_shared', False):
        return

    @wraps(original)
    def share(proposer, model):
        original(proposer, model)
        target = model.get_language_model() if hasattr(model, 'get_language_model') else model
        draft = proposer.model
        if not hasattr(target, 'lm_head') or not hasattr(draft, 'lm_head'):
            raise RuntimeError('目标或MTP模型缺少输出头')
        target_head, draft_head = target.lm_head, draft.lm_head
        target_parameters = dict(target_head.named_parameters())
        draft_parameters = dict(draft_head.named_parameters())
        # 仅在加载时进行一次内容校验；推理热路径没有比较或设备同步。
        if (target_parameters.keys() != draft_parameters.keys()
                or any(target_parameters[k].dtype != draft_parameters[k].dtype
                       or target_parameters[k].shape != draft_parameters[k].shape
                       or not torch.equal(target_parameters[k], draft_parameters[k])
                       for k in target_parameters)):
            raise RuntimeError('目标与MTP输出头不一致，拒绝共享')
        if (type(target_head.quant_method) is not type(draft_head.quant_method)
                or type(getattr(target_head.quant_method, 'quant_method', None))
                is not type(getattr(draft_head.quant_method, 'quant_method', None))):
            raise RuntimeError('目标与MTP输出头的量化方法不一致')
        draft.lm_head = target_head
        proposer._qwen38_head_shared = True

    share._qwen38_shared = True
    SpecDecodeBaseProposer._maybe_share_lm_head = share


def install_quantized_head():
    from vllm.config import get_current_vllm_config
    from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead
    from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
    from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod

    original = AscendParallelLMHead.__init__
    if getattr(original, '_qwen38_w8_head', False):
        return

    class HeadConfig:
        def __init__(self, parent):
            self.parent = parent

        def get_quant_method(self, layer, prefix):
            return AscendEmbeddingMethod(AscendW8A8DynamicLinearMethod())

        def __getattr__(self, name):
            return getattr(self.parent, name)

    @wraps(original)
    def initialize(layer, num_embeddings, embedding_dim, *args, **kwargs):
        if args or (num_embeddings, embedding_dim) != (248320, 5120):
            raise ValueError('专项输出头只支持248320×5120及关键字可选参数')
        config = get_current_vllm_config()
        description = json.loads((Path(config.model_config.model) / 'quant_model_description.json').read_text())
        if description.get('lm_head.weight') != 'W8A8_DYNAMIC':
            raise ValueError('该专项模型没有独立导出的W8输出头')
        kwargs['quant_config'] = HeadConfig(config.quant_config)
        original(layer, num_embeddings, embedding_dim, **kwargs)

    initialize._qwen38_w8_head = True
    AscendParallelLMHead.__init__ = initialize


class Qwen38NPUWorker(BudgetedNPUWorker):
    def load_model(self):
        config = self.vllm_config
        model_path = Path(config.model_config.model)
        verified = artifact(model_path, {name: version(name) for name in PACKAGES})
        identity = (config.additional_config or {}).get('qwen38_adaptation', {})
        if (identity.get('name') != ADAPTATION or identity.get('recipe') != verified['recipe']
                or identity.get('manifest_sha256') != file_sha256(model_path / MANIFEST)):
            raise ValueError('专项Worker未收到匹配的已验收模型配置')
        spec = config.speculative_config
        if (config.model_config.dtype != torch.bfloat16
                or config.parallel_config.tensor_parallel_size != 1
                or config.model_config.max_model_len != 65536
                or config.scheduler_config.max_num_seqs != 4
                or spec is None or spec.num_speculative_tokens != 2
                or '910B' not in torch.npu.get_device_name(self.device)):
            raise ValueError('专项Worker只支持Ascend910B、TP1、BF16和四路64K MTP2')
        if verified['recipe'] in {'head', 'both'}:
            install_quantized_head()
        install_scale_cache()
        install_head_sharing()
        super().load_model()
        model = self.model_runner.model
        target = model.get_language_model() if hasattr(model, 'get_language_model') else model
        draft = self.model_runner.drafter.model
        scales = [m for m in target.modules() if hasattr(m, 'weight_scale_second')]
        gdn_int8 = sum(name.endswith('linear_attn.out_proj') and module.weight.dtype == torch.int8
                       for name, module in target.named_modules() if hasattr(module, 'weight'))
        head_dtype = torch.int8 if verified['recipe'] in {'head', 'both'} else torch.bfloat16
        if (len(scales) != 128 or any(m.weight_scale_second.dtype != torch.bfloat16 for m in scales)
                or target.lm_head is not draft.lm_head or target.lm_head.weight.dtype != head_dtype
                or gdn_int8 != (48 if verified['recipe'] in {'gdn', 'both'} else 0)):
            raise RuntimeError('模型专项适配未在预期的FFN、GDN或输出头中生效')
        self._qwen38_runtime = {**identity, 'cached_scale_layers': len(scales),
                                'shared_lm_head': True, 'head_dtype': str(head_dtype),
                                'gdn_output_int8_layers': gdn_int8}
        fast_w4 = (config.additional_config or {}).get('qwen38_fast_w4', False)
        self._qwen38_runtime['w4_decode'] = (install_fast_decode(model) if fast_w4 else
            {'backend': 'upstream', 'layers': 128, 'fast_M': [], 'inner_precise': 0,
             'fallback_inner_precise': 0, 'weight_layout': 'original NZ', 'extra_weight_bytes': 0})

    def runtime_status(self):
        identity = dict(self._qwen38_runtime)
        if identity['w4_decode']['backend'] == 'native-tuned':
            identity['w4_decode'] = {**identity['w4_decode'], 'capture_dispatch': dict(_fast_w4_dispatch)}
        return {**super().runtime_status(), 'model_adaptation': identity}
