"""Qwen3.8-27B专项适配的已验收模型产物及启动约束。"""
import hashlib
import json
from pathlib import Path

from qwen_llm.config import GraphMode, ModelDType, Runner


ADAPTATION = 'qwen38-910b-v1'
MANIFEST = 'qwen38_optimization.json'
PACKAGES = ('vllm', 'vllm-ascend', 'torch', 'torch-npu', 'transformers')
MODEL_FILES = ('config.json', 'quant_model_description.json',
               'quant_model_weights.safetensors.index.json')
TEXT_SHAPE = {
    'num_hidden_layers': 64, 'hidden_size': 5120, 'intermediate_size': 17408,
    'vocab_size': 248320, 'num_attention_heads': 24, 'num_key_value_heads': 4,
    'head_dim': 256, 'linear_num_key_heads': 16, 'linear_num_value_heads': 48,
    'linear_key_head_dim': 128, 'linear_value_head_dim': 128,
    'mtp_num_hidden_layers': 1, 'mamba_ssm_dtype': 'float32',
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_shape(model_path):
    config = json.loads((Path(model_path) / 'config.json').read_text())
    text = config.get('text_config', {})
    layers = ['full_attention' if i % 4 == 3 else 'linear_attention' for i in range(64)]
    if (config.get('model_type') != 'qwen3_5'
            or any(text.get(k) != v for k, v in TEXT_SHAPE.items())
            or text.get('layer_types') != layers):
        raise ValueError('专项适配只支持已验证结构和FP32递归状态的Qwen3.8-27B dense模型')


def artifact(model_path, packages):
    model_path = Path(model_path)
    if (model_path / '.quantization.incomplete').exists():
        raise ValueError('专项模型仓库尚未导出完成')
    manifest_path = model_path / MANIFEST
    if not manifest_path.is_file():
        raise ValueError('专项适配要求独立模型仓库中的验收清单')
    result = json.loads(manifest_path.read_text())
    if (result.get('schema_version') != 1 or result.get('adaptation') != ADAPTATION
            or result.get('validated_for_inference') is not True
            or result.get('recipe') not in {'base', 'gdn', 'head', 'both'}):
        raise ValueError('模型专项验收清单不匹配或尚未通过验收')
    expected_packages = result.get('software', {})
    if any(not expected_packages.get(k) or packages.get(k) != expected_packages[k] for k in PACKAGES):
        raise ValueError('专项适配要求与验收清单一致的推理依赖版本')
    model_shape(model_path)
    expected_files = result.get('model_files', {})
    if any(expected_files.get(k) != file_sha256(model_path / k) for k in MODEL_FILES):
        raise ValueError('模型结构、量化描述或权重索引与已验收产物不一致')
    description = json.loads((model_path / 'quant_model_description.json').read_text())
    head = 'W8A8_DYNAMIC' if result['recipe'] in {'head', 'both'} else 'FLOAT'
    gdn = 'W8A8_DYNAMIC' if result['recipe'] in {'gdn', 'both'} else 'FLOAT'
    out_projections = [v for k, v in description.items()
                       if k.startswith('model.language_model.layers.')
                       and k.endswith('.linear_attn.out_proj.weight')]
    ffn = [v for k, v in description.items()
           if k.startswith('model.language_model.layers.')
           and k.endswith(('.mlp.gate_proj.weight', '.mlp.up_proj.weight', '.mlp.down_proj.weight'))]
    if (description.get('lm_head.weight') != head or len(out_projections) != 48
            or set(out_projections) != {gdn} or len(ffn) != 192 or set(ffn) != {'W4A8_DYNAMIC'}):
        raise ValueError('专项模型的敏感层精度或FFN量化配方不匹配')
    return result


def serving_options(config, packages):
    validated = artifact(config.model_path, packages)
    if (config.dtype is not ModelDType.BFLOAT16 or config.runner is not Runner.ACLGRAPH
            or config.graph_mode is not GraphMode.FULL_DECODE_ONLY
            or config.speculative_tokens != 2 or not config.prefix_caching
            or config.async_scheduling is not True or config.npugraph_ex is not True
            or config.overlap_shared_expert
            or (config.max_input_tokens, config.max_output_tokens, config.max_concurrency) != (57344, 8192, 4)):
        raise ValueError('专项适配只支持已验收的BF16、MTP2、异步解码图和四路64K配置')
    serving = validated['serving']
    if (config.batch_tokens not in serving['batch_tokens']
            or config.kv_cache_bytes not in serving['kv_cache_bytes']
            or config.gpu_memory_utilization != serving['gpu_memory_utilization']):
        raise ValueError('分块或KV内存预算未经过该专项组合验收')
    identity = {'name': ADAPTATION, 'recipe': validated['recipe'],
                'manifest_sha256': file_sha256(config.model_path / MANIFEST)}
    options = {'scheduler_cls': 'qwen_llm.engine.qwen38_cache.Qwen38CheckpointScheduler',
               'additional_config': {'qwen38_adaptation': identity}}
    return options, identity
