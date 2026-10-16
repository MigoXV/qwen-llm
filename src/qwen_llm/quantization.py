"""逐张量调用 ModelSlim 的分阶段量化，导出独立 Ascend W4A8 权重。

这是无校准数据的候选导出器；只有实际推理和能力验证通过后才可启用。
不加载完整 BF16 模型，也不修改原权重或当前推理环境。
"""
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


def pack_int4_rows(weight):
    """AscendV1 的输出维双 int4 打包，低四位来自偶数行。"""
    import torch

    if weight.ndim != 2 or weight.shape[0] % 2:
        raise ValueError('INT4 打包要求二维权重且输出维为偶数')
    if weight.min() < -8 or weight.max() > 7:
        raise ValueError('INT4 权重超出 [-8, 7]')
    value = weight.to(torch.int8)
    return ((value[0::2] & 15) | (value[1::2] << 4)).contiguous()


def unpack_int4_rows(packed):
    """还原有符号 int4，供导出校验使用。"""
    import torch

    low = packed & 15
    low = torch.where(low >= 8, low - 16, low)
    high = packed >> 4
    return torch.stack((low, high), dim=1).reshape(packed.shape[0] * 2, packed.shape[1])


def quantization_kind(name: str, scope: str) -> str:
    if scope not in {'mlp', 'all'}:
        raise ValueError('w4-scope 必须为 mlp 或 all')
    if not name.startswith('model.language_model.layers.') or not name.endswith('.weight'):
        return 'FLOAT'
    if '.mlp.' in name and name.endswith(('.gate_proj.weight', '.up_proj.weight', '.down_proj.weight')):
        return 'W4A8_DYNAMIC'
    if '.self_attn.' in name and name.endswith(('.q_proj.weight', '.k_proj.weight', '.v_proj.weight', '.o_proj.weight')):
        return 'W4A8_DYNAMIC' if scope == 'all' else 'W8A8_DYNAMIC'
    if '.linear_attn.' in name and name.endswith(('.in_proj_qkv.weight', '.in_proj_z.weight')):
        return 'W4A8_DYNAMIC' if scope == 'all' else 'W8A8_DYNAMIC'
    return 'FLOAT'


def modelslim_functions(root: Path, dependencies: Path):
    for directory in (root, dependencies):
        if not directory.is_dir():
            raise ValueError(f'量化工具目录不存在：{directory}')
        sys.path.insert(0, str(directory.resolve()))
    # 必须先初始化 llm_ptq，避免直接导入 Cython lowbit 模块时的循环依赖。
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig  # noqa: F401
    from msmodelslim.pytorch.lowbit.atomic_power_outlier import quant_one_weight_by_outliers
    return quant_one_weight_by_outliers


def quantize_w4(weight, name: str, group_size: int, quantize):
    import torch

    if weight.shape[1] % group_size:
        raise ValueError(f'{name} 的输入维不整除 group_size')
    recovered, scales, integer, offsets = quantize(
        weight, powerquant=False, fraction=0.01, num_bits=4,
        isolate_outlier_amax=False, per_channel=True, use_cuda=False,
        use_sigma=False, sigma_factor=3.0, open_outlier=False,
        group_size=group_size, w_sym=True, use_hqq=False, progressive=True)
    # 使用量化器返回的整数，避免通用 saver 的 [-8,7] 截断改变对称 [-7,7] 结果。
    packed = pack_int4_rows(integer)
    if not torch.equal(unpack_int4_rows(packed), integer.to(torch.int8)):
        raise RuntimeError(f'INT4 打包还原失败：{name}')
    rows, columns = weight.shape
    first_scale = scales[0].float().reshape(rows, 1)
    second_scale = scales[1].float().reshape(rows, columns // group_size)
    dequantized = (integer.reshape(-1, group_size).float() * second_scale.reshape(-1, 1)).reshape(weight.shape) * first_scale
    if not torch.allclose(dequantized, recovered.float(), rtol=1e-5, atol=1e-5):
        raise RuntimeError(f'ModelSlim 分阶段反量化校验失败：{name}')
    rmse = float(((dequantized - weight.float()).square().mean() / weight.float().square().mean().clamp_min(1e-20)).sqrt())
    # 当前 AscendV1 row-parallel 参数槽固定为16列；原生 TP=1 运行时不使用此 bias。
    scale_bias = 8 * dequantized.reshape(rows, 16 if name.endswith('.down_proj.weight') else 1, -1).sum(-1)
    prefix = name.removesuffix('.weight')
    tensors = {
        name: packed,
        prefix + '.weight_scale': first_scale,
        prefix + '.weight_offset': offsets[0].float().reshape(rows, 1),
        prefix + '.weight_scale_second': second_scale,
        prefix + '.weight_offset_second': offsets[1].reshape(rows, -1).to(torch.int64),
        prefix + '.scale_bias': scale_bias.float(),
    }
    if torch.any(tensors[prefix + '.weight_offset']) or torch.any(tensors[prefix + '.weight_offset_second']):
        raise RuntimeError('当前 Ascend W4A8 路径只接受对称量化')
    return tensors, rmse


def quantize_w8(weight, name: str):
    import torch

    value = weight.float()
    scale = (value.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-12)
    integer = (value / scale).round().clamp(-127, 127).to(torch.int8)
    rmse = float((((integer.float() * scale - value).square().mean()) / value.square().mean().clamp_min(1e-20)).sqrt())
    prefix = name.removesuffix('.weight')
    return {name: integer, prefix + '.weight_scale': scale,
            prefix + '.weight_offset': torch.zeros_like(scale)}, rmse


def export_w4a8(source: Path, output: Path, source_manifest: Path,
                modelslim_root: Path, dependencies: Path, scope: str = 'mlp',
                group_size: int = 256, threads: int = 16):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('量化目录必须与原权重完全分开')
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出目录非空；请保留旧结果并使用新目录')
    manifest = json.loads(source_manifest.read_text())
    if not manifest.get('verified') or Path(manifest['source']).resolve() != source:
        raise ValueError('需要已通过原始文件 SHA256 校验的清单')
    config = json.loads((source / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_5' or config['text_config']['num_hidden_layers'] != 64:
        raise ValueError('此导出入口只验证 Qwen3.8-27B 使用的64层 dense架构')
    for row in manifest['files']:
        if (source / row['path']).stat().st_size != row['bytes']:
            raise ValueError(f'原文件大小已改变：{row["path"]}')
    index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    torch.set_num_threads(threads)
    quantize = modelslim_functions(modelslim_root, dependencies)
    output.mkdir(parents=True, exist_ok=True)
    pending = output / '.quantization.incomplete'
    pending.touch()
    modelslim_sha = subprocess.check_output(['git', '-C', str(modelslim_root), 'rev-parse', 'HEAD'], text=True).strip()
    record = {
        'source': str(source), 'source_manifest': str(source_manifest),
        'source_manifest_sha256': hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        'modelslim_commit': modelslim_sha, 'w4_scope': scope, 'group_size': group_size,
        'method': 'ModelSlim data-free symmetric progressive INT8 -> group INT4; dynamic activation schemes',
        'calibration_dataset': None, 'validated_for_inference': False,
        'created_at': datetime.now(timezone.utc).isoformat(), 'layers': [], 'files': [],
    }
    description = {'version': '1.0.0', 'model_quant_type': 'W4A8_DYNAMIC', 'group_size': group_size}
    weight_map, total_size = {}, 0
    shards = sorted(set(index.values()))
    started = time.perf_counter()
    for number, shard in enumerate(shards, 1):
        exported = {}
        shard_started = time.perf_counter()
        with safe_open(source / shard, framework='pt', device='cpu') as reader:
            for name in sorted(reader.keys()):
                weight = reader.get_tensor(name)
                kind = quantization_kind(name, scope)
                if kind == 'W4A8_DYNAMIC':
                    tensors, rmse = quantize_w4(weight, name, group_size, quantize)
                elif kind == 'W8A8_DYNAMIC':
                    tensors, rmse = quantize_w8(weight, name)
                else:
                    tensors, rmse = {name: weight.contiguous()}, None
                exported.update({key: tensor.cpu().contiguous() for key, tensor in tensors.items()})
                description.update({key: kind for key in tensors})
                if rmse is not None:
                    record['layers'].append({'name': name, 'shape': list(weight.shape),
                                             'scheme': kind, 'weight_relative_rmse': rmse})
                    logger.info('%s %s weight relative RMSE=%.5f', kind, name, rmse)
                del weight, tensors
        filename = f'quant_model_weights-{number:05d}-of-{len(shards):05d}.safetensors'
        temporary = output / (filename + '.part')
        save_file(exported, temporary, metadata={'format': 'pt'})
        temporary.rename(output / filename)
        weight_map.update({key: filename for key in exported})
        total_size += sum(t.numel() * t.element_size() for t in exported.values())
        from qwen_llm.modelscope import sha256
        record['files'].append({'path': filename, 'bytes': (output / filename).stat().st_size,
                                'sha256': sha256(output / filename)})
        del exported
        gc.collect()
        (output / 'quantization_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
        logger.info('Shard %d/%d saved in %.1fs', number, len(shards), time.perf_counter() - shard_started)
    for name in ('config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json',
                 'chat_template.jinja', 'vocab.json', 'merges.txt', 'preprocessor_config.json',
                 'video_preprocessor_config.json', 'configuration.json', 'LICENSE'):
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
    (output / 'quant_model_weights.safetensors.index.json').write_text(json.dumps(
        {'metadata': {'total_size': total_size}, 'weight_map': weight_map}, ensure_ascii=False, indent=2))
    (output / 'quant_model_description.json').write_text(json.dumps(description, ensure_ascii=False, indent=2))
    record['export_complete'] = True
    record['elapsed_seconds'] = time.perf_counter() - started
    record['tensor_bytes'] = total_size
    record['scheme_counts'] = {kind: sum(r['scheme'] == kind for r in record['layers'])
                               for kind in ('W4A8_DYNAMIC', 'W8A8_DYNAMIC')}
    (output / 'quantization_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    pending.unlink()
    logger.info('Export complete: %.3f GiB; schemes=%s; inference validation pending',
                total_size / 2**30, record['scheme_counts'])
    return record
