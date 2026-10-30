"""保留已有 AWQ/GPTQ 主干，仅对词表矩阵做分组 RTN 的独立实验导出。"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import typer

app = typer.Typer()
TARGETS = ('lm_head', 'model.language_model.embed_tokens')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def quantize(weight, bits, group_size=128, chunk_rows=2048):
    import torch
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32

    rows, hidden = weight.shape
    assert hidden % group_size == 0 and hidden % (32 // bits) == 0
    packed = torch.empty(rows, hidden // (32 // bits), dtype=torch.int32)
    scales = torch.empty(rows, hidden // group_size, dtype=weight.dtype)
    error2 = norm2 = 0.0
    max_error = 0.0
    for start in range(0, rows, chunk_rows):
        x = weight[start:start + chunk_rows].cuda().float().reshape(-1, hidden // group_size, group_size)
        # 以实际保存的 BF16 scale 做舍入，避免导出/加载尺度不一致。
        scale = (x.abs().amax(-1) / (2**(bits - 1) - 1)).clamp_min(1e-12).to(weight.dtype)
        q = (x / scale.float().unsqueeze(-1)).round().clamp(-2**(bits - 1), 2**(bits - 1) - 1).to(torch.int8)
        p = pack_to_int32(q.reshape(-1, hidden), bits)
        assert torch.equal(unpack_from_int32(p, bits, torch.Size([x.shape[0], hidden])), q.reshape(-1, hidden))
        packed[start:start + x.shape[0]] = p.cpu()
        scales[start:start + x.shape[0]] = scale.cpu()
        delta = q.float() * scale.float().unsqueeze(-1) - x
        error2 += delta.square().sum().item()
        norm2 += x.square().sum().item()
        max_error = max(max_error, delta.abs().max().item())
    return packed, scales, {'relative_l2': (error2 / norm2)**0.5, 'max_abs_error': max_error,
        'source_bytes': weight.numel() * weight.element_size(),
        'quantized_bytes': packed.numel() * 4 + scales.numel() * scales.element_size() + 16}


@app.command()
def main(source: Path = typer.Option(..., exists=True), output: Path = typer.Option(...),
         bits: int = typer.Option(4)):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if bits not in (4, 8):
        raise ValueError('实验只支持 INT4/INT8')
    if output.exists() or source.resolve() in output.resolve().parents:
        raise ValueError('必须使用全新的独立输出目录')
    assert torch.cuda.device_count() == 1, '一次导出独占一张指定卡'
    output.mkdir(parents=True)
    (output / '.quantization.incomplete').touch()
    started = time.monotonic()
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    config = json.loads((source / 'config.json').read_text())
    manifest = {'source': str(source.resolve()), 'method': f'vocabulary-only W{bits}A16 group128 symmetric RTN',
        'calibration': None, 'backbone': 'unchanged AWQ/GPTQ packed tensors',
        'source_manifest_sha256': sha256(source / 'quantization_manifest.json'),
        'source_config_sha256': sha256(source / 'config.json'),
        'export_complete': False, 'validated_for_inference': False, 'tensors': {}, 'preserved_shards': []}
    for filename in sorted(set(index['weight_map'].values())):
        with safe_open(source / filename, framework='pt') as f:
            targets = [name for name in TARGETS if name + '.weight' in f.keys()]
            if not targets:
                shutil.copyfile(source / filename, output / filename)
                manifest['preserved_shards'].append(filename)
                continue
            tensors = {name: f.get_tensor(name) for name in f.keys()}
            for name in targets:
                weight = tensors.pop(name + '.weight')
                packed, scales, stats = quantize(weight, bits)
                for suffix, tensor in [('weight_packed', packed), ('weight_scale', scales),
                                       ('weight_shape', torch.tensor(weight.shape, dtype=torch.int64))]:
                    tensors[name + '.' + suffix] = tensor
                    index['weight_map'][name + '.' + suffix] = filename
                del index['weight_map'][name + '.weight']
                manifest['tensors'][name] = stats
                print(name, stats, flush=True)
            save_file(tensors, output / filename, metadata={'format': 'pt'})
            # 逐值验证所有未改动张量，不能仅根据复制代码宣称主干保持一致。
            with safe_open(output / filename, framework='pt') as exported:
                unchanged = [name for name in f.keys() if name not in {t + '.weight' for t in targets}]
                for name in unchanged:
                    assert torch.equal(f.get_tensor(name), exported.get_tensor(name)), name
                manifest['preserved_tensor_count'] = len(unchanged)
            del tensors
    qconfig = config['quantization_config']
    scheme = copy.deepcopy(next(iter(qconfig['config_groups'].values())))
    scheme['targets'] = ['re:.*lm_head$', 're:.*embed_tokens$']
    scheme['weights'].update(num_bits=bits, actorder=None)
    qconfig['config_groups']['vocabulary_rtn'] = scheme
    qconfig['ignore'] = [name for name in qconfig['ignore'] if name != 'lm_head']
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    total = 0
    for filename in set(index['weight_map'].values()):
        with safe_open(output / filename, framework='pt') as f:
            total += sum(f.get_tensor(k).numel() * f.get_tensor(k).element_size() for k in f.keys())
    index['metadata']['total_size'] = total
    (output / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json',
                 'ABLIT_META.json', 'LICENSE', 'vocab.json', 'merges.txt', 'preprocessor_config.json',
                 'processor_config.json', 'video_preprocessor_config.json'):
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
    manifest.update(export_complete=True, elapsed_seconds=time.monotonic() - started,
        files=[{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)}
               for p in sorted(output.glob('*.safetensors'))])
    (output / 'quantization_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'README.md').write_text(f'# 词表 INT{bits} 量化实验\n\n主干沿用原 AWQ/GPTQ 权重，仅词嵌入与输出层采用 group128 对称 RTN。\n\n未使用校准集；推理、质量、长上下文结果以独立实验报告为准，不能继承原模型的完整质量结论。\n')
    (output / '.quantization.incomplete').unlink()
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    app()
