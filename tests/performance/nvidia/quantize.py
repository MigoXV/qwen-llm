"""从指定原始 BF16 导出 CUDA W4A16；原目录只读，保存配方与哈希。"""
import hashlib
import json
import logging
from pathlib import Path
import time
from importlib.metadata import version

import typer

app = typer.Typer()
logging.basicConfig(level=logging.INFO)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


@app.command()
def main(
    source: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(...),
    method: str = typer.Option('awq-gptq'),
    samples: int = typer.Option(512),
    sequence_length: int = typer.Option(4096),
    calibration: Path = typer.Option(Path('outputs/nvidia-20261028/calibration.jsonl'), exists=True),
):
    if output.exists() or source.resolve() in output.resolve().parents:
        raise ValueError('使用全新的独立输出目录')
    if method not in ('awq-gptq', 'gptq'):
        raise ValueError('method 必须为 awq-gptq 或 gptq')
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.utils import load_context
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    output.mkdir(parents=True)
    (output / '.quantization.incomplete').touch()
    start = time.monotonic()
    manifest = {'source': str(source.resolve()), 'method': method, 'seed': 42,
        'samples': samples, 'sequence_length': sequence_length,
        'calibration': json.loads(calibration.with_suffix('.manifest.json').read_text()),
        'source_config_sha256': sha256(source / 'config.json'),
        'packages': {p: version(p) for p in ('torch', 'transformers', 'llmcompressor', 'compressed-tensors', 'flash-linear-attention')},
        'export_complete': False, 'validated_for_inference': False}
    (output / 'quantization_manifest.json').write_text(json.dumps(manifest, indent=2))
    torch.manual_seed(42)
    # load_context 使用压缩工具的逐层 CPU offload，校准只占 CUDA_VISIBLE_DEVICES 指定的卡。
    with load_context(Qwen3_5ForConditionalGeneration):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(str(source), dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(str(source), local_files_only=True)
    ignore = ['re:visual.*', 're:model.visual.*', 're:.*lm_head', 're:.*embed_tokens$',
              r're:.*linear_attn\.in_proj_a$', r're:.*linear_attn\.in_proj_b$', 're:.*mtp.*']
    scheme = QuantizationScheme(targets=['Linear'], weights=QuantizationArgs(
        num_bits=4, type='int', symmetric=True, strategy='group', group_size=128))
    recipe = [AWQModifier(duo_scaling='both', offload_device=torch.device('cpu'))] if method == 'awq-gptq' else []
    recipe.append(GPTQModifier(config_groups={'w4a16': scheme}, ignore=ignore))
    manifest['ignore'] = ignore
    manifest['scheme'] = scheme.model_dump(mode='json')
    manifest['awq_activation_offload'] = 'cpu' if method == 'awq-gptq' else None
    (output / 'quantization_manifest.json').write_text(json.dumps(manifest, indent=2))
    from datasets import Dataset
    dataset = Dataset.from_list([json.loads(line) for line in calibration.read_text().splitlines()])
    if len(dataset) != samples or sha256(calibration) != manifest['calibration']['sha256']:
        raise ValueError('校准数据数量或哈希不匹配')
    oneshot(model=model, processor=processor, recipe=recipe,
            dataset=dataset, shuffle_calibration_samples=False,
            max_seq_length=sequence_length, num_calibration_samples=samples,
            pipeline='sequential', sequential_offload_device='cpu',
            output_dir=str(output))
    model.save_pretrained(output)
    processor.save_pretrained(output)
    # MTP 不参与本次校准，HF 未建模的 MTP 张量从原分片逐值保留。
    from safetensors import safe_open
    from safetensors.torch import save_file
    index_path = output / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        with safe_open(output / 'model.safetensors', framework='pt') as f:
            index = {'metadata': {}, 'weight_map': {k: 'model.safetensors' for k in f.keys()}}
    original = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    missing_mtp = {}
    for name, shard in original.items():
        if ('mtp' in name) and name not in index['weight_map']:
            with safe_open(source / shard, framework='pt') as f:
                missing_mtp[name] = f.get_tensor(name)
    if missing_mtp:
        save_file(missing_mtp, output / 'mtp-preserved.safetensors', metadata={'format':'pt'})
        index['weight_map'].update({n: 'mtp-preserved.safetensors' for n in missing_mtp})
    index_path.write_text(json.dumps(index, indent=2))
    # 保留原始 tokenizer/template 和消融来源；只量化权重。
    import shutil
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                 'generation_config.json', 'ABLIT_META.json', 'LICENSE', 'vocab.json', 'merges.txt'):
        if (source / name).is_file(): shutil.copyfile(source / name, output / name)
    manifest['files'] = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                         for p in sorted(output.glob('*.safetensors'))]
    manifest['elapsed_seconds'] = time.monotonic() - start
    manifest['export_complete'] = True
    manifest['copied_missing_mtp_tensors'] = len(missing_mtp)
    manifest['preserved_mtp_tensors'] = sum('mtp' in name for name in index['weight_map'])
    (output / 'quantization_manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / '.quantization.incomplete').unlink()


if __name__ == '__main__':
    app()
