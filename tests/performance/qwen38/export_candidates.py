"""从原始 BF16 敏感层导出独立 W8 候选，其他张量逐字节保留。"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .quantization import quantize_w8


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def selected(name, recipe):
    return (recipe in {'gdn', 'both'} and name.startswith('model.language_model.layers.')
            and name.endswith('.linear_attn.out_proj.weight')) or (recipe in {'head', 'both'} and name == 'lm_head.weight')


def export(source, baseline, output, recipe):
    source, baseline, output = (Path(p).resolve() for p in (source, baseline, output))
    if recipe not in {'gdn', 'head', 'both'}:
        raise ValueError('未知候选配方')
    if any(output == p or p in output.parents or output in p.parents for p in (source, baseline)):
        raise ValueError('候选目录必须与原始和基线模型完全分开')
    if output.exists():
        raise ValueError('输出目录已存在，拒绝覆盖')
    config = json.loads((source/'config.json').read_text())
    if config.get('model_type') != 'qwen3_5' or config['text_config']['num_hidden_layers'] != 64:
        raise ValueError('仅支持 Qwen3.8-27B dense')
    source_index = json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    base_index = json.loads((baseline/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    description = json.loads((baseline/'quant_model_description.json').read_text())
    keys = {k for k in base_index if selected(k, recipe)}
    expected = {'gdn':48, 'head':1, 'both':49}[recipe]
    if len(keys) != expected or any(description[k] != 'FLOAT' for k in keys):
        raise ValueError('候选敏感层数或原精度不匹配')
    output.mkdir()
    (output/'.quantization.incomplete').touch()
    manifest = {'recipe': recipe, 'source': str(source), 'baseline': str(baseline),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'exporter_sha256': sha256(Path(__file__)), 'export_complete': False,
                'validated_for_inference': False, 'layers': [], 'files': [],
                'method': 'selected original BF16 tensors -> symmetric per-output-channel INT8, dynamic per-token activations',
                'quantization_group_size': 256, 'runtime_group_size': 128}
    new_index = dict(base_index)
    shards = sorted(set(base_index.values()))
    for shard in shards:
        targets = sorted(k for k in keys if base_index[k] == shard)
        before = sha256(baseline/shard)
        if not targets:
            subprocess.run(['cp', '--reflink=auto', '--', str(baseline/shard), str(output/shard)], check=True)
        else:
            with ExitStack() as stack:
                base = stack.enter_context(safe_open(baseline/shard, framework='pt', device='cpu'))
                originals = {name: stack.enter_context(safe_open(source/name, framework='pt', device='cpu'))
                             for name in {source_index[k] for k in targets}}
                tensors = {k: base.get_tensor(k) for k in base.keys()}
                for name in targets:
                    weight = originals[source_index[name]].get_tensor(name)
                    if not torch.equal(weight, tensors[name]):
                        raise ValueError(f'基线浮点张量已改变：{name}')
                    quantized, error = quantize_w8(weight, name)
                    tensors.update(quantized)
                    for k in quantized:
                        new_index[k] = shard
                        description[k] = 'W8A8_DYNAMIC'
                    manifest['layers'].append({'name':name,'shape':list(weight.shape),'scheme':'W8A8_DYNAMIC',
                                               'weight_relative_rmse':error})
                temp = output/(shard+'.part')
                save_file(tensors, temp, metadata={'format':'pt'})
                temp.rename(output/shard)
                del tensors
        result = sha256(output/shard)
        manifest['files'].append({'path':shard,'sha256':result,'baseline_sha256':before,
                                 'unchanged':not targets,'bytes':(output/shard).stat().st_size})
        if not targets and result != before:
            raise RuntimeError('未修改分片的 SHA256 不一致')
        print(json.dumps({'recipe':recipe,'shard':shard,'changed_layers':len(targets)}),flush=True)
    for f in baseline.iterdir():
        if f.is_file() and f.suffix not in {'.safetensors'} and f.name not in {
                'quant_model_weights.safetensors.index.json','quant_model_description.json','quantization_manifest.json'} and not f.name.startswith('.'):
            shutil.copyfile(f,output/f.name)
    total = 0
    for shard in shards:
        with safe_open(output/shard, framework='pt') as f:
            for k in f.keys():
                x=f.get_slice(k); n=1
                for d in x.get_shape():n*=d
                sizes={'F32':4,'BF16':2,'F16':2,'I8':1,'I64':8,'I32':4,'U8':1,'BOOL':1}
                total+=n*sizes[x.get_dtype()]
    (output/'quant_model_weights.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total},'weight_map':new_index},indent=2)+'\n')
    (output/'quant_model_description.json').write_text(json.dumps(description,indent=2)+'\n')
    manifest.update(export_complete=True,tensor_bytes=total)
    (output/'quantization_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (output/'README.md').write_text(f'# Qwen3.8-27B 专项候选 {recipe}\n\n来源：{source}\n\n基线：{baseline}\n\n独立候选，质量和性能验收状态见实验报告。\n')
    (output/'.quantization.incomplete').unlink()
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/home/model-bin/Qwen/Qwen3.8-27B'))
    p.add_argument('--baseline',type=Path,default=Path('/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend'))
    p.add_argument('--output-root',type=Path,default=Path('/home/model-bin/Qwen'))
    args=p.parse_args();torch.set_num_threads(2)
    for recipe,suffix in [('gdn','GDNW8'),('head','HeadW8'),('both','GDNHeadW8')]:
        export(args.source,args.baseline,args.output_root/f'Qwen3.8-27B-W4A8-Ascend-{suffix}-v1',recipe)


if __name__=='__main__':
    main()
