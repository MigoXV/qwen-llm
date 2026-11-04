"""从原BF16词表导出独立Ascend INT4候选；逐值保留已有主干。"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import time

import torch
import torch_npu
from safetensors import safe_open
from safetensors.torch import save_file
from .export_candidates import sha256


def quantize(weight):
    rows,hidden=weight.shape
    packed=torch.empty((rows,hidden//2),dtype=torch.int8)
    scales=torch.empty((rows,hidden//128),dtype=torch.bfloat16)
    error=norm=0.
    for start in range(0,rows,2048):
        if start%32768==0:print(f'quantizing row {start}/{rows}',flush=True)
        x=weight[start:start+2048].to('npu').float().reshape(-1,hidden//128,128)
        scale=(x.abs().amax(-1)/7).clamp(min=1e-12).bfloat16()
        q=(x/scale.float().unsqueeze(-1)).round().clamp(-8,7).to(torch.int8).reshape(-1,hidden)
        packed[start:start+len(q)]=(q[:,0::2].bitwise_and(15)|q[:,1::2].bitwise_and(15).bitwise_left_shift(4)).cpu()
        scales[start:start+len(q)]=scale.cpu()
        delta=q.float().reshape_as(x)*scale.float().unsqueeze(-1)-x
        error+=delta.square().sum().item();norm+=x.square().sum().item()
    return packed,scales,{'relative_l2':(error/norm)**.5,'source_bytes':weight.numel()*weight.element_size(),
        'packed_bytes':packed.numel()+scales.numel()*2}


def export(source,baseline,output):
    if output.exists() or any(output==p or p in output.parents for p in (source,baseline)):
        raise ValueError('需要全新且独立的输出目录')
    torch.set_num_threads(4);torch.npu.set_device(0)
    output.mkdir();(output/'.quantization.incomplete').touch();started=time.monotonic()
    index=json.loads((baseline/'quant_model_weights.safetensors.index.json').read_text())
    mapping=index['weight_map'];original=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    targets=('model.language_model.embed_tokens.weight','lm_head.weight')
    description=json.loads((baseline/'quant_model_description.json').read_text())
    manifest={'source':str(source),'baseline':str(baseline),'method':'Vocabulary-only group128 symmetric INT4 RTN from original BF16',
        'export_complete':False,'validated_for_inference':False,'calibration':None,'tensors':{},'files':[]}
    for shard in sorted(set(mapping.values())):
        selected=[k for k in targets if mapping[k]==shard]
        if not selected:
            subprocess.run(['cp','--reflink=auto','--',str(baseline/shard),str(output/shard)],check=True)
            assert sha256(output/shard)==sha256(baseline/shard)
        else:
            with ExitStack() as stack:
                base=stack.enter_context(safe_open(baseline/shard,framework='pt'))
                tensors={k:base.get_tensor(k) for k in base.keys()}
                changed=set()
                for key in selected:
                    f=stack.enter_context(safe_open(source/original[key],framework='pt'))
                    weight=f.get_tensor(key)
                    p,s,stats=quantize(weight);prefix=key.removesuffix('.weight')
                    for k in list(tensors):
                        if k==key or k.startswith(prefix+'.'):
                            changed.add(k);tensors.pop(k);mapping.pop(k,None);description.pop(k,None)
                    if key=='lm_head.weight':
                        # NPU W4 packs pairs of output rows, rather than hidden columns.
                        low=p.bitwise_and(15);high=p.bitwise_right_shift(4).bitwise_and(15)
                        q=torch.stack((low,high),-1).flatten(-2)
                        p=(q[0::2].bitwise_and(15)|q[1::2].bitwise_and(15).bitwise_left_shift(4)).contiguous()
                        tensors.update({key:p,prefix+'.weight_scale':torch.ones((len(s),1),dtype=torch.bfloat16),
                            prefix+'.weight_offset':torch.zeros((len(s),1),dtype=torch.bfloat16),
                            prefix+'.weight_scale_second':s,prefix+'.weight_offset_second':torch.zeros_like(s),
                            prefix+'.scale_bias':torch.zeros((len(s),1))})
                        scheme='W4A8_DYNAMIC'
                    else:
                        tensors.update({key:p,prefix+'.weight_scale':s});scheme='FLOAT'
                    for k in tensors:
                        if k.startswith(prefix+'.'):mapping[k]=shard;description[k]=scheme
                    manifest['tensors'][key]=stats
                    print(key,stats,flush=True)
                save_file(tensors,output/shard,metadata={'format':'pt'})
                with safe_open(output/shard,framework='pt') as out:
                    unchanged=[k for k in base.keys() if k not in changed]
                    assert all(torch.equal(base.get_tensor(k),out.get_tensor(k)) for k in unchanged)
                    manifest.setdefault('preserved_tensor_counts',{})[shard]=len(unchanged)
        manifest['files'].append({'path':shard,'sha256':sha256(output/shard),'unchanged':not selected})
        print(shard,flush=True)
    for f in baseline.iterdir():
        if f.is_file() and f.suffix!='.safetensors' and f.name not in {
            'quant_model_weights.safetensors.index.json','quant_model_description.json','quantization_manifest.json',
            'qwen38_optimization.json','README.md'} and not f.name.startswith('.'):
            shutil.copyfile(f,output/f.name)
    index['metadata']['total_size']=sum((output/n).stat().st_size for n in {r['path'] for r in manifest['files']})
    (output/'quant_model_weights.safetensors.index.json').write_text(json.dumps(index,indent=2)+'\n')
    (output/'quant_model_description.json').write_text(json.dumps(description,indent=2)+'\n')
    manifest.update(export_complete=True,seconds=time.monotonic()-started,
        exporter_sha256=sha256(Path(__file__)),format='Ascend packed W4 head + custom packed INT4 row-gather embedding')
    (output/'vocab4_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text('# Ascend词表INT4候选\n\n从原BF16词嵌入和输出层进行group128对称RTN；其余主干保留既有混合量化张量。没有使用校准数据。\n\n需要独立KVCapacityWorker和vocab4接线；不是可直接替换生产专项模型的已验收产物。质量、容量和速度以新实验记录为准。\n')
    (output/'.quantization.incomplete').unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/home/model-bin/Qwen/Qwen3.8-27B'))
    p.add_argument('--baseline',type=Path,default=Path('/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend-GDNHeadW8-v1'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.source,a.baseline,a.output)
