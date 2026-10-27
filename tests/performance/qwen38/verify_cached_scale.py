"""真实 FFN 权重、NZ 和全部目标形状：尺度前移与上游输出逐位对齐。"""
import argparse
import copy
import json
from pathlib import Path
import torch
import torch_npu
from safetensors import safe_open
from vllm_ascend.quantization.methods.w4a8 import AscendW4A8DynamicLinearMethod
from .adaptations import CachedScaleLinearMethod


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42);torch.npu.set_device(0)
    torch.npu.config.allow_internal_format=True
    root=Path('/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend')
    index=json.loads((root/'quant_model_weights.safetensors.index.json').read_text())['weight_map'];rows=[]
    with torch.inference_mode():
        for layer_id in (0,30,63):
            for projection in ('gate_up_proj','down_proj'):
                parts=('gate_proj','up_proj') if projection=='gate_up_proj' else ('down_proj',)
                parts_data=[]
                for part in parts:
                    prefix=f'model.language_model.layers.{layer_id}.mlp.{part}.'
                    with safe_open(root/index[prefix+'weight'],framework='pt') as f:
                        parts_data.append({k.removeprefix(prefix):f.get_tensor(k) for k in f.keys() if k.startswith(prefix)})
                loaded={k:torch.cat([d[k] for d in parts_data]) for k in parts_data[0]}
                modules=[];schemes=[]
                for cls in (AscendW4A8DynamicLinearMethod,CachedScaleLinearMethod):
                    m=torch.nn.Module()
                    for name,value in loaded.items():
                        dtype=value.dtype if name=='weight' else torch.float32 if name=='scale_bias' else torch.bfloat16
                        m.register_parameter(name,torch.nn.Parameter(value.to(dtype).npu(),requires_grad=False))
                    scheme=cls.__new__(cls);scheme.group_size=128;scheme.new_quant_version=True;scheme.tp_size=1
                    scheme.process_weights_after_loading(m);modules.append(m);schemes.append(scheme)
                assert torch_npu.get_npu_format(modules[0].weight)==torch_npu.get_npu_format(modules[1].weight)==29
                assert torch.equal(modules[0].weight_scale_second.bfloat16(),modules[1].weight_scale_second)
                k=5120 if projection=='gate_up_proj' else 17408
                for batch in (1,3,6,12,38,117,1536,2048,6144,7680):
                    x=torch.randn(batch,k,dtype=torch.bfloat16,device='npu')
                    y0=schemes[0].apply(modules[0],x);y1=schemes[1].apply(modules[1],x)
                    equal=torch.equal(y0,y1)
                    row={'layer':layer_id,'projection':projection,'M':batch,'equal':equal,'max_abs':float((y0-y1).abs().max())}
                    rows.append(row);print(json.dumps(row),flush=True)
                    assert equal,row
                del modules,schemes,x,y0,y1
                torch.npu.empty_cache()
    a.output.write_text(json.dumps({'passed':True,'rows':rows},indent=2))


if __name__=='__main__':main()
