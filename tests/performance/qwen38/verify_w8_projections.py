"""真实输出头/GDN形状，比较W8算子与原始BF16投影的误差及设备耗时。"""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from safetensors import safe_open
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import maybe_trans_nz


def read(root,index,name):
    with safe_open(root/index[name],framework='pt',device='cpu') as f:return f.get_tensor(name)


def timing(fn):
    for _ in range(3):fn()
    start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(20):fn()
    end.record();end.synchronize()
    return start.elapsed_time(end)*1000/20


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);torch.npu.set_device(0)
    torch.npu.config.allow_internal_format=True
    original=Path('/home/model-bin/Qwen/Qwen3.8-27B')
    candidate=Path('/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend-GDNHeadW8-v1')
    oi=json.loads((original/'model.safetensors.index.json').read_text())['weight_map']
    ci=json.loads((candidate/'quant_model_weights.safetensors.index.json').read_text())['weight_map'];rows=[]
    names=['lm_head.weight',*[f'model.language_model.layers.{i}.linear_attn.out_proj.weight' for i in (0,30,62)]]
    with torch.inference_mode():
        for name in names:
            reference=read(original,oi,name).npu()
            if name!='lm_head.weight':reference=maybe_trans_nz(reference)
            prefix=name.removesuffix('.weight');layer=torch.nn.Module()
            for key in ['weight','weight_scale','weight_offset']:
                value=read(candidate,ci,prefix+'.'+key)
                if key!='weight':value=value.bfloat16()
                layer.register_parameter(key,torch.nn.Parameter(value.npu(),requires_grad=False))
            scheme=AscendW8A8DynamicLinearMethod();scheme.process_weights_after_loading(layer)
            assert layer.weight.dtype==torch.int8 and torch_npu.get_npu_format(layer.weight)==29
            if name=='lm_head.weight':
                # 共享头启动校验会比较两个NZ参数，先验证该真实形状的Equal路径。
                duplicate=layer.weight.clone()
                assert torch.equal(layer.weight,duplicate)
                del duplicate
            shapes=(1,3,4,6,12) if name=='lm_head.weight' else (1,3,6,12,38,117,1536,2048,6144,7680)
            for m in shapes:
                x=torch.randn(m,reference.shape[1],dtype=torch.bfloat16,device='npu')
                base=lambda:torch.nn.functional.linear(x,reference)
                optimized=lambda:scheme.apply(layer,x)
                expected,actual=base().float(),optimized().float()
                error=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected))
                old,new=timing(base),timing(optimized)
                row={'tensor':name,'M':m,'relative_l2':error,'max_abs':float((actual-expected).abs().max()),
                     'bf16_us':old,'w8_us':new,'bf16_format':torch_npu.get_npu_format(reference),'eager_event_span_ratio':old/new,'int8_NZ':True}
                rows.append(row);print(json.dumps(row),flush=True)
                assert error<.03,row
                del x,expected,actual
            del layer,reference
            torch.npu.empty_cache()
    a.output.write_text(json.dumps({'passed':True,'relative_l2_limit':.03,'samples':'seeded normal BF16 activations; not task accuracy',
                                   'timing_mode':'20 eager operator sequences measured with NPU events; includes submission-induced device idle, not isolated kernel time or model throughput',
                                   'rows':rows},indent=2))


if __name__=='__main__':main()
