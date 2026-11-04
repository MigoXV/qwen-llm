"""短测W8已缓存的FP32尺度及输出反量化分支，不改量化权重。"""
import argparse
import ctypes
import json
from pathlib import Path
import statistics
import torch
import torch_npu
from safetensors import safe_open
from .ascendc_v2_runtime import DEFAULT_MODEL,GraphMeasurement,relative
from .benchmark import inventory
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or inventory()['owners']:raise RuntimeError('需要新目录和空闲NPU')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(42);torch.npu.set_device(0);torch.npu.config.allow_internal_format=True
    lib=ctypes.CDLL('libascendcl.so');fn=lib.aclrtSetOpExecuteTimeOut;fn.argtypes=[ctypes.c_uint32];fn.restype=ctypes.c_int
    if fn(60):raise RuntimeError('不能设置设备超时')
    index=json.loads((DEFAULT_MODEL/'quant_model_weights.safetensors.index.json').read_text())['weight_map'];rows=[]
    with torch.inference_mode():
        for name in ('model.language_model.layers.0.linear_attn.out_proj','lm_head'):
            layer=torch.nn.Module();prefix=name+'.'
            with safe_open(DEFAULT_MODEL/index[prefix+'weight'],framework='pt') as src:
                for key in src.keys():
                    if key.startswith(prefix):layer.register_parameter(key[len(prefix):],torch.nn.Parameter(src.get_tensor(key).npu(),requires_grad=False))
            scheme=AscendW8A8DynamicLinearMethod();scheme.process_weights_after_loading(layer)
            k=layer.weight.shape[0]
            def calc(x,scale,output_dtype):
                q,s=torch_npu.npu_dynamic_quant(x)
                y=torch_npu.npu_quant_matmul(q,layer.weight,scale,pertoken_scale=s,output_dtype=output_dtype)
                return y.bfloat16()
            for m in (3,6,12,38,1536):
                x=torch.randn(m,k,dtype=torch.bfloat16,device='npu');expected=scheme.apply(layer,x);torch.npu.synchronize()
                calls=128 if m<=38 else 4;base=GraphMeasurement(lambda:scheme.apply(layer,x),calls)
                for variant,scale,dtype in (('fp32-scale',layer.weight_scale_fp32,torch.bfloat16),('fp16-output',layer.weight_scale_fp32,torch.float16)):
                    row={'name':name,'M':m,'variant':variant,'passed':False}
                    try:
                        call=lambda:calc(x,scale,dtype);value=call();torch.npu.synchronize();error=relative(value,expected)
                        row.update(relative_l2=error,bitwise_equal=torch.equal(value,expected))
                        if bool(torch.isfinite(value).all()) and error<=.01:
                            g=GraphMeasurement(call,calls);u=[];v=[]
                            for i in range(5):
                                if i%2:u.append(g.sample());v.append(base.sample())
                                else:v.append(base.sample());u.append(g.sample())
                            row.update(candidate_ms=u,native_ms=v,time_ratio=statistics.median(u)/statistics.median(v),passed=relative(g.output,expected)<=.01);del g
                    except Exception as exc:row['error']=repr(exc)
                    rows.append(row);print(json.dumps(row),flush=True);(a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
                del base,x,expected
            del layer;torch.npu.empty_cache()
    (a.output/'result.json').write_text(json.dumps({'samples':rows},indent=2)+'\n')


if __name__=='__main__':main()
