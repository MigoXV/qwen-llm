"""910B 的 FP8 / INT8 KV 原生算子能力实测。"""
import argparse
import ctypes
import json
from pathlib import Path

import torch
import torch_npu

from .benchmark import inventory


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or inventory()['owners']:raise RuntimeError('需要新目录和空闲设备')
    a.output.mkdir(parents=True);torch.npu.set_device(0);torch.manual_seed(42)
    lib=ctypes.CDLL('libascendcl.so');lib.aclrtSetOpExecuteTimeOut(60)
    rows=[]
    def save():(a.output/'result.json').write_text(json.dumps(rows,indent=2)+'\n')
    with torch.inference_mode():
        q=torch.randn(1,24,1,256,device='npu',dtype=torch.bfloat16)
        k=torch.randn(1,4,128,256,device='npu',dtype=torch.bfloat16);v=torch.randn_like(k)
        def run(key,value,**kw):
            return torch_npu.npu_fused_infer_attention_score(q,key,value,input_layout='BNSD',
                actual_seq_lengths_kv=[128],num_heads=24,num_key_value_heads=4,scale=.0625,**kw)[0]
        reference=run(k,v);torch.npu.synchronize()
        for name,dtype in [('int8',torch.int8),('fp8_e4m3',torch.float8_e4m3fn)]:
            row={'dtype':name,'passed':False};rows.append(row);save()
            try:
                if name=='int8':
                    ks=k.float().abs().amax(2,keepdim=True)/127;vs=v.float().abs().amax(2,keepdim=True)/127
                    kk=(k.float()/ks).round().clamp(-128,127).to(dtype)
                    vv=(v.float()/vs).round().clamp(-128,127).to(dtype)
                else:
                    ks=k.float().abs().amax(2,keepdim=True)/448;vs=v.float().abs().amax(2,keepdim=True)/448
                    kk=(k.float()/ks).to(dtype);vv=(v.float()/vs).to(dtype)
                actual=run(kk,vv,key_antiquant_scale=ks.bfloat16(),value_antiquant_scale=vs.bfloat16(),
                    key_antiquant_mode=0,value_antiquant_mode=0)
                torch.npu.synchronize()
                error=((actual.float()-reference.float()).norm()/reference.float().norm()).item()
                row.update(relative_l2=error,finite=bool(actual.isfinite().all()),passed=error<=.05)
            except Exception as error:row['error']=str(error)
            save();print(json.dumps(row),flush=True)


if __name__=='__main__':main()
