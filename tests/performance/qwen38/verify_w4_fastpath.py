"""一分钟级筛选原生精度分支与FP16计算分支；不运行整模型质量集。"""
import argparse
import ctypes
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from .ascendc_v2_runtime import load_projection,GraphMeasurement,relative,DEFAULT_MODEL
from .benchmark import inventory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calls',type=int,default=16);p.add_argument('--layers',default='0');p.add_argument('--ms',default='1,3,6,12,38,117,1536');p.add_argument('--inner-only',action='store_true');p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.output.exists() or inventory()['owners']:raise RuntimeError('需要新输出目录和空闲NPU')
    args.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(args.seed);torch.npu.set_device(0)
    torch.npu.config.allow_internal_format=True
    lib=ctypes.CDLL('libascendcl.so');fn=lib.aclrtSetOpExecuteTimeOut;fn.argtypes=[ctypes.c_uint32];fn.restype=ctypes.c_int
    if fn(60):raise RuntimeError('不能设置设备超时')
    rows=[];started=time.monotonic()
    def native(x,w,s,precise=0):return torch_npu.npu_weight_quant_batchmatmul(x,w,s,antiquant_group_size=128,inner_precise=precise)
    with torch.inference_mode():
        for layer_id,projection in ((l,p) for l in map(int,args.layers.split(',')) for p in ('gate_up_proj','down_proj')):
            layer,scheme,integer,scale=load_projection(DEFAULT_MODEL,layer_id,projection)
            s=layer.weight_scale_second;s16=s.half();k=integer.shape[1]
            for m in map(int,args.ms.split(',')):
                x=torch.randn(m,k,dtype=torch.bfloat16,device='npu');expected=native(x,layer.weight,s);torch.npu.synchronize()
                base=GraphMeasurement(lambda:native(x,layer.weight,s),args.calls if m<=117 else 4)
                variants={'fp16':lambda:native(x.half(),layer.weight,s16).bfloat16()}
                if m<=16 or args.inner_only:
                    variants['inner1']=lambda:native(x,layer.weight,s,1)
                    variants['fp16-inner1']=lambda:native(x.half(),layer.weight,s16,1).bfloat16()
                if args.inner_only:variants={'inner1':variants['inner1']}
                for name,call in variants.items():
                    row={'layer':layer_id,'projection':projection,'M':m,'variant':name,'passed':False}
                    try:
                        actual=call();torch.npu.synchronize();error=relative(actual,expected)
                        row.update(relative_l2=error,finite=bool(torch.isfinite(actual).all()))
                        row['patterns']=[]
                        for amplitude in (0,64):
                            px=x*amplitude;reference=native(px,layer.weight,s)
                            candidate=(native(px,layer.weight,s,1) if name=='inner1' else native(px.half(),layer.weight,s16,1 if name=='fp16-inner1' else 0).bfloat16())
                            torch.npu.synchronize();row['patterns'].append(relative(candidate,reference))
                        if row['finite'] and error<=.01 and max(row['patterns'])<=.01:
                            graph=GraphMeasurement(call,args.calls if m<=117 else 4);row['graph_relative_l2']=relative(graph.output,expected)
                            a=[];b=[]
                            for i in range(5):
                                if i%2:a.append(graph.sample());b.append(base.sample())
                                else:b.append(base.sample());a.append(graph.sample())
                            row.update(candidate_ms=a,native_ms=b,time_ratio=statistics.median(a)/statistics.median(b),passed=row['graph_relative_l2']<=.01)
                            del graph
                    except Exception as exc:row['error']=repr(exc)
                    rows.append(row);print(json.dumps(row),flush=True)
                    (args.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
                del base,x,expected
            del layer,integer,scale,s,s16
            torch.npu.empty_cache()
    (args.output/'result.json').write_text(json.dumps({'samples':rows,'seconds':time.monotonic()-started,'calls_per_replay_small':args.calls,'seed':args.seed,'layers':args.layers,'M':args.ms},indent=2)+'\n')


if __name__=='__main__':main()
