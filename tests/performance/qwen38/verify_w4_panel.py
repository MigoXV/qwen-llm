"""V3全K面板复用短测；只验证算子数值与速度。"""
import argparse
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from .ascendc_v2_runtime import DEFAULT_MODEL,GraphMeasurement,load_projection,relative
from .benchmark import inventory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host-library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stream',action='store_true');args=p.parse_args()
    if args.output.exists() or inventory()['owners']:raise RuntimeError('需要新输出目录和空闲NPU')
    args.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(42);torch.npu.set_device(0);torch.npu.config.allow_internal_format=True
    lib=ctypes.CDLL('libascendcl.so');fn=lib.aclrtSetOpExecuteTimeOut;fn.argtypes=[ctypes.c_uint32];fn.restype=ctypes.c_int
    if fn(60):raise RuntimeError('不能设置设备超时')
    torch.ops.load_library(str(args.host_library));anchor=torch.empty(1,dtype=torch.bfloat16,device='npu')
    workspace=torch.ops.qwen_w4_v3.reserve(anchor,40*256*(512 if args.stream else 17408)*2);rows=[];started=time.monotonic()
    with torch.inference_mode():
        for projection in ('gate_up_proj','down_proj'):
            layer,scheme,integer,scale=load_projection(DEFAULT_MODEL,0,projection);k=integer.shape[1]
            for m in ((1536,8192) if args.stream else (3,12,1536,8192)):
                x=torch.randn(m,k,dtype=torch.bfloat16,device='npu');expected=scheme.apply(layer,x);torch.npu.synchronize()
                base=GraphMeasurement(lambda:scheme.apply(layer,x),16 if m<=117 else 4)
                configs=[(16,128,256),(16,128,512),(16,256,256)] if m<=16 else [(128,128,256),(128,128,512),(128,256,256),(256,128,256)]
                if args.stream:configs=[(256,128,256),(128,256,256)]
                op=torch.ops.qwen_w4_v3.rolling if args.stream else torch.ops.qwen_w4_v3.run
                for mt,nt,kt in configs:
                    print(json.dumps({'begin':True,'projection':projection,'M':m,'tile':[mt,nt,kt]}),flush=True)
                    call=lambda:op(x,layer.weight,layer.weight_scale_second,mt,nt,kt)
                    actual=call();torch.npu.synchronize();error=relative(actual,expected)
                    row={'projection':projection,'M':m,'tile':[mt,nt,kt],'relative_l2':error,'passed':bool(torch.isfinite(actual).all()) and error<=.01,'patterns':[]}
                    for amp in (0,64):
                        reference=scheme.apply(layer,x*amp);value=op(x*amp,layer.weight,layer.weight_scale_second,mt,nt,kt)
                        torch.npu.synchronize();e=relative(value,reference);row['patterns'].append(e);row['passed'] &= bool(torch.isfinite(value).all()) and e<=.01
                    if row['passed']:
                        graph=GraphMeasurement(call,16 if m<=117 else 4);row['graph_relative_l2']=relative(graph.output,expected)
                        a=[];b=[]
                        for i in range(5):
                            if i%2:a.append(graph.sample());b.append(base.sample())
                            else:b.append(base.sample());a.append(graph.sample())
                        row.update(candidate_ms=a,native_ms=b,time_ratio=statistics.median(a)/statistics.median(b));row['passed'] &= row['graph_relative_l2']<=.01;del graph
                    rows.append(row);print(json.dumps(row),flush=True);(args.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
                    if not row['passed']:raise RuntimeError('V3对齐失败')
                del base,x,expected,actual,reference,value
            del layer,integer,scale;torch.npu.empty_cache()
    (args.output/'result.json').write_text(json.dumps({'samples':rows,'workspace_bytes':workspace,'seconds':time.monotonic()-started,'host_sha256':sha256(args.host_library.read_bytes()).hexdigest(),'raw_library_sha256':torch.ops.qwen_w4.kernel_build_id(),'rolling':args.stream},indent=2)+'\n')


if __name__=='__main__':main()
