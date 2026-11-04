"""新进程验证V2的布局、K流水、Split-K、原生回退与图重放。"""
import argparse
import ctypes
import faulthandler
import json
import os
from pathlib import Path
import torch
import torch_npu
from .ascendc_v2 import Tiling,pack_blocked
from .ascendc_v2_runtime import DEFAULT_MODEL,Runtime,load_projection,GraphMeasurement,relative
from .benchmark import inventory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host-library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-path',type=Path,default=DEFAULT_MODEL)
    args=p.parse_args()
    if args.output.exists() or inventory()['owners']:raise RuntimeError('输出必须不存在且NPU必须空闲')
    if os.environ.get('TASK_QUEUE_ENABLE')!='1':raise RuntimeError('必须在异步队列开启时验证')
    args.output.mkdir(parents=True);(args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    faulthandler.dump_traceback_later(90,repeat=True)
    torch.set_num_threads(4);torch.manual_seed(42);torch.npu.set_device(0);torch.npu.config.allow_internal_format=True
    runtime=ctypes.CDLL('libascendcl.so');fn=runtime.aclrtSetOpExecuteTimeOut;fn.argtypes=[ctypes.c_uint32];fn.restype=ctypes.c_int
    if fn(60):raise RuntimeError('不能设置设备执行超时')
    memcpy=runtime.aclrtMemcpy
    memcpy.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
    memcpy.restype=ctypes.c_int
    def physical_bytes(tensor):
        # int8 NZ -> int32 view is valid for the native kernel but cannot be
        # converted through generic TransData/Identity. Compare its actual bytes.
        result=torch.empty(tensor.numel()*tensor.element_size(),dtype=torch.uint8)
        torch.npu.synchronize()
        if memcpy(result.data_ptr(),result.numel(),tensor.data_ptr(),result.numel(),2):
            raise RuntimeError('不能复制物理NZ字节用于布局核对')
        return result
    configs=[Tiling(64,128,16,2),Tiling(128,256,16,3),Tiling(128,512,16,2),
             Tiling(64,256,16,3,2),Tiling(128,256,16,2,8),Tiling(128,256,128,3)]
    configs += [Tiling(64,128,16,2,1,0),Tiling(128,256,16,2,1,0)]
    device=Runtime(args.host_library,configs)
    (args.output/'build.json').write_text(json.dumps({'host':str(args.host_library.resolve()),
        'raw':torch.ops.qwen_w4.kernel_library_path(),'raw_sha256':torch.ops.qwen_w4.kernel_build_id()},indent=2)+'\n')
    rows=[]
    with torch.inference_mode():
        for projection in ('gate_up_proj','down_proj'):
            layer,scheme,integer,scale=load_projection(args.model_path,0,projection)
            blocked=pack_blocked(integer).npu();n,k=integer.shape
            restored=torch.ops.qwen_w4_v2.restore(blocked,n,k,device.id)
            torch.npu.synchronize()
            restore_ok=torch.equal(physical_bytes(restored),physical_bytes(layer.weight))
            (args.output/(projection+'-restore.json')).write_text(json.dumps({'passed':restore_ok})+'\n')
            if not restore_ok:raise RuntimeError('布局逆转换不是逐位一致')
            checks=[]
            for layout,weight in ((0,layer.weight),(1,blocked)):
                dequant=torch.ops.qwen_w4_v2.dequant(weight,layer.weight_scale_second,layout,device.id)[:n*k]
                torch.npu.synchronize()
                physical=dequant.cpu().reshape(k//16,n//16,16,16)
                for group in (0,k//256,k//128-1):
                    actual=physical[group*8:(group+1)*8].permute(0,3,1,2).reshape(128,n)
                    reference=(integer[:,group*128:(group+1)*128].float()*scale[:,group].float().unsqueeze(1)).bfloat16().t()
                    checks.append({'layout':layout,'group':group,'bitwise_equal':torch.equal(actual,reference)})
                del dequant,physical,actual,reference
            (args.output/(projection+'-dequant.json')).write_text(json.dumps(checks,indent=2)+'\n')
            if not all(c['bitwise_equal'] for c in checks):raise RuntimeError('反量化中间值与CPU逐位不一致')
            for m in (1,3,12,38,1536):
                x=torch.randn(m,k,dtype=torch.bfloat16,device='npu');expected=scheme.apply(layer,x);torch.npu.synchronize()
                for layout,weight in ((0,layer.weight),(1,blocked)):
                    for config in configs:
                        if m>117 and (config.split>1 or config.mtile==16):continue
                        if m<=117 and config.mtile==128:continue
                        print(json.dumps({'begin':True,'projection':projection,'M':m,'layout':layout,'config':config.values()}),flush=True)
                        actual=device.run(x,weight,layer.weight_scale_second,config,layout);torch.npu.synchronize()
                        error=relative(actual,expected)
                        passed=bool(torch.isfinite(actual).all()) and error<=.01
                        row={'projection':projection,'M':m,'layout':layout,'config':config.values(),'relative_l2':error,'passed':passed}
                        if not passed:
                            row['expected_finite']=bool(torch.isfinite(expected).all())
                            row['actual_finite_count']=int(torch.isfinite(actual).sum())
                            row['actual_prefix']=actual[0,:16].float().cpu().tolist()
                            b=torch.ops.qwen_w4_v2.dequant(weight,layer.weight_scale_second,layout,device.id)[:n*k]
                            torch.npu.synchronize()
                            row['dequant_finite_count']=int(torch.isfinite(b).sum())
                            row['dequant_prefix']=b[:32].float().cpu().tolist()
                            prefix=b[:128*n].cpu().reshape(8,n//16,16,16).permute(0,3,1,2).reshape(128,n)
                            reference=(integer[:,:128].float()*scale[:,0].float().unsqueeze(1)).bfloat16().t()
                            wrong=(prefix!=reference)
                            row['dequant_wrong_per_k']=wrong.sum(1).tolist()
                            row['dequant_wrong_per_n64']=wrong.reshape(128,n//64,64).sum((0,2)).tolist()
                            row['reference_nz_prefix']=reference[:16,0].float().tolist()
                        if passed:
                            graph=GraphMeasurement(lambda:device.run(x,weight,layer.weight_scale_second,config,layout),3)
                            row['graph_relative_l2']=relative(graph.output,expected)
                            row['passed'] &= row['graph_relative_l2']<=.01
                            del graph
                        rows.append(row)
                        (args.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
                        print(json.dumps(row),flush=True)
                        if not row['passed']:raise RuntimeError('V2数值或图重放未通过')
                    fallback=device.native(x,weight,layer.weight_scale_second,layout);torch.npu.synchronize()
                    if relative(fallback,expected)>.01:raise RuntimeError('原生回退未通过')
            del layer,blocked,integer,x,expected,actual,fallback
            torch.npu.empty_cache()
    result={'passed':all(r['passed'] for r in rows),'cases':rows,'workspace_bytes':device.workspace_bytes,
            'host_counts':torch.ops.qwen_w4_v2.counts(),'timing_tested':False}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
