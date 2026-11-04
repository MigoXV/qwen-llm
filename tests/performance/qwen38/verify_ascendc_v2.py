"""V2逐配置数值、消融、成组图计时及三层独立复核；拒绝用旧计时结果作新基线。"""
import argparse
import ctypes
import faulthandler
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import statistics
import time
import uuid
import torch
import torch_npu
from .ascendc_v2 import Layout,LAYOUT_VERSION,SOURCE_FILES,Tiling,configurations,pack_blocked
from .ascendc_v2_runtime import Runtime,GraphMeasurement,DEFAULT_MODEL,load_projection,relative
from .benchmark import inventory

MS=(1,3,6,12,38,117,1536,2048,6144,7680,8192)
TUNE_MS=(3,6,12,38,117,1536,8192)
QUICK_MS=(3,12,1536)


def quick_configurations(m):
    mt=16 if m<=16 else 128
    return [Tiling(128,256,mt,2,1,0),Tiling(64,128,mt,2),
            Tiling(128,256,mt,2),Tiling(128,512,mt,3)]


def bucket(m):
    return 3 if m==1 else 1536 if m==2048 else 8192 if m in (6144,7680) else m


def environment():
    return {'device':torch.npu.get_device_name(0),'cann_root':str(Path(os.environ['ASCEND_HOME_PATH']).resolve()),
            'packages':{n:version(n) for n in ('torch','torch-npu','vllm','vllm-ascend','transformers')},'TASK_QUEUE_ENABLE':os.environ['TASK_QUEUE_ENABLE']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host-library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-path',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--phase',choices=('quick','tune','validate'),required=True);p.add_argument('--screen',type=Path)
    p.add_argument('--v1-record',type=Path,default=Path('/home/qwen-models/reports/qwen3.8-27b/ffn-kernel-20261103-r1/micro/result.json'))
    args=p.parse_args()
    if args.output.exists() or inventory()['owners']:raise RuntimeError('需要不存在的输出目录及空闲NPU')
    if os.environ.get('TASK_QUEUE_ENABLE')!='1':raise RuntimeError('筛选必须使用最终异步队列桥')
    screen=json.loads(args.screen.read_text()) if args.screen else None
    if args.phase=='validate' and screen is None:raise ValueError('复核需要调优结果')
    args.output.mkdir(parents=True);(args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    faulthandler.dump_traceback_later(90,repeat=True)
    torch.set_num_threads(4);torch.manual_seed(42);torch.npu.set_device(0);torch.npu.config.allow_internal_format=True
    lib=ctypes.CDLL('libascendcl.so');timeout=lib.aclrtSetOpExecuteTimeOut;timeout.argtypes=[ctypes.c_uint32];timeout.restype=ctypes.c_int
    if timeout(60):raise RuntimeError('执行超时设置失败')
    all_configs=set()
    for m in MS:
        for c in configurations(m):
            all_configs.add(c)
            if m<=117 and c.family==1:
                for split in (2,4,8):all_configs.add(Tiling(c.ntile,c.ktile,c.mtile,c.depth,split,c.family))
    all_configs=sorted(all_configs,key=lambda c:c.values());rt=Runtime(args.host_library,all_configs)
    old=json.loads(args.v1_record.read_text())
    v1_stage=torch.empty(40*128*17408,dtype=torch.bfloat16,device='npu')
    samples=[];best=[];started=time.monotonic()
    def write_sample(row):
        samples.append(row)
        with (args.output/'samples.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    def variant(layer_id,projection,m,x,weight,scales,layout,c,expected,native,probes,oracle):
        print(json.dumps({'begin':True,'layer':layer_id,'projection':projection,'M':m,'layout':layout,'configuration':c.values()}),flush=True)
        v={'configuration':c.values(),'layout':layout,'passed':False,'calls_per_replay':16 if m<=117 else 4,
           'workspace_bytes':c.workspace_bytes(m,expected.shape[1],x.shape[1])}
        actual=rt.run(x,weight,scales,c,layout);torch.npu.synchronize()
        error=relative(actual,expected);v.update(relative_l2=error,finite=bool(torch.isfinite(actual).all()))
        okay=v['finite'] and error<=.01
        if okay and oracle is not None:
            v['fp32_relative_l2']=relative(actual.float().cpu(),oracle);okay &= v['fp32_relative_l2']<=.01
        v['patterns']=[]
        if okay:
            for amplitude,px,pe in probes:
                got=rt.run(px,weight,scales,c,layout);torch.npu.synchronize()
                e=relative(got,pe);passed=bool(torch.isfinite(got).all()) and e<=.01
                v['patterns'].append({'amplitude':amplitude,'relative_l2':e,'passed':passed});okay &= passed
        if okay:
            candidate=GraphMeasurement(lambda:rt.run(x,weight,scales,c,layout),16 if m<=117 else 4)
            v['graph_relative_l2']=relative(candidate.output,expected);okay &= v['graph_relative_l2']<=.01
            native_ms=[];candidate_ms=[]
            if okay:
                for repeat in range(5):
                    if repeat%2:
                        candidate_ms.append(candidate.sample());native_ms.append(native.sample())
                    else:
                        native_ms.append(native.sample());candidate_ms.append(candidate.sample())
                v.update(candidate_ms=candidate_ms,native_ms=native_ms,graph_compatible=True)
                single=GraphMeasurement(lambda:rt.run(x,weight,scales,c,layout),1)
                v['single_replay_ms']=[single.sample() for _ in range(5)]
                v['passed']=True
                del single
            del candidate
        with (args.output/'variants.jsonl').open('a') as stream:
            stream.write(json.dumps({'layer':layer_id,'projection':projection,'M':m,**v})+'\n')
        return v
    with torch.inference_mode():
        for layer_id in ((0,) if args.phase in ('quick','tune') else (0,30,63)):
            for projection in ('gate_up_proj','down_proj'):
                layer,scheme,integer,scale=load_projection(args.model_path,layer_id,projection)
                blocked=pack_blocked(integer).npu();n,k=integer.shape
                for m in (QUICK_MS if args.phase=='quick' else TUNE_MS if args.phase=='tune' else MS):
                    x=torch.randn(m,k,dtype=torch.bfloat16,device='npu');expected=scheme.apply(layer,x);torch.npu.synchronize()
                    native=GraphMeasurement(lambda:scheme.apply(layer,x),16 if m<=117 else 4)
                    native_control=[native.sample() for _ in range(5)]
                    previous=next(r for r in old['samples'] if r['layer']==layer_id and r['M']==m and r['projection']==projection)
                    nt,kt,_=previous['tiling']
                    v1=GraphMeasurement(lambda:torch.ops.qwen_w4.nz_mm(x,layer.weight,layer.weight_scale_second,v1_stage,nt,kt),16 if m<=117 else 4)
                    v1_error=relative(v1.output,expected)
                    if v1_error>.01:raise RuntimeError('修正字节搬运后的V1对照不正确')
                    v1_ms=[v1.sample() for _ in range(5)];del v1
                    probes=[(a,x*a,scheme.apply(layer,x*a)) for a in (0,64)];torch.npu.synchronize()
                    oracle=None
                    if m in (1,12):
                        reference=(integer.reshape(n,-1,128).float()*scale.float().unsqueeze(-1)).reshape(n,k).bfloat16().float()
                        oracle=torch.nn.functional.linear(x.float().cpu(),reference)
                        del reference
                    for layout,weight in ((0,layer.weight),(1,blocked)):
                        configurations_to_test=(quick_configurations(m) if args.phase=='quick' else list(configurations(m)) if args.phase=='tune' else [Tiling(*v) for r in screen['best_configs'] if r['projection']==projection and r['layout']==layout and r['M']==bucket(m) for v in r['configurations']])
                        row={'layer':layer_id,'projection':projection,'M':m,'N':n,'K':k,'layout':layout,'variants':[],
                             'native_control_ms':native_control,'v1_ms':v1_ms,'v1_relative_l2':v1_error,
                             'v1_configuration':[nt,kt]}
                        dequant=GraphMeasurement(lambda:torch.ops.qwen_w4_v2.dequant(weight,layer.weight_scale_second,layout,rt.id)[:n*k],4)
                        row['dequant_component_ms']=[dequant.sample() for _ in range(5)]
                        del dequant
                        gc=next(c for c in configurations(m) if c.family==0)
                        gemm=GraphMeasurement(lambda:torch.ops.qwen_w4_v2.gemm(x,layer.weight_scale_second,rt.ids[gc],rt.id),16 if m<=117 else 4)
                        row['gemm_component_ms']=[gemm.sample() for _ in range(5)]
                        row['gemm_component_relative_l2']=relative(gemm.output,expected)
                        if row['gemm_component_relative_l2']>.01:raise RuntimeError('分段GEMM参考不正确')
                        del gemm
                        # Fallback is measured with inverse-layout conversion inside the graph.
                        fallback=GraphMeasurement(lambda:rt.native(x,weight,layer.weight_scale_second,layout),16 if m<=117 else 4)
                        row['fallback_ms']=[fallback.sample() for _ in range(5)]
                        row['fallback_relative_l2']=relative(fallback.output,expected)
                        row['fallback_correct']=row['fallback_relative_l2']<=.01
                        del fallback
                        if not row['fallback_correct']:raise RuntimeError('布局回退不正确')
                        for c in configurations_to_test:
                            row['variants'].append(variant(layer_id,projection,m,x,weight,layer.weight_scale_second,layout,c,expected,native,probes,oracle))
                        if args.phase in ('quick','tune') and m<=117:
                            stream=[v for v in row['variants'] if v['passed'] and v['configuration'][5]==1]
                            for v in sorted(stream,key=lambda v:statistics.median(v['candidate_ms']))[:1 if args.phase=='quick' else 2]:
                                for split in (2,4,8):
                                    c=Tiling(*v['configuration'][:4],split,1)
                                    row['variants'].append(variant(layer_id,projection,m,x,weight,layer.weight_scale_second,layout,c,expected,native,probes,oracle))
                        passed=[v for v in row['variants'] if v['passed']]
                        top=sorted(passed,key=lambda v:(statistics.median(v['candidate_ms']),v['workspace_bytes'],v['configuration']))[:2]
                        if args.phase in ('quick','tune'):best.append({'projection':projection,'M':m,'layout':layout,'configurations':[v['configuration'] for v in top]})
                        write_sample(row)
                        print(json.dumps({'complete':True,'layer':layer_id,'projection':projection,'M':m,'layout':layout,'valid_variants':len(passed)}),flush=True)
                    del native,x,expected,probes,oracle
                    torch.npu.empty_cache()
                del layer,blocked,integer
                torch.npu.empty_cache()
    qualified=[]
    if args.phase=='validate':
        groups={}
        for row in samples:
            for v in row['variants']:
                key=(row['M'],row['N'],row['K'],row['layout'],*v['configuration'])
                if v['passed']:
                    ratio=statistics.median(v['candidate_ms'])/statistics.median(v['native_ms'])
                    groups.setdefault(key,[]).append((row['layer'],ratio))
        for key,values in groups.items():
            if {x[0] for x in values}=={0,30,63} and max(x[1] for x in values)<=.9:
                qualified.append({'M':key[0],'N':key[1],'K':key[2],'layout':key[3],'configuration':key[4:],'worst_layer_time_ratio':max(x[1] for x in values)})
    result={'schema_version':2,'phase':args.phase,'correct':all(r['fallback_correct'] for r in samples),
            'run_id':uuid.uuid4().hex,'pid':os.getpid(),
            'passed_micro_gate':bool(qualified),'qualified_profiles':qualified,'samples':samples,'best_configs':best,
            'layout_version':LAYOUT_VERSION,'environment':environment(),'seconds':time.monotonic()-started,
            'raw_library_sha256':torch.ops.qwen_w4.kernel_build_id(),'host_sha256':sha256(args.host_library.read_bytes()).hexdigest(),
            'model_config_sha256':sha256((args.model_path/'config.json').read_bytes()).hexdigest(),
            'workspace_bytes':rt.workspace_bytes,'auxiliary_v1_workspace_bytes':v1_stage.numel()*2,
            'timing_schema':'grouped-16-4-v2',
            'python_source_sha256':{name:sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in SOURCE_FILES},
            'timing_method':'5 alternating paired ACL Graph event spans; 16 calls for M<=117, 4 otherwise; divided by call count. Single replay also retained. Includes device graph scheduling; not pure kernel time.'}
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
