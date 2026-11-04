"""仅安装经两次独立三层验证的V2配置；布局替换与原生回退均在Worker内。"""
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
from .ascendc_v2 import Tiling,LAYOUT_VERSION,SOURCE_FILES,pack_blocked

MS=(1,3,6,12,38,117,1536,2048,6144,7680,8192)
SHAPES=((34816,5120),(5120,17408))


def verify_sources(record):
    actual={name:sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    if record.get('python_source_sha256')!=actual:raise ValueError('V2筛选后Python源码已变更，需要重新复核')


def validate_selection(record,model_sha):
    if (record.get('schema_version')!=2 or record.get('layout_version')!=LAYOUT_VERSION
            or record.get('layout') not in (0,1) or record.get('model_config_sha256')!=model_sha
            or not record.get('passed_micro_gate')):raise ValueError('V2筛选元数据不匹配')
    runs=record.get('validation_runs',[])
    if len(runs)!=2 or len({r.get('run_id') for r in runs})!=2 or any(not r.get('run_id') for r in runs):
        raise ValueError('需要两次独立进程验证')
    expected={(l,m,n,k) for l in (0,30,63) for m in MS for n,k in SHAPES}
    for run in runs:
        samples=run.get('samples',[])
        if (run.get('phase')!='validate' or not run.get('correct') or len(samples)!=66
                or {(r['layer'],r['M'],r['N'],r['K']) for r in samples}!=expected
                or run.get('raw_library_sha256')!=record.get('raw_library_sha256')
                or run.get('host_sha256')!=record.get('host_sha256')
                or run.get('model_config_sha256')!=model_sha
                or run.get('environment')!=record.get('environment')
                or run.get('python_source_sha256')!=record.get('python_source_sha256')
                or run.get('timing_schema')!='grouped-16-4-v2'):raise ValueError('独立复核覆盖或构建不匹配')
        for r in samples:
            e=r.get('fallback_relative_l2',float('nan'))
            if r['layout']!=record['layout'] or not r.get('fallback_correct') or not math.isfinite(e) or not 0<=e<=.01:
                raise ValueError('V2原生回退验证不完整')
    profiles=record.get('profiles',[])
    if not profiles:raise ValueError('没有合格V2配置')
    seen=set()
    for profile in profiles:
        m,n,k=profile['M'],profile['N'],profile['K'];config=tuple(profile['configuration']);t=Tiling(*config)
        if (m not in MS or (n,k) not in SHAPES or (m,n,k) in seen
                or t.workspace_bytes(m,n,k)>1024**3):raise ValueError('无效或重复V2形状')
        seen.add((m,n,k))
        for run in runs:
            for layer in (0,30,63):
                row=next(r for r in run['samples'] if (r['layer'],r['M'],r['N'],r['K'])==(layer,m,n,k))
                matches=[v for v in row['variants'] if tuple(v['configuration'])==config]
                if len(matches)!=1:raise ValueError('V2配置缺少三层共同复核')
                v=matches[0];numbers=v.get('candidate_ms',[])+v.get('native_ms',[])
                patterns=v.get('patterns',[])
                if (not v.get('passed') or not v.get('graph_compatible') or len(numbers)!=10
                        or v.get('calls_per_replay')!=(16 if m<=117 else 4)
                        or not all(math.isfinite(x) and x>0 for x in numbers)
                        or not math.isfinite(v['relative_l2']) or not 0<=v['relative_l2']<=.01
                        or not math.isfinite(v['graph_relative_l2']) or not 0<=v['graph_relative_l2']<=.01
                        or len(patterns)!=2 or {p['amplitude'] for p in patterns}!={0,64}
                        or not all(p['passed'] and math.isfinite(p['relative_l2']) and 0<=p['relative_l2']<=.01 for p in patterns)
                        or (m in (1,12) and not 0<=v.get('fp32_relative_l2',float('inf'))<=.01)
                        or statistics.median(v['candidate_ms'])>.9*statistics.median(v['native_ms'])):
                    raise ValueError('V2精度、图重放或10%速度门槛未通过')
    return profiles


def install(worker):
    import torch
    import torch_npu
    from safetensors import safe_open
    from .quantization import unpack_int4_rows
    from .ascendc_v2_runtime import Runtime
    from qwen_llm.engine.qwen38_worker import CachedScaleLinearMethod
    settings=worker.vllm_config.additional_config['qwen_w4_candidate']
    path=Path(settings['selection']);host=Path(settings['host_library']);model=Path(worker.model_config.model)
    record=json.loads(path.read_text());profiles=validate_selection(record,sha256((model/'config.json').read_bytes()).hexdigest())
    verify_sources(record)
    env={'device':torch.npu.get_device_name(0),'cann_root':str(Path(os.environ['ASCEND_HOME_PATH']).resolve()),
         'packages':{n:version(n) for n in ('torch','torch-npu','vllm','vllm-ascend','transformers')},'TASK_QUEUE_ENABLE':os.environ.get('TASK_QUEUE_ENABLE','1')}
    if record['environment']!=env or sha256(host.read_bytes()).hexdigest()!=record['host_sha256']:
        raise ValueError('V2加载环境或主机库不匹配')
    configs=sorted({Tiling(*p['configuration']) for p in profiles},key=lambda c:c.values())
    configids={c:i for i,c in enumerate(configs)}
    flat=[v for p in profiles for v in (p['M'],p['N'],p['K'],record['layout'],configids[Tiling(*p['configuration'])])]
    bbytes=max((p['N']*p['K']*2 if p['configuration'][5]==0 else 20*p['configuration'][3]*p['configuration'][0]*p['configuration'][1]*2) for p in profiles)
    pbytes=max((p['M']*p['N']*p['configuration'][4]*4 if p['configuration'][4]>1 else 0) for p in profiles)
    runtime=Runtime(host,configs,flat,bbytes=bbytes,pbytes=pbytes)
    raw=Path(torch.ops.qwen_w4.kernel_library_path())
    if torch.ops.qwen_w4.kernel_build_id()!=record['raw_library_sha256'] or sha256(raw.read_bytes()).hexdigest()!=record['raw_library_sha256']:
        raise ValueError('V2原始算子构建不匹配')
    def fake(x,weight,scales,layout,workspace):return x.new_empty((x.shape[0],scales.shape[1]))
    torch.library.register_fake('qwen_w4_v2::selected',fake)
    target=worker.model_runner.model
    target=target.get_language_model() if hasattr(target,'get_language_model') else target
    converted=[];count=0
    selected_shapes={(p['N'],p['K']) for p in profiles}
    index=json.loads((model/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    for name,layer in target.named_modules():
        if not name.endswith(('.mlp.gate_up_proj','.mlp.down_proj')):continue
        adapter=layer.quant_method;original=getattr(adapter,'quant_method',None)
        if not isinstance(original,CachedScaleLinearMethod):raise ValueError('V2要求现有尺度缓存量化方法')
        k,n=layer.weight.shape[0],layer.weight_scale_second.shape[1]
        layout=record['layout'] if (n,k) in selected_shapes else 0
        if layout==1:
            layer_id=int(name.split('.layers.')[1].split('.')[0])
            parts=('gate_proj','up_proj') if name.endswith('gate_up_proj') else ('down_proj',)
            weights=[]
            for part in parts:
                key=f'model.language_model.layers.{layer_id}.mlp.{part}.weight'
                with safe_open(model/index[key],framework='pt') as source:weights.append(source.get_tensor(key))
            packed=pack_blocked(unpack_int4_rows(torch.cat(weights)))
            replacement=torch.nn.Parameter(packed.to(layer.weight.device),requires_grad=False)
            layer.weight=replacement
            converted.append(name)
            del weights,packed,replacement
        class Selected(CachedScaleLinearMethod):
            def apply(self,layer,x,bias=None,tp_rank=None):
                return torch.ops.qwen_w4_v2.selected(x,layer.weight,layer.weight_scale_second,self._layout,self._workspace)
        selected=Selected.__new__(Selected);selected.__dict__.update(original.__dict__)
        selected._layout=layout;selected._workspace=runtime.id;adapter.quant_method=selected;count+=1
    if count!=128:raise ValueError('V2未覆盖全部128个FFN投影')
    worker._ascendc_v2_runtime=runtime
    worker._ascendc_identity={'backend':'ascendc-w4a16','schema_version':2,'layout':record['layout'],
        'layout_version':LAYOUT_VERSION,'converted_layers':converted,'layers':count,
        'workspace_bytes':runtime.workspace_bytes,'profiles':profiles,'raw_library_sha256':record['raw_library_sha256'],
        'host_sha256':record['host_sha256'],'selection_sha256':sha256(path.read_bytes()).hexdigest(),
        'fallback':'inverse layout into shared original NZ scratch, then native' if record['layout'] else 'native NZ',
        'counter_names':['candidate','native','inverse_layout'],'counter_semantics':'host dispatch only; graph replay requires device trace'}
