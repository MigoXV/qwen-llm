"""仅在独立Worker中安装通过真实算子筛选的NZ后端。"""
import hashlib
import json
from pathlib import Path
import statistics
import math
import os
from importlib.metadata import version


def selection_rows(record, model_config_sha256):
    if record.get('schema_version')==2:
        from .ascendc_v2_candidate import validate_selection
        return validate_selection(record,model_config_sha256)
    if not record.get('correct') or not record.get('passed_micro_gate'):
        raise ValueError('算子筛选未通过')
    if record.get('model_config_sha256') != model_config_sha256:
        raise ValueError('筛选记录的模型配置不匹配')
    if len(record.get('samples',[])) != 66:
        raise ValueError('算子筛选覆盖不完整')
    expected={(layer,m,n,k) for layer in (0,30,63)
              for m in (1,3,6,12,38,117,1536,2048,6144,7680,8192)
              for n,k in ((34816,5120),(5120,17408))}
    actual={(s['layer'],s['M'],s['N'],s['K']) for s in record['samples']}
    if actual!=expected or not all(s['correct'] for s in record['samples']):
        raise ValueError('算子筛选形状或数值覆盖不完整')
    rows=[]
    for selected in record.get('nz_selection',[]):
        m,n,k=selected['M'],selected['N'],selected['K']
        tile=tuple(selected['tiling'])
        if ((n,k) not in {(34816,5120),(5120,17408)} or not 0<m<=8192
                or tile not in {(nt,kt,1) for nt in (64,128) for kt in (256,512)}
                or selected.get('kind') != 'nz'):
            raise ValueError('未支持的NZ形状或分块')
        verified=set()
        for sample in record['samples']:
            if (sample['M'],sample['N'],sample['K']) != (m,n,k):
                continue
            for variant in sample['variants']:
                if variant.get('kind')!='nz' or tuple(variant['tiling'])!=tile:
                    continue
                patterns=variant.get('patterns',[])
                timings=variant.get('candidate_ms',[])+sample.get('upstream_ms',[])
                if (not variant.get('correct') or not variant.get('graph_compatible')
                        or len(patterns)!=2 or not all(p['passed'] for p in patterns)
                        or len(timings)!=10 or not all(math.isfinite(t) and t>0 for t in timings)
                        or {p['amplitude'] for p in patterns}!={0,64}
                        or not all(math.isfinite(p['relative_l2']) and 0<=p['relative_l2']<=.01 for p in patterns)
                        or (m in (1,12) and 'fp32_oracle_relative_l2' not in variant)
                        or not math.isfinite(variant.get('fp32_oracle_relative_l2',0))
                        or not 0<=variant.get('fp32_oracle_relative_l2',0)<=.01
                        or statistics.median(variant['candidate_ms'])>.9*statistics.median(sample['upstream_ms'])):
                    raise ValueError('NZ配置未在全部验证层满足精度、图重放与速度门槛')
                verified.add(sample['layer'])
        if verified != {0,30,63}:
            raise ValueError('NZ配置缺少三个真实层的共同验证')
        row=[m,n,k,*tile[:2]]
        if row in rows:
            raise ValueError('重复的NZ形状')
        rows.append(row)
    if not rows:
        raise ValueError('没有通过筛选的原生NZ配置')
    return rows


_loaded_host=None


def install(worker):
    import torch
    import torch_npu
    from qwen_llm.engine.qwen38_worker import CachedScaleLinearMethod
    settings=worker.vllm_config.additional_config['qwen_w4_candidate']
    if json.loads(Path(settings['selection']).read_text()).get('schema_version')==2:
        from .ascendc_v2_candidate import install as install_v2
        return install_v2(worker)
    model=Path(worker.model_config.model)
    selection_path=Path(settings['selection']).resolve()
    host=Path(settings['host_library']).resolve()
    record=json.loads(selection_path.read_text())
    environment=record.get('environment',{})
    expected_packages={name:version(name) for name in ('torch','torch-npu','vllm','vllm-ascend','transformers')}
    if (environment.get('packages')!=expected_packages
            or environment.get('device')!=torch.npu.get_device_name(0)
            or environment.get('cann_root')!=str(Path(os.environ['ASCEND_HOME_PATH']).resolve())):
        raise ValueError('算子筛选环境与当前设备、CANN或依赖版本不一致')
    model_sha=hashlib.sha256((model/'config.json').read_bytes()).hexdigest()
    rows=selection_rows(record,model_sha)
    global _loaded_host
    if _loaded_host is not None and _loaded_host!=host:
        raise RuntimeError('每个Worker只允许加载一个实验构建')
    if _loaded_host is None:
        torch.ops.load_library(str(host))
        def fake(x,weight,scale,staging,selection_id):
            return x.new_empty((x.shape[0],scale.shape[1]))
        torch.library.register_fake('qwen_w4::nz_selected',fake)
        _loaded_host=host
    raw=Path(torch.ops.qwen_w4.kernel_library_path())
    if (torch.ops.qwen_w4.kernel_build_id()!=record['library_sha256']
            or hashlib.sha256(raw.read_bytes()).hexdigest()!=record['library_sha256']):
        raise ValueError('筛选结果与实际动态库不一致')
    module=worker.model_runner.model
    target=module.get_language_model() if hasattr(module,'get_language_model') else module
    parameters=list(target.parameters())
    staging=torch.empty(40*128*17408,dtype=torch.bfloat16,device=parameters[0].device)
    selection_id=torch.ops.qwen_w4.register_selection([value for row in rows for value in row])
    class SelectedScheme(CachedScaleLinearMethod):
        def apply(self,layer,x,bias=None,tp_rank=None):
            return torch.ops.qwen_w4.nz_selected(x,layer.weight,layer.weight_scale_second,staging,selection_id)
    changed=0
    for name,layer in target.named_modules():
        if not name.endswith(('.mlp.gate_up_proj','.mlp.down_proj')):
            continue
        adapter=getattr(layer,'quant_method',None)
        original=getattr(adapter,'quant_method',None)
        if not isinstance(original,CachedScaleLinearMethod):
            raise ValueError('目标FFN不是已验收的尺度缓存路径')
        if torch_npu.get_npu_format(layer.weight)!=29:
            raise ValueError('目标FFN权重没有保持NZ布局')
        scheme=SelectedScheme.__new__(SelectedScheme)
        scheme.__dict__.update(original.__dict__)
        adapter.quant_method=scheme
        changed+=1
    if changed!=128:
        raise ValueError(f'FFN接入数量不完整：{changed}')
    worker._ascendc_identity={
        'backend':'ascendc-w4a16','selected_shapes':rows,'layers':changed,
        'staging_bytes':staging.numel()*staging.element_size(),
        'library_sha256':record['library_sha256'],
        'host_sha256':hashlib.sha256(host.read_bytes()).hexdigest(),
        'selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        'fallback':'native group128 weight quant batchmatmul',
        'counter_semantics':'host dispatches only; graph replay requires device trace evidence'}
