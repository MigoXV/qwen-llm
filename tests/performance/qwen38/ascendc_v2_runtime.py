"""独立V2设备测试使用的权重加载、私有工作区及图计时。"""
from pathlib import Path
import json
import torch
import torch_npu
from safetensors import safe_open
from qwen_llm.engine.qwen38_worker import CachedScaleLinearMethod
from .quantization import unpack_int4_rows
from .ascendc_v2 import index_maps

DEFAULT_MODEL=Path('/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend-GDNHeadW8-v1')


def load_projection(model,layer_id,projection):
    index=json.loads((model/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    parts=('gate_proj','up_proj') if projection=='gate_up_proj' else ('down_proj',)
    loaded=[]
    for part in parts:
        prefix=f'model.language_model.layers.{layer_id}.mlp.{part}.'
        with safe_open(model/index[prefix+'weight'],framework='pt') as source:
            loaded.append({key[len(prefix):]:source.get_tensor(key) for key in source.keys() if key.startswith(prefix)})
    values={key:torch.cat([v[key] for v in loaded]) for key in loaded[0]}
    integer=unpack_int4_rows(values['weight'])
    scale=(values['weight_scale'].bfloat16().float()*values['weight_scale_second'].bfloat16().float()).bfloat16()
    layer=torch.nn.Module()
    for name,value in values.items():
        dtype=value.dtype if name=='weight' else torch.float32 if name=='scale_bias' else torch.bfloat16
        layer.register_parameter(name,torch.nn.Parameter(value.to(dtype).npu(),requires_grad=False))
    scheme=CachedScaleLinearMethod.__new__(CachedScaleLinearMethod)
    scheme.group_size,scheme.new_quant_version,scheme.tp_size=128,True,1
    scheme.process_weights_after_loading(layer)
    return layer,scheme,integer,scale


class Runtime:
    def __init__(self,library,configs,profiles=(),*,bbytes=34816*5120*2,pbytes=8*117*34816*4):
        torch.ops.load_library(str(Path(library).resolve()))
        self.configs=list(dict.fromkeys(configs))
        self.ids={value:i for i,value in enumerate(self.configs)}
        self.maps=index_maps().npu()
        self.scratch=[]
        for n,k in ((34816,5120),(5120,17408)):
            raw=torch.empty((k,n//2),dtype=torch.int8,device='npu')
            self.scratch.append(torch_npu.npu_format_cast(raw,29).view(torch.int32))
        self.id=torch.ops.qwen_w4_v2.reserve(self.maps,*self.scratch,bbytes,pbytes,
                                           [v for c in self.configs for v in c.values()],list(profiles))
        self.workspace_bytes=bbytes+max(4,pbytes)+self.maps.numel()*4+sum(t.numel()*t.element_size() for t in self.scratch)

    def run(self,x,w,s,config,layout):
        return torch.ops.qwen_w4_v2.run(x,w,s,self.ids[config],int(layout),self.id)

    def native(self,x,w,s,layout):
        return torch.ops.qwen_w4_v2.native(x,w,s,int(layout),self.id)


class GraphMeasurement:
    def __init__(self,fn,repeats):
        self.graph=torch.npu.NPUGraph();self.repeats=repeats
        torch.npu.synchronize()
        with torch.npu.graph(self.graph):
            for _ in range(repeats):self.output=fn()
        self.output.fill_(float('nan'));torch.npu.synchronize()
        self.graph.replay();torch.npu.synchronize()
        if not bool(torch.isfinite(self.output).all()):raise RuntimeError('空捕获或无效图重放')
        for _ in range(2):self.graph.replay()
        torch.npu.synchronize()

    def sample(self):
        start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
        start.record();start.synchronize();self.graph.replay();end.record();end.synchronize()
        return start.elapsed_time(end)/self.repeats


def relative(actual,expected):
    difference=actual.float()-expected.float()
    return float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
