"""主程序W4快速分派不复制权重，未命中形状保持默认模式。"""
from types import SimpleNamespace
import pytest


def test_production_fast_decode_reuses_parameters_and_falls_back_by_shape(monkeypatch):
    torch=pytest.importorskip('torch');torch_npu=pytest.importorskip('torch_npu')
    from qwen_llm.engine.qwen38_worker import CachedScaleLinearMethod,FastDecodeLinearMethod,install_fast_decode
    from qwen_llm.engine import qwen38_worker as implementation
    monkeypatch.setattr(implementation,"fast_w4_op",implementation._fast_w4_impl)
    target=torch.nn.Module();target.layers=torch.nn.ModuleList()
    for _ in range(64):
        block=torch.nn.Module();block.mlp=torch.nn.Module()
        for name in ('gate_up_proj','down_proj'):
            layer=torch.nn.Module()
            layer.weight=torch.nn.Parameter(torch.zeros(1,1,dtype=torch.int32),requires_grad=False)
            layer.weight_scale_second=torch.nn.Parameter(torch.ones(1,1,dtype=torch.bfloat16),requires_grad=False)
            method=CachedScaleLinearMethod.__new__(CachedScaleLinearMethod);method.group_size=128
            layer.quant_method=SimpleNamespace(quant_method=method);setattr(block.mlp,name,layer)
        target.layers.append(block)
    weights=[m.weight for m in target.modules() if hasattr(m,'weight')]
    monkeypatch.setattr(torch_npu,'get_npu_format',lambda w:29)
    modes=[]
    monkeypatch.setattr(torch_npu,'npu_weight_quant_batchmatmul',lambda x,*a,**kw:modes.append(kw.get('inner_precise',0)) or x)
    result=install_fast_decode(target)
    assert result['layers']==128 and result['extra_weight_bytes']==0
    assert all(a is b for a,b in zip(weights,[m.weight for m in target.modules() if hasattr(m,'weight')]))
    layer=target.layers[0].mlp.down_proj
    assert isinstance(layer.quant_method.quant_method,FastDecodeLinearMethod)
    for m in (3,6,12,38):layer.quant_method.quant_method.apply(layer,torch.ones(m,1,dtype=torch.bfloat16))
    assert modes==[0,1,1,0]


def test_dynamic_compilation_keeps_the_dispatch_operator_opaque():
    torch=pytest.importorskip('torch');pytest.importorskip('torch_npu')
    from torch.fx.experimental.proxy_tensor import make_fx
    from qwen_llm.engine.qwen38_worker import fast_w4_op
    graph=make_fx(lambda x,w,s:fast_w4_op(x,w,s),tracing_mode='fake')(
        torch.zeros(38,128,dtype=torch.bfloat16),torch.zeros(128,16,dtype=torch.int32),
        torch.ones(1,128,dtype=torch.bfloat16))
    calls=[str(node.target) for node in graph.graph.nodes if node.op=='call_function']
    assert calls==['qwen_llm.fast_w4.default']
