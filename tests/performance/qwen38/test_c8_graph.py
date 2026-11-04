from contextlib import nullcontext
from types import SimpleNamespace
import pytest


def test_graph_replay_updates_actual_sequence_lengths(monkeypatch):
    pytest.importorskip('torch');pytest.importorskip('torch_npu')
    from . import c8_graph as m
    param=('c8','attention',*[object() for _ in range(9)],3072,24,4,.0625)
    params=SimpleNamespace(attn_params={1:[param]},handles={1:['handle']},
        events={1:[SimpleNamespace(record=lambda stream:None)]},workspaces={1:object()})
    monkeypatch.setattr(m,'get_graph_params',lambda:params)
    monkeypatch.setattr(m.torch.npu,'stream',lambda stream:nullcontext())
    monkeypatch.setattr(m.torch.npu,'graph_task_update_begin',lambda *args:None)
    monkeypatch.setattr(m.torch.npu,'graph_task_update_end',lambda *args:None)
    calls=[]
    monkeypatch.setattr(m.torch_npu,'npu_fused_infer_attention_score',
        SimpleNamespace(out=lambda **kw:calls.append(kw['actual_seq_lengths_kv'])))
    original=m.AscendAttentionBackendImpl.update_graph_params
    monkeypatch.setattr(m.AscendAttentionBackendImpl,'update_graph_params',staticmethod(original))
    m.install_update()
    for lengths in ([7],[4096]):
        context=SimpleNamespace(attn_metadata={'attention':SimpleNamespace(seq_lens_list=lengths,block_tables=object())})
        m.AscendAttentionBackendImpl.update_graph_params('stream',context,1,None)
    assert calls==[[7],[4096]]
