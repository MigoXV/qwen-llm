import pytest


def test_mtp_queries_only_attend_their_own_causal_prefix():
    pytest.importorskip('torch');pytest.importorskip('torch_npu')
    from .c8_mtp_graph import causal_rows
    owners,lengths=causal_rows([2,5],[10,12],6)
    assert owners==[0,0,1,1,1,0] and lengths==[9,10,10,11,12,1]
    owners,lengths=causal_rows([0,1,4,6],[0,20,30,40],12)
    assert owners[:6]==[1,2,2,2,3,3] and lengths[:6]==[20,28,29,30,39,40]
    with pytest.raises(ValueError):causal_rows([4],[10],3)


def test_expanded_table_copies_only_columns_visible_to_queries():
    torch=pytest.importorskip('torch');pytest.importorskip('torch_npu')
    from .c8_mtp_graph import refresh_table
    source=torch.arange(4*2064,dtype=torch.int32).reshape(4,2064)
    table=torch.full((12,2096),-1,dtype=torch.int32)
    owners=[r for r in range(4) for _ in range(3)]
    refresh_table(table,source,owners,[129]*12,128)
    assert torch.equal(table[:,:2],source[owners,:2])
    assert (table[:,2:]==-1).all()


def test_shared_expansion_reuses_layers_but_refreshes_each_step(monkeypatch):
    torch=pytest.importorskip('torch');pytest.importorskip('torch_npu')
    from . import c8_mtp_graph as m
    monkeypatch.setattr(m,'sharing',True);monkeypatch.setattr(m,'row_indices',{})
    source=torch.tensor([[1,2],[3,4]],dtype=torch.int32)
    table=torch.zeros((4,2),dtype=torch.int32);owners=[0,0,1,1];lengths=[2]*4
    seen={}
    m.refresh_once(table,source,owners,lengths,1,seen)
    m.refresh_once(table,source,owners,lengths,1,seen)
    assert torch.equal(table,source[owners])
    source.add_(10)
    m.refresh_once(table,source,owners,lengths,1,{})
    assert torch.equal(table,source[owners])
    with pytest.raises(ValueError):m.refresh_once(table,source,owners,[1]*4,1,seen)
