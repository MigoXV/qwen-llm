import pytest
import torch
from .ascendc_v2 import pack_blocked,unpack_blocked,index_maps,Tiling,configurations


def test_blocked_is_lossless_and_matches_cube_nz():
    q=(torch.arange(256*384)%16-8).to(torch.int8).reshape(256,384)
    packed=pack_blocked(q)
    assert torch.equal(unpack_blocked(packed,256,384),q)
    raw=torch.stack((packed&15,packed>>4),-1).flatten().to(torch.int8)
    raw=torch.where(raw>=8,raw-16,raw)
    for n,k in ((0,0),(15,17),(63,127),(64,128),(255,383)):
        offset=((n//64)*3+k//128)*8192+((k%128)//16)*1024+(n%64)*16+k%16
        assert raw[offset]==q[n,k]


def test_index_maps_are_byte_offsets_for_each_local_element():
    maps=index_maps();nd=maps[:8192]//2
    linear=torch.arange(8192).reshape(128,64)
    desired=linear.reshape(8,16,4,16).permute(0,2,3,1).flatten()
    assert torch.equal(linear.flatten()[nd.long()],desired)
    q=(torch.arange(64*128)%16-8).to(torch.int8).reshape(64,128)
    packed=pack_blocked(q)
    words=packed.view(torch.int16).to(torch.int32)&65535
    restored=torch.empty(128,16,dtype=torch.int32)
    base=maps[-512:]//2
    for rem in range(4):
        value=sum(((words[(base+4*col).long()]>>(rem*4))&15)<<(col*4) for col in range(4))
        restored[rem::4]=value.reshape(32,16)
    expected=((q.t().to(torch.int32)&15).reshape(128,16,4)*torch.tensor([1,16,256,4096])).sum(-1)
    assert torch.equal(restored,expected)


def test_configurations_respect_resource_and_workspace_budgets():
    with pytest.raises(ValueError):Tiling(128,512,256)
    with pytest.raises(ValueError):Tiling(split=2,family=0)
    assert Tiling().workspace_bytes(12,5120,17408)==5*1024**2/2
    assert list(configurations(8192))
