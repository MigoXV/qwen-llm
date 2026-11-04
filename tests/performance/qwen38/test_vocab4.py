import pytest


def test_packed_embedding_preserves_signed_nibbles_and_group_scales():
    torch=pytest.importorskip('torch')
    from .vocab4 import unpack_rows,lookup_rows
    q=torch.arange(-8,8,dtype=torch.int8).repeat(32).reshape(2,256)
    p=q[:,0::2].bitwise_and(15)|q[:,1::2].bitwise_and(15).bitwise_left_shift(4)
    scales=torch.tensor([[1.,2.],[.5,4.]],dtype=torch.bfloat16)
    actual=unpack_rows(p,scales)
    assert actual.dtype==torch.bfloat16
    assert torch.equal(actual,q.bfloat16()*scales.repeat_interleave(128,dim=-1))
    byte=torch.arange(-128,128,dtype=torch.int16)
    lut=torch.stack((byte.bitwise_and(15),byte.bitwise_right_shift(4).bitwise_and(15)),-1)
    lut=torch.where(lut>=8,lut-16,lut).bfloat16()
    assert torch.equal(lookup_rows(p,scales,lut),actual)
