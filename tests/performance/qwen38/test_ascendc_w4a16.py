import pytest


def test_k_axis_repack_preserves_signed_values_and_row_order():
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_npu')
    from .ascendc_w4a16 import pack_k_int4
    original = torch.arange(4*128).remainder(16).sub(8).to(torch.int8).reshape(4,128)
    packed = pack_k_int4(original)
    low = (packed & 15).to(torch.int8)
    high = (packed >> 4).to(torch.int8)
    low[low >= 8] -= 16
    high[high >= 8] -= 16
    restored = torch.stack((low,high),dim=-1).reshape_as(original)
    assert torch.equal(original, restored)
    with pytest.raises(ValueError):
        pack_k_int4(torch.full((2,128),8,dtype=torch.int8))
    with pytest.raises(ValueError):
        pack_k_int4(torch.zeros((2,127),dtype=torch.int8))
