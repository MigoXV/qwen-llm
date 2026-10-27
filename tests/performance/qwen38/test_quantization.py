"""校验有符号INT4存储约定和敏感层选择，避免错误导出悄悄进入推理。"""
import pytest

torch = pytest.importorskip('torch')

from tests.performance.qwen38.quantization import pack_int4_rows, unpack_int4_rows, quantization_kind, adapt_group_scales, encode_w4a8_scales, quantize_w4


def test_signed_int4_storage_matches_ascend_output_axis_layout():
    # 每个 nibble 覆盖有符号边界；输出行打包不同于常见的输入列打包。
    value = torch.tensor([[-8, -1, 0, 7], [7, 0, -1, -8]], dtype=torch.int8)
    packed = pack_int4_rows(value)
    assert packed.tolist() == [[120, 15, -16, -121]]
    assert torch.equal(unpack_int4_rows(packed), value)


def test_packing_rejects_silent_truncation_and_out_of_range():
    with pytest.raises(ValueError):
        pack_int4_rows(torch.zeros(3, 8, dtype=torch.int8))
    with pytest.raises(ValueError):
        pack_int4_rows(torch.full((2, 8), 8, dtype=torch.int8))


def test_runtime_scale_subdivision_preserves_dequantized_weights_exactly():
    values = torch.arange(512).reshape(2, 256).float()
    scales = torch.tensor([[0.5], [2.0]])
    converted = adapt_group_scales({'layer.weight': values,
                                    'layer.weight_scale_second': scales}, 256, 128)
    reference = values * scales
    actual = (values.reshape(2, 2, 128) * converted['layer.weight_scale_second'].unsqueeze(-1)).reshape(values.shape)
    assert torch.equal(reference, actual)
    assert converted['layer.weight'] is values


def test_native_scales_decode_to_fp16_products_and_only_merge_equal_groups():
    first = torch.tensor([[0.001234], [0.007654]])
    second = torch.tensor([[17.0, 17.0, 13.0, 13.0], [15.0, 15.0, 19.0, 19.0]])
    encoded = encode_w4a8_scales(first, second)
    decoded = encoded.to(torch.int32).view(torch.float32).squeeze(0).t()
    assert encoded.dtype == torch.int64
    assert torch.equal(decoded.repeat_interleave(2, dim=1), (first * second).half().float())
    second[0, 1] += 1
    encoded = encode_w4a8_scales(first, second)
    assert encoded.shape == (1, 4, 2)
    decoded = encoded.to(torch.int32).view(torch.float32).squeeze(0).t()
    assert torch.equal(decoded, (first * second).half().float())
    with pytest.raises(ValueError):
        encode_w4a8_scales(first, -second)


def test_quantization_preserves_recurrent_controls_and_auxiliary_towers():
    prefix = 'model.language_model.layers.0.'
    assert quantization_kind(prefix + 'mlp.down_proj.weight', 'mlp') == 'W4A8_DYNAMIC'
    assert quantization_kind(prefix + 'linear_attn.in_proj_qkv.weight', 'mlp') == 'W8A8_DYNAMIC'
    assert quantization_kind(prefix + 'linear_attn.in_proj_qkv.weight', 'all') == 'W4A8_DYNAMIC'
    for name in ['linear_attn.in_proj_a.weight', 'linear_attn.in_proj_b.weight',
                 'linear_attn.out_proj.weight', 'linear_attn.A_log', 'input_layernorm.weight']:
        assert quantization_kind(prefix + name, 'all') == 'FLOAT'
    for name in ['lm_head.weight', 'mtp.layers.0.mlp.up_proj.weight', 'model.visual.blocks.0.attn.qkv.weight']:
        assert quantization_kind(name, 'all') == 'FLOAT'


@pytest.mark.parametrize('module', ['mlp.down_proj', 'self_attn.o_proj'])
def test_row_parallel_exports_match_ascend_sixteen_part_bias_contract(module):
    weight = torch.ones(16, 256)

    def exact_quantize(value, **kwargs):
        first = torch.ones(16, 1)
        second = torch.ones(16, 2)
        return value, (first, second), value, (torch.zeros_like(first), torch.zeros_like(second))

    prefix = 'model.language_model.layers.0.' + module
    tensors, _ = quantize_w4(weight, prefix + '.weight', 128, exact_quantize)
    assert tensors[prefix + '.scale_bias'].shape == (16, 16)
    assert torch.equal(tensors[prefix + '.scale_bias'].sum(1), 8 * weight.sum(1))
