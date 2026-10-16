"""校验有符号INT4存储约定和敏感层选择，避免错误导出悄悄进入推理。"""
import pytest

torch = pytest.importorskip('torch')

from qwen_llm.quantization import pack_int4_rows, unpack_int4_rows, quantization_kind


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
