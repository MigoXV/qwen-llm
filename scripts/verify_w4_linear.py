"""用真实FFN形状验证旧版Ascend W4线性层的NZ布局与反量化结果。"""
import argparse
import json
from pathlib import Path

import torch
import torch_npu
from safetensors import safe_open

from qwen_llm.quantization import unpack_int4_rows
from vllm_ascend.quantization.methods.w4a8 import AscendW4A8DynamicLinearMethod


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    torch.manual_seed(42)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    root = args.model_path
    index = json.loads((root / 'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    group = json.loads((root / 'quant_model_description.json').read_text())['group_size']
    rows = []
    with torch.inference_mode():
        for projection in ('gate_proj', 'down_proj'):
            prefix = f'model.language_model.layers.0.mlp.{projection}'
            with safe_open(root / index[prefix + '.weight'], framework='pt') as reader:
                values = {key.removeprefix(prefix + '.'): reader.get_tensor(key)
                          for key in reader.keys() if key.startswith(prefix + '.')}
            integer = unpack_int4_rows(values['weight']).float()
            # 模拟vLLM按模型dtype加载尺度，再按上游方式合并尺度。
            scale = (values['weight_scale'].bfloat16().float()
                     * values['weight_scale_second'].bfloat16().float()).bfloat16().float()
            reference_weight = (integer.reshape(integer.shape[0], -1, group)
                                * scale.unsqueeze(-1)).reshape(integer.shape)
            layer = torch.nn.Module()
            for name, value in values.items():
                if name != 'weight':
                    value = value.to(torch.float32 if name == 'scale_bias' else torch.bfloat16)
                layer.register_parameter(name, torch.nn.Parameter(value.npu(), requires_grad=False))
            scheme = AscendW4A8DynamicLinearMethod.__new__(AscendW4A8DynamicLinearMethod)
            scheme.group_size, scheme.new_quant_version, scheme.tp_size = group, True, 1
            scheme.process_weights_after_loading(layer)
            weight_format = torch_npu.get_npu_format(layer.weight)
            assert weight_format == 29, f'NZ路径未启用: {weight_format}'
            for batch in (1, 4, 128):
                inputs = torch.randn(batch, integer.shape[1]).bfloat16()
                expected = torch.nn.functional.linear(inputs.float(), reference_weight)
                actual = scheme.apply(layer, inputs.npu()).float().cpu()
                error = float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
                row = {'projection': projection, 'shape': list(integer.shape), 'batch': batch,
                       'weight_format': weight_format, 'runtime_group': group,
                       'relative_l2': error, 'max_abs': float((actual - expected).abs().max())}
                rows.append(row)
                print(json.dumps(row), flush=True)
                assert error < 0.01, row
            del layer, integer, reference_weight
            torch.npu.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'passed': True, 'relative_l2_limit': 0.01,
                                       'upstream_method_exercised': True, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
