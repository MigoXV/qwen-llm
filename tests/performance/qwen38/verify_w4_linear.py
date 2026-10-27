"""用真实FFN形状验证旧版Ascend W4线性层的NZ布局与反量化结果。"""
import argparse
import json
from pathlib import Path

import torch
import torch_npu
from safetensors import safe_open

from tests.performance.qwen38.quantization import unpack_int4_rows
from vllm_ascend.quantization.methods.w4a8 import AscendW4A8DynamicLinearMethod


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('upstream', 'gmm'), default='upstream')
    parser.add_argument('--prefill-tokens', type=int, default=0)
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
        projections = ('gate_proj', 'down_proj', 'gate_up_proj') if args.backend == 'gmm' else ('gate_proj', 'down_proj')
        for projection in projections:
            prefix = f'model.language_model.layers.0.mlp.{projection}'
            parts = ('gate_proj', 'up_proj') if projection == 'gate_up_proj' else (projection,)
            loaded = []
            for part in parts:
                part_prefix = prefix.removesuffix(projection) + part
                with safe_open(root / index[part_prefix + '.weight'], framework='pt') as reader:
                    loaded.append({key.removeprefix(part_prefix + '.'): reader.get_tensor(key)
                                   for key in reader.keys() if key.startswith(part_prefix + '.')})
            values = {key: torch.cat([part[key] for part in loaded]) for key in loaded[0]}
            integer = unpack_int4_rows(values['weight']).float()
            # 模拟vLLM按模型dtype加载尺度，再按上游方式合并尺度。
            if args.backend == 'gmm':
                scale = (values['weight_scale'].float() * values['weight_scale_second'].float()).half().float()
            else:
                scale = (values['weight_scale'].bfloat16().float()
                         * values['weight_scale_second'].bfloat16().float()).bfloat16().float()
            reference_weight = (integer.reshape(integer.shape[0], -1, group)
                                * scale.unsqueeze(-1)).reshape(integer.shape)
            layer = torch.nn.Module()
            for name, value in values.items():
                if name != 'weight':
                    value = value.to(torch.float32 if name == 'scale_bias' or args.backend == 'gmm' else torch.bfloat16)
                layer.register_parameter(name, torch.nn.Parameter(value.npu(), requires_grad=False))
            scheme_type = AscendW4A8DynamicLinearMethod
            if args.backend == 'gmm':
                from tests.performance.qwen38.gmm import NativeW4A8LinearMethod
                scheme_type = NativeW4A8LinearMethod
            scheme = scheme_type.__new__(scheme_type)
            scheme.group_size, scheme.new_quant_version, scheme.tp_size = group, True, 1
            scheme.process_weights_after_loading(layer)
            weight_format = torch_npu.get_npu_format(layer.weight)
            assert weight_format == 29, f'NZ路径未启用: {weight_format}'
            for batch in (1, 4, 128, *([args.prefill_tokens] if args.prefill_tokens else [])):
                inputs = torch.randn(batch, integer.shape[1]).bfloat16()
                device_inputs = inputs.npu()
                if args.backend == 'gmm':
                    quantized, activation_scale = torch_npu.npu_dynamic_quant(device_inputs)
                    reference_inputs = quantized.float().cpu() * activation_scale.cpu().reshape(-1, 1)
                else:
                    reference_inputs = inputs.float()
                expected = torch.nn.functional.linear(reference_inputs, reference_weight)
                actual = scheme.apply(layer, device_inputs).float().cpu()
                error = float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
                row = {'projection': projection, 'shape': list(integer.shape), 'batch': batch,
                       'weight_format': weight_format, 'runtime_group': getattr(scheme, 'effective_group_size', group),
                       'relative_l2': error, 'max_abs': float((actual - expected).abs().max())}
                rows.append(row)
                print(json.dumps(row), flush=True)
                assert error < 0.01, row
            del layer, integer, reference_weight
            torch.npu.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'passed': True, 'relative_l2_limit': 0.01,
                                       'backend': args.backend, 'scheme': scheme_type.__name__, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
