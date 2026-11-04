"""独立设备检查：框架异步队列、原生回退和图重放。不是模型接入或速度筛选。"""
import ctypes
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import torch
import torch_npu
from safetensors import safe_open
from qwen_llm.engine.qwen38_worker import CachedScaleLinearMethod
from tests.performance.qwen38.benchmark import inventory

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, default=Path('/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend-GDNHeadW8-v1'))
    args = parser.parse_args()
    output = args.output
    faulthandler.dump_traceback_later(90)
    host = args.host_library.resolve()
    assert host.is_file(), '桥动态库不存在'
    assert not output.exists(), '拒绝覆盖设备检查结果'
    assert os.environ.get('TASK_QUEUE_ENABLE') == '1'
    assert not inventory()['owners'], 'NPU仍被占用'
    output.mkdir()
    (output / 'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    runtime = ctypes.CDLL('libascendcl.so')
    timeout = runtime.aclrtSetOpExecuteTimeOut
    timeout.argtypes = [ctypes.c_uint32]
    timeout.restype = ctypes.c_int
    assert timeout(60) == 0
    torch.ops.load_library(str(host))
    # 仅注册数值测试形状；不生成服务性能筛选记录。
    table = [v for (n, k) in ((34816, 5120), (5120, 17408)) for m in (1, 2048) for v in (m, n, k, 64, 256)]
    assert torch.ops.qwen_w4.register_selection(table) == 0
    staging = torch.empty(40 * 128 * 17408, dtype=torch.bfloat16, device='npu')
    model = args.model_path
    index = json.loads((model / 'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    rows = []
    with torch.inference_mode():
        for projection in ('gate_up_proj', 'down_proj'):
            parts = ('gate_proj', 'up_proj') if projection == 'gate_up_proj' else ('down_proj',)
            loaded = []
            for part in parts:
                prefix = 'model.language_model.layers.0.mlp.' + part + '.'
                with safe_open(model / index[prefix + 'weight'], framework='pt') as f:
                    loaded.append({key[len(prefix):]: f.get_tensor(key) for key in f.keys() if key.startswith(prefix)})
            values = {key: torch.cat([v[key] for v in loaded]) for key in loaded[0]}
            layer = torch.nn.Module()
            for (name, value) in values.items():
                dtype = value.dtype if name == 'weight' else torch.float32 if name == 'scale_bias' else torch.bfloat16
                layer.register_parameter(name, torch.nn.Parameter(value.to(dtype).npu(), requires_grad=False))
            scheme = CachedScaleLinearMethod.__new__(CachedScaleLinearMethod)
            (scheme.group_size, scheme.new_quant_version, scheme.tp_size) = (128, True, 1)
            scheme.process_weights_after_loading(layer)
            k = layer.weight.shape[0]
            for m in (1, 3, 2048):
                x = torch.randn(m, k, dtype=torch.bfloat16, device='npu')
                expected = scheme.apply(layer, x)
                for mode in ('direct', 'selected'):

                    def run():
                        if mode == 'direct':
                            return torch.ops.qwen_w4.nz_mm(x, layer.weight, layer.weight_scale_second, staging, 64, 256)
                        return torch.ops.qwen_w4.nz_selected(x, layer.weight, layer.weight_scale_second, staging, 0)
                    actual = run()
                    torch.npu.synchronize()
                    relative = float(torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
                    assert bool(torch.isfinite(actual).all()) and relative <= 0.01
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        captured = run()
                    captured.fill_(float('nan'))
                    torch.npu.synchronize()
                    before = torch.ops.qwen_w4.dispatch_counts()
                    for _ in range(3):
                        graph.replay()
                    torch.npu.synchronize()
                    after = torch.ops.qwen_w4.dispatch_counts()
                    error = float(torch.linalg.vector_norm(captured.float() - expected.float()) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
                    passed = bool(torch.isfinite(captured).all()) and error <= 0.01 and (before == after)
                    row = {'projection': projection, 'M': m, 'mode': mode, 'route': 'native_fallback' if mode == 'selected' and m == 3 else 'nz_candidate', 'relative_l2': relative, 'graph_relative_l2': error, 'replays': 3, 'host_dispatches_before': before, 'host_dispatches_after': after, 'passed': passed}
                    rows.append(row)
                    (output / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
                    print(json.dumps(row), flush=True)
                    assert passed
                    del graph, captured, actual
            del layer, expected, x
            torch.npu.empty_cache()
    counts = torch.ops.qwen_w4.dispatch_counts()
    assert counts[0] > 0 and counts[1] > 0
    record = {'passed': len(rows) == 12 and all((r['passed'] for r in rows)), 'TASK_QUEUE_ENABLE': os.environ['TASK_QUEUE_ENABLE'], 'cases': rows, 'host_dispatch_counts': counts, 'host_sha256': hashlib.sha256(host.read_bytes()).hexdigest(), 'raw_sha256': torch.ops.qwen_w4.kernel_build_id(), 'model_integrated': False, 'speed_tested': False}
    faulthandler.cancel_dump_traceback_later()
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
if __name__ == '__main__':
    main()
