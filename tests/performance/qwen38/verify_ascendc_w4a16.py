"""使用真实FFN权重验证编译后的AscendC原型；门槛未通过即不接入模型。"""
import argparse
import json
from pathlib import Path
import time
import statistics
import itertools
import hashlib
import os
import ctypes
from importlib.metadata import version
import torch
import torch_npu
from safetensors import safe_open
from .quantization import unpack_int4_rows
from qwen_llm.engine.qwen38_worker import CachedScaleLinearMethod
from .ascendc_w4a16 import AscendCW4A16, AscendCNZW4A16, pack_k_int4
from .benchmark import inventory


def elapsed(fn, repeats=5):
    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        captured_output = fn()
    captured_output.fill_(float('nan'))
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    if not bool(torch.isfinite(captured_output).all()):
        raise RuntimeError('图重放没有产生有效输出，拒绝空捕获或无效捕获')
    values = []
    for _ in range(repeats):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        start.synchronize()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-path', type=Path, default=Path('/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend-GDNHeadW8-v1'))
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('拒绝覆盖原始结果')
    before = inventory()
    if before['owners']:
        raise RuntimeError('设备仍被占用，拒绝混用基线和算子实验')
    if os.environ.get('TASK_QUEUE_ENABLE') != '0':
        raise RuntimeError('直接runtime启动验证必须显式设置TASK_QUEUE_ENABLE=0')
    args.output.mkdir(parents=True)
    source_dir=args.output/'source'
    source_dir.mkdir()
    source_hashes={}
    for name in ('verify_ascendc_w4a16.py','ascendc_w4a16.py','quantization.py'):
        source=Path(__file__).with_name(name)
        (source_dir/name).write_bytes(source.read_bytes())
        source_hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    root = args.model_path
    if json.loads((root/'quant_model_description.json').read_text())['group_size'] != 128:
        raise ValueError('原型仅接受group128权重')
    index = json.loads((root/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.npu.set_device(0)
    runtime = ctypes.CDLL('libascendcl.so')
    timeout = runtime.aclrtSetOpExecuteTimeOut
    timeout.argtypes = [ctypes.c_uint32]
    timeout.restype = ctypes.c_int
    if timeout(60) != 0:
        raise RuntimeError('不能设置独立算子的60秒执行超时，拒绝运行同步原型')
    torch.npu.config.allow_internal_format = True
    results = []
    started = time.time()
    with torch.inference_mode():
        for layer_id in (0, 30, 63):
            for projection in ('gate_up_proj', 'down_proj'):
                parts = ('gate_proj', 'up_proj') if projection == 'gate_up_proj' else ('down_proj',)
                loaded = []
                for part in parts:
                    prefix = f'model.language_model.layers.{layer_id}.mlp.{part}.'
                    with safe_open(root/index[prefix+'weight'], framework='pt') as f:
                        loaded.append({key[len(prefix):]: f.get_tensor(key) for key in f.keys() if key.startswith(prefix)})
                values = {key: torch.cat([v[key] for v in loaded]) for key in loaded[0]}
                integer = unpack_int4_rows(values['weight'])
                n, k = integer.shape
                packed = pack_k_int4(integer).npu()
                scale_cpu = (values['weight_scale'].bfloat16().float() * values['weight_scale_second'].bfloat16().float()).bfloat16()
                reference_weight_cpu = (integer.reshape(n,-1,128).float()*scale_cpu.float().unsqueeze(-1)).reshape(n,k).bfloat16().float()
                scales = scale_cpu.contiguous().npu()
                workspace = torch.empty((n,k), dtype=torch.bfloat16, device='npu')
                layer = torch.nn.Module()
                for name, value in values.items():
                    dtype = value.dtype if name == 'weight' else torch.float32 if name == 'scale_bias' else torch.bfloat16
                    layer.register_parameter(name, torch.nn.Parameter(value.to(dtype).npu(), requires_grad=False))
                scheme = CachedScaleLinearMethod.__new__(CachedScaleLinearMethod)
                scheme.group_size, scheme.new_quant_version, scheme.tp_size = 128, True, 1
                scheme.process_weights_after_loading(layer)
                for m in (1, 3, 6, 12, 38, 117, 1536, 2048, 6144, 7680, 8192):
                    x = torch.randn(m,k, dtype=torch.bfloat16, device='npu')
                    output = torch.empty((m,n), dtype=torch.bfloat16, device='npu')
                    expected = scheme.apply(layer, x)
                    torch.npu.synchronize()
                    baseline_ms = elapsed(lambda: scheme.apply(layer,x))
                    variants = []
                    probe_inputs=[x*amplitude for amplitude in (0,64)]
                    probe_references=[scheme.apply(layer,probe) for probe in probe_inputs]
                    torch.npu.synchronize()
                    reference_cpu=(torch.nn.functional.linear(x.float().cpu(),reference_weight_cpu) if m in (1,12) else None)
                    settings = ([(*setting,'staged') for setting in itertools.product((64,128),(256,512),(1,2,4,8))]
                                + [(nt,kt,1,kind) for nt,kt in itertools.product((64,128),(256,512)) for kind in ('pipeline','nz')])
                    for configuration in settings:
                        setting,kind = configuration[:3],configuration[3]
                        print(json.dumps({'phase':'variant_begin','layer':layer_id,'projection':projection,
                                          'M':m,'N':n,'K':k,'tiling':setting,'kind':kind}),flush=True)
                        pipeline = kind != 'staged'
                        split = setting[2]
                        partial_bytes = split*m*n*4 if split > 1 else 4
                        if workspace.numel()*2 + partial_bytes > 1024**3:
                            variants.append({'tiling':setting,'pipeline':pipeline,'kind':kind,'correct':False,'skipped':True,
                                             'reason':'workspace_budget',
                                             'workspace_bytes':workspace.numel()*2+partial_bytes})
                            continue
                        partial = torch.empty((partial_bytes//4,),dtype=torch.float32,device='npu')
                        candidate = (AscendCNZW4A16(args.library,setting) if kind == 'nz' else
                                     AscendCW4A16(args.library,setting,pipeline=pipeline))
                        candidate_weight = layer.weight if kind == 'nz' else packed
                        candidate_scales = layer.weight_scale_second if kind == 'nz' else scales
                        actual = candidate(x,candidate_weight,candidate_scales,workspace,output,partial)
                        torch.npu.synchronize()
                        difference = actual.float() - expected.float()
                        relative = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
                        v = {'tiling':setting,'pipeline':pipeline,'kind':kind,'relative_l2':relative,'max_abs':float(difference.abs().max()),
                             'finite':bool(torch.isfinite(actual).all()),'workspace_bytes':workspace.numel()*2+partial_bytes}
                        v['correct'] = v['finite'] and relative <= .01
                        v['patterns']=[]
                        if v['correct'] and reference_cpu is not None:
                            oracle_error=float(torch.linalg.vector_norm(actual.float().cpu()-reference_cpu) /
                                               torch.linalg.vector_norm(reference_cpu).clamp_min(1e-12))
                            v['fp32_oracle_relative_l2']=oracle_error
                            v['correct']=oracle_error<=.01
                        if v['correct']:
                            for amplitude,probe,reference in zip((0,64),probe_inputs,probe_references):
                                got=candidate(probe,candidate_weight,candidate_scales,workspace,output,partial)
                                torch.npu.synchronize()
                                error=float(torch.linalg.vector_norm(got.float()-reference.float()) /
                                            torch.linalg.vector_norm(reference.float()).clamp_min(1e-12))
                                v['patterns'].append({'amplitude':amplitude,'relative_l2':error,
                                                      'passed':bool(torch.isfinite(got).all()) and error<=.01})
                            v['correct']=all(p['passed'] for p in v['patterns'])
                        if v['correct']:
                            try:
                                v['candidate_ms'] = elapsed(lambda: candidate(x,candidate_weight,candidate_scales,workspace,output,partial))
                                v['graph_compatible'] = True
                            except RuntimeError as exc:
                                v['graph_compatible'] = False
                                v['graph_error'] = repr(exc)
                                v['correct'] = False
                        variants.append(v)
                        with (args.output/'variants.jsonl').open('a') as stream:
                            stream.write(json.dumps({'layer':layer_id,'projection':projection,
                                                     'M':m,'N':n,'K':k,**v})+'\n')
                        del partial
                    valid = [v for v in variants if v['correct']]
                    best = min(valid,key=lambda v:statistics.median(v['candidate_ms'])) if valid else variants[0]
                    row = {'layer':layer_id,'projection':projection,'M':m,'N':n,'K':k,
                           **best,'upstream_ms':baseline_ms,'variants':variants}
                    results.append(row)
                    with (args.output/'samples.jsonl').open('a') as f:
                        f.write(json.dumps(row)+'\n')
                    print(json.dumps(row), flush=True)
                del layer, workspace, packed, scales, expected, actual, output, x, difference, reference_weight_cpu
                torch.npu.empty_cache()
    correct = len(results)==66 and all(r['correct'] for r in results)
    eligible = [r for r in results if r['correct'] and statistics.median(r['candidate_ms']) <= .9*statistics.median(r['upstream_ms'])]
    profile_rows = {}
    for row in results:
        for variant in row['variants']:
            if not variant['correct']:
                continue
            key=(row['M'],row['N'],row['K'],*variant['tiling'],variant['kind'])
            profile_rows.setdefault(key,[]).append((row['layer'],statistics.median(row['upstream_ms'])/statistics.median(variant['candidate_ms'])))
    profiles=[]
    for key,values in profile_rows.items():
        if len({v[0] for v in values})==3 and min(v[1] for v in values)>=1/.9:
            profiles.append({'M':key[0],'N':key[1],'K':key[2],'tiling':key[3:6],
                             'kind':key[6],'worst_layer_speed_ratio':min(v[1] for v in values)})
    stable_shapes=sorted({(p['M'],p['N'],p['K']) for p in profiles})
    selection=[]
    for shape in stable_shapes:
        candidates=[p for p in profiles if (p['M'],p['N'],p['K'])==shape and p['kind']=='nz']
        if candidates:
            selection.append(max(candidates,key=lambda p:p['worst_layer_speed_ratio']))
    record = {'correct':correct,'passed_micro_gate':correct and bool(stable_shapes),'samples':results,
              'stable_eligible_shapes':stable_shapes,'qualified_profiles':profiles,'nz_selection':selection,
              'library_sha256':hashlib.sha256(args.library.read_bytes()).hexdigest(),
              'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'python_source_sha256':source_hashes,
              'model_path':str(root.resolve()),
              'model_config_sha256':hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),
              'timing_method':'five ACL Graph replays measured with NPU events after warmup; includes graph dispatch gaps',
              'kernel_timeout_seconds':60,
              'environment':{'device':torch.npu.get_device_name(0),
                             'cann_root':str(Path(os.environ['ASCEND_HOME_PATH']).resolve()),
                             'packages':{name:version(name) for name in ('torch','torch-npu','vllm','vllm-ascend','transformers')}},
              'eligible_shapes':[{'M':r['M'],'N':r['N'],'K':r['K']} for r in eligible],
              'seconds':time.time()-started,'before':before,
              'implementation':'staged/Split-K and mixed AIV-AIC double-buffered group128 W4A16 candidates; no persistent BF16 weight cache',
              'limitations':'AIV/AIC exchange via GM staging; pipeline has Split-K=1, staged variant tests Split-K1/2/4/8; model integration requires passing this gate'}
    (args.output/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    raise SystemExit(0 if record['passed_micro_gate'] else 1)


if __name__ == '__main__':
    main()
