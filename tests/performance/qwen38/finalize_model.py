"""完整验收、复测和trace确认后，标记独立仓库；只写新仓库的元数据。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .acceptance import full_acceptance
from .export_candidates import sha256
from .runtime_candidate import ADAPTATION, MANIFEST, MODEL_FILES, PACKAGES


def read(path):
    return json.loads(path.read_text())


def finalize(root):
    root=Path(root).resolve();selection=read(root/'selection.json')
    name=f'combined-{selection["selected_recipe"]}-b{selection["selected_batch_tokens"]}'
    directory=root/name;manifest=read(directory/'manifest.json')
    acceptance=full_acceptance(root/'baseline-full',directory)
    if not acceptance['passed']:raise ValueError('完整组合尚未通过验收')
    repeat=read(root/'independent-repeat/manifest.json')
    if not all(repeat.get(k) for k in ['valid','released','port_released']):
        raise ValueError('独立新进程复测失败或未释放设备')
    if not read(root/'independent-acceptance.json')['passed']:
        raise ValueError('独立复测的质量、性能或实际缓存命中未通过门槛')
    baseline_cli=read(root/'production-baseline/manifest.json')
    if not all(baseline_cli.get(k) for k in ['valid','released','port_released']):
        raise ValueError('普通生产CLI的基线尚未验证完成')
    trace=read(root/'trace-acceptance.json')
    if not trace['passed']:raise ValueError('FFN尺度Cast消除尚未取得trace证据')
    integrity=read(root/'source-integrity-final.json')
    if not integrity['passed']:raise ValueError('原始模型完整性校验未通过')
    if not read(root/'scale-numerics-runtime.json')['passed']:
        raise ValueError('最终尺度执行方法数值检查未通过')
    if not read(root/'w8-numerics.json')['passed']:
        raise ValueError('W8投影算子数值检查未通过')
    config=manifest['config'];model=Path(config['model_path']).resolve()
    protected=[Path('/home/model-bin/Qwen/Qwen3.8-27B').resolve(),
               Path('/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend').resolve()]
    if any(model==p or p in model.parents or model in p.parents for p in protected):
        raise ValueError('禁止在原始或现有量化仓库中写验收元数据')
    if config.get('mixed_prefill_tokens') is not None:
        raise ValueError('本次发布不含未通过平衡门槛的混合分块策略')
    if (model/MANIFEST).exists():raise ValueError('专项验收清单已存在，拒绝覆盖')
    if (model/'.quantization.incomplete').exists():raise ValueError('新仓库尚未完成导出')
    quant=read(model/'quantization_manifest.json')
    if not quant['export_complete'] or quant['recipe']!=selection['selected_recipe']:
        raise ValueError('选中仓库的导出配方与最终验收不一致')
    files=[]
    for row in quant['files']:
        path=model/row['path'];baseline=protected[1]/row['path']
        independent=(not path.is_symlink() and path.stat().st_nlink==1
                     and (not baseline.exists() or (path.stat().st_dev,path.stat().st_ino)
                          != (baseline.stat().st_dev,baseline.stat().st_ino)))
        if not independent:raise ValueError('新仓库分片不是独立的普通文件')
        actual=sha256(path)
        files.append({'path':row['path'],'sha256':actual,'matched':actual==row['sha256'],
                      'independent_regular_file':independent})
    if not all(row['matched'] for row in files):raise ValueError('新仓库的权重分片已改变')
    (root/'selected-model-integrity.json').write_text(json.dumps({'passed':True,'files':files},indent=2))
    (root/'selected-model-quantization-manifest-before-finalization.json').write_text(
        json.dumps(quant,ensure_ascii=False,indent=2))
    software=manifest['ready']['runtime']['packages']
    verified={
        'schema_version':1,'adaptation':ADAPTATION,'recipe':selection['selected_recipe'],
        'validated_for_inference':True,'created_at':datetime.now(timezone.utc).isoformat(),
        'software':{name:software[name] for name in PACKAGES},
        'cann_root':manifest['ready']['runtime']['cann_root'],
        'hardware':'Ascend910B4-1; one physical card4, visible logical0',
        'model_family':'Qwen3.8-27B dense;64 layers/5120 hidden/248320 vocabulary',
        'model_files':{name:sha256(model/name) for name in MODEL_FILES},
        'weight_files':files,
        'serving':{'batch_tokens':[config['batch_tokens']],
                   'kv_cache_bytes':[config['kv_cache_bytes']],
                   'gpu_memory_utilization':config['gpu_memory_utilization'],
                   'max_input_tokens':57344,'max_output_tokens':8192,'max_concurrency':4,
                   'speculative_tokens':2,'prefix_caching':True,'async_scheduling':True,
                   'npugraph_ex':True,'graph_mode':'FULL_DECODE_ONLY','dtype':'bfloat16'},
        'evidence':{'root':str(root),'complete_candidate':name,
                    'quality':acceptance['quality'],
                    'long_correctness':acceptance['long_correctness'],
                    'stability':acceptance['stability'],
                    'independent_repeat':'independent-repeat',
                    'trace_acceptance_sha256':sha256(root/'trace-acceptance.json'),
                    'source_integrity_sha256':sha256(root/'source-integrity-final.json'),
                    'full_performance_coverage':acceptance['full_performance_coverage'],
                    'all_core_regressions_within_limit':all(r['within_regression_limit'] for r in acceptance['performance'])},
        'limitations':['Compared with the existing mixed-quantization model, not a full BF16 model.',
                       'Conditional smaller prefill chunks failed the5% long-TTFT gate and are excluded.',
                       'Quantized projections have numerical error; quality claims apply to recorded evaluations.'],
    }
    quant['validated_for_inference']=True
    quant['validation_evidence']=str(root/'combined-acceptance.json')
    pending=model/'quantization_manifest.json.part';pending.write_text(json.dumps(quant,ensure_ascii=False,indent=2)+'\n')
    pending.replace(model/'quantization_manifest.json')
    pending=model/(MANIFEST+'.part');pending.write_text(json.dumps(verified,ensure_ascii=False,indent=2)+'\n')
    pending.replace(model/MANIFEST)
    quality=read(directory/'quality.json')['categories']
    (model/'README.md').write_text(
        '# Qwen3.8-27B / Ascend910B4 专项模型\n\n'
        f'独立来源和逐分片SHA256见`quantization_manifest.json`。原始模型仓库保持不变。\n\n'
        f'实际验收结果：GSM8K {quality["gsm8k"]["passed"]}/1319，项目代码题{quality["code"]["passed"]}/64；'
        f'长文、块边界及四路前缀增长结果见`{MANIFEST}`。\n\n'
        f'至少一小时稳定性、23场景五轮性能和独立复测结果位于`{root}`。'
        '默认采用22.5GiB KV、8192分块、MTP2和FULL_DECODE_ONLY。\n\n'
        '普通生产CLI的独立启动、HTTP、性能和四路完整64K容量结果另存报告根目录；本清单不替代该验收。\n\n'
        '使用本项目已验证的专用执行适配加载；原生Qwen构造器未传递输出头量化配置。\n')
    return verified


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    result=finalize(args.root)
    print(json.dumps({k:result[k] for k in ['adaptation','recipe','validated_for_inference','model_family']},ensure_ascii=False))


if __name__=='__main__':main()
