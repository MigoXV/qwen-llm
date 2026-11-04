"""汇总V2原始记录；算子门槛与服务收益分别报告，缺失证据不算通过。"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from .ascendc_acceptance import completed,drift_gate
from .ascendc_v2_candidate import MS,SHAPES


def micro_summary(first,second):
    runs=(first,second)
    if not all(r.get('phase')=='validate' and r.get('correct') for r in runs):
        raise ValueError('需要两次完整且正确的复核')
    if not first.get('run_id') or first['run_id']==second.get('run_id'):
        raise ValueError('需要独立复核进程')
    for name in ('environment','raw_library_sha256','host_sha256','model_config_sha256','python_source_sha256','timing_schema'):
        if first.get(name)!=second.get(name):raise ValueError('复核环境/代码不一致：'+name)
    expected={(l,m,n,k,layout) for l in (0,30,63) for m in MS for n,k in SHAPES for layout in (0,1)}
    groups=defaultdict(list);ablations=defaultdict(list);counts=defaultdict(int)
    for run in runs:
        rows=run['samples']
        if len(rows)!=132 or {(r['layer'],r['M'],r['N'],r['K'],r['layout']) for r in rows}!=expected:
            raise ValueError('复核没有覆盖全部132个布局/形状组合')
        for row in rows:
            if not row['fallback_correct'] or not math.isfinite(row['fallback_relative_l2']) or row['fallback_relative_l2']>.01:
                raise ValueError('原生回退不完整')
            valid=[]
            for v in row['variants']:
                counts['executed']+=1
                if not v['passed']:
                    counts['rejected_numeric_or_graph']+=1;continue
                ratio=statistics.median(v['candidate_ms'])/statistics.median(v['native_ms'])
                if not math.isfinite(ratio) or ratio<=0:raise ValueError('非法计时')
                counts['correct']+=1;valid.append((ratio,v))
                c=v['configuration'];ablations[(row['layout'],c[5],c[4])].append(ratio)
            if not valid:raise ValueError('形状没有正确候选，不能作为完整负结果')
            ratio,v=min(valid,key=lambda pair:pair[0])
            groups[(row['projection'],row['M'],row['layout'])].append({
                'layer':row['layer'],'run_id':run['run_id'],'best_time_ratio':ratio,
                'best_configuration':v['configuration'],'native_ms':statistics.median(v['native_ms']),
                'candidate_ms':statistics.median(v['candidate_ms']),
                'v1_ms':statistics.median(row['v1_ms']),
                'dequant_ms':statistics.median(row['dequant_component_ms']),
                'gemm_ms':statistics.median(row['gemm_component_ms']),
                'fallback_ms':statistics.median(row['fallback_ms'])})
    return {'independent_runs':2,'shape_layout_cases_per_run':132,'variants':dict(counts),
        'max_workspace_bytes':max(r['workspace_bytes'] for r in runs),
        'operator_comparison':[{'projection':p,'M':m,'layout':layout,
            'best_individual_time_ratio':min(v['best_time_ratio'] for v in values),
            'median_best_time_ratio':statistics.median(v['best_time_ratio'] for v in values),
            'worst_best_time_ratio':max(v['best_time_ratio'] for v in values),
            'measurements':values} for (p,m,layout),values in sorted(groups.items())],
        'ablation':[{'layout':layout,'family':family,'split':split,'count':len(values),
            'minimum_time_ratio':min(values),'median_time_ratio':statistics.median(values),
            'maximum_time_ratio':max(values)} for (layout,family,split),values in sorted(ablations.items())],
        'interpretation':'candidate/native time ratio; lower is faster. Individual best configurations are descriptive and are not a model selection. All use grouped-16-4-v2 timing.'}


def tuning_summary(tune):
    TUNE_MS=(3,6,12,38,117,1536,8192)
    expected={(p,m,layout) for p in ('gate_up_proj','down_proj') for m in TUNE_MS for layout in (0,1)}
    rows=tune['samples']
    if (tune.get('phase')!='tune' or not tune.get('correct') or len(rows)!=28
            or {(r['projection'],r['M'],r['layout']) for r in rows}!=expected):
        raise ValueError('调优覆盖不完整')
    groups=defaultdict(list);counts=defaultdict(int)
    for row in rows:
        tested={tuple(v['configuration']) for v in row['variants']}
        if {c[5] for c in tested}!={0,1} or (row['M']<=117 and {c[4] for c in tested}!={1,2,4,8}):
            raise ValueError('调优没有覆盖全部三个阶段')
        for v in row['variants']:
            counts['executed']+=1
            if not v['passed']:
                counts['rejected_numeric_or_graph']+=1;continue
            counts['correct']+=1;c=v['configuration']
            groups[(row['layout'],c[5],c[4])].append(statistics.median(v['candidate_ms'])/statistics.median(v['native_ms']))
    return {'shape_layout_cases':28,'variants':dict(counts),
        'ablation':[{'layout':layout,'family':family,'split':split,'count':len(values),
            'minimum_time_ratio':min(values),'median_time_ratio':statistics.median(values),
            'maximum_time_ratio':max(values)} for (layout,family,split),values in sorted(groups.items())],
        'scope':'layer0 tuning only; all stages attempted. Acceptance uses separate three-layer independent validations.'}


def summarize_quick(root):
    data=json.loads((root/'quick/result.json').read_text())
    expected={(p,m,layout) for p in ('gate_up_proj','down_proj') for m in (3,12,1536) for layout in (0,1)}
    samples=data['samples']
    if (data.get('phase')!='quick' or not data.get('correct') or len(samples)!=12
            or {(r['projection'],r['M'],r['layout']) for r in samples}!=expected):
        raise ValueError('短测覆盖不完整')
    rows=[];errors=[];count=0;passed=0
    for row in samples:
        valid=[v for v in row['variants'] if v['passed']]
        if not valid or not row['fallback_correct']:raise ValueError('短测缺少正确候选或回退')
        best=min(valid,key=lambda v:statistics.median(v['candidate_ms'])/statistics.median(v['native_ms']))
        count+=len(row['variants']);passed+=len(valid)
        for v in valid:
            errors += [v['relative_l2'],v['graph_relative_l2'],v.get('fp32_relative_l2',0),*[p['relative_l2'] for p in v['patterns']]]
        rows.append({'projection':row['projection'],'M':row['M'],'layout':row['layout'],'configuration':best['configuration'],
            'native_ms':statistics.median(best['native_ms']),'candidate_ms':statistics.median(best['candidate_ms']),
            'candidate_native_time_ratio':statistics.median(best['candidate_ms'])/statistics.median(best['native_ms']),
            'dequant_component_ms':statistics.median(row['dequant_component_ms']),
            'bf16_gemm_component_ms':statistics.median(row['gemm_component_ms']),
            'v1_ms':statistics.median(row['v1_ms']),'fallback_ms':statistics.median(row['fallback_ms'])})
    if not all(math.isfinite(e) and 0<=e<=.01 for e in errors):raise ValueError('短测误差不合格')
    faster=any(r['candidate_native_time_ratio']<=.9 for r in rows)
    final=root/'controller-final.json'
    result={'schema_version':2,'scope':'short operator screen only; layer0, M=3/12/1536, gate_up/down, both layouts',
        'decision':'requires_targeted_confirmation' if faster else 'stop_after_short_operator_screen',
        'shape_layout_cases':12,'configurations_executed':count,'configurations_passed':passed,
        'max_relative_l2_including_fp32_and_amplitude_probes':max(errors),'fallback_correct':True,
        'measurement_loop_seconds_including_weight_preparation':data['seconds'],'operator_comparison':rows,
        'full_validation_completed':False,'candidate_service_tested':False,'default_promoted':False,
        'long_quality_baseline':'Interrupted on user correction; partial records retained and excluded from complete-baseline claims.',
        'build':{k:data[k] for k in ('environment','host_sha256','raw_library_sha256','model_config_sha256','python_source_sha256','workspace_bytes','timing_method')},
        'raw_record':str(root/'quick/result.json'),'smoke_record':str(root/'smoke-13/result.json'),
        'service_lifecycle':json.loads(final.read_text()) if final.exists() else {'passed':False,'reason':'restoration pending'}}
    job=root/'jobs/019-quick/result.json'
    if job.exists():result['job_wall_seconds']=json.loads(job.read_text())['seconds']
    (root/'quick-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def summarize(root):
    paths=json.loads((root/'experiment.json').read_text())
    def read(name,suffix=''):
        path=Path(paths[name]);path=path if path.is_absolute() else root/path
        return json.loads((path/suffix if suffix else path).read_text())
    first=read('validation_first','result.json');second=read('validation_second','result.json')
    micro=micro_summary(first,second);tuning=tuning_summary(read('tune','result.json'))
    base=read('baseline','manifest.json');replay=read('replay','manifest.json')
    decisions=read('selection','decision.json');quality=read('baseline','quality.json')
    native_complete=(len(base['cases'])==23 and completed(base,[c['name'] for c in base['cases']])
        and quality['math_samples']==1319 and not quality['execution_errors'] and quality['instance_unchanged'])
    long_checks={}
    for name,count in (('long-quality.json',72),('boundary-quality.json',72),('prefix-growth.json',68)):
        value=read('baseline',name);long_checks[name]={'passed':value['passed'],'total':value['total']}
        native_complete &= value['passed']==value['total']==count
        if name=='prefix-growth.json':
            long_checks[name]['allocator_steps']=len(value.get('allocator_states',[]))
            native_complete &= long_checks[name]['allocator_steps']==17
    workers=base['ready']['runtime']['workers'];config=base['config']
    capacity={'passed':bool(workers) and config['max_concurrency']==4
        and config['max_input_tokens']+config['max_output_tokens']==65536
        and config['kv_cache_bytes']==int(22.5*1024**3)
        and all(w['full_context_capacity']>=4 and w['speculative_tokens']==2
            and w['graph_mode']=='FULL_DECODE_ONLY' and w['experiment']['cached_scale_layers']==128 for w in workers)
        and bool(base.get('head_equivalence')) and all(h['equal'] and h['shared_module'] for h in base['head_equivalence']),
        'workers':[{k:w[k] for k in ('full_context_capacity','speculative_tokens','graph_mode','kv_cache_allocated_bytes','npu_allocated_bytes','npu_reserved_bytes')} for w in workers],
        'head_equivalence':base.get('head_equivalence')}
    native_complete &= capacity['passed']
    has_profiles=any(d['passed_micro_gate'] for d in decisions)
    final_path=root/'controller-final.json'
    lifecycle=json.loads(final_path.read_text()) if final_path.exists() else {'passed':False,'reason':'final service verification pending'}
    result={'schema_version':2,'baseline_complete':bool(native_complete),
        'candidate_decision':'requires_service_acceptance' if has_profiles else 'rejected_no_stable_operator_speedup',
        'end_to_end_candidate_tested':False,'default_promoted':False,
        'baseline_quality':quality['categories'],'baseline_long_validation':long_checks,'baseline_capacity':capacity,
        'baseline_performance':[{'case':c['name'],'tps':statistics.median(r['tokens_per_second'] for r in c['rounds']),
            'ttft_ms':statistics.median(t for r in c['rounds'] for t in r['ttft_ms'])} for c in base['cases']],
        'micro':micro,'tuning':tuning,'selection':decisions,'drift':drift_gate(base,replay),'lifecycle':lifecycle,
        'artifacts':paths,'implementation':{'weight_layouts':['native NZ','blocked64x128-v1'],
            'stages':['group128 vector dequant + BF16 GEMM','rolling K depth2/3 with persistent FP32 L0C','parallel Split-K 1/2/4/8 + one FP32 reduction'],
            'layout_fallback':'blocked weight is inverse packed into shared original NZ scratch; conversion is included in fallback timing',
            'int4_copy_fix':'explicit packed-byte DMA; V1 controls are remeasured after the same byte-count correction',
            'catlass_modified':False,'memory_transport':'GM-mediated AIV-to-AIC B slots on Atlas A2'},
        'limitations':['单算子比值不能解释为服务TPS提升。','没有通过两次独立三层10%门槛的配置不进入模型接入。',
            'V1与原生均按同一新计时方法重测，旧0.625比值仅为历史记录。']}
    result['experiment_complete']=(native_complete and result['drift']['passed'] and lifecycle.get('passed') and not has_profiles)
    (root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# AscendC W4A16 V2 实测对照','',f"候选结论：{result['candidate_decision']}。完整性：{result['experiment_complete']}。",'',
        f"两次独立复核，每次132个形状/布局组合；合格布局数：{sum(d['passed_micro_gate'] for d in decisions)}。",
        '比值为候选耗时/原生耗时；越小越快。表中逐层最佳值仅用于描述，模型接入另需共同配置通过门槛。','',
        '| 投影 | M | 布局 | 最快单项比值 | 六次最佳比值中位数 |','|---|---:|---|---:|---:|']
    lines += [f"| {r['projection']} | {r['M']} | {r['layout']} | {r['best_individual_time_ratio']:.3f} | {r['median_best_time_ratio']:.3f} |" for r in micro['operator_comparison']]
    lines += ['',f"原生质量：{json.dumps(result['baseline_quality'],ensure_ascii=False)}。",
        f"原生漂移通过：{result['drift']['passed']}；正式服务恢复核验：{lifecycle.get('passed')}。",'',
        '完整配置、逐轮时间、分段消融、回退转换和失败尝试均保存在原始实验目录。']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--quick',action='store_true')
    args=p.parse_args();result=summarize_quick(args.root) if args.quick else summarize(args.root)
    print(json.dumps(result,ensure_ascii=False,indent=2))
