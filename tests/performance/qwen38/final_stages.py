"""消融完成后顺序验收组合；只有已释放设备的有效原始结果可用于筛选。"""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from .acceptance import full_acceptance
from .compare import quality_gate, directory_metrics
from .matrix import run


def load(root,name,file='manifest.json'):
    return json.loads((root/name/file).read_text())


def completed(root,name):
    path=root/name/'manifest.json'
    if not path.exists():return False
    try:row=json.loads(path.read_text())
    except (ValueError,OSError):return False
    return bool(row.get('valid') and row.get('released') and row.get('port_released'))


def select(root):
    from .compare import select_batch
    baseline=load(root,'baseline-full','quality.json')
    bm=directory_metrics(root/'screen-control')
    ablations={name:quality_gate(baseline,load(root,name,'quality.json'))
               for name in ['ablation-scale-retry','ablation-head-retry','ablation-checkpoint']}
    if not all(q['passed'] for q in ablations.values()):
        raise RuntimeError('无损消融质量门槛失败，不能自动组合')
    recipes=[]
    common=['short128-c1','short128-c4','chinese8192-c1-cold','chinese8192-c4-cold']
    for recipe,name in [('gdn','quant-gdn'),('head','quant-head-retry'),('both','quant-both-retry')]:
        if not completed(root,name):
            recipes.append({'recipe':recipe,'eligible':False,'reason':'startup, execution or lifecycle failure'})
            continue
        q=quality_gate(baseline,load(root,name,'quality.json'))
        cm=directory_metrics(root/name)
        ratios={n:cm[n]['tps']/bm[n]['tps'] for n in common}
        recipes.append({'recipe':recipe,'quality':q,'ratios':ratios,
                        'score':math.prod(ratios.values())**(1/len(ratios)),
                        'eligible':q['passed'] and min(ratios.values())>=.95})
    eligible=[r for r in recipes if r['eligible']]
    best=max(eligible,key=lambda r:r['score']) if eligible else None
    recipe=best['recipe'] if best else 'base'
    features=['scale','head','checkpoint']+(['head-quant'] if recipe in {'head','both'} else [])
    model=f'qwen3.8-27b-{recipe}-w8-candidate' if best else 'qwen3.8-27b-mtp2'
    batch=select_batch(directory_metrics(root/'batch-8192'),
                       {n:directory_metrics(root/f'batch-{n}') for n in [6156,4620,3084,1548]})
    adaptive_metrics={}
    for cap in [3084,1548]:
        name=f'mixed-only-{cap}'
        if completed(root,name):adaptive_metrics[cap]=directory_metrics(root/name)
    adaptive=select_batch(directory_metrics(root/'batch-8192'),adaptive_metrics)
    fixed_best=max((r for r in batch['candidates'] if r['passed']),
                   key=lambda r:r['geometric_mean_tps_ratio'],default=None)
    adaptive_best=max((r for r in adaptive['candidates'] if r['passed']),
                      key=lambda r:r['geometric_mean_tps_ratio'],default=None)
    use_adaptive=adaptive_best is not None and (fixed_best is None or
        adaptive_best['geometric_mean_tps_ratio']>=fixed_best['geometric_mean_tps_ratio'])
    return {'selected_recipe':recipe,'selected_model':model,'optimizations':features,
            'selected_batch_tokens':8192 if use_adaptive else batch['selected_batch_tokens'],
            'selected_mixed_prefill_tokens':adaptive_best['batch_tokens'] if use_adaptive else None,
            'recipes':recipes,'lossless_ablations':ablations,'batch':batch,'adaptive_batch':adaptive,
            'status':'screened; requires full1319/long/boundary/growth/performance/stability acceptance'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();root=args.root
    deadline=time.monotonic()+6*3600
    required=['baseline-full','screen-control',*[f'batch-{n}' for n in [8192,6156,4620,3084,1548]]]
    while not all(completed(root,name) for name in required):
        if time.monotonic()>deadline:raise TimeoutError('等待消融矩阵完成超过六小时')
        jobs=root/'jobs.jsonl'
        if jobs.exists():
            for line in jobs.read_text().splitlines():
                try:job=json.loads(line)
                except ValueError:continue
                if job['name'] in required and job['exit_code']:
                    raise RuntimeError(f'前置实例失败：{job["name"]}，查看该实例日志')
        stages=(root/'stages.log').read_text()
        if 'Traceback (most recent call last)' in stages:
            raise RuntimeError('前置阶段退出失败，详见stages.log及阶段日志')
        print('WAITING '+','.join(n for n in required if not completed(root,n)),flush=True)
        time.sleep(30)
    # 纯长文保持8192，只在短提示解码与长文预填充重叠时缩块。
    for cap in [3084,1548]:
        label=f'mixed-only-{cap}'
        if not completed(root,label):
            code=run(root,label,'--suite','full','--cases',
                     'generation512-c4,chinese8192-c4-cold,mixed32k-c4',
                     '--batch-tokens','8192','--mixed-prefill-tokens',str(cap),'--no-capture')
            if code:raise RuntimeError(f'条件分块候选执行失败：{label}')
        from .compare import select_batch
        result=select_batch(directory_metrics(root/'batch-8192'),{cap:directory_metrics(root/label)})
        if result['candidates'][0]['passed']:break
    selection=select(root)
    # 单项的长请求TTFT代价可由投影计算优化抵消，必须实际测量组合。
    if selection['selected_batch_tokens']==8192 and selection['selected_mixed_prefill_tokens'] is None:
        eligible=next((r for r in selection['adaptive_batch']['candidates'] if r['batch_tokens']==3084),None)
        if (eligible and eligible['short_max_delivery_gap_ratio']<=.5
                and min(eligible['tps_ratios'].values())>=.95
                and eligible['long_cold_ttft_ratio']<=1.05):
            label=f'combination-pilot-{selection["selected_recipe"]}-mixed3084'
            if not completed(root,label):
                code=run(root,label,'--model-name',selection['selected_model'],
                    '--optimizations',','.join(selection['optimizations']),
                    '--batch-tokens','8192','--mixed-prefill-tokens','3084',
                    '--suite','full','--cases','generation512-c4,chinese8192-c4-cold,mixed32k-c4','--no-capture')
                if code:raise RuntimeError('组合条件调度试验执行失败')
            result=select_batch(directory_metrics(root/'batch-8192'),{3084:directory_metrics(root/label)})
            selection['combined_adaptive_pilot']={'name':label,**result}
            if result['candidates'][0]['passed']:
                selection['selected_mixed_prefill_tokens']=3084
    output=root/'selection.json'
    if output.exists():raise ValueError('拒绝覆盖已有筛选决策')
    output.write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    print(json.dumps(selection,ensure_ascii=False),flush=True)
    with (root/'scale-numerics-runtime.log').open('w') as log:
        subprocess.run([sys.executable,'-m','tests.performance.qwen38.verify_cached_scale',
                        '--output',str(root/'scale-numerics-runtime.json')],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    recipe=selection['selected_recipe'];budget=selection['selected_batch_tokens']
    name=f'combined-{recipe}-b{budget}'
    options=['--model-name',selection['selected_model'],'--optimizations',','.join(selection['optimizations']),
             '--batch-tokens',str(budget)]
    if selection['selected_mixed_prefill_tokens'] is not None:
        options+=['--mixed-prefill-tokens',str(selection['selected_mixed_prefill_tokens'])]
    code=run(root,name,*options,'--suite','full','--no-capture','--quality-samples','1319',
             '--quality-reference',str(root/'baseline-full/quality.json'),
             '--long-validation','--cache-boundary-validation','--prefix-growth-validation','--stability-seconds','3600')
    if code:raise RuntimeError('组合测试执行或前置质量检查失败，保留全部原始结果')
    acceptance=full_acceptance(root/'baseline-full',root/name)
    (root/'combined-acceptance.json').write_text(json.dumps(acceptance,ensure_ascii=False,indent=2))
    if not acceptance['passed']:raise RuntimeError('组合验收未通过，必须检查逐项结果')
    followups=[('independent-repeat',['--suite','full','--cases',
        'generation512-c1,generation512-c4,chinese8192-c4-cold,chinese32768-c4-hot,chinese57344-c4-hot,mixed32k-c4',
        '--no-capture','--quality-samples','0','--quality-reference',str(root/'baseline-full/quality.json')]),
        ('combined-trace',['--suite','trace','--cases','short128-c1,chinese8192-c1-cold','--capture','--record-shapes'])]
    log=(root/(name+'.log')).read_text()
    available=re.findall(r'profiled available ([0-9.]+) GiB',log)
    if available:
        expanded=min(27.5,math.floor((float(available[-1])-1.25)*2)/2)
        if expanded>22.5:
            followups.append(('expanded-kv',['--suite','full','--cases','chinese57344-c4-hot,mixed32k-c4',
                                             '--no-capture','--kv-cache-gib',str(expanded)]))
    for label,extra in followups:
        if run(root,label,*options,*extra):raise RuntimeError(f'后续验证失败：{label}')
    with (root/'production-baseline.log').open('w') as log:
        subprocess.run(['taskset','-c','0,1',sys.executable,'-m','tests.performance.qwen38.production_verify',
                        '--model','qwen3.8-27b-mtp2','--output',str(root/'production-baseline')],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print('VALIDATION COMPLETE; production promotion and canary still required',flush=True)


if __name__=='__main__':main()
