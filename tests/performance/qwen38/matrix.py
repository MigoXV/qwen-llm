"""独占设备顺序执行专项消融；每个实例独立日志、原始结果和退出码。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def run(root,name,*options):
    output=root/name
    if output.exists():raise ValueError(f'拒绝覆盖已存在的实例：{output}')
    command=[sys.executable,'-m','tests.performance.qwen38.cli','profile','--output-dir',str(output),
             '--kv-cache-gib','22.5','--rounds','5',*options]
    with (root/(name+'.log')).open('w') as log:
        code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
    row={'name':name,'command':command,'exit_code':code}
    with (root/'jobs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    return code


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--stage',choices=['ablation','quantization','baseline-final','batch'],required=True)
    a=p.parse_args();root=a.root
    if a.stage=='ablation':
        for name,opts in [('scale',['scale']),('head',['head']),('cache',['cache'])]:
            cases='short128-c1,short128-c4,chinese8192-c1-cold,chinese8192-c4-cold'
            if name=='cache':cases='short128-c1,short128-c4,chinese8192-c4-hot,chinese32768-c4-hot,chinese57344-c4-hot,mixed32k-c4'
            run(root,'ablation-'+name,'--optimizations',','.join(opts),'--suite','full','--cases',cases,'--no-capture','--quality-samples','128')
    elif a.stage=='quantization':
        run(root,'ablation-checkpoint','--optimizations','checkpoint','--suite','full',
            '--cases','short128-c1,short128-c4,chinese8192-c4-hot,chinese32768-c4-hot,chinese57344-c4-hot,mixed32k-c4',
            '--no-capture','--quality-samples','128')
        for recipe in ['gdn','head','both']:
            run(root,'quant-'+recipe,'--model-name',f'qwen3.8-27b-{recipe}-w8-candidate',
                '--suite','comparison','--no-capture','--quality-samples','128')
    elif a.stage=='baseline-final':
        for recipe in ['head','both']:
            run(root,f'quant-{recipe}-retry','--model-name',f'qwen3.8-27b-{recipe}-w8-candidate',
                '--optimizations','head-quant','--suite','comparison','--no-capture','--quality-samples','128')
        run(root,'ablation-scale-retry','--optimizations','scale','--suite','comparison','--no-capture','--quality-samples','128')
        run(root,'ablation-head-retry','--optimizations','head','--suite','comparison','--no-capture','--quality-samples','128')
        run(root,'baseline-full','--suite','full','--no-capture','--quality-samples','1319','--long-validation','--cache-boundary-validation','--prefix-growth-validation')
    else:
        numerical = root/'w8-numerics.json'
        if not numerical.exists():
            with (root/'w8-numerics.log').open('w') as log:
                subprocess.run([sys.executable,'-m','tests.performance.qwen38.verify_w8_projections','--output',str(numerical)],stdout=log,stderr=subprocess.STDOUT,check=True)
        for recipe in ['head','both']:
            if not (root/f'quant-{recipe}-retry').exists():
                run(root,f'quant-{recipe}-retry','--model-name',f'qwen3.8-27b-{recipe}-w8-candidate',
                    '--optimizations','head-quant','--suite','comparison','--no-capture','--quality-samples','128')
        run(root,'screen-control','--suite','comparison','--no-capture','--quality-samples','128')
        cases='generation512-c4,chinese8192-c4-cold,mixed32k-c4'
        for budget in [8192,6156,4620,3084,1548]:
            run(root,f'batch-{budget}','--suite','full','--cases',cases,'--batch-tokens',str(budget),'--no-capture')


if __name__=='__main__':main()
