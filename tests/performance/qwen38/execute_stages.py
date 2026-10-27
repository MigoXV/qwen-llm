"""本轮实验顺序执行，等待指定基线退出后才占用设备。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    while not (root/'migrated-baseline/analysis.json').exists():time.sleep(3)
    manifest=json.loads((root/'migrated-baseline/manifest.json').read_text())
    if not manifest.get('valid') or not manifest.get('released'):raise RuntimeError('迁移基线无效或未释放设备')
    commands=[('verify-scale',[sys.executable,'-m','tests.performance.qwen38.verify_cached_scale','--output',str(root/'scale-numerics.json')])]
    for stage in ['ablation','quantization','baseline-final','batch']:
        commands.append((stage,[sys.executable,'-m','tests.performance.qwen38.matrix',str(root),'--stage',stage]))
    for name,command in commands:
        print('START '+name,flush=True)
        with (root/(name+'-stage.log')).open('w') as log:
            code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({'stage':name,'exit_code':code}),flush=True)
        if code:raise RuntimeError(f'阶段失败：{name}')


if __name__=='__main__':main()
