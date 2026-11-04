"""单控制器管理独占NPU实验、不可变请求快照及最终服务恢复。"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();repo=Path(__file__).resolve().parents[3]
    queue=root/'queue';jobs=root/'jobs';queue.mkdir(exist_ok=True);jobs.mkdir(exist_ok=True)
    if (root/'controller.json').exists():raise ValueError('控制器记录已存在，请使用新实验目录')
    (root/'controller.json').write_text(json.dumps({'pid':os.getpid(),'started':time.time(),'repo':str(repo)})+'\n')
    (root/'controller-source.py').write_bytes(Path(__file__).read_bytes())
    def restore():
        subprocess.run(['systemctl','start','qwen-llm.service'],check=True)
        ready=None
        for _ in range(900):
            try:
                with urllib.request.urlopen('http://127.0.0.1:8100/health/ready',timeout=3) as response:ready=json.load(response)
                break
            except Exception:time.sleep(1)
        if ready is None:raise RuntimeError('正式服务未恢复就绪')
        (root/'final-ready.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2)+'\n')
        body={'model':'qwen3.8-27b-optimized-v1','input':'只回复数字：1+1等于几？','max_output_tokens':32,'reasoning':{'effort':'none'}}
        request=urllib.request.Request('http://127.0.0.1:8100/v1/responses',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=120) as response:value=json.load(response)
        (root/'final-response.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        text=''.join(c.get('text','') for item in value.get('output',[]) for c in item.get('content',[]))
        npu=subprocess.check_output(['npu-smi','info'],text=True);(root/'final-npu.txt').write_text(npu)
        owners=re.findall(r'\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|',npu)
        with socket.socket() as sock:released=sock.connect_ex(('127.0.0.1',18100))!=0
        passed=(value.get('status')=='completed' and text.strip()=='2' and value['usage']['output_tokens']>0
                and len(owners)==1 and 'qwen-llm.service' in Path('/proc/'+owners[0][0]+'/cgroup').read_text() and released)
        (root/'controller-final.json').write_text(json.dumps({'passed':passed,'owners':owners,'test_port_released':released,'text':text},ensure_ascii=False,indent=2)+'\n')
        if not passed:raise RuntimeError('服务恢复后的真实推理或设备隔离核验失败')
    active=None
    def stopped(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stopped)
    try:
        subprocess.run(['systemctl','stop','qwen-llm.service'],check=True);time.sleep(5)
        seen=set();idle=time.monotonic()
        while not (root/'finish.json').exists():
            pending=[x for x in sorted(queue.glob('*.json')) if x.name not in seen]
            if not pending:
                if time.monotonic()-idle>1800:raise RuntimeError('控制器闲置30分钟，自动恢复服务')
                time.sleep(1);continue
            for path in pending:
                idle=time.monotonic();job=json.loads(path.read_text());name=path.stem
                if not re.fullmatch(r'[a-zA-Z0-9_-]+',name):raise ValueError('任务名非法')
                destination=jobs/name;destination.mkdir()
                shutil.copy2(path,destination/'request.json')
                shutil.copytree(repo/'tests/performance/qwen38',destination/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
                (destination/'revision.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True))
                (destination/'tracked.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=repo))
                env=os.environ.copy();env.update(job.get('env',{}))
                with (destination/'stdout.log').open('w') as log:
                    active=subprocess.Popen(['/bin/bash','-c','source scripts/env_ascend_291.sh; exec "$@"','experiment',*job['argv']],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    started=time.monotonic();timed_out=False
                    progress=Path(job['progress']) if job.get('progress') else destination/'stdout.log'
                    while active.poll() is None:
                        now=time.monotonic();age=time.time()-progress.stat().st_mtime if progress.exists() else now-started
                        if now-started>job.get('timeout',7200) or (now-started>300 and age>90):
                            timed_out=True;os.killpg(active.pid,signal.SIGTERM)
                            try:active.wait(timeout=5)
                            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
                            break
                        time.sleep(1)
                    result={'exit_code':active.returncode,'timed_out':timed_out,'seconds':time.monotonic()-started}
                (destination/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(name,result,flush=True)
                active=None;seen.add(path.name);idle=time.monotonic()
                time.sleep(5)
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
        restore()


if __name__=='__main__':main()
