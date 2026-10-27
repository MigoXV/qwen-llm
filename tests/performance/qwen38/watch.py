"""只读汇报实验进度，适用于长时间硬件验证。"""
import argparse
import json
from pathlib import Path
import time


def status(root):
    active=[]
    for p in root.glob('*/manifest.json'):
        try:x=json.loads(p.read_text())
        except (ValueError,OSError):continue
        if 'released' in x:continue
        active.append((p.stat().st_mtime,p.parent,x))
    if not active:
        jobs=root/'jobs.jsonl'
        if jobs.exists():
            last=json.loads(jobs.read_text().splitlines()[-1])
            return f"暂无运行实例；最近完成 {last['name']}，退出码 {last['exit_code']}。"
        return '等待下一实例启动。'
    _,folder,x=max(active,key=lambda item:item[0])
    quality=folder/'quality.jsonl'
    n=sum(1 for _ in quality.open()) if quality.exists() else 0
    rounds=sum(len(c.get('rounds',[])) for c in x.get('cases',[]))
    last=x.get('cases',[])[-1]['name'] if x.get('cases') else '模型启动/质量检查'
    text=f"实例 {folder.name}：质量请求已完成 {n} 个，性能测量已完成 {rounds} 轮，当前 {last}。"
    if folder.name == 'production-candidate' and rounds == 15 and not x.get('capacity'):
        text=f"实例 {folder.name}：普通生产CLI五轮性能已完成，正在验证四路实际满64K容量（57344输入+8192输出）。"
    expected=x.get('validation_options',{}).get('quality_samples')
    if expected is not None and not x.get('cases'):
        text=f"实例 {folder.name}：质量请求已完成 {n}/{expected+104} 个。"
    retrieval=folder/'long-quality.json'
    if retrieval.exists() and not x.get('cases'):
        try:
            q=json.loads(retrieval.read_text());text=f"实例 {folder.name}：长文正确性已完成 {q['total']}/72 请求，通过 {q['passed']}。"
        except ValueError:pass
    for name,label,expected in [('boundary-quality.json','块边界检索',72),('prefix-growth.json','四路前缀连续增长',68)]:
        path=folder/name
        if path.exists() and not x.get('cases'):
            try:
                q=json.loads(path.read_text());text=f"实例 {folder.name}：{label}已完成 {q['total']}/{expected} 请求，通过 {q['passed']}。"
            except ValueError:pass
    stable=folder/'stability.json'
    if stable.exists():
        try:
            s=json.loads(stable.read_text());errors=sum('error' in row for r in s['rounds'] for row in r['http']['samples'])
            text=f"实例 {folder.name}：长稳态已运行 {s['elapsed_seconds']/60:.1f} 分钟，完成 {len(s['rounds'])} 轮，HTTP错误 {errors}。"
        except ValueError:pass
    return text


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(status(a.root))
