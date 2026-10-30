"""仅检查分配给目标实例的卡，允许其他卡上有独立实例。"""
import csv
import io
import subprocess
from pathlib import Path
import re


def process_resources(pid):
    root = Path('/proc')/str(pid)
    try:
        status = (root/'status').read_text()
        def value(name):
            match = re.search(rf'^{name}:\s+(\d+)', status, re.M)
            return int(match[1]) if match else None
        return {'pid': pid, 'parent_pid': value('PPid'), 'rss_bytes': (value('VmRSS') or 0)*1024,
                'open_fds': len(list((root/'fd').iterdir())), 'threads': value('Threads')}
    except OSError as exc:
        return {'pid': pid, 'error': str(exc)}


def query(kind, fields):
    raw = subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}',
                                  '--format=csv,noheader,nounits'], text=True, timeout=15)
    names = fields.split(',')
    return [dict(zip(names, [c.strip() for c in row])) for row in csv.reader(io.StringIO(raw))]


def inventory(devices, expected_pids):
    all_gpus = query('gpu', 'index,uuid,name,memory.used,memory.total,utilization.gpu,'
                     'temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,pstate')
    selected = [r for r in all_gpus if int(r['index']) in devices]
    uuids = {r['uuid'] for r in selected}
    processes = query('compute-apps', 'gpu_uuid,pid,process_name,used_gpu_memory')
    owners = [r for r in processes if r['gpu_uuid'] in uuids]
    pids = {int(r['pid']) for r in owners}
    # Worker -> Engine Core -> API：三个进程层级均记录 CPU 内存与 FD。
    process_stats = {}
    for pid in expected_pids:
        current = pid
        for _ in range(3):
            if not current or current in process_stats:break
            value = process_resources(current)
            process_stats[current] = value
            current = value.get('parent_pid')
    return {'devices': selected, 'owners': owners, 'other_devices': [r for r in all_gpus if r not in selected],
            'processes': list(process_stats.values()),
            'isolated': len(selected) == len(devices) and pids == set(expected_pids),
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
