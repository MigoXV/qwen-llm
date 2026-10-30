"""只读 Linux 进程内存取证；区分私有驻留与共享映射，避免只看 VmRSS。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import typer


def memory_snapshot(pid: int) -> dict:
    root = Path('/proc') / str(pid)
    mappings = []
    for line in (root / 'smaps').read_text().splitlines():
        if re.match(r'^[0-9a-f]+-[0-9a-f]+ ', line):
            fields = line.split(maxsplit=5)
            mapping = {'address': fields[0], 'permissions': fields[1],
                       'inode': fields[4], 'path': fields[5] if len(fields) > 5 else '[anonymous]'}
            mappings.append(mapping)
        elif ':' in line:
            key, value = line.split(':', 1)
            if key in ('Size', 'Rss', 'Pss', 'Private_Clean', 'Private_Dirty',
                       'Shared_Clean', 'Shared_Dirty', 'Anonymous', 'Swap'):
                mapping[key + '_bytes'] = int(value.split()[0]) * 1024
    return {'pid': pid, 'utc': datetime.now(timezone.utc).isoformat(),
            'status': (root / 'status').read_text(),
            'smaps_rollup': (root / 'smaps_rollup').read_text(), 'mappings': mappings}


def main(pids: list[int] = typer.Argument(...), output: Path = typer.Option(...)):
    """在请求排空后的相同边界采样；读取 smaps 不计入 TPS。"""
    result = [memory_snapshot(pid) for pid in pids]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    typer.echo(f'已保存 {len(result)} 个进程的内存映射：{output}')


if __name__ == '__main__':
    typer.run(main)
