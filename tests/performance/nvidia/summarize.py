"""从原始 HTTP 记录计算表格，避免手工抄写 TPS。"""
import json
from pathlib import Path
import statistics

import typer

from tests.performance.qwen38.compare import quality_gate

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main():pass


def summarize_rounds(rounds):
    groups = {}
    for row in rounds:
        key = (row.get('input_tokens'), row['concurrency'], row.get('position'))
        groups.setdefault(key, []).append(row)
    result = []
    for (length, concurrency, position), rows in groups.items():
        samples = [s for row in rows for s in row['samples']]
        okay = [s for s in samples if s.get('http_status') == 200]
        median = lambda values: statistics.median(values) if values else None
        result.append({'input_tokens': length, 'concurrency': concurrency, 'position': position,
            'rounds': len(rows), 'requests': len(samples), 'errors': len(samples)-len(okay),
            'aggregate_tps_median': median([r['tokens_per_second'] for r in rows]),
            'aggregate_tps_min': min(r['tokens_per_second'] for r in rows),
            'aggregate_tps_max': max(r['tokens_per_second'] for r in rows),
            'ttft_ms_median': median([s['ttft_seconds']*1000 for s in okay if s.get('ttft_seconds') is not None]),
            'decode_tps_median': median([s['decode_tokens_per_second'] for s in okay if s.get('decode_tokens_per_second') is not None]),
            'actual_output_tokens': sorted({s['response']['usage']['output_tokens'] for s in okay}),
            'preemptions': sum(r.get('preemptions_delta',0) for r in rows)
                if all('preemptions_delta' in r for r in rows) else None,
            'retrieval_passed': sum(s.get('retrieval_passed') is True for s in okay)
                if any(s.get('expected') for s in okay) else None})
    return result


@app.command()
def performance(source: Path, output: Path = typer.Option(...), last_rounds: int = typer.Option(0, min=0)):
    if source.is_dir():
        manifest = json.loads((source/'manifest.json').read_text())
        rows = [json.loads((source/r['file']).read_text()) for r in manifest['rounds'] if 'file' in r]
    else:
        manifest = json.loads(source.read_text())
        rows = manifest['rounds']
    complete = len(rows) == len(manifest['rounds'])
    if not rows:raise ValueError('没有测量轮次，不能生成性能总结')
    if last_rounds:
        groups = {}
        for row in rows:
            groups.setdefault((row.get('input_tokens'),row['concurrency'],row.get('position')),[]).append(row)
        rows = [row for group in groups.values() for row in group[-last_rounds:]]
    record = {'source': str(source.resolve()), 'service': manifest['service'],
              'valid': complete and manifest.get('valid',manifest.get('instance_unchanged',False)),
              'rows': summarize_rounds(rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    typer.echo(json.dumps(record['rows'], ensure_ascii=False, indent=2))


@app.command()
def quality(baseline: Path, candidate: Path, output: Path = typer.Option(...)):
    result = quality_gate(json.loads(baseline.read_text()), json.loads(candidate.read_text()), require_full=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    typer.echo(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:raise typer.Exit(1)


if __name__ == '__main__':app()
