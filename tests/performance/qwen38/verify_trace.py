"""离线核对FFN覆盖和尺度Cast；缺少shape记录不能视为Cast已消除。"""
import argparse
import hashlib
import json
from pathlib import Path
import re


CASES = ('short128-c1', 'chinese8192-c1-cold')
SCALE_SHAPES = {'40,34816', '136,5120'}


def shape(value):
    return ''.join(c for c in value if c not in ' \"[]')


def trace_counts(trace):
    casts = [row for row in trace['top_operators'] if 'cast' in row['name'].lower()]
    shaped = [row for row in trace['kernel_shapes'] if 'cast' in row['name'].lower()]
    scales = [row for row in shaped if shape(row['input_shapes']) in SCALE_SHAPES]
    coverage = trace.get('coverage', {})
    return {
        'path': trace['path'],
        'valid_device_trace': trace['valid_device_trace'],
        'target_forwards': coverage.get('target_forwards', 0),
        'ffn_coverage_complete': bool(coverage.get('complete') and coverage.get('w4_layers') == 128
                                      and coverage.get('expected_matmuls', 0) > 0
                                      and coverage.get('expected_matmuls') == coverage.get('observed_matmuls')),
        'all_cast_calls': sum(row['calls'] for row in casts),
        'shaped_cast_calls': sum(row['calls'] for row in shaped),
        'unknown_shape_cast_calls': sum(row['calls'] for row in shaped
                                        if not re.search(r'\d', row['input_shapes'])),
        'scale_cast_calls': sum(row['calls'] for row in scales),
        'scale_cast_by_phase': {
            phase: sum(row['calls'] for row in scales if row['phase'] == phase)
            for phase in {row['phase'] for row in scales}
        },
    }


def verify_pair(baseline, candidate):
    old, new = trace_counts(baseline), trace_counts(candidate)
    complete = all(row['valid_device_trace'] and row['ffn_coverage_complete']
                   and row['unknown_shape_cast_calls'] == 0
                   and row['all_cast_calls'] == row['shaped_cast_calls'] for row in (old, new))
    passed = (complete and old['scale_cast_calls'] == 128 * old['target_forwards']
              and new['scale_cast_calls'] == 0)
    return {'passed': passed, 'baseline': old, 'candidate': new,
            'note': 'Trace validates operator removal only; throughput comes from profiler-off rounds.'}


def verify(baseline_directory, candidate_directory):
    baseline_directory, candidate_directory = Path(baseline_directory), Path(candidate_directory)
    inputs = [baseline_directory / 'analysis.json', candidate_directory / 'analysis.json']
    old, new = [json.loads(path.read_text()) for path in inputs]
    manifests = [json.loads((directory / 'manifest.json').read_text())
                 for directory in (baseline_directory, candidate_directory)]
    rows = []
    for case in CASES:
        select = lambda data: [t for t in data['traces']
                               if Path(t['path']).parents[1].name.startswith(case + '-trace_')]
        left, right = select(old), select(new)
        if len(left) != 1 or len(right) != 1:
            rows.append({'case': case, 'passed': False, 'reason': 'missing or ambiguous trace'})
        else:
            rows.append({'case': case, **verify_pair(left[0], right[0])})
    # 早期热点采集尚未在每个manifest内记录端口；使用当时保存的整轮释放证据。
    legacy_path = baseline_directory.parents[1] / 'lifecycle-final.json'
    legacy = None
    old_port_closed = manifests[0].get('port_released')
    if old_port_closed is None and legacy_path.exists():
        legacy = json.loads(legacy_path.read_text())
        old_port_closed = (legacy.get('valid') and legacy.get('ports_closed', {}).get('18100') is True
                           and legacy.get('npu_inventory', {}).get('owners') == []
                           and legacy.get('profile_processes') == [])
        inputs.append(legacy_path)
    lifecycle = (all(m.get('valid') and m.get('released') for m in manifests)
                 and old_port_closed and manifests[1].get('port_released'))
    return {'passed': lifecycle and all(row['passed'] for row in rows), 'lifecycle_valid': lifecycle,
            'cases': rows, 'analysis_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('拒绝覆盖trace验收记录')
    result = verify(args.baseline, args.candidate)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
