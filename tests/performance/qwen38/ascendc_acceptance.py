"""AscendC 实验的双目标门槛与独立基线漂移检查；输入来自原始manifest。"""
import math

from .compare import metrics


GENERATION = tuple(f'generation512-c{c}' for c in (1, 2, 4))
CORE = (*GENERATION, 'chinese8192-c1-cold', 'chinese8192-c4-cold', 'mixed32k-c4')


def ratios(baseline, candidate, names):
    old, new = metrics(baseline), metrics(candidate)
    if any(name not in old or name not in new for name in names):
        raise ValueError('缺少必需的对照场景')
    rows = []
    for name in names:
        long_input = name.startswith(('chinese', 'repeated')) or name == 'mixed32k-c4'
        key = 'long_ttft_ms' if name == 'mixed32k-c4' else 'ttft_ms'
        values = (old[name]['tps'], new[name]['tps'], old[name][key], new[name][key])
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('对照指标必须为有限正数')
        rows.append({'case': name, 'tps_ratio': values[1] / values[0],
                     'ttft_ratio': values[3] / values[2], 'long_input': long_input})
    return rows


def completed(manifest, names):
    cases = {c['name']: c for c in manifest['cases']}
    return (all(manifest.get(k) for k in ('valid', 'released', 'port_released'))
            and all(name in cases and len(cases[name]['rounds']) >= 5
                    and all(r['valid'] for r in cases[name]['rounds']) for name in names))


def balanced_gate(baseline, candidate, *, full=False):
    names = tuple(c['name'] for c in baseline['cases']) if full else CORE
    rows = ratios(baseline, candidate, names)
    gen = [r['tps_ratio'] for r in rows if r['case'] in GENERATION]
    cold = [r['ttft_ratio'] for r in rows if r['case'].endswith('-cold')
            or r['case'] == 'mixed32k-c4']
    if len(gen) != 3 or not cold:
        raise ValueError('缺少生成吞吐或冷首字延迟目标')
    generation = math.exp(sum(map(math.log, gen)) / len(gen))
    ttft = math.exp(sum(map(math.log, cold)) / len(cold))
    regressions = [r['case'] for r in rows if r['tps_ratio'] < .95
                   or (r['long_input'] and r['ttft_ratio'] > 1.05)]
    coverage = completed(baseline, names) and completed(candidate, names)
    return {'passed': coverage and not regressions and generation >= 1.05 and ttft <= .95,
            'coverage_complete': coverage, 'generation_tps_geomean_ratio': generation,
            'cold_ttft_geomean_ratio': ttft, 'regression_cases': regressions,
            'performance': rows,
            'scope': 'performance only; quality, capacity, HBM, stability and drift require separate acceptance'}


def drift_gate(baseline, replay):
    rows = ratios(baseline, replay, CORE)
    drifting = [r['case'] for r in rows if abs(r['tps_ratio'] - 1) > .05
                or (r['long_input'] and abs(r['ttft_ratio'] - 1) > .05)]
    coverage = completed(baseline, CORE) and completed(replay, CORE)
    short_ttft = [r['case'] for r in rows if not r['long_input']
                  and abs(r['ttft_ratio'] - 1) > .05]
    return {'passed': coverage and not drifting, 'coverage_complete': coverage,
            'drifting_cases': drifting, 'performance': rows,
            'short_prompt_ttft_drift_cases': short_ttft,
            'all_reported_metrics_within_5_percent': not drifting and not short_ttft,
            'metric_policy': 'TPS for all core cases and TTFT for long inputs are gated; short-prompt TTFT differences are disclosed separately and never attributed to the candidate.'}
