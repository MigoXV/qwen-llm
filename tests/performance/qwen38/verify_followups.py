"""独立新进程和普通生产CLI的性能、质量及释放验收。"""
import argparse
import json
from pathlib import Path

from .compare import directory_metrics, metrics, quality_gate


REPEAT_CASES = ('generation512-c1', 'generation512-c4', 'chinese8192-c4-cold',
                'chinese32768-c4-hot', 'chinese57344-c4-hot', 'mixed32k-c4')
PRODUCTION_CASES = ('generation512-c1', 'generation512-c4', 'chinese57344-c4-hot')


def read(path):
    return json.loads(Path(path).read_text())


def lifecycle(manifest):
    return all(manifest.get(k) for k in ('valid', 'released', 'port_released'))


def comparison(baseline, candidate, names):
    rows = []
    for name in names:
        old, new = baseline[name], candidate[name]
        long = name.startswith('chinese') or name == 'mixed32k-c4'
        ttft_key = 'long_ttft_ms' if name == 'mixed32k-c4' else 'ttft_ms'
        tps, ttft = new['tps'] / old['tps'], new[ttft_key] / old[ttft_key]
        rows.append({'case': name, 'tps_ratio': tps, 'ttft_ratio': ttft,
                     'passed': tps >= .95 and (not long or ttft <= 1.05),
                     'baseline': old, 'candidate': new})
    return rows


def repeat(root):
    root = Path(root)
    base = read(root / 'baseline-full/manifest.json')
    new = read(root / 'independent-repeat/manifest.json')
    rows = comparison(directory_metrics(root / 'baseline-full'),
                      directory_metrics(root / 'independent-repeat'), REPEAT_CASES)
    quality = quality_gate(read(root / 'baseline-full/quality.json'), read(root / 'independent-repeat/quality.json'))
    hot = next(row for row in rows if row['case'] == 'chinese57344-c4-hot')
    hits = hot['candidate']['cache_rounds']
    complete = (set(c['name'] for c in new['cases']) == set(REPEAT_CASES)
                and all(len(c['rounds']) >= 5 for c in new['cases']))
    all_hits = len(hits) >= 5 and all(values == [55296] * 4 for values in hits)
    meaningful = any(row['tps_ratio'] >= 1.05 for row in rows) or hot['ttft_ratio'] <= .8
    return {'passed': bool(lifecycle(base) and lifecycle(new) and complete and quality['passed']
                           and all_hits and meaningful and all(row['passed'] for row in rows)),
            'coverage_complete': complete, 'quality': quality, 'all_57k_replays_hit': all_hits,
            'meaningful_gain_repeated': meaningful, 'performance': rows}


def production(root):
    root = Path(root)
    base, new = [read(root / folder / 'manifest.json')
                 for folder in ('production-baseline', 'production-candidate')]
    base_metrics, new_metrics = metrics(base), metrics(new)
    for group in (base_metrics, new_metrics):
        for row in group.values():
            row['accepted_fraction'] = None  # 普通CLI没有注入草稿计数采集。
    rows = comparison(base_metrics, new_metrics, PRODUCTION_CASES)
    workers = new['ready']['runtime']['workers']
    identity = bool(workers) and all(
        w['worker'] == 'qwen_llm.engine.qwen38_worker.Qwen38NPUWorker'
        and w['model_adaptation']['name'] == 'qwen38-910b-v1'
        and w['model_adaptation']['cached_scale_layers'] == 128
        and w['model_adaptation']['shared_lm_head']
        and w['model_adaptation']['head_dtype'] == 'torch.int8'
        and w['model_adaptation']['gdn_output_int8_layers'] == 48
        and w['full_context_capacity'] >= 4 and w['graphs_captured'] >= 3
        for w in workers)
    probes = all(m['probes']['disconnected_after_text'] for m in (base, new))
    complete = all(set(c['name'] for c in m['cases']) == set(PRODUCTION_CASES)
                   and all(len(c['rounds']) >= 5 for c in m['cases']) for m in (base, new))
    hot = next(row for row in rows if row['case'] == 'chinese57344-c4-hot')
    hits = hot['candidate']['cache_rounds']
    all_hits = len(hits) >= 5 and all(values == [55296] * 4 for values in hits)
    capacity = new.get('capacity', {})
    capacity_complete = capacity.get('full_64k_all_four_verified') is True
    same_source = base['implementation_sha256'] == new['implementation_sha256']
    return {'passed': bool(lifecycle(base) and lifecycle(new) and identity and probes and complete
                           and all_hits and capacity_complete and same_source and all(row['passed'] for row in rows)),
            'runtime_identity_verified': identity, 'http_probes_passed': probes,
            'same_production_source_and_catalog': same_source,
            'coverage_complete': complete, 'all_57k_replays_hit': all_hits,
            'four_full_64k_requests_completed': capacity_complete, 'capacity': capacity.get('summary'),
            'performance': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--production', action='store_true')
    args = parser.parse_args()
    name = 'production-acceptance.json' if args.production else 'independent-acceptance.json'
    output = args.root / name
    if output.exists():
        raise ValueError('拒绝覆盖独立验证的验收记录')
    result = production(args.root) if args.production else repeat(args.root)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'performance'}, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
