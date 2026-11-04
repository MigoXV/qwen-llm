import copy
from .ascendc_acceptance import CORE, balanced_gate, drift_gate


def manifest(tps=100, ttft=100):
    value = {'valid': True, 'released': True, 'port_released': True, 'cases': []}
    for name in CORE:
        row = {'valid': True, 'tokens_per_second': tps, 'ttft_ms': [ttft] * 4,
               'cached_tokens': [0] * 4, 'accepted_tokens': 0, 'draft_tokens': 0}
        value['cases'].append({'name': name, 'rounds': [copy.deepcopy(row) for _ in range(5)]})
    return value


def test_both_targets_and_every_case_are_required():
    baseline = manifest()
    assert balanced_gate(baseline, manifest(106, 94))['passed']
    assert not balanced_gate(baseline, manifest(106, 100))['passed']
    assert not balanced_gate(baseline, manifest(100, 94))['passed']
    candidate = manifest(106, 94)
    candidate['cases'][0]['rounds'][0]['valid'] = False
    assert not balanced_gate(baseline, candidate)['passed']
    candidate = manifest(106, 94)
    for row in candidate['cases'][0]['rounds']:
        row['tokens_per_second'] = 94
    assert CORE[0] in balanced_gate(baseline, candidate)['regression_cases']


def test_mixed_long_ttft_and_baseline_drift_cannot_be_hidden():
    baseline, replay = manifest(), manifest()
    assert drift_gate(baseline, replay)['passed']
    for row in replay['cases'][0]['rounds']:
        row['ttft_ms'] = [70] * 4
    disclosed = drift_gate(baseline, replay)
    assert disclosed['passed']
    assert disclosed['short_prompt_ttft_drift_cases'] == [CORE[0]]
    assert not disclosed['all_reported_metrics_within_5_percent']
    for row in replay['cases'][-1]['rounds']:
        row['ttft_ms'][3] = 106
    assert drift_gate(baseline, replay)['drifting_cases'] == ['mixed32k-c4']
    assert not balanced_gate(baseline, replay)['passed']
