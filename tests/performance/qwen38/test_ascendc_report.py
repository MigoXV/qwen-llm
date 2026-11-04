import json
from .summarize_ascendc import summarize


def test_incomplete_baseline_and_bad_kernel_never_become_success(tmp_path):
    baseline = tmp_path/'baseline'
    micro = tmp_path/'micro'
    baseline.mkdir()
    micro.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value))
    write(baseline/'manifest.json', {'valid':False, 'released':True, 'port_released':True, 'cases':[]})
    write(baseline/'stability.json', {'elapsed_seconds':3600})
    write(baseline/'quality.json', {'execution_errors':0, 'categories':{}})
    for name in ('long-quality.json','boundary-quality.json','prefix-growth.json'):
        write(baseline/name, {'passed':1,'total':2})
    samples = [{'projection':'down_proj','M':1,'correct':False,'relative_l2':.5} for _ in range(3)]
    write(micro/'result.json', {'samples':samples, 'passed_micro_gate':False, 'correct':False})
    result = summarize(tmp_path)
    assert not result['baseline_complete']
    assert result['candidate_decision'] == 'rejected_numerical'
    assert not result['end_to_end_candidate_tested']
    assert 'speed_ratio' not in result['operator_comparison'][0]
    assert (tmp_path/'comparison.md').is_file()
    for sample in samples:
        sample.update(correct=True, relative_l2=.002, kind='staged',
                      upstream_ms=[1.0]*5, candidate_ms=[.8]*5)
    write(micro/'result.json', {'samples':samples, 'passed_micro_gate':True,
                              'correct':True, 'nz_selection':[]})
    result = summarize(tmp_path)
    assert result['candidate_decision'] == 'rejected_no_qualified_native_nz_profile'
    assert not result['end_to_end_candidate_tested']
