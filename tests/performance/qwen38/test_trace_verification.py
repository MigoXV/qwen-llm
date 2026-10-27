from copy import deepcopy

from .verify_trace import verify_pair


def trace(scale_calls):
    return {'path': '/trace/trace_view.json', 'valid_device_trace': True,
            'coverage': {'target_forwards': 2, 'w4_layers': 128, 'complete': True,
                         'expected_matmuls': 256, 'observed_matmuls': 256},
            'top_operators': [{'name': 'Cast', 'calls': scale_calls + 3}],
            'kernel_shapes': [
                {'name': 'Cast', 'input_shapes': '\"40,34816\"', 'calls': scale_calls // 2, 'phase': 'decode'},
                {'name': 'Cast', 'input_shapes': '[136, 5120]', 'calls': scale_calls // 2, 'phase': 'prefill'},
                {'name': 'Cast', 'input_shapes': '1,5120', 'calls': 3, 'phase': 'decode'}]}


def test_cast_removal_requires_real_ffn_execution_and_all_cast_shapes():
    assert verify_pair(trace(256), trace(0))['passed']
    missing = trace(0)
    missing['kernel_shapes'] = []
    assert not verify_pair(trace(256), missing)['passed']
    missing = trace(0)
    missing['kernel_shapes'][-1]['input_shapes'] = ''
    assert not verify_pair(trace(256), missing)['passed']
    incomplete = trace(0)
    incomplete['coverage']['observed_matmuls'] = 128
    assert not verify_pair(trace(256), incomplete)['passed']


def test_remaining_scale_cast_and_incomplete_baseline_are_rejected():
    assert not verify_pair(trace(256), trace(2))['passed']
    assert not verify_pair(trace(128), trace(0))['passed']
    invalid = deepcopy(trace(0))
    invalid['valid_device_trace'] = False
    assert not verify_pair(trace(256), invalid)['passed']
