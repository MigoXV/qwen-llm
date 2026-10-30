import pytest

from tests.performance.nvidia.capacity import passed


def complete():
    return {'instance_unchanged': True, 'pool_can_hold_all':True, 'rounds': [
        {'all_full_output': True, 'all_exact_input': True, 'isolated': True, 'preemptions_delta': 0}]}


@pytest.mark.parametrize('field,value', [('all_full_output', False), ('all_exact_input', False),
                                       ('isolated', False), ('preemptions_delta', 1)])
def test_queued_recomputed_shortened_or_contaminated_result_is_not_capacity(field, value):
    record = complete()
    assert passed(record)
    record['rounds'][0][field] = value
    assert not passed(record)


def test_empty_or_restarted_service_is_not_capacity():
    assert not passed({'instance_unchanged': True, 'rounds': []})
    record = complete()
    record['instance_unchanged'] = False
    assert not passed(record)


def test_serial_completion_without_preemption_is_not_full_residency():
    record = complete()
    record['pool_can_hold_all'] = False
    assert not passed(record)
