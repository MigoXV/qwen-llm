import pytest

from qwen_llm.engine.memory import bounded_kv_memory


def test_explicit_budget_does_not_expand_to_free_memory():
    assert bounded_kv_memory(40 * 1024**3, 10 * 1024**3) == 10 * 1024**3
    assert bounded_kv_memory(40 * 1024**3, None) == 40 * 1024**3


@pytest.mark.parametrize('requested',[0,-1,41])
def test_impossible_budget_is_not_silently_reduced(requested):
    with pytest.raises(ValueError):bounded_kv_memory(40,requested)
