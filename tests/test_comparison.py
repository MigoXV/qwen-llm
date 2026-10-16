import pytest

from qwen_llm.comparison import metric


@pytest.mark.parametrize("new,higher,over", [(80, True, False), (79.9, True, True),
                                           (120, False, False), (120.1, False, True),
                                           (110, True, False), (90, False, False)])
def test_user_twenty_percent_rollback_boundary(new, higher, over):
    assert metric("test", "same-load", "metric", 100, new, higher)["over_threshold"] is over
