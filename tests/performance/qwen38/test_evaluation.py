from tests.performance.qwen38.evaluation import check_answer


def test_mentioned_correct_number_does_not_hide_wrong_answer():
    case = {"check": "exact", "expected": "97"}
    assert not check_answer(case, "97或者99")
    assert not check_answer(case, "我不确定，题目可能提到了97")
    assert check_answer(case, "９７。")


def test_json_instruction_requires_exact_object_without_explanation():
    case = {"check": "json", "expected": {"x": 1}}
    assert check_answer(case, '{"x": 1}')
    assert not check_answer(case, '{"x": 1, "y": 2}')
    assert not check_answer(case, '答案是 {"x": 1}')
    assert not check_answer(case, '{"x": true}')


def test_math_only_accepts_explicit_final_answer():
    case = {"check": "final_number", "expected": "1200"}
    assert check_answer(case, 'Work: 12*100=1200\nFINAL: 1,200')
    assert not check_answer(case, '1200 or 1300')
    assert not check_answer(case, 'Work mentions 1200\nFINAL: 1300')
