import pytest

from clinhallu.data.leakage import assert_group_disjoint


def test_same_question_context_family_is_rejected():
    first = {"question": "Q", "context": "C", "answer": "A", "hall_label": 0}
    second = {"question": " q ", "context": " c ", "answer": "B", "hall_label": 1}
    with pytest.raises(RuntimeError, match="Data leakage"):
        assert_group_disjoint(("train", [first]), ("test", [second]))
