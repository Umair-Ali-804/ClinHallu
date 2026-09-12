from clinhallu.core.constants import LABEL_GROUNDED, LABEL_HALLUCINATED


def test_label_polarity_is_frozen():
    assert LABEL_GROUNDED == 0
    assert LABEL_HALLUCINATED == 1
