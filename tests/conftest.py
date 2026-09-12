import pytest


@pytest.fixture
def raw_row():
    def make(identifier, label=0):
        return {
            "id": str(identifier),
            "question": f"question {identifier}",
            "context": f"context {identifier}",
            "answer": f"answer {identifier}",
            "hallu_label": label,
        }

    return make
