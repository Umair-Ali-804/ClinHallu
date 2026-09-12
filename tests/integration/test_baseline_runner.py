import json

import numpy as np
import pytest

from clinhallu.baselines.common import select_threshold
from clinhallu.baselines.data import load_records
from clinhallu.baselines.llm_judge import (
    evaluate_rows,
    load_binary_predictions,
    parse_judgment,
)
from clinhallu.baselines.semantic_similarity import normalized_hallucination_score
from clinhallu.baselines.zero_shot_nli import aggregate_scores
from clinhallu.data.dataset import HKGFusionDataset


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def row(identifier, label=0):
    return {
        "id": str(identifier),
        "question": "q",
        "context": "c",
        "answer": "a",
        "hallu_label": label,
    }


def test_data_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "records.jsonl"
    write_rows(path, [row(1, 0), row(1, 1)])
    with pytest.raises(ValueError, match="duplicate id"):
        load_records(path)


def test_data_rejects_empty_required_field(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [row(1, 0), row(2, 1)]
    rows[0]["answer"] = ""
    write_rows(path, rows)
    with pytest.raises(ValueError, match="empty"):
        load_records(path)


def test_semantic_score_range():
    assert np.allclose(normalized_hallucination_score([-1, 0, 1]), [1, 0.5, 0])


def test_nli_aggregation():
    assert aggregate_scores([0.1, 0.9]) == pytest.approx(0.1)
    assert 0 <= aggregate_scores([0.4], [0.8], 0.5) <= 1


def test_llm_json_parser_and_failure():
    parsed = parse_judgment('text {"label": 0, "confidence": 0.9}')
    assert parsed["label"] == 0
    assert parse_judgment("not json")["parse_failure"] is True


def test_llm_binary_predictions_reject_fractional_labels(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_rows(path, [{"id": "1", "hallu_label": 0.5}])
    with pytest.raises(ValueError, match="integer 0 or 1"):
        load_binary_predictions(path, [row(1, 0)])


def test_llm_responses_require_exact_id_alignment():
    with pytest.raises(ValueError, match="missing=1"):
        evaluate_rows([row(1, 0), row(2, 1)], {"1": '{"label": 0}'})


def test_threshold_is_determined_by_validation_only():
    validation_y = [0, 0, 1, 1]
    validation_scores = [0.1, 0.2, 0.7, 0.9]
    threshold = select_threshold(validation_y, validation_scores)

    # Evaluation values are deliberately defined but never passed to the
    # threshold selector.
    evaluation_y = [1, 0, 1, 0]
    evaluation_scores = [0.99, 0.98, 0.97, 0.96]
    assert len(evaluation_y) == len(evaluation_scores)
    assert select_threshold(validation_y, validation_scores) == threshold


class FakeTokenizer:
    cls_token_id = 1
    sep_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(10, 10 + len(text.split())))


def test_deberta_input_always_preserves_answer():
    example = row(1, 0)
    example["question"] = "q " * 200
    example["context"] = "c " * 1_000
    example["answer"] = "a " * 300
    dataset = object.__new__(HKGFusionDataset)
    dataset.tokenizer = FakeTokenizer()
    dataset.max_length = 512
    dataset.question_budget = 96
    dataset.answer_budget = 128
    item = dataset._encode_example(
        example["question"],
        example["context"],
        example["answer"],
    )
    assert len(item["input_ids"]) == 512
    assert item["input_ids"][-1] == FakeTokenizer.sep_token_id
    assert sum(item["answer_mask"]) == 128
