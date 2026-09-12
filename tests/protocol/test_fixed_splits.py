import json
from pathlib import Path

import pytest

import clinhallu.data.preprocessing as preprocessing
from clinhallu.data.preprocessing import assert_group_disjoint, run_fixed_preprocessing


def _write(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(index, label=0):
    return {
        "id": index,
        "question": f"q{index}",
        "answer": f"a{index}",
        "context": f"c{index}",
        "hallu_label": label,
    }


def _run(tmp_path, monkeypatch, train, validation, test, policy="error"):
    raw, output = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    _write(raw / "train.jsonl", train)
    _write(raw / "val.jsonl", validation)
    _write(raw / "eval_data.jsonl", test)
    monkeypatch.setattr(
        preprocessing,
        "load_config",
        lambda _: {"data": {"processed_dir": str(output), "overlap_policy": policy}},
    )
    run_fixed_preprocessing(
        "unused.yaml",
        str(raw / "train.jsonl"),
        str(raw / "val.jsonl"),
        str(raw / "eval_data.jsonl"),
        overlap_policy=policy,
    )
    return output


def test_fixed_membership_is_preserved(tmp_path, monkeypatch):
    output = _run(
        tmp_path,
        monkeypatch,
        [_row(1), _row(2, 1)],
        [_row(3)],
        [_row(4, 1)],
    )
    identifiers = [
        json.loads(line)["sample_id"] for line in (output / "train.jsonl").read_text().splitlines()
    ]
    assert identifiers == [1, 2]
    assert (output / "validation.jsonl").is_file()
    assert (output / "test.jsonl").is_file()


def test_cross_split_overlap_fails():
    row = {"question": "same", "context": "same", "answer": "x", "hall_label": 0}
    with pytest.raises(RuntimeError):
        assert_group_disjoint(("train", [row]), ("validation", [dict(row)]))


def test_overlap_cleanup_protects_test_then_validation(tmp_path, monkeypatch):
    output = _run(
        tmp_path,
        monkeypatch,
        [_row(1), _row(2), _row(3)],
        [dict(_row(2), answer="validation"), _row(4)],
        [dict(_row(3), answer="test"), dict(_row(4), answer="test wins")],
        policy="remove_lower_priority",
    )
    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    validation = [
        json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()
    ]
    assert [row["sample_id"] for row in train] == [1]
    assert [row["sample_id"] for row in validation] == [2]
