import json

from clinhallu.engine.prediction_writer import write_predictions


def test_prediction_writer_preserves_ids(tmp_path, raw_row):
    rows = [raw_row("a", 0), raw_row("b", 1)]
    destination = tmp_path / "predictions.jsonl"
    write_predictions(rows, [0, 1], [0.1, 0.8], [0.2, 0.9], [0, 1], destination)
    written = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [row["id"] for row in written] == ["a", "b"]
