import json

from clinhallu.engine.inference import evaluate


def test_completed_evaluation_is_not_repeated(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    expected = {"eval_f1": 0.9, "protocol": "frozen_test_after_validation_calibration"}
    (metrics_dir / "evaluation_metrics.json").write_text(json.dumps(expected))
    result = evaluate({"results": {"metrics_dir": str(metrics_dir)}}, force=False)
    assert result == expected
