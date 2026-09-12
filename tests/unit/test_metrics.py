import pytest
from sklearn.metrics import f1_score

from clinhallu.engine.metrics import (
    bootstrap_confidence_interval,
    compute_binary_metrics,
    multi_seed_summary,
)


def test_binary_metrics_perfect():
    y = [0, 0, 1, 1]
    m = compute_binary_metrics(y, y, [0.0, 0.0, 1.0, 1.0])
    assert m["f1"] == 1.0
    assert m["acc"] == 1.0


def test_binary_metrics_all_wrong():
    y_true = [0, 1]
    y_pred = [1, 0]
    m = compute_binary_metrics(y_true, y_pred, [0.9, 0.1])
    assert m["acc"] == 0.0


def test_binary_metrics_confusion_matrix():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    m = compute_binary_metrics(y_true, y_pred, [0.1, 0.6, 0.9, 0.8])
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["tn"] == 1 and m["fp"] == 1 and m["fn"] == 0 and m["tp"] == 2


def test_binary_metrics_keys():
    m = compute_binary_metrics([0, 1], [0, 1], [0.0, 1.0])
    assert "f1" in m
    assert "precision" in m
    assert "recall" in m
    assert "acc" in m


def test_bootstrap_ci_mean_in_interval():
    y = [0] * 50 + [1] * 50
    result = bootstrap_confidence_interval(
        y,
        y,
        metric_fn=lambda yt, yp: f1_score(yt, yp, zero_division=0),
        n_bootstrap=200,
    )
    assert result["ci_low"] <= result["point"] <= result["ci_high"]


def test_multi_seed_summary():
    seeds = [
        {"val_f1": 0.80, "val_auc": 0.85},
        {"val_f1": 0.82, "val_auc": 0.87},
        {"val_f1": 0.81, "val_auc": 0.86},
    ]
    s = multi_seed_summary(seeds)
    assert abs(s["val_f1_mean"] - 0.81) < 1e-4
    assert "val_f1_std" in s


def test_auprc_present_in_binary_metrics():
    from clinhallu.engine.metrics import compute_binary_metrics

    y = [0, 0, 1, 1]
    m = compute_binary_metrics(y, y, [0.0, 0.0, 1.0, 1.0])
    assert "auprc" in m, "AUPRC missing from binary metrics — G5 fix not applied."
    assert m["auprc"] == 1.0


def test_auprc_value_random():
    from clinhallu.engine.metrics import compute_binary_metrics

    y = [0, 1, 0, 1, 0, 1]
    probs = [0.4, 0.6, 0.3, 0.7, 0.5, 0.8]
    preds = [round(p) for p in probs]
    m = compute_binary_metrics(y, preds, probs)
    assert 0.0 <= m["auprc"] <= 1.0


def test_ece_perfect_calibration():
    from clinhallu.engine.metrics import compute_ece

    y_true = [0, 0, 1, 1]
    y_prob = [0.0, 0.0, 1.0, 1.0]
    ece = compute_ece(y_true, y_prob)
    assert ece == pytest.approx(0.0, abs=1e-3)


def test_ece_worst_calibration():
    from clinhallu.engine.metrics import compute_ece

    y_true = [1, 1, 0, 0]
    y_prob = [0.0, 0.0, 1.0, 1.0]
    ece = compute_ece(y_true, y_prob)
    assert ece > 0.5


def test_ece_in_range():
    from clinhallu.engine.metrics import compute_ece

    y = [0] * 50 + [1] * 50
    p = [0.3] * 50 + [0.7] * 50
    ece = compute_ece(y, p)
    assert 0.0 <= ece <= 1.0


def test_ece_empty():
    from clinhallu.engine.metrics import compute_ece

    assert compute_ece([], []) == 0.0
