"""Binary classification metrics and uncertainty estimates."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _aligned_arrays(y_true, y_pred, y_prob=None):
    labels = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_pred, dtype=int)
    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError("Labels and predictions must be one-dimensional")
    if len(labels) == 0 or len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have the same non-zero length")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1")
    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("Predictions must contain only 0 and 1")
    if y_prob is None:
        return labels, predictions, None
    probabilities = np.asarray(y_prob, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(labels):
        raise ValueError("Probabilities must align one-to-one with labels")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or infinity")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    return labels, predictions, probabilities


def compute_binary_metrics(
    y_true,
    y_pred,
    y_prob,
    prefix: str = "",
) -> dict:
    labels, predictions, probabilities = _aligned_arrays(y_true, y_pred, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    result = {
        f"{prefix}acc": round(float(accuracy_score(labels, predictions)), 4),
        f"{prefix}precision": round(float(precision), 4),
        f"{prefix}recall": round(float(recall), 4),
        f"{prefix}f1": round(float(f1), 4),
        f"{prefix}macro_f1": round(float(macro_f1), 4),
        f"{prefix}mcc": round(float(matthews_corrcoef(labels, predictions)), 4),
    }
    if len(np.unique(labels)) == 2:
        result[f"{prefix}auc"] = round(
            float(roc_auc_score(labels, probabilities)),
            4,
        )
        result[f"{prefix}auprc"] = round(
            float(average_precision_score(labels, probabilities)),
            4,
        )
        result[f"{prefix}brier"] = round(
            float(brier_score_loss(labels, probabilities)),
            4,
        )

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    result[f"{prefix}confusion_matrix"] = matrix.tolist()
    result[f"{prefix}tn"] = int(true_negative)
    result[f"{prefix}fp"] = int(false_positive)
    result[f"{prefix}fn"] = int(false_negative)
    result[f"{prefix}tp"] = int(true_positive)
    return result


def search_threshold_by_f1(
    y_prob,
    y_true,
    lo: float = 0.01,
    hi: float = 0.99,
    n_steps: int = 197,
) -> dict[str, float]:
    probabilities = np.asarray(y_prob, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    if len(probabilities) == 0 or len(probabilities) != len(labels):
        raise ValueError("Probabilities and labels must have the same non-zero length")
    best = (-1.0, -1.0, -0.5)
    for threshold in np.linspace(lo, hi, n_steps):
        predictions = (probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        del precision
        candidate = (float(f1), float(recall), -float(threshold))
        if candidate > best:
            best = candidate
    return {
        "threshold": round(-best[2], 4),
        "val_f1_at_threshold": round(best[0], 4),
    }


def compute_ece(y_true, y_prob, n_bins: int = 15) -> float:
    labels = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(y_prob, dtype=float)
    if len(labels) == 0:
        return 0.0
    if len(labels) != len(probabilities):
        raise ValueError("Probabilities must align one-to-one with labels")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    expected_calibration_error = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not mask.any():
            continue
        observed = float(labels[mask].mean())
        confidence = float(probabilities[mask].mean())
        expected_calibration_error += (mask.sum() / len(labels)) * abs(observed - confidence)
    return round(float(expected_calibration_error), 6)


def bootstrap_confidence_interval(
    y_true,
    y_pred,
    metric_fn: Callable,
    n_bootstrap: int = 1_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    labels, predictions, _ = _aligned_arrays(y_true, y_pred)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    point = float(metric_fn(labels, predictions))
    scores = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), size=len(labels))
        try:
            scores.append(float(metric_fn(labels[indices], predictions[indices])))
        except ValueError:
            continue
    if scores:
        lower = float(np.percentile(scores, 100 * alpha / 2))
        upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
        standard_deviation = float(np.std(scores))
    else:
        lower = upper = point
        standard_deviation = 0.0
    return {
        "point": round(point, 4),
        "ci_low": round(lower, 4),
        "ci_high": round(upper, 4),
        "std": round(standard_deviation, 4),
    }


def grouped_bootstrap_confidence_interval(
    y_true,
    y_pred,
    groups,
    metric_fn: Callable,
    n_bootstrap: int = 2_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Cluster bootstrap that keeps correlated answer variants together."""

    labels, predictions, _ = _aligned_arrays(y_true, y_pred)
    group_array = np.asarray(groups, dtype=object)
    if len(group_array) != len(labels):
        raise ValueError("Group identifiers must align one-to-one with labels")
    unique_groups = np.unique(group_array)
    if not len(unique_groups):
        return bootstrap_confidence_interval(
            labels,
            predictions,
            metric_fn,
            n_bootstrap,
            alpha,
            seed,
        )
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        indices = np.concatenate([np.flatnonzero(group_array == group) for group in sampled_groups])
        scores.append(float(metric_fn(labels[indices], predictions[indices])))
    return {
        "point": round(float(metric_fn(labels, predictions)), 4),
        "ci_low": round(float(np.percentile(scores, 100 * alpha / 2)), 4),
        "ci_high": round(float(np.percentile(scores, 100 * (1 - alpha / 2))), 4),
        "std": round(
            float(np.std(scores, ddof=1 if len(scores) > 1 else 0)),
            4,
        ),
    }


def mcnemar_test(y_true, predictions_a, predictions_b) -> dict:
    labels, first, _ = _aligned_arrays(y_true, predictions_a)
    _, second, _ = _aligned_arrays(y_true, predictions_b)
    correct_a = first == labels
    correct_b = second == labels
    n_01 = int(((~correct_a) & correct_b).sum())
    n_10 = int((correct_a & (~correct_b)).sum())
    statistic = p_value = float("nan")
    try:
        from statsmodels.stats.contingency_tables import mcnemar

        table = np.array(
            [
                [int((~correct_a & ~correct_b).sum()), n_01],
                [n_10, int((correct_a & correct_b).sum())],
            ]
        )
        result = mcnemar(table, exact=(n_01 + n_10) < 25)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except ImportError:
        pass
    return {
        "statistic": statistic,
        "p_value": p_value,
        "n_01": n_01,
        "n_10": n_10,
        "significant_at_0.05": (p_value < 0.05 if not np.isnan(p_value) else False),
    }


def multi_seed_summary(per_seed: list[dict]) -> dict[str, float]:
    if not per_seed:
        return {}
    summary = {}
    keys = sorted({key for row in per_seed for key in row})
    for key in keys:
        values = [
            float(row[key]) for row in per_seed if key in row and isinstance(row[key], (int, float))
        ]
        if values:
            summary[f"{key}_mean"] = round(float(np.mean(values)), 4)
            summary[f"{key}_std"] = round(
                float(np.std(values, ddof=1 if len(values) > 1 else 0)),
                4,
            )
    return summary
