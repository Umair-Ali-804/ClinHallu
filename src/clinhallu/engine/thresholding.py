"""Validation-only decision-threshold selection."""

from __future__ import annotations

from typing import Optional

import numpy as np


def search_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    candidates: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Maximize positive-class F1; break ties by higher recall."""

    if candidates is None:
        candidates = np.arange(0.05, 0.951, 0.01)
    best_threshold, best_f1, best_recall = 0.5, -1.0, -1.0
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1 or (np.isclose(f1, best_f1) and recall > best_recall):
            best_threshold, best_f1, best_recall = float(threshold), f1, recall
    return best_threshold, float(best_f1)
