"""Temperature scaling fitted exclusively on validation logits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .thresholding import search_threshold


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 200,
    learning_rate: float = 0.01,
) -> float:
    if logits.numel() == 0 or labels.numel() == 0:
        raise ValueError("Temperature fitting requires non-empty validation data")
    if logits.numel() != labels.numel():
        raise ValueError("Validation logits and labels must have the same length")

    # Optimizing log(T) makes positivity structural instead of relying on a
    # clamp whose zero-gradient region can trap the optimizer.
    log_temperature = torch.nn.Parameter(torch.zeros(1, device=logits.device, dtype=logits.dtype))
    optimizer = torch.optim.LBFGS([log_temperature], lr=learning_rate, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(min=1e-3).item())


def calibrate_validation(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float]:
    temperature = fit_temperature(logits, labels)
    probabilities = torch.sigmoid(logits / temperature).detach().cpu().numpy()
    threshold, f1 = search_threshold(probabilities, labels.detach().cpu().numpy())
    return temperature, threshold, f1


def save_calibration(
    directory: str | Path,
    temperature: float,
    threshold: float,
    validation_f1: float,
) -> dict:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "temperature.npy", np.array([temperature], dtype=np.float64))
    np.save(target / "threshold.npy", np.array([threshold], dtype=np.float64))
    summary = {
        "temperature": temperature,
        "threshold": threshold,
        "validation_f1_at_threshold": validation_f1,
        "fitted_on": "validation_only",
    }
    (target / "calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_calibration(directory: str | Path, *, required: bool = True) -> tuple[float, float]:
    target = Path(directory)
    temperature_path = target / "temperature.npy"
    threshold_path = target / "threshold.npy"
    missing = [str(path) for path in (temperature_path, threshold_path) if not path.is_file()]
    if missing and required:
        raise FileNotFoundError(
            "Evaluation requires validation-fitted calibration files. Missing: "
            + ", ".join(missing)
        )
    temperature = float(np.load(temperature_path)[0]) if temperature_path.is_file() else 1.0
    threshold = float(np.load(threshold_path)[0]) if threshold_path.is_file() else 0.5
    return temperature, threshold


def apply_temperature(probabilities, temperature: float) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    logits = np.log(values / (1.0 - values)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))
