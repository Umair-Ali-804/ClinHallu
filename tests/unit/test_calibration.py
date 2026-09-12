import numpy as np
import torch

from clinhallu.engine.calibration import apply_temperature, calibrate_validation


def test_calibration_returns_valid_values():
    logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    temperature, threshold, f1 = calibrate_validation(logits, labels)
    assert temperature > 0
    assert 0.05 <= threshold <= 0.95
    assert 0.0 <= f1 <= 1.0
    output = apply_temperature(np.array([0.2, 0.8]), temperature)
    assert np.all((output >= 0.0) & (output <= 1.0))
