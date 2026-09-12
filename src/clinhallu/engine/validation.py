"""Helpers shared by epoch validation and final calibration."""

from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def autocast_for(device: torch.device, enabled: bool):
    if device.type != "cuda":
        yield
        return
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast("cuda", enabled=enabled):
            yield
    else:
        with torch.cuda.amp.autocast(enabled=enabled):
            yield


def scalar_logits(output) -> torch.Tensor:
    if output.logits.dim() == 2:
        return output.logits[:, 1] - output.logits[:, 0]
    return output.logits
