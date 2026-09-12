"""Run one C1-C5 domain-adaptation condition."""

from __future__ import annotations

from clinhallu.config.paths import find_project_root

from .runner import run_experiment

CONDITION_CONFIGS = {
    name: f"configs/conditions/{name}.yaml" for name in ("c1", "c2", "c3", "c4", "c5")
}


def run_condition(condition: str, **kwargs):
    if condition not in CONDITION_CONFIGS:
        raise ValueError(f"Unknown condition {condition!r}; choose C1-C5")
    root = find_project_root()
    return run_experiment(root / CONDITION_CONFIGS[condition], **kwargs)
