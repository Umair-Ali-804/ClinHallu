"""Load layered YAML configs and resolve all paths from the project root."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .paths import absolute_path, find_project_root
from .schema import validate_config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_recursive(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in seen:
        raise ValueError(f"Circular config inheritance detected at {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    seen.add(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    parent_ref = current.pop("extends", None)
    if parent_ref is None:
        return current
    parent = Path(parent_ref)
    if not parent.is_absolute():
        parent = resolved.parent / parent
    return deep_merge(_load_recursive(parent, seen), current)


def load_config(
    config_path: str | Path,
    *,
    seed: int | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    cfg = _load_recursive(path, set())
    root = find_project_root(project_root or path.parent)
    return resolve_config(
        cfg,
        seed=seed,
        project_root=root,
        config_path=path,
    )


def resolve_config(
    config: dict[str, Any],
    *,
    seed: int | None = None,
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a loaded configuration without creating a temporary YAML file."""

    cfg = copy.deepcopy(config)
    root = find_project_root(project_root)
    if seed is not None:
        cfg.setdefault("training", {})["seed"] = int(seed)
    run_seed = int(cfg["training"]["seed"])
    experiment = cfg["experiment"]
    run_dir = absolute_path(
        Path(cfg["output"]["root"]) / experiment["kind"] / experiment["name"] / f"seed_{run_seed}",
        root,
    )

    cfg["project_root"] = str(root)
    cfg["config_path"] = str(Path(config_path).resolve()) if config_path else None
    cfg["output"]["run_dir"] = str(run_dir)
    cfg["data"].setdefault("cache_dir", "data/cache")
    for key in (
        "train_path",
        "val_path",
        "eval_path",
        "processed_dir",
        "cache_dir",
    ):
        cfg["data"][key] = str(absolute_path(cfg["data"][key], root))

    cfg.setdefault("checkpoints", {})["dir"] = str(run_dir / "checkpoints")
    cfg["checkpoints"].setdefault("best_filename", "best_model.pt")
    cfg["checkpoints"].setdefault("latest_filename", "latest_checkpoint.pt")
    # How often training writes a mid-epoch resume point. Whichever limit is
    # reached first triggers a save, so a long epoch on slow hardware is still
    # covered by the time-based bound.
    cfg["checkpoints"].setdefault("save_every_steps", 200)
    cfg["checkpoints"].setdefault("save_every_minutes", 10)
    cfg.setdefault("calibration", {})["dir"] = str(run_dir / "calibration")
    cfg["calibration"].setdefault("temperature_filename", "temperature.npy")
    cfg["calibration"].setdefault("threshold_filename", "threshold.npy")
    cfg.setdefault("results", {})["metrics_dir"] = str(run_dir / "metrics")
    cfg["results"]["predictions_dir"] = str(run_dir / "predictions")
    cfg["results"]["logs_dir"] = str(run_dir / "logs")
    cfg["results"]["training_log_csv"] = str(run_dir / "metrics" / "training_history.csv")

    validate_config(cfg)
    return cfg


def save_resolved_config(cfg: dict[str, Any], destination: str | Path) -> None:
    value = copy.deepcopy(cfg)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)
