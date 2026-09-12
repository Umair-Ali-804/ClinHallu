"""Resolve baseline YAML files and execute each method in a fresh process.

The suite writes its ledger after every individual baseline rather than once at
the end, so an interruption part-way through leaves an accurate record. The next
invocation skips the baselines that already produced predictions and continues
with the rest.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from clinhallu.baselines.registry import REGISTRY
from clinhallu.config.paths import absolute_path, find_project_root
from clinhallu.core.interrupt import INTERRUPTED_EXIT_CODE, InterruptedRun

from .ledger import Ledger, fingerprint

logger = logging.getLogger(__name__)

PATH_KEYS = {
    "api_config",
    "answers_dir",
    "output_dir",
    "cache_dir",
    "responses_path",
    "val_samples",
    "eval_samples",
}


def _flag(key: str) -> str:
    if key == "responses_path":
        return "--predictions"
    return "--" + key


def _outputs_exist(output_dir: Path) -> bool:
    """A baseline counts as finished only if it left its metrics behind."""
    if not output_dir.is_dir():
        return False
    return any(output_dir.glob("*metrics*.json"))


def run_baseline_suite(
    suite_path: str | Path,
    *,
    only: list[str] | None = None,
    dry_run: bool = False,
    resume: bool = True,
    continue_on_error: bool = True,
) -> dict:
    root = find_project_root(suite_path)
    suite_file = absolute_path(suite_path, root)
    suite = yaml.safe_load(suite_file.read_text(encoding="utf-8"))
    data = {
        key: str(absolute_path(suite["data"][f"{key}_path"], root))
        for key in ("train", "val", "eval")
    }
    selected = set(only or [])
    unknown = selected - set(suite["baselines"])
    if unknown:
        raise ValueError(f"Unknown baseline(s): {', '.join(sorted(unknown))}")

    status_path = root / "artifacts" / "baselines" / "status.json"
    ledger = Ledger(root / "artifacts" / "baselines" / "baseline_ledger.json", name="baselines")
    if not resume and not dry_run:
        ledger.reset(selected or None)

    status: dict = {}
    failures = []
    for instance_name, suite_entry in suite["baselines"].items():
        if selected and instance_name not in selected:
            continue
        if not selected and not bool(suite_entry.get("enabled", False)):
            status[instance_name] = {"status": "disabled"}
            continue

        config_path = absolute_path(suite_entry["config"], root)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        method = config["method"]
        if method not in REGISTRY:
            raise ValueError(f"Unknown method {method!r} in {config_path}")
        spec = REGISTRY[method]
        signature = fingerprint({"config": config, "data": data})
        output_dir = absolute_path(config.get("output_dir", "artifacts/baselines"), root)

        command = [
            sys.executable,
            "-m",
            spec.module,
            "--train",
            data["train"],
            "--val",
            data["val"],
            "--eval",
            data["eval"],
        ]
        for key in spec.arguments:
            if key not in config or config[key] is None:
                continue
            value = config[key]
            if key in PATH_KEYS:
                value = absolute_path(value, root)
            if isinstance(value, bool):
                if value:
                    command.append(_flag(key))
            else:
                command.extend([_flag(key), str(value)])

        if dry_run:
            status[instance_name] = {"status": "planned", "command": command}
            continue

        if resume and ledger.is_complete(instance_name, fingerprint_value=signature):
            if _outputs_exist(output_dir):
                logger.info("Baseline %s already completed; skipping.", instance_name)
                status[instance_name] = {"status": "already_completed"}
                _write_status(status_path, status)
                continue
            logger.warning(
                "Baseline %s was recorded as complete but its metrics are missing "
                "from %s; running it again.",
                instance_name,
                output_dir,
            )

        ledger.start(instance_name, fingerprint_value=signature, method=method)
        completed = subprocess.run(command, cwd=str(root), check=False)
        if completed.returncode == 0:
            ledger.complete(instance_name, fingerprint=signature, output_dir=str(output_dir))
            status[instance_name] = {"status": "completed"}
        elif completed.returncode in (INTERRUPTED_EXIT_CODE, -2):
            ledger.interrupt(instance_name)
            status[instance_name] = {"status": "interrupted"}
            _write_status(status_path, status)
            raise InterruptedRun(
                f"Baseline {instance_name} stopped cleanly. Completed baselines are "
                f"recorded in {ledger.path} and will be skipped on the next run."
            )
        else:
            ledger.fail(instance_name, f"exit code {completed.returncode}")
            status[instance_name] = {
                "status": "failed",
                "returncode": completed.returncode,
            }
            failures.append(instance_name)
            if not continue_on_error:
                _write_status(status_path, status)
                raise RuntimeError(f"Baseline run failed: {instance_name}")
        # Persist after every baseline so an abrupt stop cannot erase the record.
        _write_status(status_path, status)

    if failures:
        raise RuntimeError(
            f"Baseline run(s) failed: {', '.join(failures)}. Successful baselines are "
            "recorded and will be skipped on the next run."
        )
    return status


def _write_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(path)
