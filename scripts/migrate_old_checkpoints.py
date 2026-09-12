"""Copy checkpoints and calibration files from the supplied legacy layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_if_present(source: Path, destination: Path) -> str | None:
    if not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-checkpoint-dir", required=True)
    parser.add_argument("--new-run-dir", required=True)
    args = parser.parse_args()
    source = Path(args.old_checkpoint_dir).resolve()
    target = Path(args.new_run_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Old checkpoint directory not found: {source}")
    copied = []
    for filename in (
        "best_model.pt",
        "latest_checkpoint.pt",
        "model_only.pt",
        "model_only.safetensors",
    ):
        result = copy_if_present(source / filename, target / "checkpoints" / filename)
        if result:
            copied.append(result)
    for filename in ("temperature.npy", "threshold.npy", "calibration.json"):
        legacy_path = source / filename
        if not legacy_path.is_file():
            legacy_path = source / "calibration" / filename
        result = copy_if_present(legacy_path, target / "calibration" / filename)
        if result:
            copied.append(result)
    if not (target / "checkpoints" / "best_model.pt").is_file():
        raise FileNotFoundError("The source does not contain best_model.pt")
    if not copied:
        raise RuntimeError("No compatible checkpoint files were found")
    print(json.dumps({"status": "copied", "files": copied}, indent=2))


if __name__ == "__main__":
    main()
