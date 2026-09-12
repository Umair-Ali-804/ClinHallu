"""Report what is already finished and what a re-run would still have to do.

Answering "if I run this again, what happens?" before committing a GPU session
is the point of this command. It reads the same ledgers and checkpoints the
runners use, so what it prints is exactly what they will act on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinhallu.config.paths import find_project_root
from clinhallu.experiments.ledger import Ledger


def _checkpoint_summary(checkpoint_path: Path) -> dict:
    """Describe a checkpoint without loading its weights onto a device."""
    if not checkpoint_path.is_file():
        return {"exists": False}
    try:
        import torch

        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        try:
            import torch

            payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except Exception as exc:
            return {"exists": True, "readable": False, "error": str(exc)}
    return {
        "exists": True,
        "readable": True,
        "epoch": payload.get("epoch"),
        "phase": payload.get("phase", "epoch_complete"),
        "global_step": payload.get("global_step"),
        "batches_done": payload.get("batches_done", 0),
        "best_metric": payload.get("best_metric"),
        "size_mb": round(checkpoint_path.stat().st_size / 1e6, 1),
    }


def _run_directories(root: Path) -> list[Path]:
    artifacts = root / "artifacts"
    if not artifacts.is_dir():
        return []
    return sorted(path.parent for path in artifacts.glob("*/*/seed_*/run_manifest.json"))


def collect_status(root: Path | None = None) -> dict:
    root = Path(root) if root else find_project_root()
    report: dict = {"project_root": str(root), "runs": {}, "ledgers": {}}

    for run_dir in _run_directories(root):
        relative = str(run_dir.relative_to(root))
        stages = Ledger(run_dir / "stage_ledger.json")
        metrics = run_dir / "metrics" / "evaluation_metrics.json"
        report["runs"][relative] = {
            "stages": {unit: entry.get("status") for unit, entry in stages.data["units"].items()},
            "latest_checkpoint": _checkpoint_summary(
                run_dir / "checkpoints" / "latest_checkpoint.pt"
            ),
            "best_checkpoint": _checkpoint_summary(run_dir / "checkpoints" / "best_model.pt"),
            "evaluation_metrics": metrics.is_file(),
        }

    for label, path in {
        "ablations": root / "artifacts" / "ablations" / "ablation_ledger.json",
        "baselines": root / "artifacts" / "baselines" / "baseline_ledger.json",
    }.items():
        if path.is_file():
            report["ledgers"][label] = Ledger(path).summary()["counts"]

    pipeline_dir = root / "artifacts" / "pipeline"
    if pipeline_dir.is_dir():
        report["ledgers"]["pipeline"] = {
            path.stem: {
                unit: entry.get("status") for unit, entry in Ledger(path).data["units"].items()
            }
            for path in sorted(pipeline_dir.glob("*.json"))
            if not path.name.endswith(".tmp")
        }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit raw JSON.")
    args = parser.parse_args(argv)
    report = collect_status()

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Project: {report['project_root']}\n")
    if not report["runs"]:
        print("No runs found yet.")
    for name, run in report["runs"].items():
        stages = ", ".join(f"{k}={v}" for k, v in run["stages"].items()) or "no stages recorded"
        print(f"{name}\n  stages: {stages}")
        latest = run["latest_checkpoint"]
        if latest.get("exists") and latest.get("readable"):
            position = (
                f"mid-epoch at batch {latest['batches_done']}"
                if latest.get("phase") == "in_epoch"
                else latest.get("phase")
            )
            print(
                f"  latest checkpoint: epoch {latest['epoch']} ({position}), "
                f"step {latest['global_step']}, best={latest['best_metric']}"
            )
        elif latest.get("exists"):
            print("  latest checkpoint: present but UNREADABLE — a backup may be used")
        else:
            print("  latest checkpoint: none")
        print(f"  evaluated: {'yes' if run['evaluation_metrics'] else 'no'}\n")

    for label, value in report["ledgers"].items():
        print(f"{label}: {json.dumps(value)}")


if __name__ == "__main__":
    main()
