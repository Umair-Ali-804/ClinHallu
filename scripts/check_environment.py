"""Check the active Python/Colab environment before an expensive run."""

from __future__ import annotations

import argparse
import json
import sys

from clinhallu.config.paths import find_project_root
from clinhallu.core.environment import environment_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Python, dependencies, GPU access, and raw-data placement."
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not fail when CUDA is unavailable (useful for local diagnostics).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root()
    data = {
        name: (root / "data" / "raw" / name).is_file()
        for name in ("train.jsonl", "val.jsonl", "eval_data.jsonl")
    }
    report = environment_record()
    report["project_root"] = str(root)
    report["raw_data_present"] = data
    print(json.dumps(report, indent=2))

    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise SystemExit("ClinHallu requires Python 3.10 through 3.13.")

    missing = [name for name, version in report["packages"].items() if version is None]
    if missing:
        raise SystemExit(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Install requirements/colab.txt before running this check."
        )

    if not args.allow_cpu and not report["cuda_available"]:
        raise SystemExit(
            "CUDA is unavailable. In Colab, select Runtime > Change runtime type > "
            "T4 GPU, reconnect, and run setup again. Use --allow-cpu only for diagnostics."
        )


if __name__ == "__main__":
    main()
