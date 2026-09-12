"""Validate data counts, labels, JSONL schema and cross-split disjointness."""

from __future__ import annotations

import argparse
import json

from clinhallu.config import load_config
from clinhallu.data.validation import validate_fixed_splits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, seed=args.seed)
    report = validate_fixed_splits(
        cfg["data"]["train_path"],
        cfg["data"]["val_path"],
        cfg["data"]["eval_path"],
        expected_rows=cfg["data"].get("expected_rows"),
        minimum_class_fraction=float(cfg["data"].get("minimum_class_fraction", 0.0)),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
