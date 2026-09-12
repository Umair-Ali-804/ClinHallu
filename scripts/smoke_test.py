"""Run a one-epoch, capped training smoke test without frozen evaluation."""

from __future__ import annotations

import argparse
import json

from clinhallu.experiments.adaptation import run_condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", default="c5", choices=("c1", "c2", "c3", "c4", "c5"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    result = run_condition(args.condition, seed=args.seed, smoke_test=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
