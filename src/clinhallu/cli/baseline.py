"""Run, list, or summarize publication baselines."""

from __future__ import annotations

import argparse
import json

import yaml

from clinhallu.config.paths import absolute_path, find_project_root
from clinhallu.experiments.baseline_suite import run_baseline_suite


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="Run enabled or selected baselines")
    run_parser.add_argument("--suite", default="configs/baselines/suite.yaml")
    run_parser.add_argument("--only", nargs="+")
    run_parser.add_argument("--dry-run", action="store_true")
    list_parser = subparsers.add_parser("list", help="List configured baselines")
    list_parser.add_argument("--suite", default="configs/baselines/suite.yaml")
    summarize_parser = subparsers.add_parser("summarize", help="Build baseline summary")
    summarize_parser.add_argument("--results-root", default="artifacts/baselines")
    summarize_parser.add_argument("--output-dir", default="artifacts/reports")
    args = parser.parse_args(argv)

    if args.action == "run":
        result = run_baseline_suite(args.suite, only=args.only, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
    elif args.action == "list":
        root = find_project_root(args.suite)
        path = absolute_path(args.suite, root)
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        print(json.dumps(value["baselines"], indent=2))
    else:
        from clinhallu.reporting.summarize import summarize_baselines

        root = find_project_root()
        summary = summarize_baselines(
            absolute_path(args.results_root, root),
            absolute_path(args.output_dir, root),
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
