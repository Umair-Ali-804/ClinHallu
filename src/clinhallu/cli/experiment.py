"""CLI handlers for full conditions and ablations."""

from __future__ import annotations

import argparse
import json

from clinhallu.core.interrupt import INTERRUPTED_EXIT_CODE, InterruptedRun
from clinhallu.experiments.ablation import run_ablations
from clinhallu.experiments.adaptation import run_condition
from clinhallu.experiments.runner import run_experiment

RESUMABLE_STAGES = ("prepare", "train", "evaluate")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=13)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Continue from saved checkpoints and completed stages (default).",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Discard saved progress for this run and start over.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument(
        "--force-stage",
        nargs="+",
        choices=RESUMABLE_STAGES,
        default=[],
        help="Run these stages again even if they are recorded as complete.",
    )
    parser.add_argument("--smoke-test", action="store_true")


def _kwargs(args) -> dict:
    return {
        "seed": args.seed,
        "resume": args.resume,
        "prepare_only": args.prepare_only,
        "force_evaluation": args.force_evaluation,
        "force_stages": set(args.force_stage),
        "smoke_test": args.smoke_test,
    }


def _report(callable_, /, *call_args, **call_kwargs) -> None:
    """Run a resumable entry point and turn a clean stop into exit code 130."""
    try:
        print(json.dumps(callable_(*call_args, **call_kwargs), indent=2))
    except InterruptedRun as exc:
        print(json.dumps({"status": "interrupted", "detail": str(exc)}, indent=2))
        raise SystemExit(INTERRUPTED_EXIT_CODE) from exc


def run_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one complete configured experiment")
    parser.add_argument("--config", required=True)
    _common(parser)
    args = parser.parse_args(argv)
    _report(run_experiment, args.config, **_kwargs(args))


def condition_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one C1-C5 condition")
    parser.add_argument("condition", type=str.lower, choices=("c1", "c2", "c3", "c4", "c5"))
    _common(parser)
    args = parser.parse_args(argv)
    _report(run_condition, args.condition, **_kwargs(args))


def ablation_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run GAER++ ablations")
    parser.add_argument("--variants", nargs="+", default=["B", "C", "D", "E", "F"])
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running the remaining variants when one fails.",
    )
    _common(parser)
    args = parser.parse_args(argv)
    _report(
        run_ablations,
        args.variants,
        continue_on_error=args.continue_on_error,
        **_kwargs(args),
    )


def prepare_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate and preprocess without training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_experiment(args.config, seed=args.seed, prepare_only=True),
            indent=2,
        )
    )
