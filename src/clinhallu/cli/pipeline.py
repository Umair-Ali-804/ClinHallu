"""Sequential training, ablations, shared generation, baselines and reports.

Stages are recorded in a ledger as they complete, so re-running the same command
after a crash resumes at the first unfinished stage instead of repeating hours
of finished work. Each stage is independently resumable too, so the stage that
was interrupted continues from its own checkpoint rather than from its start.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys

from clinhallu.config.paths import find_project_root
from clinhallu.core.interrupt import INTERRUPTED_EXIT_CODE
from clinhallu.experiments.ledger import Ledger, fingerprint

STAGES = ("model", "ablations", "generate", "baselines", "report")


def commands(condition="c5", seed=13, stages=STAGES):
    if "ablations" in stages and condition != "c5":
        raise ValueError("The B-F ablation configs use C5. Use C5 or omit the ablations stage.")
    steps = {
        "model": ["condition", condition, "--seed", str(seed)],
        "ablations": ["ablate", "--variants", "B", "C", "D", "E", "F", "--seed", str(seed)],
        "generate": ["generate", "--config", "configs/generation/shared_five.yaml"],
        "baselines": ["baseline", "run"],
        "report": ["report", "build"],
    }
    return [(stage, [sys.executable, "-m", "clinhallu", *steps[stage]]) for stage in stages]


def ledger_path(root, condition, seed):
    return root / "artifacts" / "pipeline" / f"{condition}_seed{seed}.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("c1", "c2", "c3", "c4", "c5"), default="c5")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--dry-run", action="store_true")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip stages already recorded as complete (default).",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Forget recorded stage results and run every stage again.",
    )
    parser.add_argument(
        "--force-stage",
        nargs="+",
        choices=STAGES,
        default=[],
        help="Run these stages again even if they are recorded as complete.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print what is done and what remains, then exit.",
    )
    args = parser.parse_args(argv)
    root = find_project_root()

    ledger = Ledger(
        ledger_path(root, args.condition, args.seed),
        name=f"pipeline:{args.condition}:seed{args.seed}",
    )
    planned = commands(args.condition, args.seed, args.stages)

    if args.status:
        print(
            json.dumps(
                {
                    "ledger": str(ledger.path),
                    "stages": {stage: ledger.status(stage) for stage, _ in planned},
                },
                indent=2,
            )
        )
        return

    if not args.resume and not args.dry_run:
        ledger.reset(stage for stage, _ in planned)

    signature = fingerprint({"condition": args.condition, "seed": args.seed})
    for stage, command in planned:
        if (
            args.resume
            and stage not in args.force_stage
            and ledger.is_complete(stage, fingerprint_value=signature)
        ):
            print(f"[{stage}] already completed — skipping", flush=True)
            continue

        print(f"[{stage}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue

        ledger.start(stage, fingerprint_value=signature)
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode == 0:
            ledger.complete(stage, fingerprint=signature)
            continue
        if completed.returncode in (INTERRUPTED_EXIT_CODE, -2):
            ledger.interrupt(stage)
            print(
                f"[{stage}] stopped cleanly. Progress is saved; re-run the same "
                f"command to continue from here.",
                flush=True,
            )
            raise SystemExit(INTERRUPTED_EXIT_CODE)
        ledger.fail(stage, f"exit code {completed.returncode}")
        print(
            f"[{stage}] failed with exit code {completed.returncode}. Stages completed "
            f"before it are recorded in {ledger.path} and will be skipped on the next run.",
            flush=True,
        )
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
