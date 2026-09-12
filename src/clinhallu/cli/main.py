"""Top-level ClinHallu command-line interface."""

from __future__ import annotations

import sys

COMMANDS = {
    "pipeline": ("clinhallu.cli.pipeline", "main"),
    "generate": ("clinhallu.api.generate_answers", "main"),
    "api": ("clinhallu.cli.api", "main"),
    "validate": ("clinhallu.cli.validate", "main"),
    "prepare": ("clinhallu.cli.experiment", "prepare_main"),
    "train": ("clinhallu.cli.train", "main"),
    "evaluate": ("clinhallu.cli.evaluate", "main"),
    "run": ("clinhallu.cli.experiment", "run_main"),
    "condition": ("clinhallu.cli.experiment", "condition_main"),
    "ablate": ("clinhallu.cli.experiment", "ablation_main"),
    "baseline": ("clinhallu.cli.baseline", "main"),
    "report": ("clinhallu.cli.report", "main"),
    "status": ("clinhallu.cli.status", "main"),
}


def _help() -> str:
    rows = [
        "ClinHallu research CLI",
        "",
        "Usage: clinhallu <command> [options]",
        "",
        "Commands:",
        "  pipeline   Run training, ablations, generation, baselines and reports",
        "  generate   Generate five independent shared answers per record",
        "  api        Inspect current OpenRouter models",
        "  validate   Validate raw fixed splits",
        "  prepare    Validate and preprocess data",
        "  train      Train a configured model",
        "  evaluate   Run frozen evaluation",
        "  run        Prepare, train, calibrate and evaluate",
        "  condition  Run one C1-C5 condition",
        "  ablate     Run independently trained ablations",
        "  baseline   Run or summarize baselines",
        "  report     Build publication tables and figures",
        "  status     Show saved progress and what a re-run would still do",
        "",
        "Every long-running command resumes automatically. If a run stops for any",
        "reason, re-run the identical command and it continues from its last",
        "checkpoint. Use --no-resume to deliberately start over.",
    ]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(_help())
        return
    command = args.pop(0)
    target = COMMANDS.get(command)
    if target is None:
        raise SystemExit(f"Unknown command {command!r}\n\n{_help()}")
    module_name, function_name = target
    module = __import__(module_name, fromlist=[function_name])
    getattr(module, function_name)(args)


if __name__ == "__main__":
    main()
