"""Run independently trained GAER++ ablation variants, resumably.

Five variants trained back to back is the longest unattended job in the project
and therefore the one most likely to be cut short. A ledger under
``artifacts/ablations`` records each variant's outcome the moment it changes, so
re-running the same command skips the variants that finished, resumes the one
that was in flight from its own training checkpoint, and continues with the
rest.
"""

from __future__ import annotations

import logging

from clinhallu.config.paths import find_project_root
from clinhallu.core.interrupt import InterruptedRun

from .ledger import Ledger
from .runner import run_experiment

logger = logging.getLogger(__name__)

ABLATION_CONFIGS = {
    "B": "configs/ablations/b_answer_pooling.yaml",
    "C": "configs/ablations/c_cross_attention.yaml",
    "D": "configs/ablations/d_grounding.yaml",
    "E": "configs/ablations/e_evidence_pooling.yaml",
    "F": "configs/ablations/f_full_gaer_pp.yaml",
}


def ablation_ledger_path(root=None):
    root = root or find_project_root()
    return root / "artifacts" / "ablations" / "ablation_ledger.json"


def run_ablations(
    variants: list[str],
    *,
    resume: bool = True,
    continue_on_error: bool = False,
    **kwargs,
) -> list[dict]:
    """Train and evaluate each requested variant, skipping finished ones.

    Args:
        variants: variant letters, any of ``B C D E F``.
        resume: when false, recorded outcomes are discarded and every variant
            runs again from scratch.
        continue_on_error: keep going after a variant fails, so one bad variant
            does not block the other four. Failures are still raised at the end.
    """

    root = find_project_root()
    keys = []
    for variant in variants:
        key = variant.upper()
        if key not in ABLATION_CONFIGS:
            raise ValueError(f"Unknown variant {variant!r}; choose from B C D E F")
        keys.append(key)

    # A seed change produces a different run directory, so it belongs in the
    # unit key rather than in a fingerprint.
    seed = kwargs.get("seed")
    ledger = Ledger(ablation_ledger_path(root), name="ablations")
    if not resume:
        ledger.reset(f"{key}_seed{seed}" for key in keys)

    results: list[dict] = []
    failures: list[str] = []
    for key in keys:
        unit = f"{key}_seed{seed}"
        if ledger.is_complete(unit):
            entry = ledger.entry(unit)
            logger.info("Ablation %s already completed; skipping.", key)
            results.append(
                {
                    "status": "already_completed",
                    "variant": key,
                    "run_dir": entry.get("run_dir"),
                    "metrics": entry.get("metrics"),
                }
            )
            continue

        logger.info("Starting ablation %s", key)
        ledger.start(unit, variant=key, config=ABLATION_CONFIGS[key], seed=seed)
        try:
            result = run_experiment(root / ABLATION_CONFIGS[key], resume=resume, **kwargs)
        except InterruptedRun as exc:
            ledger.interrupt(unit)
            finished = [k for k in keys if ledger.is_complete(f"{k}_seed{seed}")]
            logger.warning(
                "Ablations stopped during variant %s. Completed variants are recorded "
                "in %s and will be skipped on the next run.",
                key,
                ledger.path,
            )
            raise InterruptedRun(
                f"{exc} Completed variants so far: {', '.join(finished) or 'none'}."
            ) from exc
        except Exception as exc:
            ledger.fail(unit, str(exc))
            if not continue_on_error:
                raise
            logger.error("Ablation %s failed: %s", key, exc)
            failures.append(key)
            results.append({"status": "failed", "variant": key, "error": str(exc)})
            continue

        ledger.complete(
            unit,
            variant=key,
            run_dir=result.get("run_dir"),
            metrics=result.get("metrics"),
        )
        results.append({**result, "variant": key})

    if failures:
        raise RuntimeError(
            "Ablation variant(s) failed: "
            + ", ".join(failures)
            + ". Successful variants are recorded and will be skipped on the next run."
        )
    return results
