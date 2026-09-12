"""A crash part-way through a multi-stage job must not restart the whole job."""

from __future__ import annotations

import json

import pytest

from clinhallu.core.interrupt import INTERRUPTED_EXIT_CODE, InterruptedRun
from clinhallu.experiments import ablation
from clinhallu.experiments.ledger import Ledger


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "artifacts" / "ablations").mkdir(parents=True)
    (tmp_path / "configs" / "ablations").mkdir(parents=True)
    for path in ablation.ABLATION_CONFIGS.values():
        (tmp_path / path).write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(ablation, "find_project_root", lambda *a, **k: tmp_path)
    return tmp_path


def test_completed_variants_are_skipped_on_the_next_run(project, monkeypatch):
    calls: list[str] = []
    first_attempt = {"value": True}

    def fake_run(config_path, **kwargs):
        name = str(config_path)
        calls.append(name)
        if "d_grounding" in name and first_attempt["value"]:
            first_attempt["value"] = False
            raise InterruptedRun("session ended")
        return {"status": "evaluated", "run_dir": f"run/{name}", "metrics": "m.json"}

    monkeypatch.setattr(ablation, "run_experiment", fake_run)

    # First attempt: B and C finish, D is cut off by a lost session.
    with pytest.raises(InterruptedRun):
        ablation.run_ablations(["B", "C", "D", "E", "F"], seed=13)
    assert len(calls) == 3

    # Second attempt: B and C are not touched again.
    calls.clear()
    results = ablation.run_ablations(["B", "C", "D", "E", "F"], seed=13)
    assert not any("b_answer_pooling" in c or "c_cross_attention" in c for c in calls)
    assert len(calls) == 3  # only D, E, F
    statuses = {r["variant"]: r["status"] for r in results}
    assert statuses["B"] == "already_completed"
    assert statuses["F"] == "evaluated"


def test_the_interrupted_variant_is_retried_not_skipped(project, monkeypatch):
    calls: list[str] = []

    def fake_run(config_path, **kwargs):
        calls.append(str(config_path))
        if "c_cross_attention" in str(config_path) and len(calls) == 2:
            raise InterruptedRun("stopped")
        return {"status": "evaluated", "run_dir": "r", "metrics": "m"}

    monkeypatch.setattr(ablation, "run_experiment", fake_run)
    with pytest.raises(InterruptedRun):
        ablation.run_ablations(["B", "C", "D"], seed=13)

    calls.clear()
    ablation.run_ablations(["B", "C", "D"], seed=13)
    assert any("c_cross_attention" in c for c in calls), "the cut-off variant must run again"


def test_no_resume_reruns_everything(project, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        ablation,
        "run_experiment",
        lambda config_path, **kwargs: (
            calls.append(str(config_path))
            or {"status": "evaluated", "run_dir": "r", "metrics": "m"}
        ),
    )
    ablation.run_ablations(["B", "C"], seed=13)
    calls.clear()
    ablation.run_ablations(["B", "C"], seed=13, resume=False)
    assert len(calls) == 2


def test_a_different_seed_is_tracked_separately(project, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        ablation,
        "run_experiment",
        lambda config_path, **kwargs: (
            calls.append(kwargs.get("seed"))
            or {"status": "evaluated", "run_dir": "r", "metrics": "m"}
        ),
    )
    ablation.run_ablations(["B"], seed=13)
    ablation.run_ablations(["B"], seed=42)
    assert calls == [13, 42], "seed 42 must not inherit seed 13's completion record"


def test_continue_on_error_still_records_the_successes(project, monkeypatch):
    def fake_run(config_path, **kwargs):
        if "c_cross_attention" in str(config_path):
            raise RuntimeError("CUDA out of memory")
        return {"status": "evaluated", "run_dir": "r", "metrics": "m"}

    monkeypatch.setattr(ablation, "run_experiment", fake_run)
    with pytest.raises(RuntimeError, match="failed"):
        ablation.run_ablations(["B", "C", "D"], seed=13, continue_on_error=True)

    ledger = Ledger(ablation.ablation_ledger_path(project))
    assert ledger.is_complete("B_seed13")
    assert ledger.is_complete("D_seed13")  # D still ran despite C failing
    assert not ledger.is_complete("C_seed13")


def test_pipeline_skips_completed_stages(tmp_path, monkeypatch):
    from clinhallu.cli import pipeline

    monkeypatch.setattr(pipeline, "find_project_root", lambda *a, **k: tmp_path)
    executed: list[str] = []

    class _Result:
        def __init__(self, code):
            self.returncode = code

    def fake_subprocess(command, cwd=None, check=False):
        stage = command[3]
        executed.append(stage)
        # The baselines stage dies the first time it is reached.
        return _Result(1 if stage == "baseline" and "baseline" not in done else 0)

    done: set[str] = set()
    monkeypatch.setattr(pipeline.subprocess, "run", fake_subprocess)

    with pytest.raises(SystemExit):
        pipeline.main(["--condition", "c5", "--seed", "13"])
    assert executed == ["condition", "ablate", "generate", "baseline"]

    # Second attempt: the three finished stages are not re-executed.
    done.add("baseline")
    executed.clear()
    pipeline.main(["--condition", "c5", "--seed", "13"])
    assert executed == ["baseline", "report"]


def test_pipeline_records_a_clean_stop_distinctly(tmp_path, monkeypatch):
    from clinhallu.cli import pipeline

    monkeypatch.setattr(pipeline, "find_project_root", lambda *a, **k: tmp_path)

    class _Result:
        def __init__(self, code):
            self.returncode = code

    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda command, cwd=None, check=False: _Result(
            INTERRUPTED_EXIT_CODE if command[3] == "ablate" else 0
        ),
    )
    with pytest.raises(SystemExit) as exit_info:
        pipeline.main(["--condition", "c5", "--seed", "13"])
    assert exit_info.value.code == INTERRUPTED_EXIT_CODE

    ledger_file = pipeline.ledger_path(tmp_path, "c5", 13)
    units = json.loads(ledger_file.read_text())["units"]
    assert units["model"]["status"] == "completed"
    assert units["ablations"]["status"] == "interrupted"
