"""Build publication-ready tables and primary-model figures."""

from __future__ import annotations

from pathlib import Path

from .plots import build_primary_figures
from .summarize import summarize_baselines
from .tables import collect_metrics, write_result_table


def build_publication_outputs(project_root: str | Path) -> dict:
    root = Path(project_root)
    reports = root / "artifacts" / "reports"
    tables = reports / "tables"
    summaries = reports / "summaries"
    adaptation_rows = collect_metrics(root / "artifacts" / "adaptation")
    ablation_rows = collect_metrics(root / "artifacts" / "ablations")
    write_result_table(
        adaptation_rows,
        tables / "adaptation_results.csv",
        summaries / "main_results.md",
    )
    write_result_table(
        adaptation_rows,
        tables / "main_results.csv",
        summaries / "main_results.md",
    )
    write_result_table(
        ablation_rows,
        tables / "ablation_results.csv",
        summaries / "ablation_results.md",
    )
    baseline = summarize_baselines(root / "artifacts" / "baselines", reports)
    primary_run = _latest_primary_run(root / "artifacts" / "adaptation")
    if primary_run is not None:
        build_primary_figures(primary_run, reports / "figures")
    return {
        "adaptation_runs": len(adaptation_rows),
        "ablation_runs": len(ablation_rows),
        "baseline_runs": baseline["runs"],
        "reports": str(reports),
    }


def _latest_primary_run(adaptation_root: Path) -> Path | None:
    completed = sorted(
        path.parent.parent for path in adaptation_root.glob("**/metrics/evaluation_metrics.json")
    )
    if not completed:
        return None
    c5_runs = [path for path in completed if path.parent.name == "c5"]
    return (c5_runs or completed)[-1]
