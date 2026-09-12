"""Create adaptation and ablation CSV/Markdown tables from run artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

METRIC_KEYS = (
    "eval_acc",
    "eval_precision",
    "eval_recall",
    "eval_f1",
    "eval_mcc",
    "eval_auc",
    "eval_auprc",
    "eval_brier",
    "eval_ece_calibrated",
)


def collect_metrics(root: str | Path) -> list[dict]:
    base = Path(root)
    rows = []
    for path in sorted(base.glob("**/metrics/evaluation_metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(base)
        parts = relative.parts
        name = parts[0] if parts else path.parent.name
        seed = next((part for part in parts if part.startswith("seed_")), "")
        row = {"name": name, "seed": seed}
        row.update({key: metrics.get(key) for key in METRIC_KEYS})
        rows.append(row)
    return rows


def write_result_table(rows: list[dict], csv_path: str | Path, markdown_path: str | Path) -> None:
    fields = ("name", "seed", *METRIC_KEYS)
    csv_target = Path(csv_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    with csv_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "| Run | Seed | F1 | MCC | AUROC | AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def show(row, key):
        value = row.get(key)
        return "—" if value is None else f"{float(value):.4f}"

    for row in rows:
        md.append(
            f"| {row['name']} | {row['seed']} | {show(row, 'eval_f1')} | "
            f"{show(row, 'eval_mcc')} | {show(row, 'eval_auc')} | "
            f"{show(row, 'eval_auprc')} |"
        )
    target = Path(markdown_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(md) + "\n", encoding="utf-8")
