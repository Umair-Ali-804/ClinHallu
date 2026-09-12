"""Aggregate external baseline runs into tables, summaries, and figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DISPLAY_NAMES = {
    "baseline_tfidf_lr": "TF-IDF + LR",
    "baseline_semantic_similarity": "Biomedical Semantic",
    "baseline_semantic_similarity_claim_level": "Biomedical Semantic (claim-level)",
    "baseline_nli": "NLI",
    "baseline_nli_claim_level": "NLI (claim-level)",
    "baseline_deberta_cls": "DeBERTa-v3 CLS",
    "baseline_llm_judge": "LLM-as-a-Judge",
    "baseline_selfcheck": "SelfCheckGPT (legacy)",
    "baseline_selfcheck_nli_batched": "SelfCheckGPT-NLI (batched)",
    "baseline_self_consistency_vote_batched": "Self-consistency (batched votes)",
    "gaer_plus_plus": "GAER++",
}
METRICS = ("acc", "macro_f1", "f1", "mcc", "auc", "auprc", "brier", "ece")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(root: str | Path) -> list[dict]:
    runs = []
    for metrics_path in Path(root).glob("**/metrics.json"):
        try:
            metrics = _read_json(metrics_path)
            manifest = _read_json(metrics_path.parent / "run_manifest.json")
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        predictions = metrics_path.parent / "predictions.jsonl"
        runs.append(
            {
                "model": manifest["model"],
                "model_name": manifest.get("model_name"),
                "seed": manifest.get("seed"),
                "metrics": metrics,
                "predictions": predictions if predictions.is_file() else None,
            }
        )
    return runs


def display_name(run):
    model = run["model"]
    if model == "baseline_llm_judge" or model.startswith("openrouter_judge::"):
        return "LLM judge: " + str(run.get("model_name") or model)
    return DISPLAY_NAMES.get(model, model)


def aggregate_runs(runs: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for run in runs:
        grouped[display_name(run)].append(run)

    rows = []
    for model, items in grouped.items():
        row = {
            "model": DISPLAY_NAMES.get(model, model),
            "run_count": len(items),
        }
        for metric in METRICS:
            values = [float(item["metrics"][metric]) for item in items if metric in item["metrics"]]
            row[metric] = float(np.mean(values)) if values else None
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
            )
        rows.append(row)

    order = list(DISPLAY_NAMES.values())
    return sorted(
        rows,
        key=lambda row: order.index(row["model"]) if row["model"] in order else 99,
    )


def _format_metric(row: dict, metric: str) -> str:
    if row.get(metric) is None:
        return "—"
    if row["run_count"] > 1:
        return f"{row[metric]:.4f} ± {row[f'{metric}_std']:.4f}"
    return f"{row[metric]:.4f}"


def save_tables(rows: list[dict], tables_dir: Path, summaries_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "run_count",
        *(name for metric in METRICS for name in (metric, f"{metric}_std")),
    ]
    with (tables_dir / "baseline_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (tables_dir / "baseline_results.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    headings = (
        "Model",
        "Accuracy",
        "Macro-F1",
        "F1",
        "MCC",
        "AUROC",
        "AUPRC",
        "Brier",
        "ECE",
    )
    lines = [
        "# External Baseline Comparison",
        "",
        "|" + "|".join(headings) + "|",
        "|" + "|".join(["---", *(["---:"] * 8)]) + "|",
    ]
    for row in rows:
        values = [row["model"], *(_format_metric(row, key) for key in METRICS)]
        lines.append("|" + "|".join(values) + "|")
    lines.extend(
        [
            "",
            "Trained models are reported as mean ± sample standard deviation when "
            "multiple seeds are available. Decision thresholds are selected using "
            "validation data only or fixed in advance, as recorded in each run manifest.",
        ]
    )
    (summaries_dir / "baseline_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def load_prediction_sets(runs: list[dict]) -> list[dict]:
    result = []
    for run in runs:
        if run["predictions"] is None:
            continue
        rows = [
            json.loads(line)
            for line in run["predictions"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            continue
        score_key = (
            "probability_hallucinated"
            if "probability_hallucinated" in rows[0]
            else "hallucination_score"
        )
        result.append(
            {
                "name": display_name(run),
                "gold": np.asarray([row["gold_label"] for row in rows]),
                "scores": np.asarray([row[score_key] for row in rows]),
                "predictions": np.asarray([row["predicted_label"] for row in rows]),
                "rows": rows,
            }
        )
    return result


def save_figures(prediction_sets: list[dict], figures_dir: Path) -> None:
    if not prediction_sets:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_curve,
        roc_curve,
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    names = [item["name"] for item in prediction_sets]
    f1_values = [
        f1_score(item["gold"], item["predictions"], zero_division=0) for item in prediction_sets
    ]
    mcc_values = [matthews_corrcoef(item["gold"], item["predictions"]) for item in prediction_sets]
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(max(8, len(names) * 1.25), 5))
    axis.bar(x - 0.2, f1_values, 0.4, label="F1")
    axis.bar(x + 0.2, mcc_values, 0.4, label="MCC")
    axis.set_xticks(x, names, rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("External baseline performance")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, figures_dir / "baseline_comparison")

    figure, axis = plt.subplots(figsize=(7, 5))
    for item in prediction_sets:
        if len(np.unique(item["gold"])) < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            item["gold"],
            item["scores"],
        )
        axis.plot(false_positive_rate, true_positive_rate, label=item["name"], linewidth=2)
    axis.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="ROC curves",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, figures_dir / "baseline_roc_curve")

    figure, axis = plt.subplots(figsize=(7, 5))
    for item in prediction_sets:
        if len(np.unique(item["gold"])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(
            item["gold"],
            item["scores"],
        )
        axis.plot(recall, precision, label=item["name"], linewidth=2)
    axis.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, figures_dir / "baseline_precision_recall_curve")

    columns = min(3, len(prediction_sets))
    rows_count = int(np.ceil(len(prediction_sets) / columns))
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(4 * columns, 3.5 * rows_count),
        squeeze=False,
    )
    for axis, item in zip(axes.flat, prediction_sets, strict=False):
        ConfusionMatrixDisplay(
            confusion_matrix(
                item["gold"],
                item["predictions"],
                labels=[0, 1],
            ),
            display_labels=["Grounded", "Hallucinated"],
        ).plot(ax=axis, colorbar=False)
        axis.set_title(item["name"])
    for axis in axes.flat[len(prediction_sets) :]:
        axis.axis("off")
    figure.tight_layout()
    _save_figure(figure, figures_dir / "baseline_confusion_matrix")


def _save_figure(figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(stem.with_suffix(".pdf"))
    import matplotlib.pyplot as plt

    plt.close(figure)


def save_subgroup_table(
    prediction_sets: list[dict],
    metadata_key: str,
    output_path: Path,
) -> None:
    from sklearn.metrics import f1_score

    groups = sorted(
        {
            str(row[metadata_key])
            for item in prediction_sets
            for row in item["rows"]
            if row.get(metadata_key) not in (None, "")
        }
    )
    if not groups:
        return
    lines = [
        f"# {metadata_key.replace('_', ' ').title()} F1",
        "",
        "| Model | " + " | ".join(groups) + " | Overall |",
        "|---|" + "---:|" * (len(groups) + 1),
    ]
    for item in prediction_sets:
        values = []
        for group in groups:
            subset = [row for row in item["rows"] if str(row.get(metadata_key)) == group]
            if subset:
                gold = [row["gold_label"] for row in subset]
                predicted = [row["predicted_label"] for row in subset]
                values.append(f"{f1_score(gold, predicted, zero_division=0):.4f}")
            else:
                values.append("—")
        values.append(f"{f1_score(item['gold'], item['predictions'], zero_division=0):.4f}")
        lines.append("| " + " | ".join([item["name"], *values]) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_baselines(results_root: str | Path, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    tables = output / "tables"
    figures = output / "figures"
    summaries = output / "summaries"
    runs = discover_runs(results_root)
    rows = aggregate_runs(runs)
    save_tables(rows, tables, summaries)
    prediction_sets = load_prediction_sets(runs)
    save_figures(prediction_sets, figures)
    save_subgroup_table(prediction_sets, "difficulty", summaries / "difficulty_results.md")
    save_subgroup_table(
        prediction_sets,
        "hallucination_category",
        summaries / "hallucination_category_results.md",
    )
    save_subgroup_table(prediction_sets, "category", summaries / "category_results.md")
    return {
        "runs": len(runs),
        "output_dir": str(output),
        "models": [row["model"] for row in rows],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", "--results_root", default="artifacts/baselines")
    parser.add_argument("--output-dir", "--output_dir", default="artifacts/reports")
    args = parser.parse_args(argv)
    result = summarize_baselines(args.results_root, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
