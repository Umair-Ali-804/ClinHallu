"""Plotting functions for experiment diagnostics and publication figures."""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_PALETTE = sns.color_palette("colorblind", 10)
_FONT_SIZE = 10
_DPI = 300

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": _FONT_SIZE,
        "axes.titlesize": _FONT_SIZE + 1,
        "axes.labelsize": _FONT_SIZE,
        "xtick.labelsize": _FONT_SIZE - 1,
        "ytick.labelsize": _FONT_SIZE - 1,
        "legend.fontsize": _FONT_SIZE - 1,
        "figure.dpi": _DPI,
        "savefig.dpi": _DPI,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _ensure_dir(plots_dir: str) -> Path:
    p = Path(plots_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_ablation_heatmap(
    ablation_csv: str,
    metrics: List[str],
    plots_dir: str,
    filename: str = "ablation_heatmap.pdf",
) -> None:
    df = pd.read_csv(ablation_csv)
    avail = [m for m in metrics if m in df.columns]
    if not avail:
        logger.warning("No recognised metric columns in %s — skipping heatmap.", ablation_csv)
        return

    index_col = "variant" if "variant" in df.columns else df.columns[0]
    df = df.set_index(index_col)[avail]

    labels = [c.replace("test_", "").replace("binary_", "bin_").replace("hall_", "") for c in avail]

    fig, ax = plt.subplots(figsize=(max(6, len(avail) * 1.1), max(4, len(df) * 0.6)))
    sns.heatmap(
        df.rename(columns=dict(zip(avail, labels, strict=True))),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Score"},
    )
    ax.set_title("Ablation Study — ClinHallu-Detect", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Variant")
    plt.tight_layout()

    out = _ensure_dir(plots_dir) / filename
    fig.savefig(out)
    plt.close(fig)
    logger.info("Ablation heatmap saved to %s", out)


def plot_model_comparison_bar(
    results_df: pd.DataFrame,
    metric: str,
    plots_dir: str,
    title: str = "Model Comparison",
    filename: str = "model_comparison.pdf",
) -> None:
    if metric not in results_df.columns:
        logger.warning("Metric '%s' not found in DataFrame — skipping bar chart.", metric)
        return

    df = results_df[["model", metric]].dropna().sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(6, max(3, len(df) * 0.5)))
    bars = ax.barh(
        df["model"],
        df[metric],
        color=_PALETTE[: len(df)],
        edgecolor="white",
        height=0.6,
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=_FONT_SIZE - 1)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title, pad=8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = _ensure_dir(plots_dir) / filename
    fig.savefig(out)
    plt.close(fig)
    logger.info("Bar chart saved to %s", out)


def plot_calibration_curve(
    y_true: List[int],
    y_prob: List[float],
    plots_dir: str,
    model_name: str = "ClinHallu-Detect",
    n_bins: int = 10,
    filename: str = "calibration_curve.pdf",
) -> None:
    try:
        from sklearn.calibration import calibration_curve
    except ImportError:
        logger.warning("sklearn not available — skipping calibration plot.")
        return

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", color=_PALETTE[0], label=model_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    ax2 = axes[1]
    ax2.hist(y_prob, bins=n_bins, color=_PALETTE[1], edgecolor="white")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence distribution")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = _ensure_dir(plots_dir) / filename
    fig.savefig(out)
    plt.close(fig)
    logger.info("Calibration curve saved to %s", out)


def plot_roc_curves(
    models: Dict[str, Dict],
    plots_dir: str,
    filename: str = "roc_curves.pdf",
) -> None:
    try:
        from sklearn.metrics import auc, roc_curve
    except ImportError:
        logger.warning("sklearn not available — skipping ROC plot.")
        return

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)

    for i, (name, data) in enumerate(models.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=_PALETTE[i % len(_PALETTE)],
            linewidth=1.5,
            label=f"{name} (AUC={roc_auc:.3f})",
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — ClinHallu-Detect")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = _ensure_dir(plots_dir) / filename
    fig.savefig(out)
    plt.close(fig)
    logger.info("ROC curves saved to %s", out)


def plot_training_curves(
    train_log_csv: str,
    plots_dir: str,
    filename: str = "training_curves.pdf",
) -> None:
    csv_path = Path(train_log_csv)
    if not csv_path.exists():
        logger.warning("Training log not found: %s — skipping training curves.", csv_path)
        return

    df = pd.read_csv(csv_path)

    metrics_to_plot = {
        "Loss": ("train_loss", "val_loss"),
        "Binary F1": ("train_binary_f1", "val_binary_f1"),
    }

    available = {
        title: (tc, vc)
        for title, (tc, vc) in metrics_to_plot.items()
        if tc in df.columns and vc in df.columns
    }

    if not available:
        logger.warning("No recognised metric columns in training log — skipping.")
        return

    n_plots = len(available)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 3.5))
    if n_plots == 1:
        axes = [axes]

    for ax, (title, (train_col, val_col)) in zip(axes, available.items(), strict=True):
        epochs = np.arange(len(df))
        ax.plot(epochs, df[train_col], "-o", markersize=3, color=_PALETTE[0], label="Train")
        ax.plot(epochs, df[val_col], "-s", markersize=3, color=_PALETTE[1], label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("ClinHallu-Detect — Training Curves", y=1.02)
    plt.tight_layout()

    out = _ensure_dir(plots_dir) / filename
    fig.savefig(out)
    plt.close(fig)
    logger.info("Training curves saved to %s", out)


def build_primary_figures(run_dir: str | Path, figures_dir: str | Path) -> None:
    """Build the standard figure set for one completed GAER++ run."""

    run = Path(run_dir)
    output = Path(figures_dir)
    prediction_path = run / "predictions" / "evaluation_predictions.jsonl"
    history_path = run / "metrics" / "training_history.csv"
    output.mkdir(parents=True, exist_ok=True)

    if prediction_path.is_file():
        rows = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if rows:
            labels = np.asarray([row["hallu_label"] for row in rows])
            probabilities = np.asarray([row["p_hallucinated_calibrated"] for row in rows])
            predictions = np.asarray([row["prediction"] for row in rows])
            _primary_prediction_figures(
                labels,
                probabilities,
                predictions,
                output,
            )
    if history_path.is_file():
        _primary_learning_curve(history_path, output)


def _primary_prediction_figures(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    figures_dir: Path,
) -> None:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )

    if len(np.unique(labels)) == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            labels,
            probabilities,
        )
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(false_positive_rate, true_positive_rate, linewidth=2)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set(
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            title="ROC curve",
        )
        axis.grid(alpha=0.25)
        _save_png_pdf(figure, figures_dir / "roc_curve")

        precision, recall, _ = precision_recall_curve(labels, probabilities)
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(recall, precision, linewidth=2)
        axis.set(
            xlabel="Recall",
            ylabel="Precision",
            title="Precision-recall curve",
        )
        axis.grid(alpha=0.25)
        _save_png_pdf(figure, figures_dir / "precision_recall_curve")

        observed, predicted = calibration_curve(
            labels,
            probabilities,
            n_bins=10,
            strategy="uniform",
        )
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        axis.plot(predicted, observed, "o-", linewidth=2, label="GAER++")
        axis.set(
            xlabel="Mean predicted probability",
            ylabel="Observed hallucination rate",
            title="Reliability diagram",
        )
        axis.legend()
        axis.grid(alpha=0.25)
        _save_png_pdf(figure, figures_dir / "reliability_diagram")

    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(labels, predictions, labels=[0, 1]),
        display_labels=["Grounded", "Hallucinated"],
    ).plot(ax=axis, colorbar=False)
    axis.set_title("GAER++ confusion matrix")
    figure.tight_layout()
    _save_png_pdf(figure, figures_dir / "confusion_matrix")


def _primary_learning_curve(history_path: Path, figures_dir: Path) -> None:
    history = pd.read_csv(history_path)
    if history.empty or "epoch" not in history:
        return

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for column, label in (
        ("train_loss", "Train"),
        ("validation_loss", "Validation"),
    ):
        if column in history:
            axes[0].plot(history["epoch"], history[column], "o-", label=label)
    for column, label in (
        ("train_f1", "Train F1"),
        ("validation_f1_threshold_optimized", "Validation F1"),
    ):
        if column in history:
            axes[1].plot(history["epoch"], history[column], "o-", label=label)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss")
    axes[1].set(xlabel="Epoch", ylabel="F1", title="Model selection")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    _save_png_pdf(figure, figures_dir / "learning_curve")


def _save_png_pdf(figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
