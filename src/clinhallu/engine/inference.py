"""Frozen evaluation for a trained and validation-calibrated model."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from clinhallu.config import load_config
from clinhallu.core.logging import configure_logging
from clinhallu.data.cache import load_or_build_feature_dataset
from clinhallu.data.context_selection import context_selection_from_config
from clinhallu.data.dataset import HKGFusionDataset, collate_fn_v2
from clinhallu.data.leakage import canonical_group_key
from clinhallu.models import build_model

from .calibration import load_calibration
from .checkpoints import load_checkpoint
from .metrics import (
    compute_binary_metrics,
    compute_ece,
    grouped_bootstrap_confidence_interval,
)
from .prediction_writer import write_predictions
from .validation import autocast_for, scalar_logits

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(model, loader, device: torch.device, mixed_precision: bool) -> dict:
    labels: list[int] = []
    probabilities: list[float] = []
    raw_logits: list[float] = []
    evidence_rows: list[dict] = []
    model.eval()
    for raw_batch in tqdm(loader, desc="frozen evaluation", leave=False):
        batch = {
            key: value.to(device, non_blocking=(device.type == "cuda"))
            for key, value in raw_batch.items()
        }
        batch_labels = batch.pop("labels")
        with autocast_for(device, mixed_precision):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                context_mask=batch["context_mask"],
                answer_mask=batch["answer_mask"],
            )
        labels.extend(batch_labels.detach().cpu().tolist())
        probabilities.extend(output.p_hallucinated.detach().float().cpu().tolist())
        raw_logits.extend(scalar_logits(output).detach().float().cpu().tolist())
        stats = getattr(output, "evidence_stats", None)
        if stats:
            for index in range(len(batch_labels)):
                evidence_rows.append(
                    {
                        name: float(values[index].detach().float().cpu().item())
                        for name, values in stats.items()
                    }
                )
        else:
            evidence_rows.extend({} for _ in range(len(batch_labels)))
    return {
        "labels": labels,
        "probabilities": probabilities,
        "logits": raw_logits,
        "evidence_stats": evidence_rows,
    }


def evaluate(cfg: dict, *, force: bool = False) -> dict:
    metrics_path = Path(cfg["results"]["metrics_dir"]) / "evaluation_metrics.json"
    if metrics_path.is_file() and not force:
        logger.info("Evaluation already complete: %s", metrics_path)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    seed = int(cfg["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["evaluation"].get("mixed_precision", True) and device.type == "cuda")

    checkpoint_path = Path(cfg["checkpoints"]["dir"]) / cfg["checkpoints"]["best_filename"]
    model = build_model(cfg, gradient_checkpointing=False).to(device)
    payload = load_checkpoint(str(checkpoint_path), device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    temperature, threshold = load_calibration(cfg["calibration"]["dir"], required=True)
    revision = cfg["model"].get("revision")
    tokenizer_kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder"], **tokenizer_kwargs)
    dataset_path = Path(cfg["data"]["processed_dir"]) / "test.jsonl"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Processed frozen test split not found: {dataset_path}. Run prepare first."
        )
    if cfg["data"].get("use_cache", True):
        dataset = load_or_build_feature_dataset(cfg, tokenizer, "test")
    else:
        dataset = HKGFusionDataset(
            str(dataset_path),
            tokenizer,
            max_length=int(cfg["model"]["max_length"]),
            context_selection=context_selection_from_config(cfg),
            question_budget=int(cfg["data"]["question_budget"]),
            answer_budget=int(cfg["data"]["answer_budget"]),
        )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["evaluation"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn_v2,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )
    collected = collect_predictions(model, loader, device, use_amp)
    calibrated = torch.sigmoid(
        torch.tensor(collected["logits"], dtype=torch.float32) / temperature
    ).numpy()
    predictions = (calibrated >= threshold).astype(int)

    metrics = compute_binary_metrics(
        collected["labels"],
        predictions.tolist(),
        calibrated.tolist(),
        prefix="eval_",
    )
    metrics["eval_ece_raw"] = compute_ece(collected["labels"], collected["probabilities"])
    metrics["eval_ece_calibrated"] = compute_ece(collected["labels"], calibrated.tolist())
    metrics["eval_ece_delta"] = round(metrics["eval_ece_raw"] - metrics["eval_ece_calibrated"], 6)
    groups = [canonical_group_key(sample) for sample in dataset.samples]
    confidence = grouped_bootstrap_confidence_interval(
        collected["labels"],
        predictions.tolist(),
        groups,
        metric_fn=lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
        n_bootstrap=int(cfg["evaluation"].get("bootstrap_samples", 1000)),
        seed=seed,
    )
    metrics["eval_f1_ci_low"] = confidence["ci_low"]
    metrics["eval_f1_ci_high"] = confidence["ci_high"]
    metrics["temperature"] = temperature
    metrics["threshold"] = threshold
    metrics["protocol"] = "frozen_test_after_validation_calibration"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(
        dataset.samples,
        collected["labels"],
        collected["probabilities"],
        calibrated.tolist(),
        predictions.tolist(),
        Path(cfg["results"]["predictions_dir"]) / "evaluation_predictions.jsonl",
        collected["evidence_stats"],
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Frozen evaluation saved to %s", metrics_path)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, seed=args.seed)
    configure_logging(
        cfg["logging"].get("level", "INFO"),
        Path(cfg["results"]["logs_dir"]) / "evaluation.log",
    )
    print(json.dumps(evaluate(cfg, force=args.force), indent=2))


if __name__ == "__main__":
    main()
