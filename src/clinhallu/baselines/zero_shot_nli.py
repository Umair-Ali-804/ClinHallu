"""Claim-level zero-shot natural-language-inference baseline."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .cache import cached_scores
from .common import finish_run
from .data import load_splits, sentence_split

KNOWN_LABEL_ORDERS = {
    # Official model card order: contradiction, entailment, neutral.
    "cross-encoder/nli-deberta-v3-base": (1, 0),
}


def aggregate_scores(entailment, contradiction=None, alpha: float = 0.0) -> float:
    """Turn claim-level NLI support into a bounded hallucination score."""

    maximum_entailment = float(np.max(entailment)) if len(entailment) else 0.0
    if contradiction is None or alpha <= 0:
        return float(np.clip(1.0 - maximum_entailment, 0.0, 1.0))
    maximum_contradiction = float(np.max(contradiction)) if len(contradiction) else 0.0
    score = (maximum_contradiction + alpha * (1.0 - maximum_entailment)) / (1.0 + alpha)
    return float(np.clip(score, 0.0, 1.0))


def label_indices(
    model,
    model_name: str | None = None,
    entailment_index: int | None = None,
    contradiction_index: int | None = None,
) -> tuple[int, int | None]:
    """Resolve NLI labels from model metadata or a documented override."""

    if entailment_index is not None:
        if not 0 <= entailment_index < model.config.num_labels:
            raise ValueError(f"Invalid entailment index {entailment_index}")
        if contradiction_index is not None and not (
            0 <= contradiction_index < model.config.num_labels
        ):
            raise ValueError(f"Invalid contradiction index {contradiction_index}")
        return entailment_index, contradiction_index

    labels = {int(index): str(value).lower() for index, value in model.config.id2label.items()}
    entailment = next(
        (index for index, value in labels.items() if "entail" in value),
        None,
    )
    contradiction = next(
        (index for index, value in labels.items() if "contrad" in value),
        None,
    )
    if entailment is None and model_name in KNOWN_LABEL_ORDERS:
        entailment, contradiction = KNOWN_LABEL_ORDERS[model_name]
        print(
            f"Generic labels detected; using the documented order for {model_name}: "
            "contradiction=0, entailment=1, neutral=2."
        )
    if entailment is None:
        raise ValueError(
            f"Cannot identify the entailment label from {labels}. Pass "
            "--entailment-index and --contradiction-index only after verifying "
            "the checkpoint's model card."
        )
    return entailment, contradiction


def context_chunks(text: str, max_sentences: int, window: int) -> list[str]:
    sentences = sentence_split(text, max_sentences)
    if window <= 1:
        return sentences
    return [" ".join(sentences[index : index + window]) for index in range(len(sentences))]


def score(
    records: list[dict],
    tokenizer,
    model,
    device,
    batch_size: int,
    max_sentences: int,
    context_window: int,
    alpha: float,
    model_name: str | None = None,
    entailment_index: int | None = None,
    contradiction_index: int | None = None,
) -> list[float]:
    import torch

    entailment_index, contradiction_index = label_indices(
        model,
        model_name,
        entailment_index,
        contradiction_index,
    )
    output: list[float] = []
    for block_start in range(0, len(records), 64):
        block = records[block_start : block_start + 64]
        premises: list[str] = []
        hypotheses: list[str] = []
        owners: list[int] = []
        claim_counts: list[int] = []
        claim_index = 0

        for row in block:
            claims = sentence_split(row["answer"])
            chunks = context_chunks(row["context"], max_sentences, context_window)
            claim_counts.append(len(claims))
            for claim in claims:
                premises.extend(chunks)
                hypotheses.extend([claim] * len(chunks))
                owners.extend([claim_index] * len(chunks))
                claim_index += 1

        entailment_scores = [[] for _ in range(claim_index)]
        contradiction_scores = [[] for _ in range(claim_index)]
        for start in range(0, len(premises), batch_size):
            encoded = tokenizer(
                premises[start : start + batch_size],
                hypotheses[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda",
                    enabled=device.type == "cuda",
                ):
                    probabilities = (
                        torch.softmax(
                            model(**encoded).logits,
                            dim=-1,
                        )
                        .float()
                        .cpu()
                        .numpy()
                    )
            batch_owners = owners[start : start + batch_size]
            for offset, owner in enumerate(batch_owners):
                entailment_scores[owner].append(probabilities[offset, entailment_index])
                contradiction_scores[owner].append(
                    probabilities[offset, contradiction_index]
                    if contradiction_index is not None
                    else 0.0
                )

        offset = 0
        for count in claim_counts:
            claim_scores = [
                aggregate_scores(
                    entailment_scores[offset + index],
                    contradiction_scores[offset + index],
                    alpha,
                )
                for index in range(count)
            ]
            output.append(max(claim_scores) if claim_scores else 1.0)
            offset += count
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.jsonl")
    parser.add_argument("--val", default="data/raw/val.jsonl")
    parser.add_argument("--eval", default="data/raw/eval_data.jsonl")
    parser.add_argument(
        "--model-name",
        "--model_name",
        default="cross-encoder/nli-deberta-v3-base",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16)
    parser.add_argument(
        "--max-context-sentences",
        "--max_context_sentences",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--context-window",
        "--context_window",
        type=int,
        choices=(1, 2),
        default=2,
    )
    parser.add_argument(
        "--contradiction-alpha",
        "--contradiction_alpha",
        type=float,
        default=0.0,
    )
    parser.add_argument("--entailment-index", "--entailment_index", type=int)
    parser.add_argument("--contradiction-index", "--contradiction_index", type=int)
    parser.add_argument(
        "--threshold-objective",
        "--threshold_objective",
        choices=("f1", "macro_f1"),
        default="macro_f1",
    )
    parser.add_argument("--cache-dir", "--cache_dir", default="data/cache/baselines")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="artifacts/baselines/zero_shot_nli",
    )
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    splits = load_splits(args.train, args.val, args.eval)
    _, validation, evaluation = splits
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            revision=args.revision,
        )
        .to(device)
        .eval()
    )
    entailment_index, contradiction_index = label_indices(
        model,
        args.model_name,
        args.entailment_index,
        args.contradiction_index,
    )
    config = {
        "variant": "claim_level",
        "revision": args.revision,
        "batch_size": args.batch_size,
        "max_context_sentences": args.max_context_sentences,
        "context_window": args.context_window,
        "contradiction_alpha": args.contradiction_alpha,
        "entailment_index": entailment_index,
        "contradiction_index": contradiction_index,
        "claim_aggregation": "maximum unsupported claim",
        "threshold_objective": args.threshold_objective,
        "aggregation": (
            "max_claim_one_minus_max_entailment"
            if args.contradiction_alpha == 0
            else "claim_entailment_plus_contradiction"
        ),
    }

    def scorer(rows):
        return score(
            rows,
            tokenizer,
            model,
            device,
            args.batch_size,
            args.max_context_sentences,
            args.context_window,
            args.contradiction_alpha,
            args.model_name,
            entailment_index,
            contradiction_index,
        )

    validation_scores = cached_scores(
        args.cache_dir,
        "nli_claim_level",
        args.model_name,
        args.revision,
        args.val,
        config,
        lambda: scorer(validation),
    )
    evaluation_scores = cached_scores(
        args.cache_dir,
        "nli_claim_level",
        args.model_name,
        args.revision,
        args.eval,
        config,
        lambda: scorer(evaluation),
    )
    result = finish_run(
        args.output_dir,
        "baseline_nli_claim_level",
        args.model_name,
        None,
        (args.train, args.val, args.eval),
        config,
        splits,
        validation_scores,
        evaluation_scores,
        score_is_probability=False,
        threshold_objective=args.threshold_objective,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
