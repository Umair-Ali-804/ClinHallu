"""Claim-level biomedical semantic-similarity baseline."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .cache import cached_scores
from .common import finish_run
from .data import load_splits, sentence_split


def load_sentence_encoder(model_name: str, revision: str):
    """Load current and legacy SentenceTransformers checkpoints.

    Some older checkpoints store the pooling dimension under a legacy key.
    Rebuilding their standard Transformer + mean-pooling stack preserves the
    intended operation without modifying the Hugging Face cache.
    """

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules

    try:
        return SentenceTransformer(model_name, revision=revision)
    except TypeError as exc:
        message = str(exc)
        if "Pooling.__init__() missing" not in message or "embedding_dimension" not in message:
            raise
        print(
            "Legacy pooling configuration detected; rebuilding the "
            "Transformer + mean-pooling sentence encoder."
        )
        transformer = modules.Transformer(
            model_name,
            model_args={"revision": revision},
            tokenizer_args={"revision": revision},
        )
        pooling = modules.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        return SentenceTransformer(modules=[transformer, pooling])


def normalized_hallucination_score(similarity) -> np.ndarray:
    """Map cosine support from [-1, 1] to hallucination score [1, 0]."""

    return np.clip((1.0 - np.asarray(similarity, dtype=float)) / 2.0, 0.0, 1.0)


def context_chunks(text: str, max_sentences: int, window: int) -> list[str]:
    sentences = sentence_split(text, max_sentences)
    if window <= 1:
        return sentences
    return [" ".join(sentences[index : index + window]) for index in range(len(sentences))]


def score(
    records: list[dict],
    encoder,
    batch_size: int,
    top_k: int,
    max_sentences: int,
    context_window: int,
) -> list[float]:
    """Score an answer by its weakest-supported claim."""

    output: list[float] = []
    for block_start in range(0, len(records), 128):
        block = records[block_start : block_start + 128]
        texts: list[str] = []
        example_offsets: list[list[tuple[int, int]]] = []

        for row in block:
            claims = sentence_split(row["answer"])
            chunks = context_chunks(row["context"], max_sentences, context_window)
            claim_offsets = []
            for claim in claims:
                offset = len(texts)
                texts.extend([claim, *chunks])
                claim_offsets.append((offset, len(chunks)))
            example_offsets.append(claim_offsets)

        vectors = np.asarray(
            encoder.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        for claim_offsets in example_offsets:
            claim_support = []
            for offset, chunk_count in claim_offsets:
                similarities = vectors[offset + 1 : offset + 1 + chunk_count] @ vectors[offset]
                selected = min(top_k, len(similarities))
                support = float(np.sort(similarities)[-selected:].mean()) if selected else -1.0
                claim_support.append(support)
            answer_support = min(claim_support) if claim_support else -1.0
            output.append(float(normalized_hallucination_score([answer_support])[0]))
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.jsonl")
    parser.add_argument("--val", default="data/raw/val.jsonl")
    parser.add_argument("--eval", default="data/raw/eval_data.jsonl")
    parser.add_argument("--model-name", "--model_name", default="pritamdeka/S-PubMedBert-MS-MARCO")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument("--top-k", "--top_k", type=int, default=1)
    parser.add_argument("--max-sentences", "--max_sentences", type=int, default=32)
    parser.add_argument("--context-window", "--context_window", type=int, choices=(1, 2), default=2)
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
        default="artifacts/baselines/semantic_similarity",
    )
    args = parser.parse_args(argv)

    splits = load_splits(args.train, args.val, args.eval)
    _, validation, evaluation = splits
    encoder = load_sentence_encoder(args.model_name, args.revision)
    config = {
        "variant": "claim_level",
        "top_k": args.top_k,
        "max_sentences": args.max_sentences,
        "context_window": args.context_window,
        "batch_size": args.batch_size,
        "claim_aggregation": "minimum support",
        "threshold_objective": args.threshold_objective,
        "score": "(1-min_claim_support)/2",
    }

    def scorer(rows):
        return score(
            rows,
            encoder,
            args.batch_size,
            args.top_k,
            args.max_sentences,
            args.context_window,
        )

    validation_scores = cached_scores(
        args.cache_dir,
        "semantic_claim_level",
        args.model_name,
        args.revision,
        args.val,
        config,
        lambda: scorer(validation),
    )
    evaluation_scores = cached_scores(
        args.cache_dir,
        "semantic_claim_level",
        args.model_name,
        args.revision,
        args.eval,
        config,
        lambda: scorer(evaluation),
    )
    result = finish_run(
        args.output_dir,
        "baseline_semantic_similarity_claim_level",
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
