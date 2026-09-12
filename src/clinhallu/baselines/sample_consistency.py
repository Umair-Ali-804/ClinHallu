"""Two detector adaptations using the same five context-conditioned answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clinhallu.api.generate_answers import load_shared_answers
from clinhallu.api.openrouter import atomic_json, fingerprint

from .common import finish_run, manifest, metrics, prediction_rows, write_jsonl
from .data import load_splits, sentence_split, sha256
from .zero_shot_nli import label_indices

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NLI_REVISION = "07c1fa65279cc95163bc0b3e12404b37955d804f"


class NLIPairScorer:
    def __init__(self, model_name=NLI_MODEL, revision=NLI_REVISION, batch_size=16, max_length=512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
            .to(self.device)
            .eval()
        )
        self.entailment, self.contradiction = label_indices(self.model, model_name)
        if self.contradiction is None:
            raise ValueError("Shared NLI requires both entailment and contradiction label metadata")
        self.batch_size, self.max_length = int(batch_size), int(max_length)
        if self.batch_size <= 0 or self.max_length < 4:
            raise ValueError("Invalid NLI batch_size or max_length")

    def __call__(self, pairs):
        result = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**encoded).logits.float()
                p_e = logits.softmax(-1)[:, self.entailment]
                p_c_conditional = logits[:, [self.entailment, self.contradiction]].softmax(-1)[:, 1]
            result.extend(self.torch.stack([p_e, p_c_conditional], -1).cpu().tolist())
        return result


def score_statistics(matrix, support_threshold=0.5):
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1:] != (5, 2):
        raise ValueError("NLI matrix must have shape [sentences, 5, 2]")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("NLI values must be finite and in [0,1]")
    if not 0 <= support_threshold <= 1:
        raise ValueError("support_threshold must be in [0,1]")
    sentence_scores = values[:, :, 1].mean(axis=1)
    votes = (values[:, :, 0] >= support_threshold).all(axis=0)
    return {
        "selfcheckgpt": float(sentence_scores.mean()),
        "self_consistency": float(1 - votes.mean()),
        "sample_support_votes": votes.astype(int).tolist(),
        "support_vote_count": int(votes.sum()),
        "sentence_selfcheck_scores": sentence_scores.tolist(),
    }


def obtain_nli_matrices(
    records, answers, input_path, answer_path, config, cache_dir, scorer_factory
):
    identity = fingerprint(
        {
            "protocol": "sample_premise_candidate_sentence_v1",
            "verifier": config,
            "input": sha256(input_path),
            "answers": sha256(answer_path),
        }
    )
    directory = Path(cache_dir) / "shared_nli" / identity
    matrices, scorer, hits = [], None, 0
    for index, (row, samples) in enumerate(zip(records, answers, strict=True), 1):
        claims = sentence_split(row["answer"])
        pairs = [[sample, claim] for claim in claims for sample in samples]
        key = fingerprint(pairs)
        path = directory / (key + ".json")
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["pairs_sha256"] != key:
                raise ValueError(f"NLI cache mismatch: {path}")
            matrix = saved["matrix"]
            hits += 1
        else:
            if scorer is None:
                scorer = scorer_factory()
            values = np.asarray(scorer(pairs), dtype=float)
            if values.shape != (len(pairs), 2):
                raise ValueError("NLI scorer returned the wrong number of pairs")
            matrix = values.reshape(len(claims), 5, 2).tolist()
            score_statistics(matrix)
            atomic_json(path, {"pairs_sha256": key, "matrix": matrix})
        score_statistics(matrix)
        if len(matrix) != len(claims):
            raise ValueError("NLI cache sentence count mismatch")
        matrices.append(matrix)
        if index == 1 or index % 25 == 0 or index == len(records):
            print(f"Shared NLI: {index}/{len(records)}; {hits} cache hits", flush=True)
    return matrices


def run_shared_baseline(
    method,
    train,
    val,
    evaluation,
    answers_dir,
    output_dir,
    model_name=NLI_MODEL,
    revision=NLI_REVISION,
    batch_size=16,
    max_length=512,
    support_threshold=0.5,
    cache_dir="data/cache/baselines",
    threshold_mode="fixed",
    decision_threshold=0.5,
    scorer_factory=None,
):
    if method not in ("selfcheckgpt", "self_consistency"):
        raise ValueError("Unknown shared-answer method")
    if threshold_mode not in ("fixed", "validation") or not 0 <= decision_threshold <= 1:
        raise ValueError("Invalid threshold mode or decision threshold")
    if method == "self_consistency" and (threshold_mode != "fixed" or decision_threshold != 0.5):
        raise ValueError("Self-consistency uses fixed majority voting at 0.5")
    splits = load_splits(train, val, evaluation)
    requested = {"test": (evaluation, splits[2])}
    if threshold_mode == "validation":
        requested["validation"] = (val, splits[1])
    shared = {
        key: load_shared_answers(answers_dir, key, path, rows)
        for key, (path, rows) in requested.items()
    }
    verifier = dict(
        model_name=model_name, revision=revision, batch_size=batch_size, max_length=max_length
    )
    config = {
        "method": method,
        "adaptation": "batched_context_conditioned",
        "verifier": verifier,
        "support_threshold": support_threshold,
        "threshold_mode": threshold_mode,
        "decision_threshold": decision_threshold,
        "generation_sha256": shared["test"][1]["specification_sha256"],
        "generation_model": shared["test"][1]["specification"]["model"],
        "answers_sha256": {k: sha256(v[2]) for k, v in shared.items()},
        "input_sha256": [sha256(p) for p in (train, val, evaluation)],
    }
    output = Path(output_dir)
    protocol_path = output / "scoring_protocol.json"
    identity = fingerprint(config)
    if protocol_path.is_file() and json.loads(protocol_path.read_text())["sha256"] != identity:
        raise ValueError("Scoring protocol changed; use a new output_dir")
    atomic_json(protocol_path, {"sha256": identity, "config": config})
    scorer_holder = []

    def factory():
        if not scorer_holder:
            scorer_holder.append(scorer_factory() if scorer_factory else NLIPairScorer(**verifier))
        return scorer_holder[0]

    statistics = {}
    for split, (path, rows) in requested.items():
        answers, _, answer_path = shared[split]
        matrices = obtain_nli_matrices(
            rows, answers, path, answer_path, verifier, cache_dir, factory
        )
        statistics[split] = [score_statistics(m, support_threshold) for m in matrices]
    scores = [s[method] for s in statistics["test"]]
    extras = [
        {k: v for k, v in s.items() if k not in ("selfcheckgpt", "self_consistency")}
        for s in statistics["test"]
    ]
    identifier = (
        "baseline_selfcheck_nli_batched"
        if method == "selfcheckgpt"
        else "baseline_self_consistency_vote_batched"
    )
    if threshold_mode == "validation":
        return finish_run(
            output,
            identifier,
            model_name,
            None,
            (train, val, evaluation),
            config,
            splits,
            [s[method] for s in statistics["validation"]],
            scores,
            score_is_probability=False,
            extras=extras,
        )
    result = metrics([r["hallu_label"] for r in splits[2]], scores, decision_threshold)
    for key in ("brier", "ece"):
        result.pop(key, None)
    threshold_source = (
        "fixed_majority_vote" if method == "self_consistency" else "fixed_score_threshold"
    )
    result.update(threshold_source=threshold_source, score_semantics="normalized_score")
    write_jsonl(
        output / "predictions.jsonl",
        prediction_rows(
            splits[2],
            np.asarray(scores) >= decision_threshold,
            scores,
            identifier,
            score_is_probability=False,
            extras=extras,
        ),
    )
    run_manifest = manifest(
        identifier,
        model_name,
        None,
        {"train": train, "val": val, "eval": evaluation},
        config,
        {key: len(s) for key, s in zip(("train", "val", "eval"), splits, strict=True)},
        threshold_source=threshold_source,
        score_semantics="normalized_score",
    )
    atomic_json(output / "run_manifest.json", run_manifest)
    atomic_json(output / "metrics.json", result)
    atomic_json(output / "status.json", {"status": "completed", "model": identifier})
    return result


def main_for(method, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in (
        ("train", "data/raw/train.jsonl"),
        ("val", "data/raw/val.jsonl"),
        ("eval", "data/raw/eval_data.jsonl"),
    ):
        parser.add_argument("--" + key, default=default)

    def option(name, **kwargs):
        parser.add_argument("--" + name.replace("_", "-"), "--" + name, **kwargs)

    option("answers_dir", default="data/generated/shared_five")
    option("output_dir", default="artifacts/baselines/" + method)
    option("cache_dir", default="data/cache/baselines")
    option("model_name", default=NLI_MODEL)
    parser.add_argument("--revision", default=NLI_REVISION)
    option("batch_size", type=int, default=16)
    option("max_length", type=int, default=512)
    option("support_threshold", type=float, default=0.5)
    option("threshold_mode", choices=("fixed", "validation"), default="fixed")
    option("decision_threshold", type=float, default=0.5)
    values = vars(parser.parse_args(argv))
    values["evaluation"] = values.pop("eval")
    result = run_shared_baseline(method, **values)
    print(json.dumps(result, indent=2))
