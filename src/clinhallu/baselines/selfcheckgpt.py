"""Official-style SelfCheckGPT-NLI over five independent sampled passages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clinhallu.api.generate_answers import load_shared_answers
from clinhallu.api.openrouter import atomic_json, fingerprint

from .common import finish_run, manifest, metrics, prediction_rows, write_jsonl
from .data import load_splits, sentence_split, sha256

OFFICIAL_VARIANT = "SelfCheckGPT-NLI"


class OfficialSelfCheckNLI:
    """Adapter around the SelfCheckGPT package's recommended NLI variant."""

    def __init__(self, device=None):
        import torch
        from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer = SelfCheckNLI(device=torch.device(selected))

    def predict(self, sentences, sampled_passages):
        values = self.scorer.predict(sentences=sentences, sampled_passages=sampled_passages)
        return np.asarray(values, dtype=float).tolist()


def score_records(records, samples, source_path, answer_path, cache_dir, scorer_factory=None):
    protocol = {
        "variant": OFFICIAL_VARIANT,
        "comparison": "each_candidate_sentence_against_each_sampled_passage",
        "aggregation": "mean_over_five_samples_then_mean_over_sentences",
        "source_sha256": sha256(source_path),
        "sampled_answers_sha256": sha256(answer_path),
    }
    directory = Path(cache_dir) / "official_selfcheckgpt_nli" / fingerprint(protocol)
    scorer = None
    document_scores, details = [], []
    for index, (row, sampled_passages) in enumerate(zip(records, samples, strict=True), 1):
        sentences = sentence_split(row["answer"])
        key = fingerprint({"sentences": sentences, "sampled_passages": sampled_passages})
        path = directory / f"{key}.json"
        if path.is_file():
            sentence_scores = json.loads(path.read_text(encoding="utf-8"))["sentence_scores"]
        else:
            if scorer is None:
                scorer = scorer_factory() if scorer_factory else OfficialSelfCheckNLI()
            sentence_scores = scorer.predict(sentences, sampled_passages)
            if len(sentence_scores) != len(sentences):
                raise ValueError("SelfCheckNLI returned an unexpected number of scores")
            atomic_json(path, {"sentence_scores": sentence_scores})
        values = np.asarray(sentence_scores, dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("SelfCheckGPT-NLI scores must be finite and in [0,1]")
        document_scores.append(float(values.mean()))
        details.append(
            {
                "sentence_selfcheck_scores": values.tolist(),
                "sentence_count": len(sentences),
                "sample_count": 5,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(records):
            print(f"SelfCheckGPT-NLI: {index}/{len(records)}", flush=True)
    return document_scores, details, protocol


def run_selfcheckgpt(
    train,
    val,
    evaluation,
    answers_dir,
    output_dir,
    cache_dir="data/cache/baselines",
    threshold_mode="fixed",
    decision_threshold=0.5,
    scorer_factory=None,
    **_ignored,
):
    if threshold_mode not in ("fixed", "validation") or not 0 <= decision_threshold <= 1:
        raise ValueError("Invalid threshold configuration")
    splits = load_splits(train, val, evaluation)
    requested = {"test": (evaluation, splits[2])}
    if threshold_mode == "validation":
        requested["validation"] = (val, splits[1])
    scores, details, protocols = {}, {}, {}
    for split, (path, rows) in requested.items():
        samples, _, answer_path = load_shared_answers(answers_dir, split, path, rows)
        scores[split], details[split], protocols[split] = score_records(
            rows, samples, path, answer_path, cache_dir, scorer_factory
        )
    config = {
        "method": OFFICIAL_VARIANT,
        "sample_count": 5,
        "threshold_mode": threshold_mode,
        "decision_threshold": decision_threshold,
        "protocols": protocols,
    }
    identifier = "baseline_selfcheckgpt_nli_official"
    paths = (train, val, evaluation)
    if threshold_mode == "validation":
        return finish_run(
            output_dir,
            identifier,
            OFFICIAL_VARIANT,
            None,
            paths,
            config,
            splits,
            scores["validation"],
            scores["test"],
            score_is_probability=False,
            extras=details["test"],
        )
    result = metrics([row["hallu_label"] for row in splits[2]], scores["test"], decision_threshold)
    result.pop("brier", None)
    result.pop("ece", None)
    result.update(
        threshold_source="fixed_score_threshold", score_semantics="SelfCheckGPT inconsistency score"
    )
    output = Path(output_dir)
    write_jsonl(
        output / "predictions.jsonl",
        prediction_rows(
            splits[2],
            np.asarray(scores["test"]) >= decision_threshold,
            scores["test"],
            identifier,
            score_is_probability=False,
            extras=details["test"],
        ),
    )
    atomic_json(output / "metrics.json", result)
    atomic_json(
        output / "run_manifest.json",
        manifest(
            identifier,
            OFFICIAL_VARIANT,
            None,
            {"train": train, "val": val, "eval": evaluation},
            config,
            {k: len(v) for k, v in zip(("train", "val", "eval"), splits, strict=True)},
            threshold_source="fixed_score_threshold",
            score_semantics="SelfCheckGPT inconsistency score",
        ),
    )
    atomic_json(output / "status.json", {"status": "completed", "model": identifier})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.jsonl")
    parser.add_argument("--val", default="data/raw/val.jsonl")
    parser.add_argument("--eval", default="data/raw/eval_data.jsonl")
    parser.add_argument("--answers-dir", "--answers_dir", default="data/generated/shared_five")
    parser.add_argument("--output-dir", "--output_dir", default="artifacts/baselines/selfcheckgpt")
    parser.add_argument("--cache-dir", "--cache_dir", default="data/cache/baselines")
    parser.add_argument(
        "--threshold-mode", "--threshold_mode", choices=("fixed", "validation"), default="fixed"
    )
    parser.add_argument("--decision-threshold", "--decision_threshold", type=float, default=0.5)
    # Compatibility arguments from the previous custom implementation.
    parser.add_argument("--model-name", "--model_name")
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", "--batch_size", type=int)
    parser.add_argument("--max-length", "--max_length", type=int)
    parser.add_argument("--support-threshold", "--support_threshold", type=float)
    args = vars(parser.parse_args(argv))
    args["evaluation"] = args.pop("eval")
    print(json.dumps(run_selfcheckgpt(**args), indent=2))


if __name__ == "__main__":
    main()
