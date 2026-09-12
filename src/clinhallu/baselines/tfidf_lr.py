"""TF-IDF plus logistic-regression publication baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .common import finish_run
from .data import load_splits, serialize


def run(
    train_path: str,
    validation_path: str,
    evaluation_path: str,
    output_dir: str,
    *,
    max_features: int = 100_000,
    seed: int = 13,
) -> dict:
    splits = load_splits(train_path, validation_path, evaluation_path)
    train, validation, evaluation = splits

    # Keep the publication settings for real data while allowing the tiny
    # fixtures used by smoke tests and tutorials to form a vocabulary.
    min_df = 2 if len(train) >= 20 else 1
    max_df = 0.98 if len(train) >= 50 else 1.0
    config = {
        "ngram_range": [1, 2],
        "min_df": min_df,
        "max_df": max_df,
        "sublinear_tf": True,
        "max_features": max_features,
        "class_weight": "balanced",
        "max_iter": 2_000,
    }
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        max_features=max_features,
        lowercase=True,
        strip_accents="unicode",
    )
    train_features = vectorizer.fit_transform(serialize(row) for row in train)
    model = LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
        random_state=seed,
    ).fit(train_features, [row["hallu_label"] for row in train])

    validation_scores = model.predict_proba(
        vectorizer.transform(serialize(row) for row in validation)
    )[:, 1]
    evaluation_scores = model.predict_proba(
        vectorizer.transform(serialize(row) for row in evaluation)
    )[:, 1]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, output / "model.joblib")
    return finish_run(
        output,
        "baseline_tfidf_lr",
        "TfidfVectorizer+LogisticRegression",
        seed,
        (train_path, validation_path, evaluation_path),
        config,
        splits,
        validation_scores,
        evaluation_scores,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.jsonl")
    parser.add_argument("--val", default="data/raw/val.jsonl")
    parser.add_argument("--eval", default="data/raw/eval_data.jsonl")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="artifacts/baselines/tfidf_lr",
    )
    parser.add_argument("--max-features", "--max_features", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    result = run(
        args.train,
        args.val,
        args.eval,
        args.output_dir,
        max_features=args.max_features,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
