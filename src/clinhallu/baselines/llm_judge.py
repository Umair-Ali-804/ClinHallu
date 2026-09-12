"""LLM-as-judge baseline using local generation or supplied predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .common import finish_fixed_prediction_run, finish_run
from .data import load_splits, sha256

PROMPT = """You are a medical hallucination detection evaluator.
Classify the candidate answer using only the provided context.

0 = fully supported / non-hallucinated
1 = unsupported, fabricated, contradictory, reversed, or hallucinated

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Return one JSON object only: {{"hallu_label": 0 or 1, "confidence": 0.0 to 1.0}}"""


def parse_judgment(text: str) -> dict:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", str(text)):
        try:
            value, _ = decoder.raw_decode(str(text)[match.start() :])
            raw_label = value.get("hallu_label", value.get("label"))
            label = int(raw_label)
            confidence = float(value.get("confidence", 1.0))
            if label not in (0, 1) or not 0.0 <= confidence <= 1.0:
                continue
            return {
                "label": label,
                "confidence": confidence,
                "reason": str(value.get("reason", "")),
                "parse_failure": False,
            }
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, AttributeError):
            continue
    return {
        "label": 1,
        "confidence": 0.5,
        "reason": "parse failure; conservative hallucinated fallback",
        "parse_failure": True,
    }


def _read_json_or_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Judge prediction/response file not found: {source}")
    text = source.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Judge file is empty: {source}")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = value.get("predictions", value.get("results", [value]))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Judge file must contain JSON objects: {source}")
    return value


def load_responses(path: str | Path) -> dict[str, str]:
    output = {}
    for row in _read_json_or_jsonl(path):
        identifier = str(row["id"])
        if identifier in output:
            raise ValueError(f"Duplicate judge response id: {identifier}")
        output[identifier] = str(row.get("response", row.get("output", row)))
    return output


def load_binary_predictions(path: str | Path, expected_records: list[dict]) -> list[int]:
    values = {}
    for row in _read_json_or_jsonl(path):
        identifier = str(row["id"])
        if identifier in values:
            raise ValueError(f"Duplicate judge prediction id: {identifier}")
        raw = row.get("hallu_label", row.get("label", row.get("prediction")))
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            numeric = float("nan")
        if isinstance(raw, bool) or numeric not in (0.0, 1.0):
            raise ValueError(f"Prediction for id {identifier} must be integer 0 or 1")
        values[identifier] = int(numeric)
    expected_ids = [str(row["id"]) for row in expected_records]
    missing = [identifier for identifier in expected_ids if identifier not in values]
    extras = sorted(set(values) - set(expected_ids))
    if missing or extras:
        raise ValueError(
            f"Judge IDs do not match evaluation data: missing={len(missing)}, extra={len(extras)}"
        )
    return [values[identifier] for identifier in expected_ids]


def generate_local(records, model_name, revision, max_new_tokens, max_input_tokens):
    import torch
    from transformers import pipeline

    generator = pipeline(
        "text-generation",
        model=model_name,
        revision=revision,
        device=0 if torch.cuda.is_available() else -1,
        dtype="auto",
    )
    responses = {}
    for row in records:
        prompt = PROMPT.format(**row)
        result = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
            truncation=True,
            max_length=max_input_tokens,
        )[0]["generated_text"]
        responses[str(row["id"])] = result
    return responses


def save_responses(path: str | Path, responses: dict[str, str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for identifier, response in responses.items():
            handle.write(
                json.dumps({"id": identifier, "response": response}, ensure_ascii=False) + "\n"
            )


def evaluate_rows(rows, responses):
    expected = {str(row["id"]) for row in rows}
    observed = {str(identifier) for identifier in responses}
    missing = expected - observed
    extras_found = observed - expected
    if missing or extras_found:
        raise ValueError(
            "Judge responses do not match the requested split: "
            f"missing={len(missing)}, extra={len(extras_found)}"
        )
    scores, extras = [], []
    for row in rows:
        parsed = parse_judgment(responses.get(str(row["id"]), ""))
        score = parsed["confidence"] if parsed["label"] == 1 else 1 - parsed["confidence"]
        scores.append(score)
        extras.append(
            {
                "judge_label": parsed["label"],
                "judge_confidence": parsed["confidence"],
                "judge_reason": parsed["reason"],
                "parse_failure": parsed["parse_failure"],
            }
        )
    return scores, extras


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.jsonl")
    parser.add_argument("--val", default="data/raw/val.jsonl")
    parser.add_argument("--eval", default="data/raw/eval_data.jsonl")
    parser.add_argument("--model-name", "--model_name", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--provider",
        choices=("local_huggingface", "cached_predictions", "openrouter"),
        default="local_huggingface",
    )
    parser.add_argument("--predictions", help="JSON/JSONL containing id and hallu_label")
    parser.add_argument("--val-responses", "--val_responses")
    parser.add_argument("--eval-responses", "--eval_responses")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", "--max_input_tokens", type=int, default=4096)
    parser.add_argument("--output-dir", "--output_dir", default="artifacts/baselines/llm_judge")
    parser.add_argument("--api-config", "--api_config", default="configs/api/judge.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.provider == "openrouter":
        from .openrouter_judge import run_openrouter_judge

        result = run_openrouter_judge(
            args.train,
            args.val,
            args.eval,
            args.output_dir,
            args.model_name,
            args.api_config,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
        return

    splits = load_splits(args.train, args.val, args.eval)
    _, validation, evaluation = splits
    if args.predictions:
        labels = load_binary_predictions(args.predictions, evaluation)
        extras = [{"judge_label": label} for label in labels]
        result = finish_fixed_prediction_run(
            args.output_dir,
            "baseline_llm_judge",
            args.model_name,
            (args.train, args.val, args.eval),
            splits,
            labels,
            extras,
        )
        print(json.dumps(result, indent=2))
        return
    if args.provider == "cached_predictions":
        raise ValueError("provider=cached_predictions requires --predictions")

    output = Path(args.output_dir)
    identity = json.dumps(
        {
            "model": args.model_name,
            "revision": args.revision,
            "prompt": PROMPT,
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "val": sha256(args.val),
            "eval": sha256(args.eval),
        },
        sort_keys=True,
    )
    key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    cached_validation = output / "cache" / key / "raw_validation_responses.jsonl"
    cached_evaluation = output / "cache" / key / "raw_evaluation_responses.jsonl"
    validation_path = Path(args.val_responses) if args.val_responses else cached_validation
    evaluation_path = Path(args.eval_responses) if args.eval_responses else cached_evaluation
    validation_responses = (
        load_responses(validation_path)
        if validation_path.exists()
        else generate_local(
            validation,
            args.model_name,
            args.revision,
            args.max_new_tokens,
            args.max_input_tokens,
        )
    )
    evaluation_responses = (
        load_responses(evaluation_path)
        if evaluation_path.exists()
        else generate_local(
            evaluation,
            args.model_name,
            args.revision,
            args.max_new_tokens,
            args.max_input_tokens,
        )
    )
    if not validation_path.exists():
        save_responses(cached_validation, validation_responses)
    if not evaluation_path.exists():
        save_responses(cached_evaluation, evaluation_responses)
    validation_scores, _ = evaluate_rows(validation, validation_responses)
    evaluation_scores, extras = evaluate_rows(evaluation, evaluation_responses)
    failures = sum(item["parse_failure"] for item in extras)
    config = {
        "provider": args.provider,
        "revision": args.revision,
        "temperature": 0.0,
        "prompt": PROMPT,
        "parse_failures": failures,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    result = finish_run(
        output,
        "baseline_llm_judge",
        args.model_name,
        None,
        (args.train, args.val, args.eval),
        config,
        splits,
        validation_scores,
        evaluation_scores,
        extras=extras,
    )
    result["parse_failures"] = failures
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
