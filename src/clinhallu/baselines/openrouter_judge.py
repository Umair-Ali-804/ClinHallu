"""Evaluation-only LLM judges through OpenRouter, with strict cached outputs."""

import json
import math
from pathlib import Path

import yaml

from clinhallu.api.openrouter import (
    OpenRouterClient,
    ResponseCache,
    atomic_json,
    build_request,
    fingerprint,
    usage_summary,
)
from clinhallu.api.prompts import messages_for, read_prompts
from clinhallu.config.paths import absolute_path, find_project_root

from .common import manifest, metrics, prediction_rows, write_jsonl
from .data import load_splits, sha256

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "hallu_label": {"type": "integer", "enum": [0, 1]},
        "probability_hallucinated": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["hallu_label", "probability_hallucinated"],
    "additionalProperties": False,
}


def validate_judgment(value):
    if not isinstance(value, dict):
        raise ValueError("Judge output must be one JSON object")
    required = {"hallu_label", "probability_hallucinated"}
    missing = required - set(value)
    if missing:
        raise ValueError("Judge output is missing required field(s): " + ", ".join(sorted(missing)))
    label, score = value["hallu_label"], value["probability_hallucinated"]
    if type(label) is not int or label not in (0, 1):
        raise ValueError("Judge label must be integer 0 or 1")
    if type(score) not in (float, int) or not math.isfinite(float(score)) or not 0 <= score <= 1:
        raise ValueError("Judge probability must be a finite number in [0,1]")
    if label != int(score >= 0.5):
        raise ValueError("Judge label must agree with probability >= 0.5")
    return label, float(score), sorted(set(value) - required)


def run_openrouter_judge(
    train, val, evaluation, output_dir, model_name, api_config, client=None, dry_run=False
):
    root = find_project_root(api_config)
    cfg = yaml.safe_load(absolute_path(api_config, root).read_text(encoding="utf-8"))
    prompts = read_prompts(cfg["prompts"], root)
    settings = cfg.get("request", {})
    splits = load_splits(train, val, evaluation)
    rows = splits[2]
    output = Path(output_dir)
    cache = ResponseCache(output / "api_cache")
    protocol = {
        "model": model_name,
        "prompts": prompts,
        "request": settings,
        "schema": JUDGE_SCHEMA,
        "threshold": 0.5,
        "input_sha256": [sha256(p) for p in (train, val, evaluation)],
    }
    identity = fingerprint(protocol)
    protocol_path = output / "api_protocol.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text())["sha256"] != identity:
        raise ValueError("Judge data, model or prompt changed; use a new output_dir")
    requests = [
        build_request(
            model_name,
            messages_for(prompts, {k: row[k] for k in ("question", "context", "answer")}),
            JUDGE_SCHEMA,
            "judgment",
            settings,
        )
        for row in rows
    ]
    if dry_run:
        unique = {fingerprint(r): r for r in requests}
        return {
            "status": "dry_run",
            "evaluation_rows": len(rows),
            "new_requests_needed": sum(not cache.contains(r) for r in unique.values()),
        }
    atomic_json(protocol_path, {"sha256": identity, "protocol": protocol})
    client = client or OpenRouterClient(
        timeout=settings.get("timeout", 120),
        rate_limit_retries=settings.get("rate_limit_retries", 0),
    )
    labels, scores, responses, extras = [], [], {}, []
    for index, request in enumerate(requests, 1):
        parsed, response, reused = cache.obtain(request, client)
        try:
            label, score, ignored_fields = validate_judgment(parsed)
        except ValueError as exc:
            raise ValueError(f"{exc}; raw completion saved at {cache.path(request)}") from exc
        labels.append(label)
        scores.append(score)
        responses[fingerprint(request)] = response
        extras.append(
            {
                "judge_label": label,
                "completion_id": response.get("completion_id"),
                "actual_model": response.get("model"),
                "provider": response.get("provider"),
                "ignored_output_fields": ignored_fields,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"{model_name}: {index}/{len(rows)} evaluation judgments", flush=True)
    identifier = "openrouter_judge::" + model_name
    result = metrics([r["hallu_label"] for r in rows], scores, 0.5)
    result.update(
        threshold_source="fixed_prompt_decision", score_semantics="self_reported_probability"
    )
    write_jsonl(
        output / "predictions.jsonl",
        prediction_rows(rows, labels, scores, identifier, extras=extras),
    )
    atomic_json(
        output / "judge_predictions.json",
        [
            {"id": row["id"], "hallu_label": label, "probability_hallucinated": score}
            for row, label, score in zip(rows, labels, scores, strict=True)
        ],
    )
    run_manifest = manifest(
        identifier,
        model_name,
        None,
        {"train": train, "val": val, "eval": evaluation},
        protocol,
        {key: len(split) for key, split in zip(("train", "val", "eval"), splits, strict=True)},
        threshold_source="fixed_prompt_decision",
        score_semantics="self_reported_probability",
        usage=usage_summary(responses),
        api_protocol_sha256=identity,
    )
    atomic_json(output / "run_manifest.json", run_manifest)
    atomic_json(output / "metrics.json", result)
    atomic_json(output / "status.json", {"status": "completed", "model": identifier})
    return result
