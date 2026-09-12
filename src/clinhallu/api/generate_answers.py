"""Generate five independent sampled passages for official-style SelfCheckGPT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from clinhallu.baselines.data import load_records, sha256
from clinhallu.config.paths import absolute_path, find_project_root

from .openrouter import (
    OpenRouterClient,
    ResponseCache,
    atomic_json,
    build_request,
    fingerprint,
    usage_summary,
)
from .prompts import messages_for, read_prompts

ANSWER_KEYS = tuple(f"answer_{i:02d}" for i in range(1, 6))
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
SPLIT_FILES = {"validation": "validation_answers.json", "test": "evaluation_answers.json"}


def validate_answer(value):
    if not isinstance(value, dict) or set(value) != {"answer"}:
        raise ValueError("Expected exactly one nonempty answer field")
    if not isinstance(value["answer"], str) or not value["answer"].strip():
        raise ValueError("Sampled answer must be a nonempty string")
    return value["answer"].strip()


def run_generation(
    config_path, splits=("test",), dry_run=False, max_new_requests=None, client=None
):
    root = find_project_root(config_path)
    cfg = yaml.safe_load(absolute_path(config_path, root).read_text(encoding="utf-8"))
    prompts = read_prompts(cfg["prompts"], root)
    settings = cfg.get("request", {})
    output = absolute_path(cfg["output_dir"], root)
    cache = ResponseCache(absolute_path(cfg["cache_dir"], root))
    if max_new_requests is not None and max_new_requests < 0:
        raise ValueError("max_new_requests must be nonnegative")
    splits = tuple(dict.fromkeys(splits))
    if not splits or set(splits) - set(SPLIT_FILES):
        raise ValueError("Choose validation and/or test")
    sample_count = int(cfg.get("sample_count", 5))
    if sample_count != 5:
        raise ValueError("This protocol requires exactly five sampled passages")
    base_seed = int(settings.pop("seed", 13))
    spec = {
        "protocol": "selfcheckgpt_five_independent_samples_v1",
        "model": cfg["model_name"],
        "prompts": prompts,
        "request": settings,
        "schema": ANSWER_SCHEMA,
        "conditioning": "question_and_context_only",
        "sample_count": sample_count,
        "independent_requests": True,
        "sample_seeds": [base_seed + i for i in range(sample_count)],
    }
    identity = fingerprint(spec)
    manifest_path = output / "generation_manifest.json"
    manifest = {"specification": spec, "specification_sha256": identity, "splits": {}}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["specification_sha256"] != identity:
            raise ValueError("Generation model, settings or prompts changed; use a new output_dir")
    prepared = {}
    for split in splits:
        path = absolute_path(cfg["data"][split], root)
        rows = load_records(path, require_labels=False)
        digest = sha256(path)
        previous = manifest["splits"].get(split, {})
        if previous.get("input_sha256", digest) != digest:
            raise ValueError(f"{split} data changed; use a new output_dir")
        requests = []
        for row in rows:
            row_requests = []
            for sample_index in range(sample_count):
                sample_settings = dict(settings, seed=base_seed + sample_index)
                row_requests.append(
                    build_request(
                        cfg["model_name"],
                        messages_for(
                            prompts, {"question": row["question"], "context": row["context"]}
                        ),
                        ANSWER_SCHEMA,
                        "sampled_answer",
                        sample_settings,
                    )
                )
            requests.append(row_requests)
        prepared[split] = (rows, requests, digest)
    unique = {
        fingerprint(req): req
        for _, row_reqs, _ in prepared.values()
        for reqs in row_reqs
        for req in reqs
    }
    if dry_run:
        unique_pairs = {
            (row["question"], row["context"]) for rows, _, _ in prepared.values() for row in rows
        }
        return {
            "status": "dry_run",
            "rows": {s: len(v[0]) for s, v in prepared.items()},
            "unique_question_context_pairs": len(unique_pairs),
            "unique_requests": len(unique),
            "cached_completions": sum(cache.contains(r) for r in unique.values()),
            "new_requests_needed": sum(not cache.contains(r) for r in unique.values()),
            "answers_per_request": 1,
            "samples_per_record": sample_count,
            "output_dir": str(output),
        }
    client = client or OpenRouterClient(
        timeout=settings.get("timeout", 120),
        rate_limit_retries=settings.get("rate_limit_retries", 0),
    )
    # Record input identity even if interrupted before this split is complete.
    for split, (_, _, digest) in prepared.items():
        manifest["splits"].setdefault(split, {"input_sha256": digest, "complete": False})
    atomic_json(manifest_path, manifest)
    new_count, responses, outputs = 0, {}, {}
    for split, (rows, requests, digest) in prepared.items():
        exported = []
        for index, (row, row_requests) in enumerate(zip(rows, requests, strict=True), 1):
            answers = {}
            for sample_index, request in enumerate(row_requests):
                if (
                    not cache.contains(request)
                    and max_new_requests is not None
                    and new_count >= max_new_requests
                ):
                    status = {
                        "status": "incomplete",
                        "new_requests": new_count,
                        "outputs": outputs,
                        "resume": "Run the same command again",
                    }
                    atomic_json(output / "generation_status.json", status)
                    return status
                parsed, response, reused = cache.obtain(request, client)
                try:
                    answer = validate_answer(parsed)
                except ValueError as exc:
                    raise ValueError(
                        f"{exc}; raw completion saved at {cache.path(request)}"
                    ) from exc
                new_count += int(not reused)
                responses[fingerprint(request)] = response
                answers[ANSWER_KEYS[sample_index]] = answer
            exported.append({"id": row["id"], "question": row["question"], **answers})
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(f"{split}: {index}/{len(rows)} rows; {new_count} new completions", flush=True)
        answer_path = output / SPLIT_FILES[split]
        atomic_json(answer_path, exported)
        manifest["splits"][split] = {
            "input_sha256": digest,
            "answers_sha256": sha256(answer_path),
            "filename": answer_path.name,
            "rows": len(rows),
            "complete": True,
            "request_hashes": sorted({fingerprint(r) for rr in requests for r in rr}),
        }
        atomic_json(manifest_path, manifest)
        outputs[split] = str(answer_path)
    status = {
        "status": "completed",
        "new_requests": new_count,
        "outputs": outputs,
        "usage_for_requested_splits": usage_summary(responses),
    }
    atomic_json(output / "generation_status.json", status)
    return status


def load_shared_answers(directory, split, input_path, records):
    directory = Path(directory)
    manifest_path = directory / "generation_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Generate shared answers first: missing generation_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    info = manifest.get("splits", {}).get(split, {})
    if not info.get("complete"):
        raise ValueError(f"Generate complete {split} answers first")
    path = directory / SPLIT_FILES[split]
    if info["input_sha256"] != sha256(input_path) or info["answers_sha256"] != sha256(path):
        raise ValueError("Shared answers or source data changed; generate a new protocol directory")
    exported = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(exported, list) or len(exported) != len(records):
        raise ValueError("Shared answers must be an aligned JSON array")
    by_id = {}
    for row in exported:
        if not isinstance(row, dict) or set(row) != {"id", "question", *ANSWER_KEYS}:
            raise ValueError("Unexpected shared-answer fields")
        identifier = str(row["id"])
        if identifier in by_id:
            raise ValueError("Duplicate shared-answer ID")
        for key in ANSWER_KEYS:
            validate_answer({"answer": row[key]})
        by_id[identifier] = row
    if set(by_id) != {str(r["id"]) for r in records}:
        raise ValueError("Shared-answer IDs differ from input")
    aligned = []
    for row in records:
        item = by_id[str(row["id"])]
        if item["question"] != row["question"]:
            raise ValueError("Shared-answer question mismatch")
        aligned.append([item[k] for k in ANSWER_KEYS])
    return aligned, manifest, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generation/shared_five.yaml")
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=["test"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-new-requests", type=int)
    args = parser.parse_args(argv)
    status = run_generation(args.config, args.splits, args.dry_run, args.max_new_requests)
    print(json.dumps(status, indent=2))
    if status["status"] == "incomplete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
