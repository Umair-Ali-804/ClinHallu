"""Offline API protocol tests: no network or paid completions."""

import io
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import yaml

from clinhallu.api.generate_answers import (
    ANSWER_KEYS,
    ANSWER_SCHEMA,
    load_shared_answers,
    run_generation,
)
from clinhallu.api.openrouter import OpenRouterClient, ResponseCache, build_request, parse_object
from clinhallu.baselines.data import load_records
from clinhallu.baselines.openrouter_judge import (
    JUDGE_SCHEMA,
    run_openrouter_judge,
    validate_judgment,
)
from clinhallu.baselines.sample_consistency import run_shared_baseline, score_statistics
from clinhallu.baselines.selfcheckgpt import run_selfcheckgpt
from clinhallu.experiments.baseline_suite import run_baseline_suite
from clinhallu.reporting.summarize import aggregate_runs

SOURCE = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"\n')
    shutil.copytree(SOURCE / "prompts", tmp_path / "prompts")
    (tmp_path / "configs").mkdir()
    for split in ("train", "val", "eval"):
        rows = [
            dict(
                id=(11 if split == "val" and i == 0 else f"{split}{i}"),
                question=f"What happened in {split}?",
                context=f"{split}: the patient improved.",
                answer="The patient improved." if i == 0 else "unsupported: the patient worsened.",
                hallu_label=i,
            )
            for i in range(2)
        ]
        (tmp_path / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = yaml.safe_load((SOURCE / "configs/generation/shared_five.yaml").read_text())
    cfg["data"] = {"validation": "val.jsonl", "test": "eval.jsonl"}
    cfg["output_dir"] = "answers"
    cfg["cache_dir"] = "cache/generation"
    (tmp_path / "configs/generate.yaml").write_text(yaml.safe_dump(cfg))
    shutil.copy(SOURCE / "configs/api/judge.yaml", tmp_path / "configs/judge.yaml")
    monkeypatch.setenv("CLINHALLU_ROOT", str(tmp_path))
    return tmp_path


class FakeAPI:
    def __init__(self, invalid=False):
        self.calls = []
        self.invalid = invalid

    def complete(self, request):
        assert "n" not in request
        self.calls.append(request)
        text = request["messages"][1]["content"]
        fields = json.loads(text[text.index("{") :])
        assert "hallu_label" not in fields and "id" not in fields
        if "answer" not in fields:
            assert set(fields) == {"question", "context"}
            result = {"answer": "The patient improved."}
        else:
            assert set(fields) == {"question", "context", "answer"}
            label = int("unsupported" in fields["answer"])
            result = {"hallu_label": label, "probability_hallucinated": 0.9 if label else 0.1}
        return {
            "text": "invalid" if self.invalid else json.dumps(result),
            "finish_reason": "stop",
            "completion_id": str(len(self.calls)),
            "model": request["model"],
            "usage": {"prompt_tokens": 30, "completion_tokens": 50, "cost": 0},
        }


def fake_nli_factory():
    def score(pairs):
        return [[0.01, 0.99] if "unsupported" in claim else [0.99, 0.01] for _, claim in pairs]

    return score


class FakeOfficialNLI:
    def predict(self, sentences, sampled_passages):
        return [0.99 if "unsupported" in sentence else 0.01 for sentence in sentences]


def shared(project, method, **kwargs):
    if method == "selfcheckgpt":
        return run_selfcheckgpt(
            *(str(project / f"{s}.jsonl") for s in ("train", "val", "eval")),
            project / "answers",
            project
            / (
                "result_"
                + method
                + ("_val" if kwargs.get("threshold_mode") == "validation" else "")
            ),
            cache_dir=project / "cache/selfcheck",
            scorer_factory=lambda: FakeOfficialNLI(),
            **kwargs,
        )
    return run_shared_baseline(
        method,
        *(str(project / f"{s}.jsonl") for s in ("train", "val", "eval")),
        project / "answers",
        project
        / ("result_" + method + ("_val" if kwargs.get("threshold_mode") == "validation" else "")),
        cache_dir=project / "cache/nli",
        **kwargs,
    )


def test_single_generation_reused_by_both_baselines(project):
    cfg = project / "configs/generate.yaml"
    api = FakeAPI()
    dry = run_generation(cfg, dry_run=True, client=api)
    assert dry["new_requests_needed"] == 5 and not api.calls
    result = run_generation(cfg, client=api)
    assert result["new_requests"] == 5 and len(api.calls) == 5
    path = project / "answers/evaluation_answers.json"
    records = json.loads(path.read_text())
    assert len(records) == 2 and set(records[0]) == {"id", "question", *ANSWER_KEYS}
    assert not (project / "answers/validation_answers.json").exists()
    first = shared(project, "selfcheckgpt")
    # Both methods reuse the same generated passages without another API call.
    # Their local scorers are intentionally distinct: official SelfCheckNLI for
    # SelfCheckGPT and pairwise NLI voting for self-consistency.
    second = shared(project, "self_consistency", scorer_factory=fake_nli_factory)
    assert first["f1"] == second["f1"] == 1 and first["threshold"] == second["threshold"] == 0.5
    assert "brier" not in first and "ece" not in second
    assert len(api.calls) == 5
    assert run_generation(cfg, client=api)["new_requests"] == 0


def test_generation_limit_resume_and_integer_ids(project):
    api = FakeAPI()
    cfg = project / "configs/generate.yaml"
    first = run_generation(cfg, splits=("validation", "test"), max_new_requests=1, client=api)
    assert first["status"] == "incomplete" and len(api.calls) == 1
    assert not (project / "answers/evaluation_answers.json").exists()
    result = run_generation(cfg, splits=("validation", "test"), client=api)
    assert result["status"] == "completed" and len(api.calls) == 10
    assert json.loads((project / "answers/validation_answers.json").read_text())[0]["id"] == 11


def test_optional_validation_threshold_requires_validation_answers(project):
    api = FakeAPI()
    cfg = project / "configs/generate.yaml"
    run_generation(cfg, client=api)
    with pytest.raises(ValueError, match="validation"):
        shared(project, "selfcheckgpt", threshold_mode="validation")
    run_generation(cfg, splits=("validation",), client=api)
    result = shared(project, "selfcheckgpt", threshold_mode="validation")
    assert result["validation_f1_at_threshold"] == 1
    assert len(api.calls) == 10


def test_prompt_change_rejected(project):
    cfg = project / "configs/generate.yaml"
    run_generation(cfg, client=FakeAPI())
    p = project / "prompts/shared_five_system.txt"
    p.write_text(p.read_text() + " new instructions")
    with pytest.raises(ValueError, match="changed"):
        run_generation(cfg, client=FakeAPI())


def test_tampered_answer_file_rejected(project):
    run_generation(project / "configs/generate.yaml", client=FakeAPI())
    p = project / "answers/evaluation_answers.json"
    p.write_text(p.read_text() + " ")
    rows = load_records(project / "eval.jsonl")
    with pytest.raises(ValueError, match="changed"):
        load_shared_answers(project / "answers", "test", project / "eval.jsonl", rows)


def test_source_context_change_rejected(project):
    cfg = project / "configs/generate.yaml"
    run_generation(cfg, client=FakeAPI())
    p = project / "eval.jsonl"
    p.write_text(p.read_text().replace("patient improved", "patient recovered"))
    with pytest.raises(ValueError, match="changed"):
        run_generation(cfg, client=FakeAPI())


def test_invalid_completion_saved_without_automatic_regeneration(project):
    api = FakeAPI(invalid=True)
    for _ in range(2):
        with pytest.raises(ValueError, match="cached completion"):
            run_generation(project / "configs/generate.yaml", client=api)
    assert len(api.calls) == 1


def test_truncated_completion_reports_finish_reason_and_is_not_retried(tmp_path):
    class TruncatedAPI:
        def __init__(self):
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            return {
                "text": "",
                "finish_reason": "length",
                "usage": {"completion_tokens": 512},
            }

    request = {"model": "openai/gpt-5-mini", "messages": [], "max_tokens": 512}
    api = TruncatedAPI()
    cache = ResponseCache(tmp_path / "cache")
    for _ in range(2):
        with pytest.raises(
            ValueError,
            match=r"finish_reason='length'.*completion_tokens=512.*lower reasoning.effort",
        ):
            cache.obtain(request, api)
    assert api.calls == 1


def test_openrouter_transport_one_sample_per_request(monkeypatch):
    seen = []

    def urlopen(request, timeout):
        seen.append(json.loads(request.data))
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps({"answer": "answer"})},
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    req = build_request("openai/gpt-4.1-mini", [], ANSWER_SCHEMA, "sampled_answer", {})
    response = OpenRouterClient(api_key="fake-key").complete(req)
    assert len(seen) == 1 and "n" not in seen[0]
    assert set(parse_object(response["text"])) == {"answer"}
    assert seen[0]["response_format"]["json_schema"]["strict"] is True


def test_judge_only_evaluation_continuous_scores_and_resume(project):
    api = FakeAPI()
    paths = [str(project / f"{s}.jsonl") for s in ("train", "val", "eval")]
    args = (*paths, project / "judge", "openai/gpt-4.1-mini", project / "configs/judge.yaml")
    metrics = run_openrouter_judge(*args, client=api)
    assert len(api.calls) == 2 and metrics["auc"] == 1
    assert metrics["brier"] == pytest.approx(0.01)
    assert run_openrouter_judge(*args, client=api) == metrics and len(api.calls) == 2
    saved = json.loads((project / "judge/run_manifest.json").read_text())
    assert saved["model"] == "openrouter_judge::openai/gpt-4.1-mini"


def test_strict_json_and_judgment_validation():
    for value in (
        {"hallu_label": True, "probability_hallucinated": 0.9},
        {"hallu_label": 0.5, "probability_hallucinated": 0.9},
        {"hallu_label": 0, "probability_hallucinated": 0.9},
        {"hallu_label": 1, "probability_hallucinated": float("nan")},
    ):
        with pytest.raises(ValueError):
            validate_judgment(value)
    assert validate_judgment(
        {
            "hallu_label": 1,
            "probability_hallucinated": 0.9,
            "reason": "provider-added field",
        }
    ) == (1, 0.9, ["reason"])
    for text in ('{"x":1,"x":2}', '{"x":NaN}', 'preamble {"x":1}'):
        with pytest.raises(ValueError):
            parse_object(text)


def test_majority_votes_require_each_candidate_sentence_supported():
    m = np.full((2, 5, 2), [0.9, 0.1])
    m[1, 3:] = [0.1, 0.9]
    result = score_statistics(m)
    assert result["sample_support_votes"] == [1, 1, 1, 0, 0]
    assert result["self_consistency"] == pytest.approx(0.4)
    assert result["selfcheckgpt"] == pytest.approx(0.26)


def test_judge_models_are_not_averaged_together():
    rows = aggregate_runs(
        [
            {"model": "baseline_llm_judge", "model_name": name, "metrics": {"f1": score}}
            for name, score in [("judge_a", 0.1), ("judge_b", 0.9)]
        ]
    )
    assert len(rows) == 2 and {r["model"] for r in rows} == {
        "LLM judge: judge_a",
        "LLM judge: judge_b",
    }


def test_all_thirteen_suite_commands_are_wired(monkeypatch):
    monkeypatch.setenv("CLINHALLU_ROOT", str(SOURCE))
    status = run_baseline_suite(SOURCE / "configs/baselines/suite.yaml", dry_run=True)
    assert len(status) == 13
    assert sum(r["status"] == "planned" for r in status.values()) == 5
    assert sum(r["status"] == "disabled" for r in status.values()) == 8
    assert "--answers_dir" in status["self_consistency"]["command"]
    selected = run_baseline_suite(
        SOURCE / "configs/baselines/suite.yaml",
        only=["openrouter_gpt_5_mini"],
        dry_run=True,
    )
    assert "openrouter" in selected["openrouter_gpt_5_mini"]["command"]


def test_requested_judge_models_and_gpt5_parameters_are_exact():
    suite = yaml.safe_load((SOURCE / "configs/baselines/suite.yaml").read_text())
    expected = {
        "openai/gpt-5-mini",
        "google/gemini-2.5-flash",
        "deepseek/deepseek-chat",
        "meta-llama/llama-3.3-70b-instruct",
        "qwen/qwen3-32b",
        "meta-llama/llama-4-maverick",
        "qwen/qwen3-30b-a3b",
        "google/gemini-2.5-flash-lite",
    }
    observed = set()
    for name, entry in suite["baselines"].items():
        if not name.startswith("openrouter_"):
            continue
        config = yaml.safe_load((SOURCE / entry["config"]).read_text())
        observed.add(config["model_name"])
        settings = yaml.safe_load((SOURCE / config["api_config"]).read_text())["request"]
        request = build_request(config["model_name"], [], JUDGE_SCHEMA, "judgment", settings)
        if config["model_name"] == "openai/gpt-5-mini":
            assert "temperature" not in request
            assert request["max_tokens"] == 4096
            assert request["reasoning"] == {"effort": "minimal", "exclude": True}
    assert observed == expected
