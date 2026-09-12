"""Actual tiny-model training and evaluation, without downloads or API calls."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from transformers import (
    BertTokenizerFast,
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
    DebertaV2Model,
)

from clinhallu.baselines.tfidf_lr import main as tfidf_main
from clinhallu.experiments.runner import run_experiment
from clinhallu.reporting import build_publication_outputs

SOURCE = Path(__file__).resolve().parents[2]


@pytest.fixture
def local_project(tmp_path, monkeypatch):
    monkeypatch.setenv("CLINHALLU_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    torch.set_num_threads(1)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="toy"\n')
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "the",
        "patient",
        "improved",
        "worsened",
        "what",
        "happened",
        "question",
        "context",
        "answer",
        ".",
        "?",
    ]
    (encoder / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(encoder / "vocab.txt"), model_max_length=64)
    tokenizer.save_pretrained(encoder)
    model_cfg = DebertaV2Config(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        relative_attention=True,
        position_buckets=16,
        max_relative_positions=64,
    )
    DebertaV2Model(model_cfg).save_pretrained(encoder)
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    for split, count in [("train", 8), ("val", 4), ("eval", 4)]:
        rows = [
            {
                "id": f"{split}_{i}",
                "question": f"what happened {split} {i}?",
                "context": f"the patient improved. context {split} {i}.",
                "answer": "the patient improved." if i % 2 == 0 else "the patient worsened.",
                "hallu_label": i % 2,
            }
            for i in range(count)
        ]
        (raw / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    configs = tmp_path / "configs"
    configs.mkdir()
    cfg = yaml.safe_load((SOURCE / "configs/base.yaml").read_text())
    cfg["model"].update(
        encoder=str(encoder),
        revision=None,
        hidden_dim=16,
        classifier_hidden_dim=16,
        max_length=64,
        num_attention_heads=2,
    )
    cfg["lora"].update(r=2, lora_alpha=4)
    cfg["data"].update(
        train_path="data/raw/train.jsonl",
        val_path="data/raw/val.jsonl",
        eval_path="data/raw/eval.jsonl",
        question_budget=12,
        answer_budget=12,
        expected_rows={"train": 8, "validation": 4, "test": 4},
    )
    cfg["data"]["context_selection"]["max_context_tokens"] = 32
    cfg["training"].update(
        max_epochs=1,
        batch_size=2,
        validation_batch_size=2,
        gradient_accumulation_steps=3,
        mixed_precision=False,
        gradient_checkpointing=False,
    )
    cfg["evaluation"].update(batch_size=2, mixed_precision=False, bootstrap_samples=10)
    (configs / "base.yaml").write_text(yaml.safe_dump(cfg))
    shutil.copytree(SOURCE / "configs/ablations", configs / "ablations")
    return tmp_path


def assert_evaluated(result):
    assert result["status"] == "evaluated"
    run = Path(result["run_dir"])
    metrics = json.loads(Path(result["metrics"]).read_text())
    rows = [
        json.loads(s)
        for s in (run / "predictions/evaluation_predictions.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4 and len({r["id"] for r in rows}) == 4
    assert (run / "checkpoints/model_only.safetensors").is_file()
    assert np.isfinite(metrics["eval_f1"])
    assert 0 <= metrics["threshold"] <= 1 and metrics["temperature"] > 0
    return run


def test_full_training_calibration_evaluation_resume_and_report(local_project, monkeypatch):
    cfg = local_project / "configs/base.yaml"
    result = run_experiment(cfg)
    assert_evaluated(result)

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed experiment should reuse artifacts")

    monkeypatch.setattr("clinhallu.experiments.runner._run", forbidden)
    assert run_experiment(cfg)["run_dir"] == result["run_dir"]
    raw = local_project / "data/raw"
    tfidf_main(
        [
            "--train",
            str(raw / "train.jsonl"),
            "--val",
            str(raw / "val.jsonl"),
            "--eval",
            str(raw / "eval.jsonl"),
            "--output-dir",
            str(local_project / "artifacts/baselines/tfidf_lr"),
        ]
    )
    build_publication_outputs(local_project)
    assert (local_project / "artifacts/reports/tables/baseline_results.csv").is_file()
    assert list((local_project / "artifacts/reports/figures").glob("*.png"))


@pytest.mark.parametrize(
    "filename",
    [
        "b_answer_pooling",
        "c_cross_attention",
        "d_grounding",
        "e_evidence_pooling",
        "f_full_gaer_pp",
    ],
)
def test_every_ablation_trains_and_evaluates(local_project, filename):
    result = run_experiment(local_project / f"configs/ablations/{filename}.yaml")
    run = assert_evaluated(result)
    assert "ablations" in run.parts


def test_smoke_run_never_evaluates_frozen_test(local_project):
    result = run_experiment(local_project / "configs/base.yaml", smoke_test=True)
    assert result["status"] == "smoke_test_complete"
    assert not (Path(result["run_dir"]) / "metrics/evaluation_metrics.json").exists()


def test_real_local_nli_and_semantic_baselines(local_project):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules

    from clinhallu.baselines.sample_consistency import NLIPairScorer
    from clinhallu.baselines.semantic_similarity import main as semantic_main
    from clinhallu.baselines.zero_shot_nli import main as nli_main

    encoder = local_project / "encoder"
    nli = local_project / "nli"
    cfg = DebertaV2Config.from_pretrained(encoder)
    cfg.num_labels = 3
    cfg.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    cfg.label2id = {v: k for k, v in cfg.id2label.items()}
    DebertaV2ForSequenceClassification(cfg).save_pretrained(nli)
    BertTokenizerFast.from_pretrained(encoder).save_pretrained(nli)
    values = np.asarray(
        NLIPairScorer(str(nli), revision=None, batch_size=2, max_length=64)(
            [
                ["the patient improved.", "the patient improved."],
                ["the patient improved.", "the patient worsened."],
            ]
        )
    )
    assert values.shape == (2, 2) and np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()
    transformer = modules.Transformer(str(encoder), max_seq_length=64)
    semantic = local_project / "semantic"
    SentenceTransformer(modules=[transformer, modules.Pooling(16)]).save(str(semantic))
    raw = local_project / "data/raw"
    for name, main, model in [("nli", nli_main, nli), ("semantic", semantic_main, semantic)]:
        out = local_project / "artifacts/baselines" / name
        main(
            [
                "--train",
                str(raw / "train.jsonl"),
                "--val",
                str(raw / "val.jsonl"),
                "--eval",
                str(raw / "eval.jsonl"),
                "--model-name",
                str(model),
                "--cache-dir",
                str(local_project / "cache"),
                "--output-dir",
                str(out),
            ]
        )
        rows = (out / "predictions.jsonl").read_text().splitlines()
        assert len(rows) == 4
        assert 0 <= json.loads((out / "metrics.json").read_text())["auc"] <= 1
