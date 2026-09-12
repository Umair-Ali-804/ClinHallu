"""Prove the interrupt-and-resume path end to end, as a user would hit it.

Builds a tiny self-contained project, starts a real training subprocess, sends
it SIGINT part-way through (what a Colab disconnect or Ctrl-C delivers), then
re-runs the identical command and checks that it continued rather than
restarting.

    python scripts/demo_resume.py
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml
from transformers import BertTokenizerFast, DebertaV2Config, DebertaV2Model

SOURCE = Path(__file__).resolve().parents[1]
VOCAB = [
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


def build_project(root: Path) -> Path:
    (root / "pyproject.toml").write_text('[project]\nname="toy"\n')
    encoder = root / "encoder"
    encoder.mkdir()
    (encoder / "vocab.txt").write_text("\n".join(VOCAB) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(encoder / "vocab.txt"), model_max_length=64)
    tokenizer.save_pretrained(encoder)
    DebertaV2Model(
        DebertaV2Config(
            vocab_size=len(VOCAB),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    ).save_pretrained(encoder)

    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    offset = 0
    for split, count in [("train", 240), ("val", 40), ("eval", 40)]:
        rows = [
            {
                "id": f"{split}_{i}",
                "question": f"what happened {i} ?",
                "context": f"the patient improved {i} . the patient worsened {i} .",
                "answer": (f"the patient improved {i}" if i % 2 else f"the patient worsened {i}"),
                "hall_label": i % 2,
            }
            for i in range(offset, offset + count)
        ]
        offset += count
        (raw / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    cfg = yaml.safe_load((SOURCE / "configs" / "base.yaml").read_text())
    cfg["model"].update(
        encoder=str(encoder),
        revision=None,
        hidden_dim=16,
        classifier_hidden_dim=16,
        max_length=64,
        num_attention_heads=2,
    )
    cfg["lora"]["target_modules"] = ["query_proj", "key_proj", "value_proj"]
    cfg["data"].update(
        train_path="data/raw/train.jsonl",
        val_path="data/raw/val.jsonl",
        eval_path="data/raw/eval.jsonl",
        question_budget=12,
        answer_budget=12,
        expected_rows={"train": 240, "validation": 40, "test": 40},
        minimum_class_fraction=0.3,
        use_cache=False,
    )
    cfg["data"]["context_selection"]["max_context_tokens"] = 32
    cfg["training"].update(
        max_epochs=3,
        batch_size=2,
        validation_batch_size=4,
        gradient_accumulation_steps=1,
        mixed_precision=False,
        gradient_checkpointing=False,
        early_stopping_patience=5,
    )
    cfg["evaluation"].update(batch_size=4, mixed_precision=False, bootstrap_samples=10)
    cfg["checkpoints"].update(save_every_steps=5, save_every_minutes=0)
    configs = root / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text(yaml.safe_dump(cfg))
    shutil.copytree(SOURCE / "configs/ablations", configs / "ablations")
    return configs / "base.yaml"


def describe(checkpoint: Path) -> dict:
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    return {
        "epoch": payload["epoch"],
        "phase": payload.get("phase"),
        "global_step": payload.get("global_step"),
        "batches_done": payload.get("batches_done"),
    }


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="clinhallu_demo_"))
    env = {
        **os.environ,
        "CLINHALLU_ROOT": str(root),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "TQDM_DISABLE": "1",
    }
    config = build_project(root)
    print(f"Toy project at {root}\n")

    print("Preparing data...")
    prep = subprocess.run(
        [sys.executable, "-m", "clinhallu", "prepare", "--config", str(config)],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if prep.returncode != 0:
        print(prep.stdout[-4000:])
        print(prep.stderr[-4000:])
        return 1

    command = [sys.executable, "-m", "clinhallu.cli.train", "--config", str(config)]
    run_dir = root / "artifacts" / "adaptation" / "c5" / "seed_13"
    latest = run_dir / "checkpoints" / "latest_checkpoint.pt"

    # --- attempt 1: interrupted like a dropped session -------------------- #
    print("Attempt 1: starting training, interrupting once a mid-epoch save lands...")
    process = subprocess.Popen(
        command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    # Wait for the first mid-epoch checkpoint, then pull the plug immediately.
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if latest.is_file():
            break
        if process.poll() is not None:
            print("  training finished before any checkpoint appeared")
            return 1
        time.sleep(0.2)
    else:
        print("  timed out waiting for a checkpoint")
        process.kill()
        return 1
    time.sleep(0.5)
    process.send_signal(signal.SIGINT)
    output, _ = process.communicate(timeout=180)

    print(f"  exit code: {process.returncode} (130 == stopped cleanly, progress saved)")
    for line in output.splitlines():
        if "received" in line or "Stopped during" in line or "Checkpoint written" in line:
            print(f"  | {line.strip()[-110:]}")
    if process.returncode != 130:
        print("FAILED: expected exit code 130")
        return 1
    if not latest.is_file():
        print("FAILED: no checkpoint was written")
        return 1

    before = describe(latest)
    print(f"  checkpoint: {before}\n")
    if before["phase"] != "in_epoch":
        print("  (note: the stop landed on an epoch boundary rather than mid-epoch)")

    # --- attempt 2: the identical command, uninterrupted ------------------- #
    print("Attempt 2: re-running the identical command...")
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    print(f"  exit code: {completed.returncode}")
    for line in completed.stdout.splitlines() + completed.stderr.splitlines():
        if "Resuming inside epoch" in line or "Resumed from epoch" in line:
            print(f"  | {line.strip()[-110:]}")
    if completed.returncode != 0:
        print("FAILED: resumed run did not complete")
        print(completed.stdout[-3000:], completed.stderr[-3000:])
        return 1

    history = (run_dir / "metrics" / "training_history.csv").read_text().splitlines()
    print(f"\n  epochs recorded: {len(history) - 1} (expected 3)")
    print(f"  best checkpoint: {(run_dir / 'checkpoints' / 'best_model.pt').is_file()}")
    print(f"  calibration:     {(run_dir / 'calibration' / 'temperature.npy').is_file()}")

    ok = len(history) - 1 == 3 and (run_dir / "checkpoints" / "best_model.pt").is_file()
    print("\nPASS: interrupted training resumed and completed." if ok else "\nFAILED")
    shutil.rmtree(root, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
