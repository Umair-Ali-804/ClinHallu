"""Exact segment tokenization, datasets, collation, and data loaders."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import PreTrainedTokenizerBase

from clinhallu.core.seeding import worker_init_fn

from .context_selection import (
    ContextSelectionConfig,
    context_selection_from_config,
    select_context_sentences,
)
from .resumable_sampler import ResumableRandomSampler

logger = logging.getLogger(__name__)


class HKGFusionDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        require_labels: bool = True,
        context_selection: Optional[ContextSelectionConfig] = None,
        question_budget: int = 96,
        answer_budget: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.require_labels = require_labels
        # The primary config enables answer-aware selection; ablations can
        # disable it without changing dataset code.
        self.context_selection = context_selection or ContextSelectionConfig(enabled=False)
        # Question and answer receive hard caps. Context uses the remaining
        # budget after four special tokens.
        self.question_budget = question_budget
        self.answer_budget = answer_budget
        self.samples = self._load(jsonl_path)

        if require_labels:
            missing = [i for i, s in enumerate(self.samples) if "hall_label" not in s]
            if missing:
                raise KeyError(
                    f"{jsonl_path}: {len(missing)} record(s) (e.g. index {missing[0]}) "
                    f"have no 'hall_label' field. Use require_labels=False for "
                    f"label-free inference."
                )

        logger.info(
            "Loaded %d samples from %s (require_labels=%s, question_budget=%d, "
            "answer_budget=%d, max_length=%d)",
            len(self.samples),
            jsonl_path,
            require_labels,
            self.question_budget,
            self.answer_budget,
            self.max_length,
        )

    @staticmethod
    def _load(path: str) -> List[Dict]:
        records: List[Dict] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _truncate_preserving_ends(ids: List[int], budget: int) -> List[int]:
        """Truncate an answer while preserving both its beginning and end."""
        if len(ids) <= budget:
            return ids
        if budget <= 0:
            return []
        head = budget // 2
        tail = budget - head
        return ids[:head] + ids[-tail:] if tail > 0 else ids[:head]

    def _encode_example(self, question: str, context: str, answer: str) -> Dict[str, List[int]]:
        """Build the exact token sequence
            [CLS] Question [SEP] Context [SEP] Answer [SEP]
        from independently tokenized segments and explicit masks."""
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        if cls_id is None or sep_id is None:
            raise ValueError(
                f"Tokenizer {self.tokenizer.__class__.__name__} is missing "
                f"cls_token_id/sep_token_id -- GAER++'s exact segment "
                f"construction requires both."
            )

        q_ids = self.tokenizer.encode(question, add_special_tokens=False)
        a_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        if len(q_ids) > self.question_budget:
            q_ids = q_ids[: self.question_budget]
        if len(a_ids) > self.answer_budget:
            a_ids = self._truncate_preserving_ends(a_ids, self.answer_budget)

        num_special_tokens = 4  # [CLS] + 3x[SEP]
        context_budget = self.max_length - num_special_tokens - len(q_ids) - len(a_ids)
        context_budget = max(context_budget, 0)

        c_ids = self.tokenizer.encode(context, add_special_tokens=False)
        if len(c_ids) > context_budget:
            # Context has already been ranked/selected upstream (when context
            # selection is enabled) so the highest-scoring sentences appear
            # first; a simple prefix truncation to the remaining budget keeps
            # the most relevant material. When context selection is
            # disabled, this is a plain beginning-truncation of the raw
            # context. The cut point comes from the explicit segment budget.
            c_ids = c_ids[:context_budget]

        input_ids = [cls_id] + q_ids + [sep_id] + c_ids + [sep_id] + a_ids + [sep_id]

        q_start = 1
        q_end = q_start + len(q_ids)
        c_start = q_end + 1  # skip [SEP]
        c_end = c_start + len(c_ids)
        a_start = c_end + 1  # skip [SEP]
        a_end = a_start + len(a_ids)
        # (final [SEP] at index a_end, unmasked in all three segment masks)

        seq_len = len(input_ids)
        question_mask = [0] * seq_len
        context_mask = [0] * seq_len
        answer_mask = [0] * seq_len
        for i in range(q_start, q_end):
            question_mask[i] = 1
        for i in range(c_start, c_end):
            context_mask[i] = 1
        for i in range(a_start, a_end):
            answer_mask[i] = 1

        assert len(input_ids) <= self.max_length, (
            f"Constructed sequence length {len(input_ids)} exceeds max_length "
            f"{self.max_length} -- budget arithmetic bug."
        )
        assert sum(question_mask) == len(q_ids)
        assert sum(context_mask) == len(c_ids)
        assert sum(answer_mask) == len(a_ids)

        return {
            "input_ids": input_ids,
            "question_mask": question_mask,
            "context_mask": context_mask,
            "answer_mask": answer_mask,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        question = s["question"]
        context = s["context"]
        answer = s["answer"]
        if self.context_selection.enabled:
            context = select_context_sentences(
                question=question,
                context=context,
                answer=answer,
                cfg=self.context_selection,
                tokenizer=self.tokenizer,
            )

        enc = self._encode_example(question, context, answer)
        seq_len = len(enc["input_ids"])

        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "question_mask": torch.tensor(enc["question_mask"], dtype=torch.float32),
            "context_mask": torch.tensor(enc["context_mask"], dtype=torch.float32),
            "answer_mask": torch.tensor(enc["answer_mask"], dtype=torch.float32),
        }
        if "hall_label" in s:
            item["hall_label"] = torch.tensor(int(s["hall_label"]), dtype=torch.long)
        return item


def collate_fn_v2(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)
    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    question_mask = torch.zeros(B, max_len, dtype=torch.float32)
    context_mask = torch.zeros(B, max_len, dtype=torch.float32)
    answer_mask = torch.zeros(B, max_len, dtype=torch.float32)

    for i, item in enumerate(batch):
        L = item["input_ids"].size(0)
        input_ids[i, :L] = item["input_ids"]
        attention_mask[i, :L] = item["attention_mask"]
        question_mask[i, :L] = item["question_mask"]
        context_mask[i, :L] = item["context_mask"]
        answer_mask[i, :L] = item["answer_mask"]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "question_mask": question_mask,
        "context_mask": context_mask,
        "answer_mask": answer_mask,
    }
    if all("hall_label" in item for item in batch):
        out["labels"] = torch.stack([item["hall_label"] for item in batch])
    return out


def build_dataloaders_v2(
    cfg: Dict,
    tokenizer: PreTrainedTokenizerBase,
    device: Optional[torch.device] = None,
    smoke_test: bool = False,
    smoke_test_n: int = 1000,
) -> Tuple[DataLoader, DataLoader]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = cfg["data"]["processed_dir"]
    batch_size = cfg["training"]["batch_size"]
    max_length = cfg["model"]["max_length"]
    num_workers = cfg["data"].get("num_workers", 0)
    pin = device.type == "cuda"
    question_budget = cfg["data"].get("question_budget", 96)
    answer_budget = cfg["data"].get("answer_budget", 128)

    context_selection = context_selection_from_config(cfg)

    # A smoke test intentionally tokenizes only its small subset; building a
    # full 14k-row cache first would defeat the purpose of a quick check.
    if cfg["data"].get("use_cache", True) and not smoke_test:
        from .cache import load_or_build_feature_dataset

        train_ds = load_or_build_feature_dataset(cfg, tokenizer, "train")
        val_ds = load_or_build_feature_dataset(cfg, tokenizer, "validation")
    else:
        train_ds = HKGFusionDataset(
            f"{processed_dir}/train.jsonl",
            tokenizer,
            max_length,
            context_selection=context_selection,
            question_budget=question_budget,
            answer_budget=answer_budget,
        )
        val_ds = HKGFusionDataset(
            f"{processed_dir}/validation.jsonl",
            tokenizer,
            max_length,
            context_selection=context_selection,
            question_budget=question_budget,
            answer_budget=answer_budget,
        )

    if smoke_test:
        # Cap to ~1000 train examples / a small validation slice, so a
        # full pass (load -> tokenize -> forward -> backward -> optimizer ->
        # validate -> F1 -> threshold -> checkpoint) can be verified quickly
        # before committing to a full run.
        train_ds = Subset(train_ds, list(range(min(smoke_test_n, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(max(smoke_test_n // 5, 50), len(val_ds)))))
        logger.warning(
            "SMOKE TEST MODE: train=%d val=%d (capped). Do NOT treat "
            "these metrics as representative -- this only verifies the pipeline runs.",
            len(train_ds),
            len(val_ds),
        )

    loader_options = {
        "collate_fn": collate_fn_v2,
        "num_workers": num_workers,
        "pin_memory": pin,
        "worker_init_fn": worker_init_fn,
        "persistent_workers": num_workers > 0,
    }
    # A per-epoch deterministic sampler rather than shuffle=True: the order for
    # epoch k depends only on (seed, k), so an interrupted run can resume in the
    # middle of an epoch without repeating or skipping examples.
    train_sampler = ResumableRandomSampler(train_ds, seed=int(cfg["training"]["seed"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        **loader_options,
    )
    validation_batch_size = int(cfg["training"].get("validation_batch_size", batch_size * 2))
    val_loader = DataLoader(
        val_ds,
        batch_size=validation_batch_size,
        shuffle=False,
        **loader_options,
    )
    logger.info("v2 DataLoaders built — train:%d  val:%d", len(train_ds), len(val_ds))
    return train_loader, val_loader
