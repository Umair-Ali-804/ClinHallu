from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from clinhallu.engine.trainer import run_epoch


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 1)

    def forward(self, input_ids, attention_mask, context_mask, answer_mask, labels=None):
        features = input_ids.float().mean(dim=1, keepdim=True)
        logits = self.projection(features).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return SimpleNamespace(
            logits=logits,
            probs=probabilities,
            p_hallucinated=probabilities,
            loss=loss,
        )


def test_one_training_epoch_runs():
    samples = []
    for index in range(4):
        samples.append(
            {
                "input_ids": torch.tensor([index + 1, index + 2]),
                "attention_mask": torch.ones(2, dtype=torch.long),
                "context_mask": torch.tensor([1.0, 0.0]),
                "answer_mask": torch.tensor([0.0, 1.0]),
                "labels": torch.tensor(index % 2),
            }
        )
    loader = DataLoader(samples, batch_size=2)
    model = TinyDetector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        optimizer=optimizer,
        gradient_accumulation_steps=2,
    )
    assert len(result["labels"]) == 4
    assert result["metrics"]["loss"] >= 0
