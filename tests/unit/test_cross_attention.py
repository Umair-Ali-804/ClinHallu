import torch

from clinhallu.models.components.cross_attention import BidirectionalCrossAttention


def test_cross_attention_shapes_and_skip_reverse():
    hidden = torch.randn(2, 10, 16)
    real = torch.ones(2, 10, dtype=torch.long)
    context = torch.zeros_like(real)
    context[:, 1:5] = 1
    answer = torch.zeros_like(real)
    answer[:, 6:9] = 1
    module = BidirectionalCrossAttention(16, 4, 0.0).eval()
    evidence, reverse, _, reverse_weights = module(
        hidden, context, answer, real, compute_reverse=False
    )
    assert evidence.shape == hidden.shape
    assert reverse is None and reverse_weights is None
    assert torch.isfinite(evidence).all()
