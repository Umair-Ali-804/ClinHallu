import torch

from clinhallu.models.components.grounding import GroundingAndEvidencePooling


def test_grounding_is_finite_with_masked_tokens():
    module = GroundingAndEvidencePooling(hidden_dim=8)
    answer = torch.randn(2, 6, 8, requires_grad=True)
    context = torch.randn(2, 6, 8, requires_grad=True)
    mask = torch.zeros(2, 6)
    mask[:, 1:4] = 1
    real = torch.ones_like(mask)
    evidence, grounding, stats = module(answer, context, mask, real)
    assert evidence.shape == (2, 8)
    assert torch.isfinite(grounding).all()
    assert torch.isfinite(stats.stack_primary()).all()
    evidence.sum().backward()
