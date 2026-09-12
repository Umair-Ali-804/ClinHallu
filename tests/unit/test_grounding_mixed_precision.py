import torch

from clinhallu.models.components.grounding import GroundingAndEvidencePooling


def test_half_precision_single_token_and_empty_answer_have_finite_gradients():
    module = GroundingAndEvidencePooling(hidden_dim=8).half()
    h = torch.randn(2, 4, 8, dtype=torch.float16, requires_grad=True)
    z = torch.randn_like(h, requires_grad=True)
    answer = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]])
    evidence, _, stats = module(h, z, answer, torch.ones_like(answer))
    loss = evidence.float().sum() + stats.std.sum() + stats.entropy.sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(h.grad).all() and torch.isfinite(z.grad).all()
    assert torch.isfinite(module.grounding_proj.weight.grad).all()
