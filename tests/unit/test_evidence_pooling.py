import torch

from clinhallu.models.components.evidence_pooling import EvidenceAwarePooling


def test_legacy_evidence_pooling_shape():
    module = EvidenceAwarePooling(12)
    hidden = torch.randn(3, 7, 12)
    mask = torch.zeros(3, 7)
    mask[:, 2:6] = 1
    support = torch.rand(3, 7)
    output = module(hidden, mask, support)
    assert output.shape == (3, 12)
    assert torch.isfinite(output).all()
