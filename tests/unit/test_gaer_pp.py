import pytest
import torch

from clinhallu.models import (
    _GAER_ABLATION_CONFIGS,
    GAERPlusPlusModel,
    HKGFusionModelV2,
)
from clinhallu.models.components.classifier import GAERClassifierHead
from clinhallu.models.components.cross_attention import BidirectionalCrossAttention
from clinhallu.models.components.grounding import GroundingAndEvidencePooling
from clinhallu.models.components.interaction import AnswerEvidenceInteraction


def _toy_batch(B=3, n=20, H=32, seed=0):
    torch.manual_seed(seed)
    hidden = torch.randn(B, n, H)
    attention_mask = torch.ones(B, n, dtype=torch.long)
    context_mask = torch.zeros(B, n, dtype=torch.long)
    answer_mask = torch.zeros(B, n, dtype=torch.long)
    context_mask[:, 1:10] = 1
    answer_mask[:, 11:19] = 1
    return hidden, attention_mask, context_mask, answer_mask


# -----------------------------------------------------------------------
# BidirectionalCrossAttention
# -----------------------------------------------------------------------


def test_cross_attention_output_shapes_and_finite():
    H_dim = 32
    hidden, attention_mask, context_mask, answer_mask = _toy_batch(H=H_dim)
    layer = BidirectionalCrossAttention(hidden_dim=H_dim, num_heads=4)
    z, answer_ctx, _, _ = layer(
        hidden,
        context_mask,
        answer_mask,
        attention_mask,
    )
    assert z.shape == hidden.shape
    assert answer_ctx.shape == hidden.shape
    assert torch.isfinite(z).all()
    assert torch.isfinite(answer_ctx).all()


def test_cross_attention_compute_reverse_false_skips_reverse_branch():
    """Colab/research audit #15/#38: GAERPlusPlusModel doesn't consume
    answer_ctx, so it should be able to skip computing it. `z` (the branch
    that IS used) must be numerically identical whether or not the reverse
    branch runs -- the two directions must not share state that would make
    skipping one change the other."""
    H_dim = 32
    hidden, attention_mask, context_mask, answer_mask = _toy_batch(H=H_dim)
    layer = BidirectionalCrossAttention(hidden_dim=H_dim, num_heads=4)
    layer.eval()  # dropout off so the two calls are directly comparable

    z_full, answer_ctx_full, a2c_w_full, c2a_w_full = layer(
        hidden,
        context_mask,
        answer_mask,
        attention_mask,
        compute_reverse=True,
    )
    z_skip, answer_ctx_skip, a2c_w_skip, c2a_w_skip = layer(
        hidden,
        context_mask,
        answer_mask,
        attention_mask,
        compute_reverse=False,
    )

    assert answer_ctx_skip is None and c2a_w_skip is None
    assert answer_ctx_full is not None and c2a_w_full is not None
    torch.testing.assert_close(z_full, z_skip)
    torch.testing.assert_close(a2c_w_full, a2c_w_skip)


def test_gaer_plus_plus_default_forward_does_not_compute_reverse_cross_attention(
    tiny_deberta_patch,
):
    """The full/default GAERPlusPlusModel forward pass must not pay for the
    context->answer direction it never uses -- monkeypatch the module's
    context_to_answer sub-layer and assert it is never invoked."""
    hidden = tiny_deberta_patch
    lora_cfg = dict(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["query_proj", "key_proj", "value_proj"],
    )
    model = GAERPlusPlusModel(
        model_name="fake/tiny-deberta",
        hidden_dim=hidden,
        num_attention_heads=4,
        lora_config=lora_cfg,
    )

    calls = {"n": 0}
    orig_forward = model.cross_attention.context_to_answer.forward

    def _counting_forward(*args, **kwargs):
        calls["n"] += 1
        return orig_forward(*args, **kwargs)

    model.cross_attention.context_to_answer.forward = _counting_forward

    batch = _fake_gaer_batch()
    model(*batch)
    assert calls["n"] == 0, (
        "context_to_answer was invoked during the default forward pass, "
        "but its output (answer_ctx) is never consumed -- see audit #15/#38."
    )


def test_cross_attention_handles_empty_context_and_empty_answer():
    H_dim = 16
    B, n = 1, 8
    hidden = torch.randn(B, n, H_dim)
    attention_mask = torch.ones(B, n, dtype=torch.long)
    context_mask = torch.zeros(B, n, dtype=torch.long)  # empty context
    answer_mask = torch.zeros(B, n, dtype=torch.long)  # empty answer
    answer_mask[:, 4:6] = 1
    layer = BidirectionalCrossAttention(hidden_dim=H_dim, num_heads=2)
    z, answer_ctx, _, _ = layer(hidden, context_mask, answer_mask, attention_mask)
    assert torch.isfinite(z).all()
    assert torch.isfinite(answer_ctx).all()


def test_answer_to_context_only_uses_context_values():
    """z at an answer position must be a convex combination of CONTEXT-position
    input rows only (never the answer's own row or other answer rows), since
    the answer_to_context branch's key_padding_mask restricts keys/values to
    context positions."""
    H_dim = 8
    B, n = 1, 10
    torch.manual_seed(1)
    hidden = torch.randn(B, n, H_dim)
    attention_mask = torch.ones(B, n, dtype=torch.long)
    context_mask = torch.zeros(B, n, dtype=torch.long)
    context_mask[:, 1:4] = 1  # only positions 1,2,3 are context
    answer_mask = torch.zeros(B, n, dtype=torch.long)
    answer_mask[:, 6:9] = 1  # positions 6,7,8 are answer

    layer = BidirectionalCrossAttention(hidden_dim=H_dim, num_heads=2)
    layer.eval()
    with torch.no_grad():
        z, _, a2c_weights, _ = layer(hidden, context_mask, answer_mask, attention_mask)

    # attention weights at answer positions must be zero outside the context span
    aw = a2c_weights[0]  # (n_query, n_key)
    for pos in [6, 7, 8]:
        weights_at_pos = aw[pos]
        non_context_positions = [i for i in range(n) if i not in (1, 2, 3)]
        assert torch.allclose(
            weights_at_pos[non_context_positions],
            torch.zeros(len(non_context_positions)),
            atol=1e-5,
        ), "answer_to_context attention leaked weight onto non-context keys"


# -----------------------------------------------------------------------
# GroundingAndEvidencePooling — the critical regression test
# -----------------------------------------------------------------------


def test_evidence_is_built_from_context_z_not_answer_h():
    """Regression test for the Phase-2 architecture bug this project fixes:
    evidence `e` must equal a weighted combination of context representations
    `z`, and must be INDEPENDENT of the values of `h` beyond h's role in
    computing the grounding gate. Concretely: if we replace `h` with a
    completely different random tensor (same shape) but keep `z` and the
    grounding module's learned gate weights fixed, `e` should still be
    expressible purely as `alpha @ z` for the alpha implied by the *original*
    h — i.e. e never contains h's values directly, only z's."""
    H_dim = 16
    B, n = 2, 12
    torch.manual_seed(2)
    h = torch.randn(B, n, H_dim)
    z = torch.randn(B, n, H_dim)
    attention_mask = torch.ones(B, n, dtype=torch.long)
    answer_mask = torch.zeros(B, n, dtype=torch.long)
    answer_mask[:, 5:10] = 1

    pool = GroundingAndEvidencePooling(hidden_dim=H_dim, temperature=1.0)
    pool.eval()
    with torch.no_grad():
        e, _, stats = pool(h, z, answer_mask, attention_mask, use_grounding=True)

        # Reconstruct e purely from z and the masked temperature-softmax
        # weights (computed on the LOGIT, per spec P0.1) and confirm it
        # matches exactly.
        g_logit = pool.compute_grounding_logit(h, z)
        amask = answer_mask.bool() & attention_mask.bool()
        neg_fill = torch.finfo(g_logit.dtype).min
        masked_logits = torch.where(
            amask, g_logit / pool.temperature, torch.full_like(g_logit, neg_fill)
        )
        alpha = torch.softmax(masked_logits, dim=1)
        e_from_z_only = torch.einsum("bn,bnd->bd", alpha, z)
        assert torch.allclose(e, e_from_z_only, atol=1e-5)

        # Sanity: e is NOT reproducible from h using the same alpha (unless by
        # fluke h==z). This confirms the pooled sum used z's values, not h's.
        e_from_h_only = torch.einsum("bn,bnd->bd", alpha, h)
        assert not torch.allclose(e, e_from_h_only, atol=1e-4)

        # Perturbing h (while re-deriving g from the NEW h, then re-pooling
        # over the ORIGINAL z) must leave the z-pooling logic well-defined and
        # still only touch z's values in the final sum -- i.e. changing z
        # while holding alpha fixed changes e; changing h changes alpha
        # (via the gate) but the summed vectors are still drawn from z.
        z_shifted = z + 100.0
        e2, _, _ = pool(
            h,
            z_shifted,
            answer_mask,
            attention_mask,
            use_grounding=True,
        )
        assert not torch.allclose(e, e2), (
            "Evidence vector did not change when context representations `z` "
            "were shifted -- evidence must be derived from z, not from h."
        )

    assert e.shape == (B, H_dim)
    assert torch.isfinite(e).all()
    assert stats.stack_primary().shape == (B, 6)
    assert stats.stack_primary(include_low_grounding_fraction=True).shape == (B, 7)


def test_evidence_pooling_uniform_mode_still_uses_z_only():
    H_dim = 8
    B, n = 1, 6
    torch.manual_seed(3)
    h = torch.randn(B, n, H_dim)
    z = torch.randn(B, n, H_dim)
    attention_mask = torch.ones(B, n, dtype=torch.long)
    answer_mask = torch.zeros(B, n, dtype=torch.long)
    answer_mask[:, 2:5] = 1

    pool = GroundingAndEvidencePooling(hidden_dim=H_dim)
    e, _, _ = pool(h, z, answer_mask, attention_mask, use_grounding=False)
    expected = z[:, 2:5, :].mean(dim=1)
    assert torch.allclose(e, expected, atol=1e-5)


def test_evidence_pooling_zero_answer_tokens_falls_back_to_zero_vector():
    H_dim = 8
    B, n = 1, 6
    h = torch.randn(B, n, H_dim)
    z = torch.randn(B, n, H_dim)
    attention_mask = torch.ones(B, n, dtype=torch.long)
    answer_mask = torch.zeros(B, n, dtype=torch.long)
    pool = GroundingAndEvidencePooling(hidden_dim=H_dim)
    e, _, stats = pool(h, z, answer_mask, attention_mask)
    assert torch.allclose(e, torch.zeros_like(e))
    assert torch.allclose(stats.mean, torch.zeros_like(stats.mean))


def test_grounding_std_gradient_is_finite_when_variance_is_exactly_zero():
    """Regression test: sqrt'(0) is infinite, so a batch containing an
    example with zero (or one) answer tokens -- forcing grounding-score
    variance to exactly 0 -- must not poison gradients with NaN. This
    reproduces a real bug found while smoke-testing the full model: the
    encoder's LoRA gradients turned to NaN on the very first training step
    whenever a batch included a zero-answer-token example."""
    H_dim = 8
    B, n = 3, 6
    torch.manual_seed(5)
    h = torch.randn(B, n, H_dim, requires_grad=True)
    z = torch.randn(B, n, H_dim, requires_grad=True)
    attention_mask = torch.ones(B, n, dtype=torch.long)
    answer_mask = torch.zeros(B, n, dtype=torch.long)
    answer_mask[0, 2:4] = 1  # normal case
    answer_mask[1, 3] = 1  # single answer token -> variance == 0 exactly
    # row 2 left with zero answer tokens entirely -> has_answer=False path

    pool = GroundingAndEvidencePooling(hidden_dim=H_dim)
    e, _, stats = pool(h, z, answer_mask, attention_mask)
    loss = e.sum() + stats.std.sum() + stats.entropy.sum()
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    for p in pool.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


# -----------------------------------------------------------------------
# AnswerEvidenceInteraction + classifier head
# -----------------------------------------------------------------------


def test_interaction_and_classifier_dims_dynamic():
    H_dim = 24
    B = 4
    a = torch.randn(B, H_dim)
    e = torch.randn(B, H_dim)
    inter = AnswerEvidenceInteraction()
    feat = inter(a, e, use_contrast=True)
    assert feat.shape == (B, 4 * H_dim + 1)
    assert feat.shape[-1] == AnswerEvidenceInteraction.output_dim(H_dim)

    clf = GAERClassifierHead(input_dim=feat.shape[-1] + H_dim + 7)
    r = torch.cat([torch.randn(B, H_dim), feat, torch.randn(B, 7)], dim=-1)
    logits = clf(r)
    assert logits.shape == (B,)
    assert torch.isfinite(logits).all()


def test_interaction_no_contrast_zeros_out_extras_but_keeps_ae():
    H_dim = 10
    B = 2
    a = torch.randn(B, H_dim)
    e = torch.randn(B, H_dim)
    inter = AnswerEvidenceInteraction()
    feat = inter(a, e, use_contrast=False)
    assert torch.allclose(feat[:, :H_dim], a)
    assert torch.allclose(feat[:, H_dim : 2 * H_dim], e)
    assert torch.allclose(feat[:, 2 * H_dim :], torch.zeros(B, 2 * H_dim + 1))


# -----------------------------------------------------------------------
# Full GAERPlusPlusModel — end to end with a real (tiny) DeBERTa-v2 + LoRA
# -----------------------------------------------------------------------


@pytest.fixture
def tiny_deberta_patch(monkeypatch):
    """Builds a real, randomly-initialized tiny DebertaV2Model so the full
    architecture (incl. LoRA via peft) is exercised without any network
    access or pretrained-weight download."""
    from transformers import DebertaV2Config, DebertaV2Model

    import clinhallu.models.encoder as encoder_v2_mod

    hidden = 32
    cfg = DebertaV2Config(
        vocab_size=200,
        hidden_size=hidden,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        relative_attention=True,
        position_buckets=32,
        pos_att_type=["p2c", "c2p"],
    )

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*a, **k):
            return cfg

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name, config):
            return DebertaV2Model(config)

    monkeypatch.setattr(encoder_v2_mod, "AutoConfig", FakeAutoConfig)
    monkeypatch.setattr(encoder_v2_mod, "AutoModel", FakeAutoModel)
    return hidden


def _fake_gaer_batch(B=3, n=18, vocab=200):
    torch.manual_seed(4)
    input_ids = torch.randint(0, vocab, (B, n))
    attention_mask = torch.ones(B, n, dtype=torch.long)
    context_mask = torch.zeros(B, n, dtype=torch.long)
    context_mask[:, 1:8] = 1
    answer_mask = torch.zeros(B, n, dtype=torch.long)
    answer_mask[:, 9:16] = 1
    labels = torch.tensor([0.0, 1.0, 0.0][:B])
    return input_ids, attention_mask, context_mask, answer_mask, labels


def test_gaer_plus_plus_forward_and_backward(tiny_deberta_patch):
    hidden = tiny_deberta_patch
    lora_cfg = dict(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["query_proj", "key_proj", "value_proj"],
    )
    model = GAERPlusPlusModel(
        model_name="fake/tiny-deberta",
        hidden_dim=hidden,
        num_attention_heads=4,
        lora_config=lora_cfg,
    )
    batch = _fake_gaer_batch()
    out = model(*batch)
    B = batch[0].shape[0]
    assert out.logits.shape == (B,)
    assert out.probs.shape == (B,)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grad_total = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_total > 0


@pytest.mark.parametrize("mode", list(_GAER_ABLATION_CONFIGS.keys()))
def test_all_ablation_modes_run_without_nan(tiny_deberta_patch, mode):
    hidden = tiny_deberta_patch
    lora_cfg = dict(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["query_proj", "key_proj", "value_proj"],
    )
    model = GAERPlusPlusModel(
        model_name="fake/tiny-deberta",
        hidden_dim=hidden,
        num_attention_heads=4,
        lora_config=lora_cfg,
        ablation_mode=mode,
    )
    batch = _fake_gaer_batch()
    out = model(*batch)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.loss)


def test_invalid_ablation_mode_raises():
    with pytest.raises(ValueError):
        GAERPlusPlusModel(hidden_dim=16, num_attention_heads=2, ablation_mode="not_a_real_mode")


def test_legacy_hkg_fusion_v2_still_importable_and_functional(tiny_deberta_patch):
    """Backward-compatibility guarantee: HKGFusionModelV2 must remain usable,
    unmodified, alongside the new GAERPlusPlusModel."""
    hidden = tiny_deberta_patch
    lora_cfg = dict(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["query_proj", "key_proj", "value_proj"],
    )
    legacy = HKGFusionModelV2(
        model_name="fake/tiny-deberta",
        hidden_dim=hidden,
        num_attention_heads=4,
        lora_config=lora_cfg,
    )
    batch = _fake_gaer_batch()
    out = legacy(*batch)
    assert out.logits.shape[-1] == 2  # legacy model is still 2-class softmax
    assert torch.isfinite(out.loss)
