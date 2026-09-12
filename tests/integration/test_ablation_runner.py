from clinhallu.experiments.ablation import ABLATION_CONFIGS
from clinhallu.models.gaer_pp import _GAER_ABLATION_CONFIGS


def test_all_registered_ablation_configs_have_model_modes():
    expected = {
        "B": "B_answer_pooling",
        "C": "C_cross_attention",
        "D": "D_grounding",
        "E": "E_evidence_pooling",
        "F": "F_full_gaer_plus_plus",
    }
    assert set(ABLATION_CONFIGS) == set(expected)
    assert all(mode in _GAER_ABLATION_CONFIGS for mode in expected.values())
