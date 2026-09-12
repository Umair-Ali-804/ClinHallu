"""GAER++ and backward-compatible legacy model APIs."""

from .factory import build_model
from .gaer_pp import _GAER_ABLATION_CONFIGS, GAERPlusPlusModel
from .legacy_hkg import HKGFusionModelV2
from .outputs import GAERPlusPlusOutput, HKGFusionOutput

__all__ = [
    "GAERPlusPlusModel",
    "GAERPlusPlusOutput",
    "HKGFusionModelV2",
    "HKGFusionOutput",
    "_GAER_ABLATION_CONFIGS",
    "build_model",
]
