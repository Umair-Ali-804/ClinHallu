"""Data schemas, validation, preprocessing and model-ready datasets.

Dataset imports are lazy so lightweight validation and preprocessing commands
do not initialize PyTorch or Transformers.
"""

from .validation import validate_fixed_splits

__all__ = [
    "HKGFusionDataset",
    "build_dataloaders_v2",
    "collate_fn_v2",
    "validate_fixed_splits",
]


def __getattr__(name: str):
    if name in {"HKGFusionDataset", "build_dataloaders_v2", "collate_fn_v2"}:
        from .dataset import HKGFusionDataset, build_dataloaders_v2, collate_fn_v2

        return {
            "HKGFusionDataset": HKGFusionDataset,
            "build_dataloaders_v2": build_dataloaders_v2,
            "collate_fn_v2": collate_fn_v2,
        }[name]
    raise AttributeError(name)
