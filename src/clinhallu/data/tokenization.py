"""Public tokenization helpers.

The exact segment construction remains implemented by ``HKGFusionDataset``;
this module provides a stable import location without duplicating that logic.
"""

from .dataset import HKGFusionDataset


def encode_record(dataset: HKGFusionDataset, question: str, context: str, answer: str):
    return dataset._encode_example(question, context, answer)
