
from .dataset import HKGFusionDataset


def encode_record(dataset: HKGFusionDataset, question: str, context: str, answer: str):
    return dataset._encode_example(question, context, answer)
