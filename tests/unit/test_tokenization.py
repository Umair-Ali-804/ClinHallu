from clinhallu.data.dataset import HKGFusionDataset


class FakeTokenizer:
    cls_token_id = 1
    sep_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return list(range(10, 10 + len(text.split())))


def test_exact_segment_budgets_preserve_answer_ends():
    dataset = object.__new__(HKGFusionDataset)
    dataset.tokenizer = FakeTokenizer()
    dataset.max_length = 32
    dataset.question_budget = 4
    dataset.answer_budget = 8
    encoded = dataset._encode_example("q " * 20, "c " * 50, "a " * 20)
    assert len(encoded["input_ids"]) == 32
    assert sum(encoded["question_mask"]) == 4
    assert sum(encoded["answer_mask"]) == 8
