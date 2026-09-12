from clinhallu.models.components.answer_pooling import AnswerAttentionPool


def test_answer_pooling_keeps_original_checkpoint_key_names():
    keys = set(AnswerAttentionPool(8, 4).state_dict())
    assert keys == {"proj.weight", "proj.bias", "score.weight"}
