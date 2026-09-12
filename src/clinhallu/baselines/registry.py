"""Registry of supported publication baseline implementations."""

from .base import BaselineSpec

REGISTRY = {
    "tfidf_lr": BaselineSpec(
        "tfidf_lr",
        "clinhallu.baselines.tfidf_lr",
        ("max_features", "seed", "output_dir"),
    ),
    "semantic_similarity": BaselineSpec(
        "semantic_similarity",
        "clinhallu.baselines.semantic_similarity",
        (
            "model_name",
            "revision",
            "batch_size",
            "top_k",
            "max_sentences",
            "context_window",
            "threshold_objective",
            "cache_dir",
            "output_dir",
        ),
    ),
    "zero_shot_nli": BaselineSpec(
        "zero_shot_nli",
        "clinhallu.baselines.zero_shot_nli",
        (
            "model_name",
            "revision",
            "batch_size",
            "max_context_sentences",
            "context_window",
            "contradiction_alpha",
            "threshold_objective",
            "cache_dir",
            "output_dir",
        ),
    ),
    "selfcheckgpt": BaselineSpec(
        "selfcheckgpt",
        "clinhallu.baselines.selfcheckgpt",
        (
            "model_name",
            "revision",
            "batch_size",
            "max_length",
            "answers_dir",
            "support_threshold",
            "cache_dir",
            "output_dir",
            "threshold_mode",
            "decision_threshold",
        ),
    ),
    "self_consistency": BaselineSpec(
        "self_consistency",
        "clinhallu.baselines.self_consistency",
        (
            "model_name",
            "revision",
            "batch_size",
            "max_length",
            "answers_dir",
            "support_threshold",
            "cache_dir",
            "output_dir",
            "threshold_mode",
            "decision_threshold",
        ),
    ),
    "llm_judge": BaselineSpec(
        "llm_judge",
        "clinhallu.baselines.llm_judge",
        (
            "model_name",
            "revision",
            "provider",
            "responses_path",
            "api_config",
            "max_new_tokens",
            "max_input_tokens",
            "output_dir",
        ),
    ),
}
