from clinhallu.data.context_selection import ContextSelectionConfig, select_context_sentences


def test_context_selection_retains_relevant_sentence():
    context = "Unrelated material. Aspirin reduces platelet aggregation. Another sentence."
    cfg = ContextSelectionConfig(enabled=True, max_sentences=2, max_context_tokens=40)
    selected = select_context_sentences(
        "What does aspirin reduce?", context, "platelet aggregation", cfg
    )
    assert "Aspirin reduces platelet aggregation" in selected
