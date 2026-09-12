"""Label-free, answer-aware sentence selection for long contexts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Mapping

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9%]+(?:\.[0-9]+)?")
_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,}|[0-9]+(?:\.[0-9]+)?%?|p\s*[<>=]\s*0?\.[0-9]+)\b")

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "was",
    "were",
    "are",
    "of",
    "in",
    "on",
    "to",
    "and",
    "or",
    "for",
    "with",
    "as",
    "by",
    "at",
    "this",
    "that",
    "it",
    "be",
    "been",
    "has",
    "have",
    "had",
    "not",
    "no",
}


@dataclass
class ContextSelectionConfig:
    enabled: bool = True
    max_context_tokens: int = 280
    max_sentences: int = 8
    # Sentence score:
    #   score(s_j) = 0.45*lexical_similarity(question, s_j)
    #              + 0.45*lexical_similarity(answer, s_j)
    #              + 0.10*entity_overlap(question+answer, s_j)
    # Lexical similarity is Jaccard overlap over lowercased non-stopwords.
    question_weight: float = 0.45
    answer_weight: float = 0.45
    entity_weight: float = 0.10
    # Always keep at least this many leading sentences even if their score is
    # low -- context often opens with framing/background that later sentences
    # depend on for coherence, and this avoids selecting a disjoint bag of
    # high-scoring fragments.
    min_leading_sentences: int = 1


def context_selection_from_config(config: Mapping) -> ContextSelectionConfig:
    """Build a typed selector config from a project configuration mapping."""

    value = config.get("data", {}).get("context_selection", {}) or {}
    return ContextSelectionConfig(
        enabled=bool(value.get("enabled", True)),
        max_context_tokens=int(value.get("max_context_tokens", 280)),
        max_sentences=int(value.get("max_sentences", 8)),
        question_weight=float(value.get("question_weight", 0.45)),
        answer_weight=float(value.get("answer_weight", 0.45)),
        entity_weight=float(value.get("entity_weight", 0.10)),
        min_leading_sentences=int(value.get("min_leading_sentences", 1)),
    )


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    return parts if parts else [text]


def _tokenset(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def _entityset(text: str) -> set:
    return {m.group(0) for m in _ENTITY_RE.finditer(text)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def select_context_sentences(
    question: str,
    context: str,
    answer: str,
    cfg: ContextSelectionConfig | None = None,
    tokenizer=None,
) -> str:
    """Answer-aware, unsupervised sentence-level context selection.

    Scores each context sentence by lexical overlap (Jaccard on lowercased
    non-stopword tokens) with the question and answer, plus a bonus for
    "entity-like" overlap (capitalized terms, numbers, percentages,
    p-values).
    No labels are used anywhere in this scoring. Selected sentences are
    re-ordered to their ORIGINAL position in the context (never shuffled),
    and selection stops once either `max_sentences` or an approximate
    `max_context_tokens` budget (whitespace-token count, or tokenizer count
    if a tokenizer is supplied) is reached. If nothing scores above zero
    (e.g. a context totally unrelated to the Q/A -- which itself may be a
    signal of hallucination and should NOT be silently discarded), we fall
    back to keeping the original context so an unrelated passage is not
    silently discarded.
    """
    cfg = cfg or ContextSelectionConfig()
    if not cfg.enabled or not context.strip():
        return context

    sentences = _split_sentences(context)
    if len(sentences) <= cfg.min_leading_sentences:
        return context

    q_tokens = _tokenset(question)
    a_tokens = _tokenset(answer)
    q_entities = _entityset(question)
    a_entities = _entityset(answer)

    scored = []
    for idx, sent in enumerate(sentences):
        s_tokens = _tokenset(sent)
        s_entities = _entityset(sent)
        score = (
            cfg.question_weight * _jaccard(s_tokens, q_tokens)
            + cfg.answer_weight * _jaccard(s_tokens, a_tokens)
            + cfg.entity_weight * _jaccard(s_entities, q_entities | a_entities)
        )
        scored.append((idx, sent, score))

    if all(s[2] == 0.0 for s in scored):
        # No lexical signal at all -- keep the full context rather than
        # guessing; an unrelated context is informative, not noise.
        return context

    ranked = sorted(scored, key=lambda x: x[2], reverse=True)

    selected_idx = set(range(min(cfg.min_leading_sentences, len(sentences))))
    token_budget = cfg.max_context_tokens
    used_tokens = sum(_approx_len(sentences[i], tokenizer) for i in selected_idx)

    for idx, sent, _score in ranked:
        if len(selected_idx) >= cfg.max_sentences:
            break
        if idx in selected_idx:
            continue
        sent_len = _approx_len(sent, tokenizer)
        if used_tokens + sent_len > token_budget and selected_idx:
            continue
        selected_idx.add(idx)
        used_tokens += sent_len

    ordered = [sentences[i] for i in sorted(selected_idx)]
    return " ".join(ordered)


def _approx_len(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return len(text.split())
