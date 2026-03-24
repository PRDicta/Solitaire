"""
The Librarian — Recall Confidence Assessment

Evaluates the quality of a recall result set to decide whether the current
tier's results are sufficient or whether the search should widen to the next tier.

Four signals:
1. top_score: Absolute strength of the best candidate after reranking
2. score_gap: Distance between #1 and #4 — clear winners vs uniform mediocrity
3. entity_hit_rate: Did any result actually contain the extracted proper nouns?
4. answer_signal: For question-type queries, do results contain the answer pattern?

Used by the tiered recall waterfall in cmd_auto_recall. Not to be confused with
src/core/confidence.py which handles entry-level provenance scoring.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from .reranker import ScoredCandidate
from .entity_extractor import ExtractedEntities


@dataclass
class RecallConfidence:
    """Confidence assessment of a recall tier's result set."""
    top_score: float            # Best candidate composite score after reranking
    score_gap: float            # Gap between #1 and #4 (0 if < 4 results)
    entity_hit_rate: float      # Fraction of query entities found in results (0-1)
    answer_signal: float        # 0-1: how likely the results contain the actual answer
    confident: bool             # Above threshold?
    tier: int                   # Which tier produced this assessment


def assess_confidence(
    scored_results: List[ScoredCandidate],
    query_entities: Optional[ExtractedEntities] = None,
    threshold: float = 0.45,
    tier: int = 1,
    original_message: str = "",
) -> RecallConfidence:
    """
    Assess whether a set of scored recall results is confident enough to return.

    Args:
        scored_results: Reranked candidates from the current tier
        query_entities: Extracted entities from the original query (for hit rate)
        threshold: Composite confidence threshold (0-1). Below this, widen search.
        tier: Which tier produced these results (for tracking)
        original_message: The user's original message (for question pattern detection)

    Returns:
        RecallConfidence with the assessment
    """
    if not scored_results:
        return RecallConfidence(
            top_score=0.0, score_gap=0.0, entity_hit_rate=0.0,
            answer_signal=0.0, confident=False, tier=tier,
        )

    top_score = scored_results[0].composite_score

    # Score gap: distance between #1 and #4.
    if len(scored_results) >= 4:
        score_gap = scored_results[0].composite_score - scored_results[3].composite_score
    else:
        score_gap = scored_results[0].composite_score - scored_results[-1].composite_score

    entity_hit_rate = _compute_entity_hit_rate(scored_results, query_entities)
    answer_signal = _compute_answer_signal(scored_results, original_message)

    # Composite confidence: top_score modulated by answer quality signals.
    composite = top_score

    # If entity_hit_rate is 0 and we had entities to look for, penalize.
    has_entities = (
        query_entities is not None
        and query_entities.proper_nouns
        and len(query_entities.proper_nouns) > 0
    )
    if has_entities and entity_hit_rate == 0.0:
        composite *= 0.5
    elif has_entities and entity_hit_rate < 0.5:
        composite *= 0.75

    # Answer signal penalty: for question queries, low answer_signal means
    # the results probably don't contain the actual answer.
    # This is the most important gate: a top score of 1.0 is meaningless
    # if the results don't contain the attribute being asked about.
    if _is_question_query(original_message):
        if answer_signal == 0.0:
            # Zero attribute matches. Force widening regardless of score.
            composite *= 0.3
        elif answer_signal < 0.3:
            composite *= 0.5

    # Score gap bonus: clear differentiation in the pool
    if score_gap > 0.15:
        composite += 0.05

    confident = composite >= threshold

    return RecallConfidence(
        top_score=top_score, score_gap=score_gap,
        entity_hit_rate=entity_hit_rate, answer_signal=answer_signal,
        confident=confident, tier=tier,
    )


def _is_question_query(message: str) -> bool:
    """Detect if the message is asking a specific factual question."""
    msg = message.lower().strip()
    # Direct question patterns
    if msg.endswith("?"):
        return True
    question_starts = [
        "what is", "what's", "what was", "what are",
        "who is", "who's", "who was", "who are",
        "where is", "where does", "where did",
        "when did", "when was", "when is",
        "how old", "how many", "how much",
        "do you know", "do you remember", "can you tell me",
    ]
    return any(msg.startswith(q) for q in question_starts)


# Patterns that extract the attribute being asked about.
# "What is the user's last name?" -> attribute = "last name"
# "What is the user's email?" -> attribute = "email"
_ATTRIBUTE_PATTERNS = [
    re.compile(r"what(?:'s| is| was| are) (\w+)(?:'s|'s) (.+?)(?:\?|$)", re.I),
    re.compile(r"(?:do you (?:know|remember)) (\w+)(?:'s|'s) (.+?)(?:\?|$)", re.I),
    re.compile(r"what (?:is|was) the (.+?) (?:of|for) (\w+)", re.I),
]


def _compute_answer_signal(
    scored_results: List[ScoredCandidate],
    original_message: str,
) -> float:
    """
    For question-type queries, check if the results contain content that
    looks like it answers the question. Returns 0-1.

    The key insight: entity_hit_rate checks whether the owner's name appears in results,
    but when asking "What is the user's last name?", the owner name will appear in
    hundreds of entries. What matters is whether "last name" (the attribute)
    appears in any result.
    """
    if not original_message or not _is_question_query(original_message):
        return 1.0  # Not a question, signal is vacuously satisfied

    # Extract the attribute being asked about
    attributes = []
    for pattern in _ATTRIBUTE_PATTERNS:
        match = pattern.search(original_message)
        if match:
            # The attribute is usually the last captured group
            attr = match.group(match.lastindex).strip().lower()
            attributes.append(attr)

    # Also extract key non-entity nouns from the question
    # "What is the user's last name?" -> "last name"
    # Strip the entity names and question words to get the attribute
    msg_lower = original_message.lower()
    _q_words = {'what', 'who', 'where', 'when', 'how', 'is', 'was', 'are',
                'were', 'the', 'a', 'an', 'do', 'does', 'did', 'you', 'know',
                'remember', 'can', 'tell', 'me', 'my', 'his', 'her', 'their',
                'our', 'your', "'s", "'s", 'with', 'about', 'for', 'from',
                'that', 'this', 'have', 'has', 'had', 'been', 'being',
                'will', 'would', 'could', 'should', 'and', 'but', 'not',
                'working', 'doing', 'going', 'using', 'getting'}
    # Remove entity names from consideration
    entity_names = set()
    if scored_results and scored_results[0].entry:
        # We don't have entity list here, so extract from message directly
        pass

    # Extract attribute words: non-question, non-entity words from the message.
    # For "What is the user's last name?", entities are ["{owner}"] and
    # attribute words are ["last", "name"].
    words = re.findall(r'\b\w+\b', msg_lower)

    # Build entity set from pattern matches (group 1 is usually the entity)
    _entity_words = set()
    for pattern in _ATTRIBUTE_PATTERNS:
        match = pattern.search(original_message)
        if match and match.lastindex and match.lastindex >= 2:
            # First group is typically the entity name
            for w in match.group(1).lower().split():
                _entity_words.add(w)

    non_q_words = [
        w for w in words
        if w not in _q_words and w not in _entity_words and len(w) > 2
    ]
    if non_q_words:
        # The attribute phrase (e.g. "last name")
        attr_phrase = " ".join(non_q_words)
        attributes.append(attr_phrase)
        # Also add individual words for partial matching
        for w in non_q_words:
            if len(w) > 3:  # Skip very short words
                attributes.append(w)

    if not attributes:
        return 1.0  # Can't determine what's being asked, assume OK

    # Deduplicate attributes
    attributes = list(dict.fromkeys(attributes))

    # Check if any attribute appears in any result
    combined = " ".join(
        (sc.entry.content or "").lower() for sc in scored_results
    )

    hits = sum(1 for attr in attributes if attr in combined)
    return min(1.0, hits / max(1, len(attributes)))


def _compute_entity_hit_rate(
    scored_results: List[ScoredCandidate],
    query_entities: Optional[ExtractedEntities],
) -> float:
    """What fraction of query entities appear in any result content?"""
    if not query_entities or not query_entities.proper_nouns:
        return 1.0

    entities_to_find = [e.lower() for e in query_entities.proper_nouns]
    if not entities_to_find:
        return 1.0

    combined = " ".join(
        (sc.entry.content or "").lower() for sc in scored_results
    )

    hits = sum(1 for e in entities_to_find if e in combined)
    return hits / len(entities_to_find)
