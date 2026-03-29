"""
Solitaire -- Conflict Detection Utilities

Shared heuristic conflict detection used by both:
- Reranker (post-filter on retrieval results)
- IngestionContradictionDetector (at ingestion time)

Three detection modes, all regex-based, no LLM calls:
1. Numeric: same context words, different percentage/number values (>5% delta)
2. Preference/status: different state assertions for the same subject
3. Negation: one text negates what the other asserts

Plus a temporal supersession detector for ingestion-time use.
"""
import re
from typing import Set, Optional


# -- Stop words for entity extraction -----------------------------------

_ENTITY_STOP = {
    'the', 'this', 'that', 'with', 'from', 'have', 'been',
    'being', 'about', 'into', 'through', 'during', 'also',
    'just', 'very', 'really', 'than', 'then', 'some', 'only',
    'which', 'these', 'those', 'other', 'more', 'most', 'such',
    'each', 'every', 'both', 'between', 'after', 'before',
    'would', 'could', 'should', 'will', 'does', 'what',
    'were', 'where', 'when', 'while', 'there', 'their',
    'they', 'them', 'your', 'here', 'still', 'already',
}


def extract_claim_entities(content: str, tags: Optional[list] = None) -> Set[str]:
    """
    Extract entity-like tokens for pairwise comparison.

    Returns lowercased significant words (4+ chars, non-stopword) that
    could anchor a factual claim.
    """
    text = content.lower()
    if tags:
        text += " " + " ".join(tags).lower()
    words = set(re.findall(r'\b[a-z]{4,}\b', text))
    return words - _ENTITY_STOP


def detect_claim_conflict(text_a: str, text_b: str) -> Optional[str]:
    """
    Heuristic check: do two texts contain conflicting factual claims?

    Returns the conflict type string if detected, None otherwise.
    Conflict types: 'numeric', 'preference', 'negation'.
    """
    ta = text_a.lower()
    tb = text_b.lower()

    if numeric_conflict(ta, tb):
        return "numeric"
    if preference_conflict(ta, tb):
        return "preference"
    if negation_conflict(ta, tb):
        return "negation"
    return None


def numeric_conflict(text_a: str, text_b: str) -> bool:
    """Detect conflicting numeric claims between two texts."""
    pct_pattern = re.compile(
        r'(\w+(?:\s+\w+){0,3})\s+'
        r'(?:~|about |approximately |around |at |to |revised to )?'
        r'(\d+(?:\.\d+)?)\s*[%％]'
    )

    claims_a = [(m.group(1).lower(), float(m.group(2))) for m in pct_pattern.finditer(text_a)]
    claims_b = [(m.group(1).lower(), float(m.group(2))) for m in pct_pattern.finditer(text_b)]

    if not claims_a or not claims_b:
        return False

    _stop = {"is", "at", "the", "a", "an", "to", "of", "in", "for", "on", "it", "was", "are"}

    for ctx_a, val_a in claims_a:
        words_a = set(ctx_a.split()) - _stop
        for ctx_b, val_b in claims_b:
            words_b = set(ctx_b.split()) - _stop
            if words_a & words_b and abs(val_a - val_b) > 5:
                return True
    return False


def preference_conflict(text_a: str, text_b: str) -> bool:
    """
    Detect preference or state contradictions.

    Patterns like "prefers X" vs "switched to Y" or "prefers Y",
    where X and Y are different values for the same subject.
    """
    pref_pattern = re.compile(
        r'(?:prefers?|switched?\s+to|changed?\s+to|moved?\s+to|uses?|wants?)\s+'
        r'([a-z]+(?:\s+[a-z]+){0,2})'
    )

    prefs_a = [m.group(1).strip() for m in pref_pattern.finditer(text_a)]
    prefs_b = [m.group(1).strip() for m in pref_pattern.finditer(text_b)]

    if not prefs_a or not prefs_b:
        return False

    for pa in prefs_a:
        for pb in prefs_b:
            if pa != pb:
                return True
    return False


def negation_conflict(text_a: str, text_b: str) -> bool:
    """
    Detect negation-based contradictions.

    Entry A: "client uses weekly reports"
    Entry B: "client no longer uses weekly reports"

    Also handles morphological variation via individual significant-word matching.
    """
    _negation_markers = [
        r'no longer\s+',
        r'not\s+',
        r'stopped\s+',
        r'discontinued\s+',
        r'removed\s+',
        r'dropped\s+',
    ]

    _trivial = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'shall',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'and', 'but', 'or', 'not', 'that', 'this', 'it', 'its',
        'using', 'used', 'uses', 'use',
    }

    def _check_direction(source: str, target: str) -> bool:
        for marker in _negation_markers:
            for m in re.finditer(marker + r'(\w+(?:\s+\w+){0,4})', source):
                negated_phrase = m.group(1).strip()

                # Full phrase match (strongest signal)
                if len(negated_phrase) >= 4 and negated_phrase in target:
                    return True

                # Individual significant word match (morphological variation)
                words = [w for w in negated_phrase.split()
                         if len(w) >= 4 and w not in _trivial]
                if words and all(w in target for w in words):
                    return True
        return False

    return _check_direction(text_b, text_a) or _check_direction(text_a, text_b)


# -- Temporal supersession (ingestion-time only) -------------------------

_TEMPORAL_MARKERS = re.compile(
    r'\b(?:now|currently|as of|switched to|changed to|no longer|stopped|'
    r'recently|just started|moving to|moved to)\b',
    re.IGNORECASE,
)


def has_temporal_markers(text: str) -> bool:
    """Check if text contains temporal markers suggesting supersession."""
    return bool(_TEMPORAL_MARKERS.search(text))
