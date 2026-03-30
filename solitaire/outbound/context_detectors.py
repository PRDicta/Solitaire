"""Layer 5: Context coherence detectors.

Detect context coherence failures: topics the user raised that the
response ignores, and fabricated back-references ("as I mentioned"
when nothing was mentioned).

These detectors need transcript context — user message and prior
assistant turns.

No LLM calls. Keyword extraction and string matching.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass
class ContextDetection:
    """A single context coherence detection."""
    category: str           # e.g., "thread_dropping"
    severity: str           # "info" | "warning"
    count: int              # occurrences
    samples: List[str]      # up to 3 samples
    detail: str             # human-readable explanation
    score: Optional[float]  # metric where applicable
    confidence: str = "high"  # "high" | "low" — low routes to model verification


# ═══════════════════════════════════════════════════════════════════════════
# PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

# Back-reference claims that need verification
RE_BACK_REFERENCE = [
    re.compile(r"\bas I (?:mentioned|noted|said|described|explained|discussed)\b", re.IGNORECASE),
    re.compile(r"\bas (?:we|I) (?:discussed|talked about|covered|went over)\b", re.IGNORECASE),
    re.compile(r"\b(?:earlier|previously|before),? (?:I|we) (?:mentioned|noted|discussed)\b", re.IGNORECASE),
    re.compile(r"\byou (?:mentioned|noted|said|asked about) (?:earlier|previously|before)\b", re.IGNORECASE),
    re.compile(r"\bgoing back to (?:what|my|our|the)\b", re.IGNORECASE),
    re.compile(r"\bto (?:reiterate|recap) (?:what|my)\b", re.IGNORECASE),
]

# Stop words to exclude from topic extraction
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "need",
    "i", "you", "we", "they", "he", "she", "it", "my", "your", "our",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "how", "when", "where", "why",
    "and", "or", "but", "if", "then", "else", "so", "yet", "not", "no",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under",
    "also", "just", "than", "too", "very", "really", "quite",
    "all", "each", "every", "both", "few", "more", "most", "some", "any",
    "there", "here", "now", "then", "still", "already",
    "think", "know", "want", "like", "get", "make", "see", "look",
    "going", "come", "take", "give", "tell", "say", "said",
    "let", "sure", "right", "well", "yeah", "yes", "ok", "okay",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't",
    "shouldn't", "can't", "isn't", "aren't", "wasn't", "weren't",
    "it's", "that's", "there's", "here's", "what's", "let's",
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _extract_topics(text: str, min_word_len: int = 4) -> Set[str]:
    """Extract substantive topic words from text.

    Returns lowercased words that are at least min_word_len chars,
    not in the stop list, and not purely numeric.
    """
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", text.lower())
    return {
        w for w in words
        if len(w) >= min_word_len
        and w not in STOP_WORDS
        and not w.isdigit()
    }


def _extract_named_entities(text: str) -> Set[str]:
    """Extract likely named entities (capitalized multi-word sequences).

    Simple heuristic: sequences of 2+ capitalized words, or single
    capitalized words that aren't sentence starters.
    """
    # Multi-word capitalized sequences
    entities = set()
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        entities.add(m.group(0).lower())
    return entities


def _snippet_around(text: str, match: re.Match, max_len: int = 60) -> str:
    """Extract a snippet around a regex match."""
    start = max(0, match.start() - 10)
    end = min(len(text), match.end() + 10)
    s = text[start:end].strip()
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s


# ═══════════════════════════════════════════════════════════════════════════
# DETECTORS
# ═══════════════════════════════════════════════════════════════════════════

def detect_thread_dropping(
    text: str,
    user_text: str,
    min_user_topics: int = 3,
    min_dropped: int = 2,
) -> Optional[ContextDetection]:
    """Detect user topics that the response ignores.

    Extracts topic words and named entities from the user message,
    then checks how many appear in the response. Only fires when:
    - User message has enough distinct topics (min_user_topics)
    - Enough topics are completely absent from response (min_dropped)

    This avoids false positives on short user messages or responses
    that address the topics using different vocabulary.
    """
    if not user_text:
        return None

    user_topics = _extract_topics(user_text)
    user_entities = _extract_named_entities(user_text)

    # Combine topics and entities
    all_user_topics = user_topics | user_entities
    if len(all_user_topics) < min_user_topics:
        return None

    response_text_lower = text.lower()
    dropped = []
    for topic in all_user_topics:
        if topic not in response_text_lower:
            dropped.append(topic)

    if len(dropped) < min_dropped:
        return None

    # Calculate drop ratio
    drop_ratio = len(dropped) / len(all_user_topics)
    if drop_ratio < 0.5:
        # Less than half dropped isn't a strong signal
        return None

    severity = "warning" if drop_ratio > 0.7 else "info"
    # Keyword-based topic matching is inherently fuzzy — a response can
    # address a topic using different vocabulary. High confidence only
    # when the drop is overwhelming (>70%).
    confidence = "high" if drop_ratio > 0.7 else "low"
    samples = sorted(dropped)[:3]

    return ContextDetection(
        category="thread_dropping",
        severity=severity,
        count=len(dropped),
        samples=samples,
        detail=f"{len(dropped)}/{len(all_user_topics)} user topics not addressed "
               f"in response ({drop_ratio:.0%} drop rate). "
               f"Dropped: {', '.join(samples)}.",
        score=drop_ratio,
        confidence=confidence,
    )


def detect_inaccurate_reference(
    text: str,
    prior_turns: str = "",
) -> Optional[ContextDetection]:
    """Detect fabricated back-references.

    Catches "as I mentioned" / "we discussed" claims that don't match
    anything in prior turns. Only fires when prior_turns is available.
    """
    if not prior_turns:
        return None

    matches = []
    for pattern in RE_BACK_REFERENCE:
        for m in pattern.finditer(text):
            matches.append(m)

    if not matches:
        return None

    # For each back-reference, check if the surrounding content appears
    # in prior turns. Extract the topic after the back-reference phrase.
    fabricated = []
    prior_lower = prior_turns.lower()

    for m in matches:
        # Get text after the back-reference (next 5-15 words)
        after_start = m.end()
        after_text = text[after_start:after_start + 150].strip()
        after_words = re.findall(r"\b[a-zA-Z]{4,}\b", after_text.lower())[:5]

        if not after_words:
            continue

        # Check if any of these topic words appear in prior turns
        found_in_prior = any(w in prior_lower for w in after_words)
        if not found_in_prior:
            fabricated.append(m)

    if not fabricated:
        return None

    samples = [_snippet_around(text, m) for m in fabricated[:3]]
    severity = "warning" if len(fabricated) >= 2 else "info"
    # Single fabricated reference could be vocabulary mismatch in prior
    # turn matching. Multiple is more definitive.
    confidence = "high" if len(fabricated) >= 2 else "low"

    return ContextDetection(
        category="inaccurate_reference",
        severity=severity,
        count=len(fabricated),
        samples=samples,
        detail=f"{len(fabricated)} back-reference(s) not supported by prior turns. "
               f"Don't claim to have mentioned something you haven't.",
        score=None,
        confidence=confidence,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_context_scan(
    text: str,
    user_text: str = "",
    prior_turns: str = "",
) -> List[ContextDetection]:
    """Run all context coherence detectors. Returns list of detections.

    Args:
        text: The assistant's response (preprocessed).
        user_text: The user's message that prompted this response.
        prior_turns: Concatenated prior assistant turns for reference checking.
    """
    results = []

    d = detect_thread_dropping(text, user_text)
    if d is not None:
        results.append(d)

    d = detect_inaccurate_reference(text, prior_turns)
    if d is not None:
        results.append(d)

    return results
