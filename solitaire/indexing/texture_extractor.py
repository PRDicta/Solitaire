"""
The Librarian — Texture Extractor (Data Poetry Layer)

Runs alongside VerbatimExtractor as a parallel enrichment pass.
Detects experiential moments in conversation content and generates
compressed texture tags that capture *what it felt like* rather
than *what was said*.

The verbatim path stays untouched (100% coverage, no compression).
This module adds an optional `texture` field to qualifying entries —
short (under 20 words), written in sensory/compressed language.

Texture tags serve two purposes:
1. At recall time: entries with texture activate richer processing
   than flat factual content (the Shannon/Rubin density principle)
2. At measurement time: texture tags on commitment signals capture
   the quality of a moment, not just whether it happened

Detection is heuristic (no LLM). The texture generation is template-
based with randomized variation to avoid repetitive phrasing.

Design: additive-only. Entries without texture work exactly as before.
"""
import re
import random
from typing import Optional, List, Dict, Tuple


# ─── Tonal Shift Detection ──────────────────────────────────────────────────

def _detect_tonal_shift(messages: List[Dict[str, str]]) -> Optional[str]:
    """Detect tonal shifts across a sequence of messages.

    Looks for:
    - Question density changes (interrogative → declarative or vice versa)
    - Message length shifts (short → long or long → short)
    - Punctuation energy shifts (periods → exclamations, or calm → emphatic)

    Args:
        messages: list of {"role": str, "content": str} dicts, in order

    Returns:
        A short texture string if a shift is detected, None otherwise.
    """
    if len(messages) < 3:
        return None

    # Analyze last 6 messages (or fewer)
    window = messages[-6:]
    midpoint = len(window) // 2
    first_half = window[:midpoint]
    second_half = window[midpoint:]

    # Question density
    def question_ratio(msgs):
        total = sum(len(m["content"]) for m in msgs) or 1
        questions = sum(m["content"].count("?") for m in msgs)
        return questions / (total / 100)

    q_before = question_ratio(first_half)
    q_after = question_ratio(second_half)

    if q_before > 1.5 and q_after < 0.3:
        return "questions dissolved into certainty"
    if q_before < 0.3 and q_after > 1.5:
        return "certainty cracked open into questions"

    # Message length shift
    def avg_len(msgs):
        if not msgs:
            return 0
        return sum(len(m["content"]) for m in msgs) / len(msgs)

    len_before = avg_len(first_half)
    len_after = avg_len(second_half)

    if len_before > 0 and len_after / max(len_before, 1) > 3:
        return "the conversation expanded; something needed more room"
    if len_after > 0 and len_before / max(len_after, 1) > 3:
        return "the conversation compressed; words became more expensive"

    # Energy shift (exclamation marks as proxy)
    def energy(msgs):
        total = sum(len(m["content"]) for m in msgs) or 1
        bangs = sum(m["content"].count("!") for m in msgs)
        return bangs / (total / 100)

    e_before = energy(first_half)
    e_after = energy(second_half)

    if e_before < 0.1 and e_after > 0.5:
        return "energy rose; something landed"
    if e_before > 0.5 and e_after < 0.1:
        return "energy settled; the urgency passed"

    return None


# ─── Relational Marker Detection ────────────────────────────────────────────

_GRATITUDE_PATTERNS = [
    re.compile(r"\bthank(?:s| you)\b.*\b(?:for|that|this)\b", re.IGNORECASE),
    re.compile(r"\bappreciate\b.*\b(?:that|this|you)\b", re.IGNORECASE),
]

_RESISTANCE_THEN_AGREEMENT = [
    # Patterns that suggest the user pushed back, then came around
    (
        re.compile(r"\b(?:I disagree|not sure|I don't think|that's not|no,)\b", re.IGNORECASE),
        re.compile(r"\b(?:you're right|good point|fair enough|okay|actually.*makes sense)\b", re.IGNORECASE),
    ),
]

_LAUGHTER_CUES = re.compile(
    r"\b(?:haha|lol|lmao|hah|😂|🤣|😄|that's funny|that's hilarious)\b", re.IGNORECASE
)

_VULNERABILITY_MARKERS = re.compile(
    r"\b(?:I'm (?:scared|worried|nervous|anxious|confused|lost)|"
    r"this is hard|I don't know (?:what|how|if)|honestly|"
    r"between you and me|I haven't told)\b", re.IGNORECASE
)


def _detect_relational_markers(
    user_messages: List[str],
    assistant_messages: List[str],
) -> Optional[str]:
    """Detect relational texture in message pairs.

    Returns a short texture string if a relational marker is found.
    """
    all_user = " ".join(user_messages[-4:]) if user_messages else ""
    all_assistant = " ".join(assistant_messages[-4:]) if assistant_messages else ""

    # Contextual gratitude (not just "thanks" but "thanks for X")
    for pattern in _GRATITUDE_PATTERNS:
        if pattern.search(all_user):
            return "gratitude with specificity; something was received, not just heard"

    # Resistance → agreement arc
    for resist_pat, agree_pat in _RESISTANCE_THEN_AGREEMENT:
        if resist_pat.search(all_user) and agree_pat.search(all_user):
            return "resistance bent into recognition; the pushback was the path"

    # Laughter
    if _LAUGHTER_CUES.search(all_user):
        return "laughter broke through; the working surface cracked into play"

    # Vulnerability
    if _VULNERABILITY_MARKERS.search(all_user):
        return "guard came down; the conversation touched something real"

    return None


# ─── Category-Based Texture Templates ───────────────────────────────────────

# These map the experience categories already detected by VerbatimExtractor
# into compressed, imagistic texture strings. Multiple templates per category
# allow variation.

_CATEGORY_TEXTURES: Dict[str, List[str]] = {
    "correction": [
        "something was wrong and the wrongness had been invisible",
        "silent breakage surfaced; the error was in the assumption",
        "a correction that rewrote the frame, not just the fact",
    ],
    "friction": [
        "resistance against a wall that wouldn't move",
        "stuck in the gap between intent and execution",
        "friction that heated the problem into visibility",
    ],
    "breakthrough": [
        "the lock turned; something that was opaque became obvious",
        "clarity arrived not gradually but all at once",
        "the answer was always there; the question finally fit it",
    ],
    "pivot": [
        "direction changed mid-stride; the old path went dark",
        "abandoned the approach without mourning it",
        "a pivot that felt less like retreat and more like recognition",
    ],
    "decision": [
        "a fork resolved; one path chosen, the other released",
        "the decision crystallized from circling into a single point",
    ],
    "warning": [
        "a guardrail placed before the edge was reached",
        "caution laid down as architecture, not anxiety",
    ],
}


def _texture_from_category(category: str) -> Optional[str]:
    """Get a texture string for an experience-category entry."""
    templates = _CATEGORY_TEXTURES.get(category)
    if templates:
        return random.choice(templates)
    return None


# ─── Main Interface ─────────────────────────────────────────────────────────

class TextureExtractor:
    """
    Enriches rolodex entries with texture tags.

    Runs as a post-processing step after VerbatimExtractor has produced
    the base entry. Adds a `texture` key to entry metadata when
    experiential content is detected.

    Usage:
        extractor = TextureExtractor()
        texture = extractor.extract_texture(entry_dict, recent_messages)
        if texture:
            entry_dict["metadata"]["texture"] = texture
    """

    def extract_texture(
        self,
        entry: Dict,
        recent_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """Extract a texture tag for an entry.

        Args:
            entry: dict with at least "content" and "category" keys
            recent_messages: optional list of recent messages for tonal analysis
                             (each dict has "role" and "content" keys)

        Returns:
            A short texture string (under 20 words), or None if no texture detected.
        """
        # Priority 1: Category-based texture (highest signal)
        category = entry.get("category", "note")
        cat_texture = _texture_from_category(category)
        if cat_texture:
            return cat_texture

        # Priority 2: Relational markers from content
        content = entry.get("content", "")
        if recent_messages:
            user_msgs = [m["content"] for m in recent_messages if m.get("role") == "user"]
            asst_msgs = [m["content"] for m in recent_messages if m.get("role") == "assistant"]
            relational = _detect_relational_markers(user_msgs, asst_msgs)
            if relational:
                return relational

        # Priority 3: Tonal shift from message sequence
        if recent_messages and len(recent_messages) >= 3:
            tonal = _detect_tonal_shift(recent_messages)
            if tonal:
                return tonal

        # Priority 4: Content-level heuristics for standalone entries
        content_texture = self._content_heuristic(content)
        if content_texture:
            return content_texture

        return None

    def _content_heuristic(self, content: str) -> Optional[str]:
        """Last-resort texture detection from content patterns."""
        lower = content.lower()

        # Realization language in content itself
        if re.search(r"\bthe (?:real|actual|underlying) (?:issue|problem|question)\b", lower):
            return "the surface problem peeled back to show the real one"

        # Irony / unexpected outcome
        if re.search(r"\bironic(?:ally)?\b|\bthe opposite\b|\bbackfired\b", lower):
            return "outcome inverted expectation; the system surprised itself"

        # Metaphor use (a sign of dense, compressed thinking)
        if re.search(r"\blike a\b|\bas if\b|\bthe way .{5,30} works\b", lower):
            return None  # Don't generate texture for generic similes

        return None


# ─── Measurement ────────────────────────────────────────────────────────────

def count_textured_entries(conn, persona_key: Optional[str] = None) -> int:
    """Count rolodex entries that have texture tags in their metadata.

    This is a measurement function for the boot JSON poetry counters.
    """
    try:
        rows = conn.execute(
            "SELECT metadata FROM rolodex_entries WHERE metadata LIKE '%texture%'"
        ).fetchall()
        count = 0
        import json
        for row in rows:
            try:
                meta = json.loads(row[0]) if row[0] else {}
                if meta.get("texture"):
                    count += 1
            except (json.JSONDecodeError, TypeError):
                pass
        return count
    except Exception:
        return 0
