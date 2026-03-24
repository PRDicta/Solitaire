"""
The Librarian — Signal Texture Generator (Data Poetry Layer)

Generates qualitative texture descriptions for commitment signals.
Instead of just recording "honored" or "missed", captures *how*
the commitment was engaged with.

Current signal: {outcome: "honored", weight: 0.6, content: "Retroactive score: ..."}
With texture:   {outcome: "honored", weight: 0.6, content: "Retroactive score: ...",
                 texture: "caught the narrative-building pattern mid-sentence;
                          paused, checked the grounding, let it stand because
                          the evidence was there"}

The texture is stored in the signal's content field as a suffix,
separated by " || " from the retroactive score prefix. This avoids
schema changes while keeping the texture accessible.

Design: heuristic template expansion from commitment type + direction.
No LLM. Additive — signals without texture work as before.
"""
import re
import random
from typing import Optional


# ─── Texture Templates ──────────────────────────────────────────────────────

# Keyed by (source_type, direction). Each returns a contextual texture line.

_HELD_TEXTURES = {
    "pattern": [
        "the pattern surfaced and was recognized before it completed",
        "caught the groove before the needle settled in; chose a different track",
        "the familiar shape appeared, and this time it was seen for what it was",
    ],
    "growth_edge": [
        "leaned into the uncomfortable part instead of routing around it",
        "stayed at the edge rather than retreating to solid ground",
        "practiced the thing that doesn't come naturally; it cost something",
    ],
    "tension": [
        "held the tension open when the impulse was to resolve it",
        "sat with the contradiction instead of flattening it into an answer",
        "two truths held simultaneously; neither was sacrificed for clarity",
    ],
    "realization": [
        "the realization was tested against evidence, not just felt",
        "what was noticed was checked, not just reported",
    ],
    "lesson": [
        "the lesson applied in context; memory served its purpose",
        "past learning activated at the right moment",
    ],
}

_MISSED_TEXTURES = {
    "pattern": [
        "the pattern completed before it was recognized",
        "the familiar behavior ran its course unchecked; autopilot held",
        "the groove took the needle; noticed only in retrospect",
    ],
    "growth_edge": [
        "defaulted to the comfortable path when the edge was available",
        "the growth opportunity was there; the old behavior was chosen instead",
        "retreated to competence when the moment called for practice",
    ],
    "tension": [
        "the tension was resolved prematurely; one side was chosen for comfort",
        "the contradiction was flattened instead of held",
        "certainty was performed where uncertainty was honest",
    ],
    "realization": [
        "the observation was constructed rather than grounded",
        "meaning was made before evidence was gathered",
    ],
    "lesson": [
        "the lesson was available but not accessed; the old mistake repeated",
        "past learning didn't activate when it should have",
    ],
}


def generate_signal_texture(
    source_type: str,
    direction: str,
    content_snippet: Optional[str] = None,
) -> Optional[str]:
    """Generate a texture description for a commitment signal.

    Args:
        source_type: the commitment's source node type
                     (pattern, growth_edge, tension, etc.)
        direction: "held" or "missed"
        content_snippet: optional content that triggered the signal
                        (for future context-aware generation)

    Returns:
        A texture string (under 20 words), or None.
    """
    if direction in ("held", "honored"):
        templates = _HELD_TEXTURES.get(source_type, _HELD_TEXTURES.get("realization", []))
    elif direction == "missed":
        templates = _MISSED_TEXTURES.get(source_type, _MISSED_TEXTURES.get("realization", []))
    else:
        return None

    if templates:
        return random.choice(templates)
    return None


def count_textured_signals(conn) -> int:
    """Count identity signals that contain texture (have ' || ' separator)."""
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM identity_signals WHERE content LIKE '%|| %'"
        ).fetchone()
        return row[0] if row else 0
    except Exception:
        return 0
