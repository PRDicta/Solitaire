"""
The Librarian — Memory Weight (System 3)

Weights entries by significance, confidence, emotional valence, and revisit cost.
Uses heuristic pattern matching to score entries without ML.

All scoring is deterministic and pattern-based, making it explainable and tunable.
"""

import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime, timedelta


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class MemoryWeight:
    """
    Quantifies how significant, certain, and emotionally laden a memory entry is.

    Attributes:
        significance: 0.0-1.0. How important is this entry to retain and retrieve?
        confidence: 0.0-1.0. How confident are we that this information is still true?
        emotional_valence: -1.0 to 1.0. Emotional tone (-1 negative, 0 neutral, 1 positive).
        revisit_cost: "low" | "medium" | "high". How costly is it to revisit/reverify?
    """
    significance: float = 0.3
    confidence: float = 1.0
    emotional_valence: float = 0.0
    revisit_cost: str = "low"

    def __post_init__(self):
        """Validate field bounds."""
        if not 0.0 <= self.significance <= 1.0:
            raise ValueError(f"significance must be 0.0-1.0, got {self.significance}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")
        if not -1.0 <= self.emotional_valence <= 1.0:
            raise ValueError(f"emotional_valence must be -1.0 to 1.0, got {self.emotional_valence}")
        if self.revisit_cost not in ("low", "medium", "high"):
            raise ValueError(f"revisit_cost must be 'low'|'medium'|'high', got {self.revisit_cost}")


# ─── Scoring Functions ───────────────────────────────────────────────────────

def score_weight(
    content: str,
    role: str = "assistant",
    category: str = "note",
    flags: Optional[Dict[str, Any]] = None,
) -> MemoryWeight:
    """
    Score a memory entry using heuristic patterns.

    Args:
        content: The entry text to analyze.
        role: "user" | "assistant". Who created this entry.
        category: Entry category from EntryCategory enum (as string).
        flags: Optional dict with keys:
            - is_user_knowledge: bool
            - is_correction: bool
            - is_milestone: bool

    Returns:
        MemoryWeight with scored significance, confidence, emotional_valence, revisit_cost.
    """
    if flags is None:
        flags = {}

    # Normalize content for pattern matching
    content_lower = content.lower()

    # ─── Trust Milestones ───────────────────────────────────────────────────
    # Patterns: "no notes", "approved", "accepted without", "trust your", "run with it", "your call"
    trust_patterns = [
        r"\bno\s+notes\b",
        r"\bapproved\b",
        r"\baccepted\s+without\b",
        r"\btrust\s+your\b",
        r"\brun\s+with\s+it\b",
        r"\byour\s+call\b",
        r"\bi\s+trust\s+you\b",
        r"\bgo\s+with\b",
    ]
    if any(re.search(p, content_lower) for p in trust_patterns):
        return MemoryWeight(
            significance=0.9,
            confidence=1.0,
            emotional_valence=0.8,
            revisit_cost="high",
        )

    # ─── Architectural Decisions ────────────────────────────────────────────
    # Patterns: "naming", "architecture", "design decision", "scope", "roadmap", "milestone"
    arch_patterns = [
        r"\bnaming\b",
        r"\barchitecture\b",
        r"\barchitectural\b",
        r"\bdesign\s+decision\b",
        r"\bscope\b",
        r"\broadmap\b",
        r"\bmilestone\b",
        r"\bfoundational\b",
        r"\bfoundation\b",
        r"\bcore\s+principle\b",
    ]
    if any(re.search(p, content_lower) for p in arch_patterns) or flags.get("is_milestone"):
        return MemoryWeight(
            significance=0.8,
            confidence=1.0,
            emotional_valence=0.0,
            revisit_cost="medium",
        )

    # ─── Corrections / Pivots ───────────────────────────────────────────────
    # Patterns: "actually", "change of plan", "pivot", "rethinking", "scratch that"
    correction_patterns = [
        r"\bactually\b",
        r"\bchange\s+of\s+plan\b",
        r"\bpivot\b",
        r"\brethinking\b",
        r"\bscratch\s+that\b",
        r"\bnever\s+mind\b",
        r"\bwaited\s+?—\s*i\b",
        r"\bwait\b.*?\bactually\b",
    ]
    if any(re.search(p, content_lower) for p in correction_patterns) or flags.get("is_correction"):
        return MemoryWeight(
            significance=0.7,
            confidence=0.95,
            emotional_valence=0.0,
            revisit_cost="low",
        )

    # ─── User Knowledge ─────────────────────────────────────────────────────
    # Explicit user_knowledge category
    if category == "user_knowledge" or flags.get("is_user_knowledge"):
        return MemoryWeight(
            significance=0.6,
            confidence=1.0,
            emotional_valence=0.0,
            revisit_cost="low",
        )

    # ─── Regular Implementation ─────────────────────────────────────────────
    # Code samples, implementation details, examples
    if category in ("implementation", "example", "code"):
        return MemoryWeight(
            significance=0.4,
            confidence=1.0,
            emotional_valence=0.0,
            revisit_cost="low",
        )

    # ─── Routine Context ────────────────────────────────────────────────────
    # Fallback: generic notes, references, facts
    return MemoryWeight(
        significance=0.2,
        confidence=1.0,
        emotional_valence=0.0,
        revisit_cost="low",
    )


def apply_confidence_decay(
    weight: MemoryWeight,
    age_days: float,
    is_user_knowledge: bool = False,
    reinforced: bool = False,
) -> MemoryWeight:
    """
    Apply confidence decay based on age without reinforcement.

    Decay schedule (without reinforcement):
        - 0-7 days: no decay
        - 7-14 days: -0.1 (total)
        - 14-30 days: -0.2 (total)
        - 30+ days: -0.3 (total)

    Modifiers:
        - is_user_knowledge: halve decay rate
        - reinforced: floor confidence at 0.8

    Args:
        weight: Original MemoryWeight to decay.
        age_days: Days since creation/last reinforcement.
        is_user_knowledge: Whether this is user knowledge (slower decay).
        reinforced: Whether this entry was recently reinforced (higher floor).

    Returns:
        New MemoryWeight with decayed confidence.
    """
    new_weight = MemoryWeight(
        significance=weight.significance,
        confidence=weight.confidence,
        emotional_valence=weight.emotional_valence,
        revisit_cost=weight.revisit_cost,
    )

    # Compute base decay
    if age_days <= 7:
        decay = 0.0
    elif age_days <= 14:
        decay = 0.1
    elif age_days <= 30:
        decay = 0.2
    else:
        decay = 0.3

    # Halve decay for user knowledge
    if is_user_knowledge:
        decay *= 0.5

    # Apply decay
    new_confidence = new_weight.confidence - decay

    # Enforce floor based on reinforcement
    if reinforced:
        new_confidence = max(0.8, new_confidence)
    else:
        new_confidence = max(0.1, new_confidence)

    new_weight.confidence = new_confidence
    return new_weight


def to_dict(w: MemoryWeight) -> Dict[str, Any]:
    """
    Convert MemoryWeight to a JSON-serializable dict.

    Args:
        w: MemoryWeight instance.

    Returns:
        Dict with keys: significance, confidence, emotional_valence, revisit_cost.
    """
    return asdict(w)


def from_dict(d: Dict[str, Any]) -> MemoryWeight:
    """
    Reconstruct MemoryWeight from a dict (e.g., from JSON).

    Args:
        d: Dict with keys matching MemoryWeight fields.

    Returns:
        MemoryWeight instance.

    Raises:
        KeyError or ValueError if required fields are missing or invalid.
    """
    return MemoryWeight(
        significance=float(d["significance"]),
        confidence=float(d["confidence"]),
        emotional_valence=float(d["emotional_valence"]),
        revisit_cost=str(d["revisit_cost"]),
    )


def merge_weight_into_metadata(
    existing_metadata: Dict[str, Any],
    weight: MemoryWeight,
) -> Dict[str, Any]:
    """
    Add or update the "weight" key in metadata without destroying other keys.

    Args:
        existing_metadata: Existing metadata dict (will not be modified).
        weight: MemoryWeight to merge.

    Returns:
        New metadata dict with "weight" key added/updated.
    """
    updated = existing_metadata.copy()
    updated["weight"] = to_dict(weight)
    return updated


def extract_weight_from_metadata(
    metadata: Dict[str, Any],
) -> Optional[MemoryWeight]:
    """
    Extract MemoryWeight from metadata dict if present.

    Args:
        metadata: Metadata dict that may contain a "weight" key.

    Returns:
        MemoryWeight if "weight" key exists and is valid, otherwise None.
    """
    if "weight" not in metadata:
        return None

    try:
        return from_dict(metadata["weight"])
    except (KeyError, ValueError, TypeError):
        return None


# ─── Utility: Compute Age ───────────────────────────────────────────────────

def compute_age_days(created_at: datetime, now: Optional[datetime] = None) -> float:
    """
    Compute age of an entry in days.

    Args:
        created_at: Datetime the entry was created.
        now: Reference time (defaults to utcnow()).

    Returns:
        Age in fractional days.
    """
    if now is None:
        now = datetime.utcnow()

    delta = now - created_at
    return delta.total_seconds() / 86400.0  # 86400 = seconds per day
