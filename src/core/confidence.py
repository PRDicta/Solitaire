"""
The Librarian -- Confidence Scoring (Hindsight-inspired)

Unified confidence contract for all entry types. Implements:

1. Reinforcement: repeated observation of the same fact/opinion strengthens
   confidence. Each reinforcement bumps the score toward 1.0 with diminishing
   returns (logarithmic).

2. Decay: entries that haven't been reinforced lose confidence over time.
   Decay rate varies by provenance (user-stated decays slower than
   assistant-inferred) and category (user_knowledge decays slowest).

3. Authority: provenance-based trust floor. User-stated content starts
   higher and decays less than assistant-inferred content.

4. Retrieval integration: produces a single 0.0-1.0 confidence score
   that the reranker can use as a scoring signal.

Inspired by the Hindsight paper's CARA (Confidence-Accuracy-Recency-Authority)
framework, simplified for a non-ML pipeline.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


# -- Authority baselines by provenance ------------------------------------

PROVENANCE_AUTHORITY = {
    "user-stated": 1.0,        # User said it directly. Highest trust.
    "system": 0.95,            # System-generated (e.g., boot, enrichment).
    "assistant-inferred": 0.7, # The assistant inferred it. Reasonable but fallible.
    "unknown": 0.5,            # No provenance. Minimal trust.
}

# -- Decay rates (confidence lost per day without reinforcement) ----------
# Lower is slower. Applied as: confidence -= rate * days_since_reinforcement
# Floored by authority baseline (user-stated content never drops below 0.4).

DECAY_RATES = {
    "user_knowledge": 0.003,     # ~90 days to lose 0.3 points
    "decision": 0.005,           # ~60 days
    "preference": 0.005,
    "fact": 0.008,               # ~37 days
    "definition": 0.008,
    "behavioral": 0.002,         # Very slow -- behavioral patterns are stable
    "disposition_drift": 0.010,  # Faster -- drift signals are time-sensitive
    "_default": 0.007,           # ~43 days for unlisted categories
}

# -- Confidence floors by provenance --------------------------------------
# Content never decays below this. User-stated facts remain retrievable
# even if stale. Assistant-inferred content can decay to near-zero.

CONFIDENCE_FLOORS = {
    "user-stated": 0.4,
    "system": 0.3,
    "assistant-inferred": 0.15,
    "unknown": 0.1,
}


@dataclass
class ConfidenceScore:
    """
    Unified confidence assessment for any entry.

    Fields:
        base: Initial confidence from authority (provenance-derived).
        reinforcement_count: How many times this content has been re-observed.
        reinforcement_bonus: Cumulative bonus from reinforcement (log scale).
        decay_applied: Cumulative confidence lost to time decay.
        effective: The final score used by retrieval. Always 0.0-1.0.
        last_reinforced_at: When this entry was last reinforced.
    """
    base: float = 0.7
    reinforcement_count: int = 0
    reinforcement_bonus: float = 0.0
    decay_applied: float = 0.0
    effective: float = 0.7
    last_reinforced_at: Optional[str] = None  # ISO datetime string

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": round(self.base, 4),
            "reinforcement_count": self.reinforcement_count,
            "reinforcement_bonus": round(self.reinforcement_bonus, 4),
            "decay_applied": round(self.decay_applied, 4),
            "effective": round(self.effective, 4),
            "last_reinforced_at": self.last_reinforced_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConfidenceScore":
        return cls(
            base=float(d.get("base", 0.7)),
            reinforcement_count=int(d.get("reinforcement_count", 0)),
            reinforcement_bonus=float(d.get("reinforcement_bonus", 0.0)),
            decay_applied=float(d.get("decay_applied", 0.0)),
            effective=float(d.get("effective", 0.7)),
            last_reinforced_at=d.get("last_reinforced_at"),
        )


# -- Core functions -------------------------------------------------------

def initial_confidence(
    provenance: str = "unknown",
    category: str = "note",
) -> ConfidenceScore:
    """
    Compute the starting confidence for a new entry.

    Authority baseline comes from provenance. User-stated content starts
    at 1.0; assistant-inferred at 0.7; unknown at 0.5.
    """
    base = PROVENANCE_AUTHORITY.get(provenance, 0.5)
    return ConfidenceScore(
        base=base,
        reinforcement_count=0,
        reinforcement_bonus=0.0,
        decay_applied=0.0,
        effective=base,
        last_reinforced_at=datetime.utcnow().isoformat(),
    )


def reinforce(
    score: ConfidenceScore,
    now: Optional[datetime] = None,
) -> ConfidenceScore:
    """
    Strengthen confidence after re-observing the same content.

    Uses logarithmic diminishing returns: the first few reinforcements
    matter most. The 1st adds ~0.15, the 5th adds ~0.05, the 20th adds ~0.02.

    Formula: bonus = 0.3 * log(1 + count) / log(1 + 20)
    Cap at 0.3 total bonus. This means even unknown-provenance content
    can reach 0.8 effective with enough reinforcement (0.5 base + 0.3 bonus).
    """
    if now is None:
        now = datetime.utcnow()

    new_count = score.reinforcement_count + 1
    # Logarithmic bonus: diminishing returns, capped at 0.3
    bonus = 0.3 * math.log1p(new_count) / math.log1p(20)
    bonus = min(bonus, 0.3)

    # Reset decay on reinforcement (content just proved it's still relevant)
    effective = min(1.0, score.base + bonus)

    return ConfidenceScore(
        base=score.base,
        reinforcement_count=new_count,
        reinforcement_bonus=round(bonus, 4),
        decay_applied=0.0,  # Reset decay
        effective=round(effective, 4),
        last_reinforced_at=now.isoformat(),
    )


def apply_decay(
    score: ConfidenceScore,
    days_elapsed: float,
    category: str = "note",
    provenance: str = "unknown",
) -> ConfidenceScore:
    """
    Apply time-based confidence decay.

    Decay is linear per day, varying by category. User knowledge decays
    slowest; disposition drift decays fastest. Reinforced entries decay
    from their reinforced level, not from base.

    The floor prevents entries from becoming completely invisible.
    Provenance determines the floor: user-stated content stays retrievable
    even when stale.
    """
    rate = DECAY_RATES.get(category, DECAY_RATES["_default"])
    floor = CONFIDENCE_FLOORS.get(provenance, 0.1)

    # Decay from the pre-decay effective score (base + bonus)
    pre_decay = score.base + score.reinforcement_bonus
    total_decay = rate * days_elapsed
    effective = max(floor, pre_decay - total_decay)

    return ConfidenceScore(
        base=score.base,
        reinforcement_count=score.reinforcement_count,
        reinforcement_bonus=score.reinforcement_bonus,
        decay_applied=round(total_decay, 4),
        effective=round(effective, 4),
        last_reinforced_at=score.last_reinforced_at,
    )


def compute_effective(
    score: ConfidenceScore,
    days_since_reinforcement: float,
    category: str = "note",
    provenance: str = "unknown",
) -> float:
    """
    Convenience: compute the current effective confidence for retrieval.

    This is the function the reranker calls. It takes the stored score
    and the time elapsed since last reinforcement, and returns a single
    0.0-1.0 value.
    """
    decayed = apply_decay(score, days_since_reinforcement, category, provenance)
    return decayed.effective


# -- Metadata integration -------------------------------------------------

def extract_confidence_from_metadata(metadata: Dict[str, Any]) -> Optional[ConfidenceScore]:
    """Extract ConfidenceScore from an entry's metadata dict."""
    conf_data = metadata.get("confidence")
    if conf_data is None:
        return None
    try:
        if isinstance(conf_data, (int, float)):
            # Legacy format: confidence stored as a raw number, not a dict
            return ConfidenceScore(base=float(conf_data))
        return ConfidenceScore.from_dict(conf_data)
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def merge_confidence_into_metadata(
    metadata: Dict[str, Any],
    score: ConfidenceScore,
) -> Dict[str, Any]:
    """Add or update the confidence score in metadata without destroying other keys."""
    updated = metadata.copy()
    updated["confidence"] = score.to_dict()
    return updated
