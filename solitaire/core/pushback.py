"""
System 6: Pushback Decision Framework

Provides intelligent, values-aligned pushback on user decisions through
evidence-weighted evaluation, consequence assessment, and contextual
sensitivity (energy, recency, taste).

The Chief Librarian uses this system to distinguish between:
- NUDGE: gentle note-taking, low stakes
- CHALLENGE: substantive disagreement, moderate stakes
- BLOCK: hard stop with values conflict, high stakes
- PROTECT: intervention against self-contradiction, critical stakes
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional


class PushbackTier(str, Enum):
    """Severity tier of pushback intervention."""
    NUDGE = "nudge"              # Low evidence OR low consequence
    CHALLENGE = "challenge"       # Medium evidence AND medium+ consequence
    BLOCK = "block"              # High evidence AND high+ consequence
    PROTECT = "protect"          # Critical — user contradicting own values/goals


class EvidenceStrength(str, Enum):
    """Quality and weight of supporting evidence."""
    ANECDOTAL = "anecdotal"            # weight 0.2 — single observation, weak prior
    SINGLE_PRIOR = "single_prior"      # weight 0.4 — one documented decision
    MULTIPLE_PRIOR = "multiple_prior"  # weight 0.6 — pattern across 2-3 instances
    EMPIRICAL = "empirical"            # weight 0.8 — external data, research
    USER_VALUE = "user_value"          # weight 1.0 — contradicts stated value/goal


class ConsequenceLevel(str, Enum):
    """Magnitude of impact from the decision."""
    TRIVIAL = "trivial"           # naming, formatting, preference
    MODERATE = "moderate"         # timeline, scope, resource allocation
    HIGH = "high"                 # client-facing, financial, architectural
    CRITICAL = "critical"         # contradicts user's stated values/goals


# Evidence weight mapping
EVIDENCE_WEIGHTS = {
    EvidenceStrength.ANECDOTAL: 0.2,
    EvidenceStrength.SINGLE_PRIOR: 0.4,
    EvidenceStrength.MULTIPLE_PRIOR: 0.6,
    EvidenceStrength.EMPIRICAL: 0.8,
    EvidenceStrength.USER_VALUE: 1.0,
}

# Framing templates by tier
# Design principle: state the disagreement, then the reasoning. No diplomatic preamble.
# The tier controls intensity, not politeness. Even NUDGE is direct — just brief.
FRAMING_TEMPLATES = {
    PushbackTier.NUDGE: "{reasoning}. Your call.",
    PushbackTier.CHALLENGE: "I think that's wrong. {reasoning}. Risk: {consequence}.",
    PushbackTier.BLOCK: "I'd flag this seriously. {reasoning}. This contradicts {evidence}. What changed?",
    PushbackTier.PROTECT: "Hold on. You said: \"{evidence_quote}\". This cuts against that. Conscious choice or drift?",
}


@dataclass
class PushbackEvaluation:
    """Result of a pushback decision analysis."""
    should_pushback: bool
    tier: PushbackTier
    evidence_strength: EvidenceStrength
    consequence_level: ConsequenceLevel
    reasoning: str                          # Why the Chief is pushing back
    supporting_entry_ids: List[str] = field(default_factory=list)  # Rolodex entries
    suggested_framing: str = ""             # How to phrase it at this tier
    recency_override: bool = False          # True if discussed in last 2 sessions
    energy_deferral: bool = False           # True if deferred due to low user energy


@dataclass
class PushbackEvent:
    """Record of a pushback intervention."""
    timestamp: str          # ISO format
    session_id: str
    tier: str              # PushbackTier value
    topic: str
    reasoning: str
    outcome: str = "pending"  # pending/accepted/overridden/deferred


def evaluate_pushback(
    evidence_strength: EvidenceStrength,
    consequence_level: ConsequenceLevel,
    user_energy: str,  # "high"/"medium"/"low" from rhythm system
    recently_considered: bool,  # Discussed in last 2 sessions?
    is_taste: bool = False,  # If True, never pushback
    reasoning: str = "",  # Explanation for the pushback
    evidence_quote: str = "",  # Supporting prior statement
    supporting_entry_ids: Optional[List[str]] = None,
) -> PushbackEvaluation:
    """
    Core decision function. Evaluates whether and how to push back.

    Rules:
    ------
    - If is_taste: always return should_pushback=False
    - If recently_considered and user decided: should_pushback=False, set recency_override
    - If user_energy == "low" and tier would be NUDGE/CHALLENGE: defer with
      energy_deferral=True; suggested_framing notes "flagging for later"
    - Never pushback if evidence is ANECDOTAL and consequence is TRIVIAL (emotional decisions)

    Tier determination:
    -------------------
    - evidence_weight < 0.4 OR consequence TRIVIAL → NUDGE
    - evidence_weight >= 0.4 AND consequence MODERATE+ → CHALLENGE
    - evidence_weight >= 0.6 AND consequence HIGH+ → BLOCK
    - evidence_weight >= 0.8 AND consequence CRITICAL → PROTECT

    All framings end with user-has-final-say language and explicit permission to override.

    Args:
        evidence_strength: Quality of supporting evidence
        consequence_level: Impact magnitude
        user_energy: "high", "medium", or "low"
        recently_considered: Was this topic discussed in the last 2 sessions?
        is_taste: If True, always skip pushback (taste is user's domain)
        reasoning: Brief explanation of the pushback
        evidence_quote: Direct quote from a prior user statement (for PROTECT tier)
        supporting_entry_ids: Rolodex entry IDs backing this position

    Returns:
        PushbackEvaluation with should_pushback, tier, suggested_framing, flags
    """
    if supporting_entry_ids is None:
        supporting_entry_ids = []

    # Guard: never pushback on taste
    if is_taste:
        return PushbackEvaluation(
            should_pushback=False,
            tier=PushbackTier.NUDGE,
            evidence_strength=evidence_strength,
            consequence_level=consequence_level,
            reasoning="Taste is your domain.",
            supporting_entry_ids=supporting_entry_ids,
        )

    # Guard: if recently decided, skip (recency override)
    if recently_considered:
        return PushbackEvaluation(
            should_pushback=False,
            tier=PushbackTier.NUDGE,
            evidence_strength=evidence_strength,
            consequence_level=consequence_level,
            reasoning="You've already considered this recently.",
            supporting_entry_ids=supporting_entry_ids,
            recency_override=True,
        )

    # Guard: never pushback on emotional decisions (anecdotal + trivial)
    if (evidence_strength == EvidenceStrength.ANECDOTAL and
        consequence_level == ConsequenceLevel.TRIVIAL):
        return PushbackEvaluation(
            should_pushback=False,
            tier=PushbackTier.NUDGE,
            evidence_strength=evidence_strength,
            consequence_level=consequence_level,
            reasoning="Emotional decision; supporting you as-is.",
            supporting_entry_ids=supporting_entry_ids,
        )

    # Calculate evidence weight
    evidence_weight = EVIDENCE_WEIGHTS[evidence_strength]

    # Determine tier based on evidence + consequence
    if evidence_weight < 0.4 or consequence_level == ConsequenceLevel.TRIVIAL:
        tier = PushbackTier.NUDGE
    elif (evidence_weight >= 0.4 and
          consequence_level in (ConsequenceLevel.MODERATE, ConsequenceLevel.HIGH, ConsequenceLevel.CRITICAL)):
        if evidence_weight >= 0.6 and consequence_level in (ConsequenceLevel.HIGH, ConsequenceLevel.CRITICAL):
            if evidence_weight >= 0.8 and consequence_level == ConsequenceLevel.CRITICAL:
                tier = PushbackTier.PROTECT
            else:
                tier = PushbackTier.BLOCK
        else:
            tier = PushbackTier.CHALLENGE
    else:
        tier = PushbackTier.NUDGE

    # Check energy deferral: if low energy and tier is NUDGE/CHALLENGE, defer
    should_defer = (user_energy == "low" and tier in (PushbackTier.NUDGE, PushbackTier.CHALLENGE))

    # Format suggested framing
    suggested_framing = _format_framing(
        tier=tier,
        reasoning=reasoning,
        consequence=consequence_level.value,
        evidence_quote=evidence_quote,
        deferred=should_defer,
    )

    return PushbackEvaluation(
        should_pushback=not should_defer,  # Don't pushback if deferring due to energy
        tier=tier,
        evidence_strength=evidence_strength,
        consequence_level=consequence_level,
        reasoning=reasoning,
        supporting_entry_ids=supporting_entry_ids,
        suggested_framing=suggested_framing,
        energy_deferral=should_defer,
    )


def _format_framing(
    tier: PushbackTier,
    reasoning: str,
    consequence: str = "",
    evidence_quote: str = "",
    deferred: bool = False,
) -> str:
    """
    Internal helper to format pushback framing with context.

    Args:
        tier: PushbackTier enum
        reasoning: Explanation of the pushback
        consequence: ConsequenceLevel value
        evidence_quote: Direct quote from prior user statement
        deferred: If True, frame as "flagging for later"

    Returns:
        Formatted framing string
    """
    if deferred:
        return f"Parking this for when energy is higher: {reasoning}."

    if tier == PushbackTier.NUDGE:
        return f"{reasoning}. Your call."

    elif tier == PushbackTier.CHALLENGE:
        consequence_str = f" Risk: {consequence}." if consequence else ""
        return f"I think that's wrong. {reasoning}.{consequence_str}"

    elif tier == PushbackTier.BLOCK:
        return f"I'd flag this seriously. {reasoning}. This contradicts prior thinking. What changed?"

    elif tier == PushbackTier.PROTECT:
        if evidence_quote:
            return (f"Hold on. You said: \"{evidence_quote}\". "
                   f"This cuts against that. "
                   f"Conscious choice or drift?")
        else:
            return (f"Hold on. What you're proposing cuts against a stated value. "
                   f"Conscious choice or drift?")

    return reasoning


def format_pushback(evaluation: PushbackEvaluation) -> str:
    """
    Return the suggested_framing string from a PushbackEvaluation.

    Args:
        evaluation: PushbackEvaluation result

    Returns:
        Suggested framing string
    """
    return evaluation.suggested_framing


def log_pushback_event(
    state_path: str,
    event: PushbackEvent,
) -> None:
    """
    Append a pushback event to the chief_state.json pushback_log array.

    Reads the state file, appends the event, keeps max 50 events (FIFO),
    and writes back. Creates pushback_log if missing.

    Args:
        state_path: Path to chief_state.json
        event: PushbackEvent to log
    """
    state_file = Path(state_path)

    # Read existing state or create new
    if state_file.exists():
        with open(state_file, "r") as f:
            state = json.load(f)
    else:
        state = {}

    # Initialize or get pushback_log
    if "pushback_log" not in state:
        state["pushback_log"] = []

    # Append event
    event_dict = asdict(event)
    state["pushback_log"].append(event_dict)

    # Keep max 50 (FIFO)
    if len(state["pushback_log"]) > 50:
        state["pushback_log"] = state["pushback_log"][-50:]

    # Write back
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def get_pushback_stats(state_path: str) -> dict:
    """
    Read pushback_log from state file and return summary statistics.

    Args:
        state_path: Path to chief_state.json

    Returns:
        Dictionary with keys:
        - total: Total pushback events
        - by_tier: {nudge: N, challenge: N, block: N, protect: N}
        - by_outcome: {accepted: N, overridden: N, deferred: N, pending: N}
        - override_rate: Float (overridden / (accepted + overridden)), or 0 if no decisions
    """
    state_file = Path(state_path)

    if not state_file.exists():
        return {
            "total": 0,
            "by_tier": {"nudge": 0, "challenge": 0, "block": 0, "protect": 0},
            "by_outcome": {"accepted": 0, "overridden": 0, "deferred": 0, "pending": 0},
            "override_rate": 0.0,
        }

    with open(state_file, "r") as f:
        state = json.load(f)

    pushback_log = state.get("pushback_log", [])

    # Count by tier
    by_tier = {tier.value: 0 for tier in PushbackTier}
    # Count by outcome
    by_outcome = {"accepted": 0, "overridden": 0, "deferred": 0, "pending": 0}

    for event in pushback_log:
        tier = event.get("tier", "nudge")
        outcome = event.get("outcome", "pending")

        if tier in by_tier:
            by_tier[tier] += 1
        if outcome in by_outcome:
            by_outcome[outcome] += 1

    # Calculate override rate
    decided = by_outcome["accepted"] + by_outcome["overridden"]
    override_rate = 0.0
    if decided > 0:
        override_rate = by_outcome["overridden"] / decided

    return {
        "total": len(pushback_log),
        "by_tier": by_tier,
        "by_outcome": by_outcome,
        "override_rate": override_rate,
    }


def update_pushback_outcome(
    state_path: str,
    timestamp: str,
    outcome: str,  # accepted/overridden/deferred
) -> bool:
    """
    Update the outcome of a pushback event by timestamp.

    Reads the state file, finds the matching event, updates its outcome,
    and writes back.

    Args:
        state_path: Path to chief_state.json
        timestamp: ISO timestamp to match
        outcome: New outcome value (accepted/overridden/deferred)

    Returns:
        True if event was found and updated, False otherwise
    """
    state_file = Path(state_path)

    if not state_file.exists():
        return False

    with open(state_file, "r") as f:
        state = json.load(f)

    pushback_log = state.get("pushback_log", [])

    # Find and update
    found = False
    for event in pushback_log:
        if event.get("timestamp") == timestamp:
            event["outcome"] = outcome
            found = True
            break

    # Write back if found
    if found:
        state["pushback_log"] = pushback_log
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    return found
