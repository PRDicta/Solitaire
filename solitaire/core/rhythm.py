"""
System 2: Conversational Rhythm

Detects user energy, session depth, and topic weight to guide adaptive response behavior.
Provides response guidance based on conversational state.
"""

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Literal


@dataclass
class RhythmState:
    """Current conversational rhythm state."""

    user_energy: Literal["high", "medium", "low"] = "medium"
    session_depth: Literal["fresh", "working", "deep", "winding_down"] = "fresh"
    topic_weight: Literal["light", "working", "heavy", "sensitive"] = "working"
    message_count: int = 0
    session_start_ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_message_ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def detect_energy(
    message: str, role: str, message_count: int, elapsed_minutes: float
) -> RhythmState:
    """
    Detect user energy, session depth, and topic weight from message patterns.

    Args:
        message: The user's message text
        role: User role (e.g., "user", "system")
        message_count: Total messages in session so far
        elapsed_minutes: Minutes elapsed since session start

    Returns:
        RhythmState with detected user_energy, session_depth, and topic_weight
    """
    state = RhythmState()
    state.message_count = message_count

    # Detect user_energy
    state.user_energy = _detect_user_energy(message)

    # Detect session_depth
    state.session_depth = _detect_session_depth(message_count, elapsed_minutes, message)

    # Detect topic_weight
    state.topic_weight = _detect_topic_weight(message)

    # Update timestamps
    state.last_message_ts = datetime.utcnow().isoformat()

    return state


def _detect_user_energy(message: str) -> Literal["high", "medium", "low"]:
    """
    Detect user energy level from message patterns.

    High energy:
    - message length > 500 chars
    - multiple questions
    - exclamation marks
    - brainstorming language

    Low energy:
    - message length < 80 chars
    - tiredness signals
    - delegation language

    Returns "medium" for everything else.
    """
    lower_msg = message.lower()

    # High energy signals
    if len(message) > 500:
        return "high"

    # Count questions and exclamations
    question_count = message.count("?")
    exclamation_count = message.count("!")

    if question_count > 1 or (question_count > 0 and exclamation_count > 0):
        return "high"

    # Brainstorming language patterns
    brainstorm_patterns = [
        r"\bwhat if\b",
        r"\blet's try\b",
        r"\bidea\b",
        r"\bwhat about\b",
        r"\bmaybe\b.*\?",
        r"\btrying\b",
        r"\bexplore\b",
    ]
    if any(re.search(pattern, lower_msg) for pattern in brainstorm_patterns):
        return "high"

    # Low energy signals - tiredness
    tiredness_patterns = [
        r"\btired\b",
        r"\blong day\b",
        r"\bexhausted\b",
        r"\bwinding down\b",
        r"\bone more\b",
        r"\blast thing\b",
        r"\bfatigued\b",
        r"\bdraining\b",
    ]
    if any(re.search(pattern, lower_msg) for pattern in tiredness_patterns):
        return "low"

    # Low energy signals - delegation
    delegation_patterns = [
        r"\bjust do it\b",
        r"\brun with it\b",
        r"\byour call\b",
        r"\bhandle it\b",
        r"\byou decide\b",
        r"\bwhatever\b",
        r"\bfine by me\b",
    ]
    if any(re.search(pattern, lower_msg) for pattern in delegation_patterns):
        return "low"

    # Low energy - very short message
    if len(message) < 80:
        return "low"

    return "medium"


def _detect_session_depth(
    message_count: int, elapsed_minutes: float, message: str
) -> Literal["fresh", "working", "deep", "winding_down"]:
    """
    Detect session depth from message count, elapsed time, and content.

    Fresh: < 5 messages or < 15 minutes elapsed
    Working: 5-20 messages
    Deep: 20+ messages
    Winding down: keyword detection
    """
    lower_msg = message.lower()

    # Check for winding down keywords first
    winding_patterns = [
        r"\bwrap up\b",
        r"\bone more\b",
        r"\blast item\b",
        r"\bthat's it for\b",
        r"\bcall it\b",
        r"\bfinish up\b",
        r"\bwrap this up\b",
    ]
    if any(re.search(pattern, lower_msg) for pattern in winding_patterns):
        return "winding_down"

    # Depth based on message count and time
    if message_count < 5 or elapsed_minutes < 15:
        return "fresh"

    if message_count >= 20:
        return "deep"

    # 5-20 messages
    return "working"


def _detect_topic_weight(message: str) -> Literal["light", "working", "heavy", "sensitive"]:
    """
    Detect topic weight from keyword patterns.

    Heavy: architecture, strategy, financial, pricing, roadmap, critical, redesign
    Sensitive: personal, frustrated, worried, trust, honest, disappointed, feeling
    Light: quick, simple, minor, trivial, just a, formatting
    Working: default
    """
    lower_msg = message.lower()

    # Heavy topic indicators
    heavy_patterns = [
        r"\barchitecture\b",
        r"\bstrategy\b",
        r"\bfinancial\b",
        r"\bpricing\b",
        r"\broadmap\b",
        r"\bcritical\b",
        r"\bredesign\b",
        r"\bmajor\b",
        r"\bfundamental\b",
        r"\bcore\b",
    ]
    if any(re.search(pattern, lower_msg) for pattern in heavy_patterns):
        return "heavy"

    # Sensitive topic indicators
    sensitive_patterns = [
        r"\bpersonal\b",
        r"\bfrustrated\b",
        r"\bworried\b",
        r"\btrust\b",
        r"\bhonest\b",
        r"\bdisappointed\b",
        r"\bfeeling\b",
        r"\bscared\b",
        r"\buneasy\b",
        r"\bconcerned\b",
        r"\bconfident\b",  # "not confident"
    ]
    if any(re.search(pattern, lower_msg) for pattern in sensitive_patterns):
        return "sensitive"

    # Light topic indicators
    light_patterns = [
        r"\bquick\b",
        r"\bsimple\b",
        r"\bminor\b",
        r"\btrivial\b",
        r"\bjust a\b",
        r"\bformatting\b",
        r"\bsmall\b",
        r"\bquestion mark\b",
        r"\btypo\b",
    ]
    if any(re.search(pattern, lower_msg) for pattern in light_patterns):
        return "light"

    return "working"


def get_response_guidance(state: RhythmState, persona_rhythm=None) -> dict:
    """
    Get response guidance based on conversational rhythm state.

    Args:
        state: RhythmState from detect_energy().
        persona_rhythm: Optional RhythmConfig from persona profile (Phase 5).
            When provided, persona preferences modulate the detected state.

    Returns guidance for:
    - verbosity: "compressed" | "normal" | "expanded"
    - questions_allowed: True | False
    - warmth_modifier: float (-0.1 to +0.1)
    - humor_allowed: True | False
    - new_topics_allowed: True | False
    - tangent_tolerance: float (0.0-1.0, from persona or default 0.5)
    - action_bias: float (0.0-1.0, from persona or default 0.5)
    - elaboration_trigger: str ("ask" | "offer" | "automatic")
    """
    # Map persona default_verbosity to guidance verbosity
    _verbosity_map = {
        "terse": "compressed",
        "moderate": "normal",
        "thorough": "expanded",
    }
    default_verbosity = "normal"
    if persona_rhythm and hasattr(persona_rhythm, 'default_verbosity'):
        default_verbosity = _verbosity_map.get(persona_rhythm.default_verbosity, "normal")

    guidance = {
        "verbosity": default_verbosity,
        "questions_allowed": True,
        "warmth_modifier": 0.0,
        "humor_allowed": True,
        "new_topics_allowed": True,
        "tangent_tolerance": persona_rhythm.tangent_tolerance if persona_rhythm else 0.5,
        "action_bias": persona_rhythm.action_bias if persona_rhythm else 0.5,
        "elaboration_trigger": persona_rhythm.elaboration_trigger if persona_rhythm else "ask",
    }

    # Energy-based adjustments
    if state.user_energy == "high":
        guidance["verbosity"] = "expanded"
        guidance["questions_allowed"] = True
        guidance["humor_allowed"] = True
        guidance["new_topics_allowed"] = True
        guidance["warmth_modifier"] = 0.05

    elif state.user_energy == "low":
        guidance["verbosity"] = "compressed"
        guidance["questions_allowed"] = False
        guidance["humor_allowed"] = False
        guidance["new_topics_allowed"] = False
        guidance["warmth_modifier"] = 0.1

    # Session depth adjustments
    if state.session_depth == "fresh":
        guidance["warmth_modifier"] = max(
            guidance["warmth_modifier"], 0.05
        )  # Be warmer on entry
        guidance["new_topics_allowed"] = True

    elif state.session_depth == "deep":
        guidance["questions_allowed"] = True
        guidance["new_topics_allowed"] = False  # Stay focused

    elif state.session_depth == "winding_down":
        guidance["verbosity"] = "compressed"
        guidance["questions_allowed"] = False
        guidance["new_topics_allowed"] = False
        guidance["warmth_modifier"] = min(
            guidance["warmth_modifier"] + 0.05, 0.1
        )  # Be warmer as we wind down

    # Topic weight adjustments
    if state.topic_weight == "heavy":
        guidance["verbosity"] = "expanded"
        guidance["questions_allowed"] = True
        guidance["humor_allowed"] = False  # Heavy topics need seriousness
        guidance["warmth_modifier"] = max(guidance["warmth_modifier"], 0.0)

    elif state.topic_weight == "sensitive":
        guidance["verbosity"] = "normal"
        guidance["questions_allowed"] = False  # Don't probe sensitive topics
        guidance["humor_allowed"] = False
        guidance["warmth_modifier"] = min(
            guidance["warmth_modifier"] + 0.1, 0.1
        )  # Max warmth
        guidance["new_topics_allowed"] = False

    elif state.topic_weight == "light":
        guidance["verbosity"] = "compressed"
        guidance["questions_allowed"] = True
        guidance["humor_allowed"] = True
        guidance["warmth_modifier"] = max(guidance["warmth_modifier"], -0.05)

    # ─── Phase 5: Persona rhythm modulation ────────────────────────────
    if persona_rhythm:
        silence = getattr(persona_rhythm, 'silence_comfort', 0.5)

        # High silence comfort + low energy → lean harder into compressed
        if silence >= 0.7 and state.user_energy == "low":
            guidance["verbosity"] = "compressed"
            guidance["questions_allowed"] = False

        # Low silence comfort + any state → resist compressed, pad if needed
        if silence <= 0.3 and guidance["verbosity"] == "compressed":
            if state.topic_weight not in ("light",):
                guidance["verbosity"] = "normal"

        # High action bias overrides elaboration tendency
        action = getattr(persona_rhythm, 'action_bias', 0.5)
        if action >= 0.7 and guidance["verbosity"] == "expanded":
            # Action-biased personas don't over-explain; they do
            guidance["verbosity"] = "normal"

        # Low tangent tolerance restricts new topics
        tangent = getattr(persona_rhythm, 'tangent_tolerance', 0.5)
        if tangent <= 0.3 and state.session_depth in ("deep", "working"):
            guidance["new_topics_allowed"] = False

    return guidance


def to_dict(state: RhythmState) -> dict:
    """Convert RhythmState to dictionary for JSON serialization."""
    return asdict(state)


def from_dict(d: dict) -> RhythmState:
    """Create RhythmState from dictionary."""
    return RhythmState(
        user_energy=d.get("user_energy", "medium"),
        session_depth=d.get("session_depth", "fresh"),
        topic_weight=d.get("topic_weight", "working"),
        message_count=d.get("message_count", 0),
        session_start_ts=d.get(
            "session_start_ts", datetime.utcnow().isoformat()
        ),
        last_message_ts=d.get("last_message_ts", datetime.utcnow().isoformat()),
    )
