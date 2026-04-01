"""
The Librarian — Persona System
Cognitive profiles, YAML persistence, drift overlay, and cross-session state.

A persona is the Librarian's DNA — a multi-dimensional cognitive profile
that governs how it observes, communicates, learns, and pushes back.
The baseline lives in persona.yaml; live drift entries in the rolodex
nudge it over time. On boot, baseline + accumulated drift = active profile.

The persistent persona state layer (persona_state.json) captures:
- Effective trait snapshots at session end
- Per-trait drift history across sessions
- Baseline ratchet candidates for long-running stable drift
- Session lineage for persona evolution tracking
"""
import json as _json
import math
import yaml
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any


# ─── Constants ──────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0"

DEFAULT_TRAITS = {
    "observance": 0.5,
    "assertiveness": 0.5,
    "conviction": 0.5,
    "warmth": 0.5,
    "humor": 0.5,
    "initiative": 0.5,
    "empathy": 0.5,
}

VALID_TRAIT_NAMES = set(DEFAULT_TRAITS.keys())


# ─── Data Structures ───────────────────────────────────────────────────────

@dataclass
class TraitProfile:
    """The seven trait dimensions of a cognitive profile."""
    observance: float = 0.5
    assertiveness: float = 0.5
    conviction: float = 0.5
    warmth: float = 0.5
    humor: float = 0.5
    initiative: float = 0.5
    empathy: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "TraitProfile":
        filtered = {k: float(v) for k, v in d.items() if k in VALID_TRAIT_NAMES}
        return cls(**filtered)

    def get(self, trait: str) -> float:
        if trait not in VALID_TRAIT_NAMES:
            raise ValueError(f"Unknown trait: {trait}. Valid: {VALID_TRAIT_NAMES}")
        return getattr(self, trait)

    def set(self, trait: str, value: float) -> None:
        if trait not in VALID_TRAIT_NAMES:
            raise ValueError(f"Unknown trait: {trait}. Valid: {VALID_TRAIT_NAMES}")
        clamped = max(0.0, min(1.0, value))
        setattr(self, trait, clamped)

    def clamp_all(self) -> None:
        """Ensure all traits are within [0.0, 1.0]."""
        for trait in VALID_TRAIT_NAMES:
            val = getattr(self, trait)
            setattr(self, trait, max(0.0, min(1.0, val)))


@dataclass
class DomainEnvelope:
    """What this Librarian is *for* — shapes attention and skill acquisition."""
    primary: str = "general"
    secondary: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)
    skill_orientation: Dict[str, List[str]] = field(default_factory=lambda: {
        "active": [],
        "watching": [],
        "ignored": [],
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainEnvelope":
        return cls(
            primary=d.get("primary", "general"),
            secondary=d.get("secondary", []),
            excluded=d.get("excluded", []),
            skill_orientation=d.get("skill_orientation", {
                "active": [], "watching": [], "ignored": [],
            }),
        )


@dataclass
class DriftConfig:
    """Controls how the disposition filter adjusts traits over time."""
    max_nudge_per_event: float = 0.03
    max_session_drift: float = 0.15
    base_decay: float = 0.25
    signal_threshold: float = 0.6
    reinforcement_factor: float = 0.3

    def effective_decay(self, reinforcement_count: int) -> float:
        """Compute decay rate scaled by reinforcement.

        Formula: effective_decay = base_decay / (1 + reinforcement_count × factor)
        One-off signals decay fast. Repeated signals persist.
        """
        return self.base_decay / (1.0 + reinforcement_count * self.reinforcement_factor)

    def compute_effective_nudge(
        self,
        original_nudge: float,
        reinforcement_count: int,
        sessions_elapsed: int,
    ) -> float:
        """Apply reinforcement-tiered decay to a drift nudge."""
        decay = self.effective_decay(reinforcement_count)
        return original_nudge * ((1.0 - decay) ** sessions_elapsed)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DriftConfig":
        return cls(
            max_nudge_per_event=d.get("max_nudge_per_event", 0.03),
            max_session_drift=d.get("max_session_drift", 0.15),
            base_decay=d.get("base_decay", 0.25),
            signal_threshold=d.get("signal_threshold", 0.6),
            reinforcement_factor=d.get("reinforcement_factor", 0.3),
        )


@dataclass
class SharingConfig:
    """Cross-Librarian profile signal sharing settings."""
    receives_profile_signals: bool = True
    sends_profile_signals: bool = True
    siloed: bool = False
    consent_given_at: Optional[str] = None

    def is_sharing_enabled(self) -> bool:
        """Check if any sharing is active (siloed overrides everything)."""
        if self.siloed:
            return False
        return self.receives_profile_signals or self.sends_profile_signals

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SharingConfig":
        return cls(
            receives_profile_signals=d.get("receives_profile_signals", True),
            sends_profile_signals=d.get("sends_profile_signals", True),
            siloed=d.get("siloed", False),
            consent_given_at=d.get("consent_given_at"),
        )


VALID_GREETING_STYLES = {"check-in", "social", "direct", "contextual"}


@dataclass
class GreetingProtocol:
    """Greeting behavior configuration for a persona.

    Controls how the persona opens a session — warmth level, memory weaving,
    small talk tolerance, and canned examples the LLM riffs on.
    """
    style: str = "direct"  # check-in | social | direct | contextual
    warmth_threshold: float = 0.55
    memory_reference: bool = True
    small_talk_tolerance: int = 2  # Max exchanges before nudging toward work
    examples: Dict[str, str] = field(default_factory=lambda: {
        "high_warmth": "Good to see you back. What's on your mind?",
        "mid_warmth": "What can I help with?",
        "low_warmth": "Ready when you are.",
        "contextual": "I see you were working on [THREAD]. Where did that land?",
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "style": self.style,
            "warmth_threshold": self.warmth_threshold,
            "memory_reference": self.memory_reference,
            "small_talk_tolerance": self.small_talk_tolerance,
            "examples": dict(self.examples),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GreetingProtocol":
        style = d.get("style", "direct")
        if style not in VALID_GREETING_STYLES:
            style = "direct"
        return cls(
            style=style,
            warmth_threshold=d.get("warmth_threshold", 0.55),
            memory_reference=d.get("memory_reference", True),
            small_talk_tolerance=d.get("small_talk_tolerance", 2),
            examples=d.get("examples", {}),
        )


# ─── Phase 5: Behavioral Genome ───────────────────────────────────────────

VALID_VERBOSITY_LEVELS = {"terse", "moderate", "thorough"}
VALID_ELABORATION_TRIGGERS = {"ask", "offer", "automatic"}
VALID_FLAG_STYLES = {"inline", "footnote", "separate"}
VALID_MEMORY_WEIGHT_TIERS = {"high_weight", "normal_weight", "low_weight"}


@dataclass
class AttentionItem:
    """A single attention pattern — something the persona notices unprompted.

    Each item describes a category of observation (e.g., "numerical_inconsistency")
    with a human-readable description and a minimum observance threshold.
    The LLM interprets these declaratively; no runtime regex matching.
    """
    category: str = ""
    description: str = ""
    min_observance: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "min_observance": self.min_observance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AttentionItem":
        return cls(
            category=d.get("category", ""),
            description=d.get("description", ""),
            min_observance=d.get("min_observance", 0.5),
        )


@dataclass
class AttentionConfig:
    """What the persona notices unprompted. Driven by domain + observance trait.

    always_flag: Categories the persona should flag without being asked.
    never_flag: Categories the persona should explicitly ignore.
    flag_style: How flags are surfaced — inline (in response), footnote, or separate.
    """
    always_flag: List[AttentionItem] = field(default_factory=list)
    never_flag: List[str] = field(default_factory=list)
    flag_style: str = "inline"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "always_flag": [item.to_dict() for item in self.always_flag],
            "never_flag": list(self.never_flag),
            "flag_style": self.flag_style,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AttentionConfig":
        flag_style = d.get("flag_style", "inline")
        if flag_style not in VALID_FLAG_STYLES:
            flag_style = "inline"
        return cls(
            always_flag=[
                AttentionItem.from_dict(item)
                for item in d.get("always_flag", [])
            ],
            never_flag=d.get("never_flag", []),
            flag_style=flag_style,
        )

    def get_active_flags(self, effective_observance: float) -> List[AttentionItem]:
        """Return attention items whose min_observance is met by the effective trait."""
        return [
            item for item in self.always_flag
            if effective_observance >= item.min_observance
        ]


@dataclass
class RhythmConfig:
    """Persona-specific conversational rhythm parameters.

    These override or modulate the generic rhythm detector's output.
    The detector reads the room; these parameters define the persona's
    natural conversational posture.

    default_verbosity: Baseline response length preference.
    tangent_tolerance: 0.0-1.0. How far off-topic the persona will follow.
    silence_comfort: 0.0-1.0. How comfortable the persona is with pauses/short replies.
    action_bias: 0.0-1.0. Preference for doing over discussing.
    elaboration_trigger: When to expand on a topic — only if asked, offer to, or automatically.
    """
    default_verbosity: str = "moderate"
    tangent_tolerance: float = 0.5
    silence_comfort: float = 0.5
    action_bias: float = 0.5
    elaboration_trigger: str = "ask"

    def __post_init__(self):
        if self.default_verbosity not in VALID_VERBOSITY_LEVELS:
            self.default_verbosity = "moderate"
        if self.elaboration_trigger not in VALID_ELABORATION_TRIGGERS:
            self.elaboration_trigger = "ask"
        self.tangent_tolerance = max(0.0, min(1.0, self.tangent_tolerance))
        self.silence_comfort = max(0.0, min(1.0, self.silence_comfort))
        self.action_bias = max(0.0, min(1.0, self.action_bias))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_verbosity": self.default_verbosity,
            "tangent_tolerance": self.tangent_tolerance,
            "silence_comfort": self.silence_comfort,
            "action_bias": self.action_bias,
            "elaboration_trigger": self.elaboration_trigger,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RhythmConfig":
        return cls(
            default_verbosity=d.get("default_verbosity", "moderate"),
            tangent_tolerance=d.get("tangent_tolerance", 0.5),
            silence_comfort=d.get("silence_comfort", 0.5),
            action_bias=d.get("action_bias", 0.5),
            elaboration_trigger=d.get("elaboration_trigger", "ask"),
        )


@dataclass
class MemoryPriorities:
    """Persona-level memory recall weighting.

    Controls which categories of recalled content get boosted or suppressed
    during auto-recall. The persona's domain focus shapes what memories
    matter most in context.

    high_weight: Categories that get a 2x recall boost.
    normal_weight: Categories at standard recall weight (1x).
    low_weight: Categories that get a 0.5x recall reduction.
    """
    high_weight: List[str] = field(default_factory=list)
    normal_weight: List[str] = field(default_factory=list)
    low_weight: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "high_weight": list(self.high_weight),
            "normal_weight": list(self.normal_weight),
            "low_weight": list(self.low_weight),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryPriorities":
        return cls(
            high_weight=d.get("high_weight", []),
            normal_weight=d.get("normal_weight", []),
            low_weight=d.get("low_weight", []),
        )

    def get_weight_for_category(self, category: str) -> float:
        """Return the recall weight multiplier for a given category.

        Returns 2.0 for high_weight, 1.0 for normal_weight, 0.5 for low_weight.
        Unrecognized categories default to 1.0.
        """
        if category in self.high_weight:
            return 2.0
        if category in self.low_weight:
            return 0.5
        return 1.0


@dataclass
class ConvictionOverride:
    """A trigger condition for conviction-based intervention."""
    condition: str = ""
    action: str = ""
    min_conviction: float = 0.5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConvictionOverride":
        return cls(
            condition=d.get("condition", ""),
            action=d.get("action", ""),
            min_conviction=d.get("min_conviction", 0.5),
        )


@dataclass
class InitiativeTrigger:
    """A trigger condition for proactive initiative."""
    condition: str = ""
    action: str = ""
    min_initiative: float = 0.5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InitiativeTrigger":
        return cls(
            condition=d.get("condition", ""),
            action=d.get("action", ""),
            min_initiative=d.get("min_initiative", 0.5),
        )


VALID_URGENCY_LEVELS = {"background", "immediate", "deferred"}
VALID_COST_LEVELS = {"low", "medium", "high"}
VALID_SOURCE_TYPES = {"indexed", "web"}


@dataclass
class AcquisitionSource:
    """A source endpoint for anticipatory knowledge acquisition.

    Sources are tried in order: indexed skill packs first, then pre-vetted
    external URLs, then web search as a fallback.
    """
    type: str = "indexed"           # indexed | web
    path: str = ""                  # For indexed: relative path under personas/{key}/
    query_template: str = ""        # For web: search template with {context} placeholder
    trusted_domains: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.path:
            d["path"] = self.path
        if self.query_template:
            d["query_template"] = self.query_template
        if self.trusted_domains:
            d["trusted_domains"] = list(self.trusted_domains)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AcquisitionSource":
        src_type = d.get("type", "indexed")
        if src_type not in VALID_SOURCE_TYPES:
            src_type = "indexed"
        return cls(
            type=src_type,
            path=d.get("path", ""),
            query_template=d.get("query_template", ""),
            trusted_domains=d.get("trusted_domains", []),
        )


@dataclass
class AcquisitionConfig:
    """What to acquire and where to look for it."""
    skill_domain: str = ""
    sources: List[AcquisitionSource] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_domain": self.skill_domain,
            "sources": [s.to_dict() for s in self.sources],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AcquisitionConfig":
        return cls(
            skill_domain=d.get("skill_domain", ""),
            sources=[
                AcquisitionSource.from_dict(s)
                for s in d.get("sources", [])
            ],
        )


@dataclass
class AcquisitionTrigger:
    """A trigger condition for anticipatory knowledge acquisition (Tier 3).

    When conversational patterns match, the persona predicts an upcoming
    knowledge need and proactively acquires content. The urgency level
    controls whether acquisition happens silently (background), with a
    brief pause (immediate), or at the next natural break (deferred).
    The estimated_cost gates user confirmation for expensive operations.
    """
    condition: str = ""                             # Human-readable trigger name
    patterns: List[str] = field(default_factory=list)  # Regex patterns to match
    acquire: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    urgency: str = "background"                     # background | immediate | deferred
    narrate: bool = False                           # Whether to tell the user what's happening
    estimated_cost: str = "low"                     # low | medium | high

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "patterns": list(self.patterns),
            "acquire": self.acquire.to_dict(),
            "urgency": self.urgency,
            "narrate": self.narrate,
            "estimated_cost": self.estimated_cost,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AcquisitionTrigger":
        urgency = d.get("urgency", "background")
        if urgency not in VALID_URGENCY_LEVELS:
            urgency = "background"
        cost = d.get("estimated_cost", "low")
        if cost not in VALID_COST_LEVELS:
            cost = "low"
        acquire_raw = d.get("acquire", {})
        if isinstance(acquire_raw, dict):
            acquire = AcquisitionConfig.from_dict(acquire_raw)
        else:
            acquire = AcquisitionConfig()
        return cls(
            condition=d.get("condition", ""),
            patterns=d.get("patterns", []),
            acquire=acquire,
            urgency=urgency,
            narrate=d.get("narrate", False),
            estimated_cost=cost,
        )


@dataclass
class BehavioralTriggers:
    """Specific conditions that activate trait-driven responses."""
    conviction_overrides: List[ConvictionOverride] = field(default_factory=list)
    initiative_triggers: List[InitiativeTrigger] = field(default_factory=list)
    acquisition_triggers: List[AcquisitionTrigger] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conviction_overrides": [asdict(c) for c in self.conviction_overrides],
            "initiative_triggers": [asdict(i) for i in self.initiative_triggers],
            "acquisition_triggers": [a.to_dict() for a in self.acquisition_triggers],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BehavioralTriggers":
        return cls(
            conviction_overrides=[
                ConvictionOverride.from_dict(c)
                for c in d.get("conviction_overrides", [])
            ],
            initiative_triggers=[
                InitiativeTrigger.from_dict(i)
                for i in d.get("initiative_triggers", [])
            ],
            acquisition_triggers=[
                AcquisitionTrigger.from_dict(a)
                for a in d.get("acquisition_triggers", [])
            ],
        )


@dataclass
class PersonaIdentity:
    """Display identity for a Librarian instance."""
    name: str = "Librarian"
    role: str = "general-assistant"
    description: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonaIdentity":
        return cls(
            name=d.get("name", "Librarian"),
            role=d.get("role", "general-assistant"),
            description=d.get("description", ""),
        )


# ─── Drift Entry ───────────────────────────────────────────────────────────

@dataclass
class DriftEntry:
    """A single disposition drift observation, stored in the rolodex.

    Each drift entry records a signal event, which traits were affected,
    and the nudge applied. On boot, these are aggregated with decay to
    produce the effective profile overlay.
    """
    signal: str                           # Signal key (e.g., "pushback_accepted")
    traits_affected: Dict[str, float]     # {trait_name: signed nudge value}
    active_profile_snapshot: Dict[str, float]  # Profile state after nudge
    trigger_context: str                  # Human-readable description
    confidence: float                     # Signal detection confidence
    reinforcement_count: int = 0          # How many prior same-signal entries exist
    session_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_content_dict(self) -> Dict[str, Any]:
        """Serialize for storage in rolodex_entries.content (JSON)."""
        return {
            "signal": self.signal,
            "traits_affected": self.traits_affected,
            "active_profile_snapshot": self.active_profile_snapshot,
            "trigger_context": self.trigger_context,
            "confidence": self.confidence,
            "reinforcement_count": self.reinforcement_count,
        }

    @classmethod
    def from_content_dict(cls, d: Dict[str, Any], **kwargs) -> "DriftEntry":
        """Deserialize from rolodex_entries.content JSON."""
        return cls(
            signal=d.get("signal", ""),
            traits_affected=d.get("traits_affected", {}),
            active_profile_snapshot=d.get("active_profile_snapshot", {}),
            trigger_context=d.get("trigger_context", ""),
            confidence=d.get("confidence", 0.0),
            reinforcement_count=d.get("reinforcement_count", 0),
            **kwargs,
        )


# ─── Persistent Persona State ─────────────────────────────────────────────

@dataclass
class TraitHistoryEntry:
    """A single session's contribution to a trait's evolution."""
    session_id: str
    effective_value: float
    drift_delta: float  # Drift from baseline at session end
    timestamp: str      # ISO format

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraitHistoryEntry":
        return cls(
            session_id=d.get("session_id", ""),
            effective_value=d.get("effective_value", 0.5),
            drift_delta=d.get("drift_delta", 0.0),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class RatchetCandidate:
    """Tracks a trait that may be ready to have its baseline updated.

    When a trait drifts consistently in the same direction for
    `min_sessions` sessions, it becomes a ratchet candidate.
    Once confirmed (by user or auto-threshold), the baseline
    updates and the drift history resets for that trait.
    """
    trait: str
    direction: str                    # "up" or "down"
    consecutive_sessions: int = 0     # Sessions drifting in this direction
    average_delta: float = 0.0        # Mean drift across those sessions
    proposed_baseline: float = 0.0    # Suggested new baseline value
    first_seen: str = ""              # ISO timestamp
    last_seen: str = ""               # ISO timestamp

    # Thresholds for ratchet readiness
    MIN_SESSIONS: int = 5             # Minimum consistent sessions to propose
    MIN_AVG_DELTA: float = 0.02       # Minimum average drift magnitude
    AUTO_APPLY_SESSIONS: int = 50     # Auto-apply without approval after this many

    @property
    def ready(self) -> bool:
        """Is this candidate ready for ratcheting (manual approval)?"""
        return (
            self.consecutive_sessions >= self.MIN_SESSIONS
            and abs(self.average_delta) >= self.MIN_AVG_DELTA
        )

    @property
    def auto_apply(self) -> bool:
        """Has this candidate accumulated enough evidence to auto-apply?"""
        return (
            self.consecutive_sessions >= self.AUTO_APPLY_SESSIONS
            and abs(self.average_delta) >= self.MIN_AVG_DELTA
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trait": self.trait,
            "direction": self.direction,
            "consecutive_sessions": self.consecutive_sessions,
            "average_delta": round(self.average_delta, 4),
            "proposed_baseline": round(self.proposed_baseline, 4),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RatchetCandidate":
        return cls(
            trait=d.get("trait", ""),
            direction=d.get("direction", "up"),
            consecutive_sessions=d.get("consecutive_sessions", 0),
            average_delta=d.get("average_delta", 0.0),
            proposed_baseline=d.get("proposed_baseline", 0.5),
            first_seen=d.get("first_seen", ""),
            last_seen=d.get("last_seen", ""),
        )


@dataclass
class SkillAccessRecord:
    """Records a skill pack's usage within a single session (Phase 4)."""
    session_id: str
    access_count: int
    timestamp: str  # ISO date

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillAccessRecord":
        return cls(
            session_id=d.get("session_id", ""),
            access_count=d.get("access_count", 0),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class PersonaState:
    """Cross-session persistent state for a persona.

    Lives alongside persona.yaml as persona_state.json.
    Updated at session end, loaded on boot.

    The persona.yaml is the *design* — the intentional configuration.
    The persona_state.json is the *evolution* — what actually happened.
    """
    # Last known effective traits (snapshot at session end)
    effective_traits: Dict[str, float] = field(default_factory=dict)

    # Per-trait history: last N sessions of drift data
    trait_history: Dict[str, List[TraitHistoryEntry]] = field(default_factory=dict)

    # Baseline ratchet tracking
    ratchet_candidates: Dict[str, RatchetCandidate] = field(default_factory=dict)

    # Ratchets that have been applied (audit trail)
    ratchet_log: List[Dict[str, Any]] = field(default_factory=list)

    # Session lineage
    total_sessions: int = 0
    last_session_id: str = ""
    last_updated: str = ""

    # Schema version for forward compatibility
    state_version: str = "1.0"

    # How many sessions of history to retain per trait
    MAX_HISTORY_PER_TRAIT: int = 20

    # ─── Phase 4: Portfolio Evolution Fields ─────────────────────
    # Skill pack usage across sessions: {pack_name: [SkillAccessRecord]}
    skill_usage_history: Dict[str, List[SkillAccessRecord]] = field(default_factory=dict)

    # Reflection tracking
    last_reflection_at: str = ""
    reflection_count: int = 0

    # Pending portfolio evolution candidates (persisted across sessions)
    pending_promotions: List[Dict[str, Any]] = field(default_factory=list)
    pending_demotions: List[Dict[str, Any]] = field(default_factory=list)

    # Dismissed candidates (won't suggest again)
    dismissed_candidates: List[str] = field(default_factory=list)

    # Max skill access records per pack
    MAX_HISTORY_PER_PACK: int = 20

    def record_session(
        self,
        session_id: str,
        effective_traits: Dict[str, float],
        baseline_traits: Dict[str, float],
    ) -> List[RatchetCandidate]:
        """Record end-of-session persona state. Returns any new ratchet candidates.

        Args:
            session_id: Current session ID.
            effective_traits: The effective trait values at session end.
            baseline_traits: The persona.yaml baseline trait values.

        Returns:
            List of RatchetCandidates that are newly ready for ratcheting.
        """
        now = datetime.now(timezone.utc).isoformat()
        self.effective_traits = {k: round(v, 4) for k, v in effective_traits.items()}
        self.total_sessions += 1
        self.last_session_id = session_id
        self.last_updated = now

        newly_ready = []

        for trait in VALID_TRAIT_NAMES:
            eff = effective_traits.get(trait, 0.5)
            base = baseline_traits.get(trait, 0.5)
            delta = round(eff - base, 4)

            # Record history
            entry = TraitHistoryEntry(
                session_id=session_id,
                effective_value=round(eff, 4),
                drift_delta=delta,
                timestamp=now,
            )
            if trait not in self.trait_history:
                self.trait_history[trait] = []
            self.trait_history[trait].append(entry)

            # Trim to max history
            if len(self.trait_history[trait]) > self.MAX_HISTORY_PER_TRAIT:
                self.trait_history[trait] = self.trait_history[trait][-self.MAX_HISTORY_PER_TRAIT:]

            # Update ratchet candidates
            self._update_ratchet(trait, delta, eff, now)

        # Check for newly ready candidates
        for trait, candidate in self.ratchet_candidates.items():
            if candidate.ready:
                newly_ready.append(candidate)

        return newly_ready

    def _update_ratchet(
        self, trait: str, delta: float, effective: float, timestamp: str
    ) -> None:
        """Update ratchet tracking for a single trait."""
        if abs(delta) < 0.005:
            # No meaningful drift — reset candidate
            self.ratchet_candidates.pop(trait, None)
            return

        direction = "up" if delta > 0 else "down"
        existing = self.ratchet_candidates.get(trait)

        if existing and existing.direction == direction:
            # Same direction — extend the streak
            existing.consecutive_sessions += 1
            # Running average
            n = existing.consecutive_sessions
            existing.average_delta = (
                existing.average_delta * (n - 1) + delta
            ) / n
            existing.proposed_baseline = round(
                effective - (delta * 0.5), 4  # Split the difference
            )
            existing.last_seen = timestamp
        else:
            # New direction or first observation — start fresh
            self.ratchet_candidates[trait] = RatchetCandidate(
                trait=trait,
                direction=direction,
                consecutive_sessions=1,
                average_delta=delta,
                proposed_baseline=round(effective - (delta * 0.5), 4),
                first_seen=timestamp,
                last_seen=timestamp,
            )

    def apply_ratchet(
        self, trait: str, new_baseline: float
    ) -> Optional[Dict[str, Any]]:
        """Apply a baseline ratchet for a trait. Returns audit log entry or None."""
        candidate = self.ratchet_candidates.get(trait)
        if not candidate:
            return None

        log_entry = {
            "trait": trait,
            "old_baseline": round(new_baseline - candidate.average_delta, 4),
            "new_baseline": round(new_baseline, 4),
            "direction": candidate.direction,
            "sessions_observed": candidate.consecutive_sessions,
            "average_delta": candidate.average_delta,
            "applied_at": datetime.now(timezone.utc).isoformat(),
        }
        self.ratchet_log.append(log_entry)

        # Clear the candidate and trait history (fresh start post-ratchet)
        self.ratchet_candidates.pop(trait, None)
        self.trait_history.pop(trait, None)

        return log_entry

    def get_trait_trend(self, trait: str, last_n: int = 5) -> Optional[Dict[str, Any]]:
        """Get recent trend data for a trait."""
        history = self.trait_history.get(trait, [])
        if not history:
            return None

        recent = history[-last_n:]
        deltas = [h.drift_delta for h in recent]
        values = [h.effective_value for h in recent]

        return {
            "trait": trait,
            "sessions_sampled": len(recent),
            "current_effective": values[-1] if values else None,
            "mean_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0,
            "trend_direction": (
                "up" if sum(1 for d in deltas if d > 0) > len(deltas) / 2
                else "down" if sum(1 for d in deltas if d < 0) > len(deltas) / 2
                else "stable"
            ),
            "min_effective": round(min(values), 4) if values else None,
            "max_effective": round(max(values), 4) if values else None,
        }

    # ─── Phase 4: Reflection Recording ──────────────────────────────

    def record_reflection(self, report) -> None:
        """Record a reflection report's findings into persistent state.

        Updates skill_usage_history, pending promotions/demotions,
        and reflection metadata.

        Args:
            report: A ReflectionReport from session_reflection.py.
        """
        now = datetime.now(timezone.utc).isoformat()
        self.last_reflection_at = now
        self.reflection_count += 1

        # Record skill usage from this session
        for usage in getattr(report, 'skill_packs_used', []):
            pack_name = usage.pack_name if hasattr(usage, 'pack_name') else usage.get("pack_name", "")
            access_count = (
                usage.access_count_this_session
                if hasattr(usage, 'access_count_this_session')
                else usage.get("access_count_this_session", 0)
            )
            session_id = getattr(report, 'session_id', '')

            record = SkillAccessRecord(
                session_id=session_id,
                access_count=access_count,
                timestamp=now[:10],  # Just the date portion
            )

            if pack_name not in self.skill_usage_history:
                self.skill_usage_history[pack_name] = []
            self.skill_usage_history[pack_name].append(record)

            # Trim per-pack history
            if len(self.skill_usage_history[pack_name]) > self.MAX_HISTORY_PER_PACK:
                self.skill_usage_history[pack_name] = (
                    self.skill_usage_history[pack_name][-self.MAX_HISTORY_PER_PACK:]
                )

        # Update pending promotions (merge, don't duplicate)
        for candidate in getattr(report, 'promotion_candidates', []):
            c_dict = candidate.to_dict() if hasattr(candidate, 'to_dict') else candidate
            pack_name = c_dict.get("pack_name", "")
            if pack_name in self.dismissed_candidates:
                continue
            # Replace existing entry for same pack
            self.pending_promotions = [
                p for p in self.pending_promotions
                if p.get("pack_name") != pack_name
            ]
            self.pending_promotions.append(c_dict)

        # Update pending demotions
        for candidate in getattr(report, 'demotion_candidates', []):
            c_dict = candidate.to_dict() if hasattr(candidate, 'to_dict') else candidate
            pack_name = c_dict.get("pack_name", "")
            if pack_name in self.dismissed_candidates:
                continue
            self.pending_demotions = [
                p for p in self.pending_demotions
                if p.get("pack_name") != pack_name
            ]
            self.pending_demotions.append(c_dict)

    def dismiss_candidate(self, pack_name: str) -> bool:
        """Dismiss a promotion/demotion candidate so it won't be suggested again."""
        if pack_name in self.dismissed_candidates:
            return False
        self.dismissed_candidates.append(pack_name)
        # Remove from pending lists
        self.pending_promotions = [
            p for p in self.pending_promotions if p.get("pack_name") != pack_name
        ]
        self.pending_demotions = [
            p for p in self.pending_demotions if p.get("pack_name") != pack_name
        ]
        return True

    # ─── JSON I/O ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "state_version": self.state_version,
            "effective_traits": self.effective_traits,
            "trait_history": {
                trait: [e.to_dict() for e in entries]
                for trait, entries in self.trait_history.items()
            },
            "ratchet_candidates": {
                trait: c.to_dict()
                for trait, c in self.ratchet_candidates.items()
            },
            "ratchet_log": self.ratchet_log,
            "total_sessions": self.total_sessions,
            "last_session_id": self.last_session_id,
            "last_updated": self.last_updated,
            # Phase 4 fields
            "skill_usage_history": {
                pack: [r.to_dict() for r in records]
                for pack, records in self.skill_usage_history.items()
            },
            "last_reflection_at": self.last_reflection_at,
            "reflection_count": self.reflection_count,
            "pending_promotions": self.pending_promotions,
            "pending_demotions": self.pending_demotions,
            "dismissed_candidates": self.dismissed_candidates,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonaState":
        """Deserialize from a parsed JSON dict."""
        state = cls(
            state_version=d.get("state_version", "1.0"),
            effective_traits=d.get("effective_traits", {}),
            total_sessions=d.get("total_sessions", 0),
            last_session_id=d.get("last_session_id", ""),
            last_updated=d.get("last_updated", ""),
            ratchet_log=d.get("ratchet_log", []),
            # Phase 4 fields
            last_reflection_at=d.get("last_reflection_at", ""),
            reflection_count=d.get("reflection_count", 0),
            pending_promotions=d.get("pending_promotions", []),
            pending_demotions=d.get("pending_demotions", []),
            dismissed_candidates=d.get("dismissed_candidates", []),
        )

        # Deserialize trait history
        for trait, entries in d.get("trait_history", {}).items():
            state.trait_history[trait] = [
                TraitHistoryEntry.from_dict(e) for e in entries
            ]

        # Deserialize ratchet candidates
        for trait, cdata in d.get("ratchet_candidates", {}).items():
            state.ratchet_candidates[trait] = RatchetCandidate.from_dict(cdata)

        # Phase 4: Deserialize skill usage history
        for pack, records in d.get("skill_usage_history", {}).items():
            state.skill_usage_history[pack] = [
                SkillAccessRecord.from_dict(r) for r in records
            ]

        return state

    def save(self, path: str) -> None:
        """Write persona state to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            _json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PersonaState":
        """Load persona state from a JSON file. Returns empty state if not found."""
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = _json.load(f)
            return cls.from_dict(data)
        except (_json.JSONDecodeError, IOError):
            return cls()


# ─── Persona Profile (Top-Level) ──────────────────────────────────────────

@dataclass
class PersonaProfile:
    """Complete cognitive profile for a Librarian instance.

    This is the in-memory representation. It can be:
    - Loaded from a persona.yaml (baseline)
    - Overlaid with drift entries from the rolodex (effective)
    - Exported back to YAML for portability
    """
    schema_version: str = SCHEMA_VERSION
    identity: PersonaIdentity = field(default_factory=PersonaIdentity)
    traits: TraitProfile = field(default_factory=TraitProfile)
    domain: DomainEnvelope = field(default_factory=DomainEnvelope)
    triggers: BehavioralTriggers = field(default_factory=BehavioralTriggers)
    drift: DriftConfig = field(default_factory=DriftConfig)
    sharing: SharingConfig = field(default_factory=SharingConfig)
    greeting: GreetingProtocol = field(default_factory=GreetingProtocol)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    rhythm: RhythmConfig = field(default_factory=RhythmConfig)
    memory_priorities: MemoryPriorities = field(default_factory=MemoryPriorities)
    resident_knowledge_path: Optional[str] = None
    resident_budget_tokens: int = 4000
    custom_signals: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=lambda: {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "created_by": "",
        "template_source": None,
        "model_affinity": None,
        "minimum_librarian_version": "1.4.0",
    })

    # ─── Runtime State (not persisted to YAML) ─────────────────────
    _baseline_traits: Optional[TraitProfile] = field(
        default=None, repr=False, compare=False
    )
    _session_drift_total: Dict[str, float] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self):
        """Snapshot baseline traits on creation."""
        if self._baseline_traits is None:
            self._baseline_traits = deepcopy(self.traits)
        if not self._session_drift_total:
            self._session_drift_total = {t: 0.0 for t in VALID_TRAIT_NAMES}

    @property
    def baseline(self) -> TraitProfile:
        """The original baseline traits (from YAML, before drift)."""
        return self._baseline_traits

    def apply_nudge(self, trait: str, nudge: float) -> bool:
        """Apply a single drift nudge to the active profile.

        Respects max_nudge_per_event and max_session_drift guards.
        Returns True if the nudge was applied, False if it was clamped to zero.
        """
        if trait not in VALID_TRAIT_NAMES:
            return False

        # Clamp individual nudge
        clamped_nudge = max(
            -self.drift.max_nudge_per_event,
            min(self.drift.max_nudge_per_event, nudge)
        )

        # Check session drift budget
        current_session_drift = abs(self._session_drift_total.get(trait, 0.0))
        remaining_budget = self.drift.max_session_drift - current_session_drift
        if remaining_budget <= 0:
            return False

        # Reduce nudge to fit within remaining budget
        if abs(clamped_nudge) > remaining_budget:
            clamped_nudge = math.copysign(remaining_budget, clamped_nudge)

        # Apply
        current = self.traits.get(trait)
        self.traits.set(trait, current + clamped_nudge)
        self._session_drift_total[trait] = (
            self._session_drift_total.get(trait, 0.0) + clamped_nudge
        )
        return True

    def apply_drift_overlay(
        self,
        drift_entries: List[DriftEntry],
        current_session_count: int,
        session_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """Apply accumulated drift entries onto the baseline, with decay.

        Args:
            drift_entries: All disposition_drift entries from the rolodex.
            current_session_count: Total session count for decay calculation.
            session_counts: Optional map of entry session_id → session index.

        Returns:
            Dict of net drift applied per trait.
        """
        net_drift: Dict[str, float] = {t: 0.0 for t in VALID_TRAIT_NAMES}

        # Reset traits to baseline before overlaying
        self.traits = deepcopy(self._baseline_traits)

        for entry in drift_entries:
            # Determine sessions elapsed since this drift entry
            entry_session = session_counts.get(entry.session_id, 0) if session_counts else 0
            sessions_elapsed = max(0, current_session_count - entry_session)

            for trait, original_nudge in entry.traits_affected.items():
                if trait not in VALID_TRAIT_NAMES:
                    continue
                effective = self.drift.compute_effective_nudge(
                    original_nudge=original_nudge,
                    reinforcement_count=entry.reinforcement_count,
                    sessions_elapsed=sessions_elapsed,
                )
                net_drift[trait] = net_drift.get(trait, 0.0) + effective

        # Apply net drift to baseline
        for trait, total_nudge in net_drift.items():
            current = self.traits.get(trait)
            self.traits.set(trait, current + total_nudge)

        self.traits.clamp_all()
        return net_drift

    def get_effective_trait(self, trait: str) -> float:
        """Get the current effective value of a trait (baseline + drift)."""
        return self.traits.get(trait)

    def get_drift_delta(self) -> Dict[str, float]:
        """Get the difference between current effective and baseline traits."""
        delta = {}
        for trait in VALID_TRAIT_NAMES:
            effective = self.traits.get(trait)
            baseline = self._baseline_traits.get(trait)
            diff = effective - baseline
            if abs(diff) > 0.001:
                delta[trait] = round(diff, 4)
        return delta

    def should_trigger_fire(
        self, trigger_type: str, trigger_min: float
    ) -> bool:
        """Check if a behavioral trigger should fire given current effective profile.

        Option A: triggers check effective profile, not baseline.
        The min threshold is the safety floor.
        """
        trait_map = {
            "conviction": "conviction",
            "initiative": "initiative",
            "observance": "observance",
        }
        trait = trait_map.get(trigger_type)
        if trait is None:
            return False
        return self.get_effective_trait(trait) >= trigger_min

    # ─── YAML I/O ──────────────────────────────────────────────────

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Serialize to a dict suitable for YAML output."""
        d = {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "traits": self._baseline_traits.to_dict(),  # Always save baseline
            "domain": self.domain.to_dict(),
            "triggers": self.triggers.to_dict(),
            "drift": self.drift.to_dict(),
            "sharing": self.sharing.to_dict(),
            "greeting": self.greeting.to_dict(),
            "attention": self.attention.to_dict(),
            "rhythm": self.rhythm.to_dict(),
            "memory_priorities": self.memory_priorities.to_dict(),
        }
        if self.resident_knowledge_path:
            d["resident_knowledge_path"] = self.resident_knowledge_path
        if self.resident_budget_tokens != 4000:
            d["resident_budget_tokens"] = self.resident_budget_tokens
        if self.custom_signals:
            d["custom_signals"] = self.custom_signals
        d["meta"] = self.meta
        return d

    def save_yaml(self, path: str) -> None:
        """Write the persona baseline to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(
                self.to_yaml_dict(),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    @classmethod
    def from_yaml(cls, path: str) -> "PersonaProfile":
        """Load a persona from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Persona file not found: {path}")

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaProfile":
        """Construct a PersonaProfile from a parsed YAML dict."""
        profile = cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            identity=PersonaIdentity.from_dict(data.get("identity", {})),
            traits=TraitProfile.from_dict(data.get("traits", {})),
            domain=DomainEnvelope.from_dict(data.get("domain", {})),
            triggers=BehavioralTriggers.from_dict(data.get("triggers", {})),
            drift=DriftConfig.from_dict(data.get("drift", {})),
            sharing=SharingConfig.from_dict(data.get("sharing", {})),
            greeting=GreetingProtocol.from_dict(data.get("greeting", {})),
            attention=AttentionConfig.from_dict(data.get("attention", {})),
            rhythm=RhythmConfig.from_dict(data.get("rhythm", {})),
            memory_priorities=MemoryPriorities.from_dict(data.get("memory_priorities", {})),
            resident_knowledge_path=data.get("resident_knowledge_path"),
            resident_budget_tokens=data.get("resident_budget_tokens", 4000),
            custom_signals=data.get("custom_signals", []),
            meta=data.get("meta", {}),
        )
        # Snapshot baseline after loading
        profile._baseline_traits = deepcopy(profile.traits)
        profile._session_drift_total = {t: 0.0 for t in VALID_TRAIT_NAMES}
        return profile

    @classmethod
    def default(cls) -> "PersonaProfile":
        """Create a default persona with neutral traits (all 0.5)."""
        return cls()

    def export_effective_as_yaml(self, path: str) -> None:
        """Export the current effective profile (baseline + drift) as a new YAML.

        Useful for creating a new template from a well-calibrated instance.
        """
        export = deepcopy(self)
        export._baseline_traits = deepcopy(self.traits)  # Bake drift into baseline
        export.save_yaml(path)

    # ─── Persistent State Management ──────────────────────────────

    _state: Optional[PersonaState] = field(
        default=None, repr=False, compare=False
    )

    @property
    def state(self) -> Optional[PersonaState]:
        """The persistent cross-session state, if loaded."""
        return self._state

    def load_state(self, path: str) -> PersonaState:
        """Load persistent persona state from disk.

        If the state file contains effective_traits from a prior session,
        those are available for reference but do NOT override the
        baseline + drift overlay calculation. The state is observational,
        not authoritative over the drift system.
        """
        self._state = PersonaState.load(path)
        return self._state

    def save_state(self, path: str, session_id: str) -> Dict[str, Any]:
        """Snapshot current persona state to disk at session end.

        Records effective traits, updates trait history, evaluates
        ratchet candidates. Returns a summary of what was persisted.

        Args:
            path: File path for persona_state.json.
            session_id: Current session ID for lineage tracking.

        Returns:
            Dict with persistence summary including any ratchet candidates.
        """
        if self._state is None:
            self._state = PersonaState()

        effective = self.traits.to_dict()
        baseline = self._baseline_traits.to_dict()

        newly_ready = self._state.record_session(
            session_id=session_id,
            effective_traits=effective,
            baseline_traits=baseline,
        )

        self._state.save(path)

        summary = {
            "persisted": True,
            "total_sessions": self._state.total_sessions,
            "effective_traits": {k: round(v, 4) for k, v in effective.items()},
            "drift_delta": self.get_drift_delta(),
        }

        if newly_ready:
            summary["ratchet_candidates_ready"] = [
                c.to_dict() for c in newly_ready
            ]

        return summary

    def apply_ratchet(
        self, trait: str, persona_yaml_path: str
    ) -> Optional[Dict[str, Any]]:
        """Apply a baseline ratchet: update persona.yaml baseline for a trait.

        This bakes stable drift into the baseline so the trait starts
        closer to its natural level. The drift entries in the rolodex
        then have less work to do (and decay matters less).

        Args:
            trait: The trait to ratchet.
            persona_yaml_path: Path to persona.yaml to update.

        Returns:
            Audit log entry, or None if no candidate exists.
        """
        if self._state is None:
            return None

        candidate = self._state.ratchet_candidates.get(trait)
        if not candidate:
            return None

        new_baseline = candidate.proposed_baseline

        # Update the in-memory baseline
        self._baseline_traits.set(trait, new_baseline)

        # Update the effective trait to maintain the same delta relationship
        # (the overlay will recalculate on next boot)
        self.traits.set(trait, new_baseline + (
            self.traits.get(trait) - self._baseline_traits.get(trait)
        ) if False else new_baseline)  # Just set to new baseline; drift will re-overlay

        # Record in state
        log_entry = self._state.apply_ratchet(trait, new_baseline)

        # Persist updated baseline to persona.yaml
        self.save_yaml(persona_yaml_path)

        return log_entry

    def get_evolution_summary(self) -> Optional[Dict[str, Any]]:
        """Get a summary of persona evolution across sessions."""
        if self._state is None or self._state.total_sessions == 0:
            return None

        trends = {}
        for trait in VALID_TRAIT_NAMES:
            trend = self._state.get_trait_trend(trait)
            if trend:
                trends[trait] = trend

        ready_ratchets = {
            trait: c.to_dict()
            for trait, c in self._state.ratchet_candidates.items()
            if c.ready
        }

        pending_ratchets = {
            trait: c.to_dict()
            for trait, c in self._state.ratchet_candidates.items()
            if not c.ready and c.consecutive_sessions >= 2
        }

        return {
            "total_sessions": self._state.total_sessions,
            "last_session": self._state.last_session_id,
            "last_updated": self._state.last_updated,
            "trends": trends,
            "ratchets_ready": ready_ratchets if ready_ratchets else None,
            "ratchets_pending": pending_ratchets if pending_ratchets else None,
            "ratchet_history": (
                self._state.ratchet_log[-5:]
                if self._state.ratchet_log else None
            ),
        }

    # ─── Display ───────────────────────────────────────────────────

    def format_ascii_profile(self, show_drift: bool = True) -> str:
        """Render the cognitive profile as ASCII bar chart."""
        lines = []
        lines.append(f"╔══ {self.identity.name} ({self.identity.role}) ══╗")
        lines.append("")

        bar_width = 20
        for trait in VALID_TRAIT_NAMES:
            effective = self.traits.get(trait)
            baseline = self._baseline_traits.get(trait)
            drift_val = effective - baseline

            filled = int(effective * bar_width)
            empty = bar_width - filled
            bar = "█" * filled + "░" * empty

            label = f"{trait:<16}"
            value = f"{effective:.2f}"

            if show_drift and abs(drift_val) > 0.001:
                direction = "↑" if drift_val > 0 else "↓"
                drift_str = f"  ({direction}{abs(drift_val):.2f} from {baseline:.2f})"
            else:
                drift_str = ""

            lines.append(f"  {label} {bar}  {value}{drift_str}")

        lines.append("")
        lines.append(f"  Domain: {self.domain.primary}")
        if self.domain.secondary:
            lines.append(f"  Also: {', '.join(self.domain.secondary)}")

        return "\n".join(lines)


# ─── Persona Registry ────────────────────────────────────────────────────

@dataclass
class PersonaRegistryEntry:
    """A single persona's metadata from the registry."""
    key: str                          # e.g., "default", "custom", "specialist"
    file: str                         # YAML filename relative to personas/
    display_name: str                 # e.g., "Chief Librarian"
    short_label: str                  # e.g., "Chief" (for response header)
    description: str
    domain_filter: Dict[str, Any]     # include: [...], writes_as: "..."
    subfolders: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    detection_patterns: List[str] = field(default_factory=list)
    emoji: Optional[str] = None
    requires_disclaimer: bool = False
    disclaimer_text: str = ""

    @classmethod
    def from_dict(cls, key: str, d: Dict[str, Any]) -> "PersonaRegistryEntry":
        return cls(
            key=key,
            file=d.get("file", f"{key}.yaml"),
            display_name=d.get("display_name", key.title()),
            short_label=d.get("short_label", key.title()),
            description=d.get("description", ""),
            domain_filter=d.get("domain_filter", {
                "include": [key, "shared"],
                "writes_as": key,
            }),
            subfolders=d.get("subfolders", []),
            skills=d.get("skills", []),
            detection_patterns=d.get("detection_patterns", []),
            emoji=d.get("emoji"),
            requires_disclaimer=d.get("requires_disclaimer", False),
            disclaimer_text=d.get("disclaimer_text", ""),
        )

    def to_selection_dict(self) -> Dict[str, str]:
        """Format for AskUserQuestion option display."""
        return {
            "label": self.display_name,
            "description": self.description,
        }


class PersonaRegistry:
    """Loads and manages the multi-persona registry.

    The registry lives at personas/persona_registry.yaml and maps
    persona keys to their YAML files, domain scoping rules, and
    display metadata.
    """

    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self.personas_dir = str(Path(registry_path).parent)
        self._entries: Dict[str, PersonaRegistryEntry] = {}
        self._default_key: str = "default"
        self._domain_rules: Dict[str, Any] = {}
        self._template_creation_enabled: bool = True
        self._load()

    def _load(self) -> None:
        """Parse the registry YAML."""
        p = Path(self.registry_path)
        if not p.exists():
            return

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._default_key = data.get("default", "default")
        self._domain_rules = data.get("domain_rules", {})
        self._template_creation_enabled = data.get("template_creation_enabled", True)

        for key, pdata in data.get("personas", {}).items():
            self._entries[key] = PersonaRegistryEntry.from_dict(key, pdata)

    @property
    def default_key(self) -> str:
        return self._default_key

    @property
    def domain_rules(self) -> Dict[str, Any]:
        return self._domain_rules

    @property
    def template_creation_enabled(self) -> bool:
        return self._template_creation_enabled

    def list_personas(self) -> List[PersonaRegistryEntry]:
        """Return all registered personas in stable order."""
        return list(self._entries.values())

    def get(self, key: str) -> Optional[PersonaRegistryEntry]:
        """Look up a persona by key."""
        return self._entries.get(key)

    def get_default(self) -> Optional[PersonaRegistryEntry]:
        """Return the default persona entry."""
        return self._entries.get(self._default_key)

    def load_persona(self, key: str) -> Optional[PersonaProfile]:
        """Load a PersonaProfile from the registry by key.

        Returns None if the key doesn't exist or the YAML is missing.
        """
        entry = self._entries.get(key)
        if not entry:
            return None

        yaml_path = str(Path(self.personas_dir) / entry.file)
        if not Path(yaml_path).exists():
            return None

        return PersonaProfile.from_yaml(yaml_path)

    def get_persona_yaml_path(self, key: str) -> Optional[str]:
        """Return the full path to a persona's YAML file."""
        entry = self._entries.get(key)
        if not entry:
            return None
        return str(Path(self.personas_dir) / entry.file)

    def get_persona_state_path(self, key: str) -> Optional[str]:
        """Return the path where this persona's state file lives."""
        entry = self._entries.get(key)
        if not entry:
            return None
        state_file = entry.file.replace(".yaml", "_state.json")
        return str(Path(self.personas_dir) / state_file)

    def get_domain_filter(self, key: str) -> List[str]:
        """Return the domain include list for a persona."""
        entry = self._entries.get(key)
        if not entry:
            return ["shared"]
        return entry.domain_filter.get("include", [key, "shared"])

    def get_write_domain(self, key: str) -> str:
        """Return the domain tag this persona writes entries as."""
        entry = self._entries.get(key)
        if not entry:
            return "shared"
        return entry.domain_filter.get("writes_as", key)

    def detect_persona(self, text: str) -> Optional[str]:
        """Attempt to auto-detect which persona a message is for.

        Returns persona key if a strong match is found, None otherwise.
        Uses the detection_patterns from each persona entry.
        """
        import re
        scores: Dict[str, int] = {}

        for key, entry in self._entries.items():
            count = 0
            for pattern in entry.detection_patterns:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    count += len(matches)
                except re.error:
                    continue
            if count > 0:
                scores[key] = count

        if not scores:
            return None

        # Return the highest-scoring persona, but only if it's clearly ahead
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) == 1:
            return sorted_scores[0][0]
        if sorted_scores[0][1] >= sorted_scores[1][1] * 2:
            return sorted_scores[0][0]

        return None  # Ambiguous — don't auto-switch

    def format_selection_options(self) -> List[Dict[str, str]]:
        """Format all personas for AskUserQuestion display."""
        options = []
        for entry in self._entries.values():
            options.append(entry.to_selection_dict())
        return options
