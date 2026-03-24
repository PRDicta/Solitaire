"""
The Librarian — Disposition Filter
Live signal detection and nudge engine that runs alongside every ingest call.

The filter examines ingested content for dispositional signals — cues that
should nudge the active cognitive profile. It writes drift entries to the
rolodex and applies micro-adjustments to the in-memory persona in real time.

This is the engine behind "over time, your Librarian just fits better."
"""
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from .persona import (
    PersonaProfile,
    DriftEntry,
    VALID_TRAIT_NAMES,
)


# ─── Signal Definitions ────────────────────────────────────────────────────

@dataclass
class SignalDefinition:
    """A dispositional signal the filter watches for."""
    key: str                                  # Unique signal identifier
    affects: List[str]                        # Trait names affected
    direction: List[str]                      # "increase" or "decrease" per trait
    weight: float = 1.0                       # Multiplier on nudge magnitude
    patterns: List[str] = field(default_factory=list)  # Regex patterns for detection
    requires_role: Optional[str] = None       # "user", "assistant", or None (both)
    requires_flag: Optional[str] = None       # CLI flag (e.g., "--corrects")
    description: str = ""
    min_confidence: Optional[float] = None    # Per-signal confidence threshold (overrides persona default)


# ─── Default Signal Library ────────────────────────────────────────────────

DEFAULT_SIGNALS: List[SignalDefinition] = [
    SignalDefinition(
        key="correction_received",
        affects=["conviction", "observance"],
        direction=["decrease", "increase"],
        weight=1.0,
        requires_flag="corrects",
        description="User corrected a wrong fact — confidence was too high",
    ),
    SignalDefinition(
        key="positive_acknowledgment",
        affects=["conviction", "initiative"],
        direction=["increase", "increase"],
        weight=0.5,
        patterns=[
            r"\b(exactly|precisely|perfect|nailed it|spot on|that'?s? (right|it|correct))\b",
            r"^(yes|yep|yup|yeah)[.!,\s]*$",
            r"\bgood (point|call|catch)\b",
        ],
        requires_role="user",
        description="User affirmed the Librarian's output or approach",
    ),
    SignalDefinition(
        key="pushback_accepted",
        affects=["conviction"],
        direction=["increase"],
        weight=1.2,
        patterns=[
            r"\b(you'?re right|good point|fair point|I stand corrected)\b",
            r"\b(didn'?t (think|consider) (of |about )?that)\b",
            r"\b(actually[,.]? (yeah|yes|true|fair))\b",
        ],
        requires_role="user",
        description="User accepted the Librarian's pushback — conviction rewarded",
    ),
    SignalDefinition(
        key="pushback_rejected",
        affects=["conviction"],
        direction=["decrease"],
        weight=0.8,
        patterns=[
            r"\b(just do it|I (hear|understand) you but)\b",
            r"\b(no[,.]? (just|I want|do it))\b",
            r"\b(I disagree|that'?s not|don'?t (agree|think so))\b",
            r"\b(override|overrule|ignore that)\b",
        ],
        requires_role="user",
        description="User rejected pushback — conviction should soften",
    ),
    SignalDefinition(
        key="user_requests_more_detail",
        affects=["observance"],
        direction=["increase"],
        weight=0.6,
        patterns=[
            r"\b(tell me more|expand on|elaborate|go deeper|more detail)\b",
            r"\b(can you explain|what do you mean|unpack)\b",
        ],
        requires_role="user",
        description="User wants more depth — increase observance",
    ),
    SignalDefinition(
        key="user_requests_brevity",
        affects=["assertiveness"],
        direction=["increase"],
        weight=0.6,
        patterns=[
            r"\b(keep it (short|brief|concise)|too (much|long|verbose))\b",
            r"\b(shorter|tl;?dr|bottom line|cut to)\b",
        ],
        requires_role="user",
        description="User wants conciseness — increase assertiveness (more direct)",
    ),
    SignalDefinition(
        key="user_override_without_explanation",
        affects=["assertiveness", "conviction"],
        direction=["decrease", "decrease"],
        weight=0.8,
        patterns=[
            r"^(no|nope|wrong)[.!]?\s*$",
            r"^(just do (it|what I (said|asked)))[.!]?\s*$",
        ],
        requires_role="user",
        description="User overrode without reasoning — pull back",
    ),
    SignalDefinition(
        key="warmth_appreciated",
        affects=["warmth"],
        direction=["increase"],
        weight=0.6,
        patterns=[
            r"\b(thank(s| you)|appreciate|that (means|helps) a lot)\b",
            r"\b(beautifully said|well (said|put))\b",
        ],
        requires_role="user",
        description="User responded positively to emotional tone",
    ),
    SignalDefinition(
        key="humor_landed",
        affects=["humor"],
        direction=["increase"],
        weight=0.5,
        patterns=[
            r"\b(lol|haha|hah|😂|🤣|that'?s funny|love that)\b",
            r"\b(laughing|cracking up|hilarious)\b",
        ],
        requires_role="user",
        description="User responded positively to humor",
    ),
    SignalDefinition(
        key="ai_tone_flagged",
        affects=["assertiveness", "observance"],
        direction=["increase", "increase"],
        weight=1.3,
        patterns=[
            r"\b(sounds? (like )?AI|AI[- ]?(ish|like|generated|written|sounding))\b",
            r"\b(too (robotic|formal|generic|corporate|polished))\b",
            r"\b(doesn'?t sound (human|natural|like (you|me|a person)))\b",
            r"\b(chat ?gpt|gpt[- ]?(ish|like|vibes?))\b",
            r"\b(reads? like a (bot|machine|language model))\b",
        ],
        requires_role="user",
        description="User flagged output as AI-sounding — increase directness and scrutiny of own writing",
    ),
    SignalDefinition(
        key="user_sharing_expertise",
        affects=["empathy", "observance"],
        direction=["increase", "increase"],
        weight=0.5,
        patterns=[
            r"\b(in my experience|I('ve| have) (found|learned|noticed|seen))\b",
            r"\b(what (actually|really) (happens|works)|the (trick|thing) is)\b",
            r"\b(I('ve| have) been (doing|working on|building))\b",
            r"\b(when I was|back when I|years ago I)\b",
        ],
        requires_role="user",
        description="User sharing domain expertise or personal experience — increase attentiveness, creates curiosity opportunity (cat 22)",
    ),
    SignalDefinition(
        key="user_low_energy",
        affects=["assertiveness"],
        direction=["increase"],
        weight=0.4,
        patterns=[
            r"^(ok|okay|k|sure|fine|got it|yep|yup|yeah|right|mhm)[.!,]?\s*$",
            r"^(sounds good|works for me|go ahead)[.!,]?\s*$",
        ],
        requires_role="user",
        description="User sent a brief acknowledgment — increase directness, reduce verbosity (cat 19 energy matching)",
    ),
]


# ─── Disposition Filter ────────────────────────────────────────────────────

class DispositionFilter:
    """Live signal detection and nudge engine.

    Sits alongside the ingest pipeline. For every ingested message,
    it checks for dispositional signals and applies micro-adjustments
    to the active persona profile.
    """

    def __init__(
        self,
        persona: PersonaProfile,
        signals: Optional[List[SignalDefinition]] = None,
        session_id: str = "",
    ):
        self.persona = persona
        self.signals = list(signals) if signals else list(DEFAULT_SIGNALS)
        self.session_id = session_id
        self._signal_map: Dict[str, SignalDefinition] = {
            s.key: s for s in self.signals
        }
        # Compiled regex patterns for performance
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for sig in self.signals:
            if sig.patterns:
                self._compiled_patterns[sig.key] = [
                    re.compile(p, re.IGNORECASE) for p in sig.patterns
                ]

    def evaluate(
        self,
        content: str,
        role: str,
        flags: Optional[Dict[str, Any]] = None,
        reinforcement_counts: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> List[DriftEntry]:
        """Evaluate a message for dispositional signals.

        Args:
            content: The message text being ingested.
            role: "user" or "assistant".
            flags: Optional CLI flags (e.g., {"corrects": "entry-id"}).
            reinforcement_counts: Prior counts per signal+trait from rolodex.
                Format: {signal_key: {trait: count}}

        Returns:
            List of DriftEntry objects to be stored in the rolodex.
            Empty list if no signals detected.
        """
        flags = flags or {}
        reinforcement_counts = reinforcement_counts or {}
        detected: List[DriftEntry] = []

        for signal in self.signals:
            # Role filter
            if signal.requires_role and signal.requires_role != role:
                continue

            # Flag-based detection
            if signal.requires_flag:
                if signal.requires_flag not in flags:
                    continue
                confidence = 1.0  # Flag-based signals are high confidence
            else:
                # Pattern-based detection
                confidence = self._check_patterns(signal.key, content)
                # Per-signal threshold overrides persona default
                threshold = signal.min_confidence if signal.min_confidence is not None else self.persona.drift.signal_threshold
                if confidence < threshold:
                    continue

            # Build drift entry
            traits_affected = {}
            snapshot = {}

            for trait, direction in zip(signal.affects, signal.direction):
                if trait not in VALID_TRAIT_NAMES:
                    continue

                # Compute nudge magnitude
                base_nudge = self.persona.drift.max_nudge_per_event * signal.weight
                if direction == "decrease":
                    base_nudge = -base_nudge

                # Get reinforcement count for this signal+trait
                r_count = reinforcement_counts.get(signal.key, {}).get(trait, 0)

                # Apply nudge to active profile
                applied = self.persona.apply_nudge(trait, base_nudge)
                if applied:
                    traits_affected[trait] = round(base_nudge, 6)
                    snapshot[trait] = round(self.persona.traits.get(trait), 4)

            if traits_affected:
                # Get overall reinforcement count (max across affected traits)
                max_reinforcement = max(
                    reinforcement_counts.get(signal.key, {}).get(t, 0)
                    for t in traits_affected
                ) if reinforcement_counts.get(signal.key) else 0

                entry = DriftEntry(
                    signal=signal.key,
                    traits_affected=traits_affected,
                    active_profile_snapshot=snapshot,
                    trigger_context=self._build_context_string(
                        signal, content, role
                    ),
                    confidence=confidence,
                    reinforcement_count=max_reinforcement + 1,  # This is a new reinforcement
                    session_id=self.session_id,
                    created_at=datetime.utcnow(),
                )
                detected.append(entry)

        return detected

    def _check_patterns(self, signal_key: str, content: str) -> float:
        """Check content against compiled patterns for a signal.

        Returns a confidence score (0.0-1.0).
        Multiple pattern matches increase confidence.
        """
        patterns = self._compiled_patterns.get(signal_key, [])
        if not patterns:
            return 0.0

        matches = sum(1 for p in patterns if p.search(content))
        if matches == 0:
            return 0.0

        # Confidence scales with number of matching patterns, caps at 1.0
        return min(1.0, 0.6 + (matches - 1) * 0.15)

    def _build_context_string(
        self, signal: SignalDefinition, content: str, role: str
    ) -> str:
        """Build a human-readable context string for the drift entry."""
        # Truncate content for storage
        preview = content[:120].replace("\n", " ")
        if len(content) > 120:
            preview += "..."
        return f"{role} message triggered '{signal.key}': \"{preview}\""

    def get_reinforcement_counts_query(self) -> str:
        """Return the SQL query to fetch reinforcement counts from the rolodex.

        The caller (librarian.py or CLI) runs this against the DB and passes
        results to evaluate() as reinforcement_counts.
        """
        return """
            SELECT
                json_extract(content, '$.signal') as signal,
                json_each.key as trait,
                COUNT(*) as count
            FROM rolodex_entries,
                 json_each(json_extract(content, '$.traits_affected'))
            WHERE category = 'disposition_drift'
            GROUP BY signal, trait
        """

    def load_custom_signals(self, signal_dicts: List[Dict[str, Any]]) -> int:
        """Load additional signal definitions from persona.yaml custom_signals.

        Validates each signal definition:
        - key is required and must not collide with existing signals
        - affects and direction must have matching lengths
        - all affected traits must be valid trait names
        - patterns are compiled to regex (invalid patterns are skipped)

        Returns:
            Number of signals successfully loaded.
        """
        loaded = 0
        for d in signal_dicts:
            key = d.get("key")
            if not key:
                continue  # Skip signals without a key

            # Skip duplicates — custom signals can override defaults by key
            affects = d.get("affects", [])
            direction = d.get("direction", [])

            # Validate affects/direction alignment
            if len(affects) != len(direction):
                continue  # Mismatched — skip rather than crash

            # Validate trait names
            valid_affects = []
            valid_direction = []
            for trait, dire in zip(affects, direction):
                if trait in VALID_TRAIT_NAMES and dire in ("increase", "decrease"):
                    valid_affects.append(trait)
                    valid_direction.append(dire)

            if not valid_affects:
                continue  # No valid traits to affect

            # Compile patterns, skip invalid regex
            raw_patterns = d.get("patterns", [])
            compiled = []
            valid_patterns = []
            for p in raw_patterns:
                try:
                    compiled.append(re.compile(p, re.IGNORECASE))
                    valid_patterns.append(p)
                except re.error:
                    continue  # Skip bad regex

            # Need either patterns or a flag — otherwise signal can never fire
            requires_flag = d.get("requires_flag")
            if not valid_patterns and not requires_flag:
                continue

            sig = SignalDefinition(
                key=key,
                affects=valid_affects,
                direction=valid_direction,
                weight=d.get("weight", 1.0),
                patterns=valid_patterns,
                requires_role=d.get("requires_role"),
                requires_flag=requires_flag,
                description=d.get("description", ""),
                min_confidence=d.get("min_confidence", 0.5),  # Custom signals default to 0.5 (trusted)
            )

            # If key exists, replace it (custom overrides default)
            if key in self._signal_map:
                self.signals = [s for s in self.signals if s.key != key]

            self.signals.append(sig)
            self._signal_map[sig.key] = sig
            if compiled:
                self._compiled_patterns[sig.key] = compiled

            loaded += 1

        return loaded

    def get_signal_inventory(self) -> Dict[str, Dict[str, Any]]:
        """Return a summary of all loaded signals (default + custom).

        Useful for CLI display and debugging.
        """
        inventory = {}
        default_keys = {s.key for s in DEFAULT_SIGNALS}
        for sig in self.signals:
            inventory[sig.key] = {
                "affects": list(zip(sig.affects, sig.direction)),
                "weight": sig.weight,
                "pattern_count": len(sig.patterns),
                "requires_role": sig.requires_role,
                "requires_flag": sig.requires_flag,
                "source": "default" if sig.key in default_keys else "custom",
                "description": sig.description,
            }
        return inventory
