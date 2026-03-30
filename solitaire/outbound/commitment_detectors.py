"""Layer 4: Commitment adherence detectors.

Detect violations of persona behavioral commitments: diplomatic preambles
(when conviction is high), position collapse (capitulation without reasoning),
and hedging overload (excessive hedge phrases for a high-conviction persona).

Layers 4 detectors are gated on persona traits (conviction, assertiveness)
and some require transcript context (prior assistant turn for collapse detection).

No LLM calls. Regex matching, word counting, and trait comparison.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from solitaire.outbound.surface_detectors import RE_DIPLOMATIC_PREAMBLE


@dataclass
class CommitmentDetection:
    """A single commitment adherence detection."""
    category: str           # e.g., "diplomatic_preamble_commitment"
    severity: str           # "info" | "warning"
    count: int              # occurrences
    samples: List[str]      # up to 3 matched snippets
    detail: str             # human-readable explanation
    score: Optional[float]  # metric where applicable
    confidence: str = "high"  # "high" | "low" — low routes to model verification


# ═══════════════════════════════════════════════════════════════════════════
# PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

# Capitulation markers — signs of position collapse
RE_CAPITULATION = [
    re.compile(r"\byou'?re (?:absolutely |completely )?right\b", re.IGNORECASE),
    re.compile(r"\bI (?:stand )?correct(?:ed)?\b", re.IGNORECASE),
    re.compile(r"\bI (?:should have|shouldn'?t have)\b", re.IGNORECASE),
    re.compile(r"\bmy (?:mistake|apologi(?:es|ze)|bad)\b", re.IGNORECASE),
    re.compile(r"\b(?:fair enough|good point|you make a (?:good |fair |valid )?point)\b", re.IGNORECASE),
    re.compile(r"\bI (?:agree|concede|concur)\b", re.IGNORECASE),
    re.compile(r"\b(?:actually|on reflection),? (?:you'?re|that'?s) (?:right|correct|fair)\b", re.IGNORECASE),
]

# Prior-turn position markers — signs the assistant held a position
RE_PRIOR_POSITION = [
    re.compile(r"\bI (?:think|believe|would (?:argue|suggest|recommend))\b", re.IGNORECASE),
    re.compile(r"\bthe (?:better|right|correct) (?:approach|way|call)\b", re.IGNORECASE),
    re.compile(r"\bI'?d (?:push back|disagree|argue|recommend against)\b", re.IGNORECASE),
    re.compile(r"\b(?:should|shouldn'?t|must|need to)\b", re.IGNORECASE),
]

# Hedge phrases
HEDGE_PHRASES = [
    re.compile(r"\bperhaps\b", re.IGNORECASE),
    re.compile(r"\bmaybe\b", re.IGNORECASE),
    re.compile(r"\bmight (?:be |want )\b", re.IGNORECASE),
    re.compile(r"\bcould (?:be |potentially )\b", re.IGNORECASE),
    re.compile(r"\bit seems (?:like |that )?\b", re.IGNORECASE),
    re.compile(r"\bI'?m not (?:sure|certain)\b", re.IGNORECASE),
    re.compile(r"\bpossibly\b", re.IGNORECASE),
    re.compile(r"\barguably\b", re.IGNORECASE),
    re.compile(r"\bit (?:could|might) be (?:worth|that)\b", re.IGNORECASE),
    re.compile(r"\bone could argue\b", re.IGNORECASE),
    re.compile(r"\bthere'?s an argument (?:for|that)\b", re.IGNORECASE),
    re.compile(r"\bin some (?:cases|ways|sense)\b", re.IGNORECASE),
    re.compile(r"\bto some (?:extent|degree)\b", re.IGNORECASE),
]


def _snippet(text: str, match: re.Match, max_len: int = 60) -> str:
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

def detect_diplomatic_preamble_commitment(
    text: str,
    conviction: float = 0.5,
) -> Optional[CommitmentDetection]:
    """Diplomatic preamble gated on conviction.

    Layer 1 catches diplomatic preambles as a surface tell (always fires).
    Layer 4 gates on conviction > 0.6 — a high-conviction persona using
    diplomatic preambles is a commitment violation, not just a style issue.
    """
    if conviction <= 0.6:
        return None

    matches = list(RE_DIPLOMATIC_PREAMBLE.finditer(text))
    if not matches:
        return None

    samples = [_snippet(text, m) for m in matches[:3]]
    return CommitmentDetection(
        category="diplomatic_preamble_commitment",
        severity="warning",
        count=len(matches),
        samples=samples,
        detail=f"{len(matches)} diplomatic preamble(s) from a persona with "
               f"conviction={conviction:.2f}. High-conviction personas state "
               f"disagreement directly.",
        score=conviction,
    )


def detect_position_collapse(
    text: str,
    prior_assistant_text: str = "",
    conviction: float = 0.5,
) -> Optional[CommitmentDetection]:
    """Detect capitulation without reasoning.

    Compares current response against prior assistant turn. If the prior
    turn held a position and the current turn capitulates without new
    evidence or reasoning, flag it.

    Only fires when conviction > 0.6 and prior turn exists.
    """
    if conviction <= 0.6 or not prior_assistant_text:
        return None

    # Check if prior turn held a position
    prior_had_position = any(
        p.search(prior_assistant_text) for p in RE_PRIOR_POSITION
    )
    if not prior_had_position:
        return None

    # Check for capitulation markers in current turn
    capitulation_matches = []
    for pattern in RE_CAPITULATION:
        for m in pattern.finditer(text):
            capitulation_matches.append(m)

    if not capitulation_matches:
        return None

    # Check if the capitulation comes with reasoning (because/since/given)
    reasoning_pattern = re.compile(
        r"\b(?:because|since|given that|the reason|after (?:reviewing|looking|considering))\b",
        re.IGNORECASE,
    )
    has_reasoning = bool(reasoning_pattern.search(text))

    if has_reasoning:
        # Capitulation with reasoning is fine — the persona was genuinely persuaded
        return None

    samples = [_snippet(text, m) for m in capitulation_matches[:3]]
    # Position collapse detection is inherently ambiguous — the structural
    # check for reasoning (because/since/given) is porous. The model may
    # have legitimate reasons that don't start with a conjunction. Always
    # route to model verification.
    return CommitmentDetection(
        category="position_collapse",
        severity="warning",
        count=len(capitulation_matches),
        samples=samples,
        detail=f"Position reversal without reasoning. Prior turn held a position; "
               f"current turn capitulates ({len(capitulation_matches)} marker(s)) "
               f"without citing new evidence. Conviction={conviction:.2f}.",
        score=conviction,
        confidence="low",
    )


def detect_hedging_overload(
    text: str,
    conviction: float = 0.5,
    threshold_per_100w: float = 3.0,
) -> Optional[CommitmentDetection]:
    """Detect excessive hedging for a high-conviction persona.

    Counts hedge phrases per 100 words. Only fires when conviction > 0.7.
    """
    if conviction <= 0.7:
        return None

    word_count = len(text.split())
    if word_count < 30:
        return None

    hedge_matches = []
    for pattern in HEDGE_PHRASES:
        for m in pattern.finditer(text):
            hedge_matches.append(m)

    hedge_count = len(hedge_matches)
    per_100w = (hedge_count / word_count) * 100

    if per_100w <= threshold_per_100w:
        return None

    samples = [_snippet(text, m) for m in hedge_matches[:3]]
    severity = "warning" if per_100w > threshold_per_100w * 1.5 else "info"

    return CommitmentDetection(
        category="hedging_overload",
        severity=severity,
        count=hedge_count,
        samples=samples,
        detail=f"{hedge_count} hedge phrase(s) in {word_count} words "
               f"({per_100w:.1f}/100w, threshold: {threshold_per_100w}/100w). "
               f"Conviction={conviction:.2f} persona should commit to claims.",
        score=per_100w,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_commitment_scan(
    text: str,
    prior_assistant_text: str = "",
    conviction: float = 0.5,
) -> List[CommitmentDetection]:
    """Run all commitment adherence detectors. Returns list of detections.

    Args:
        text: The assistant's response (preprocessed).
        prior_assistant_text: The previous assistant response (for collapse detection).
        conviction: Persona conviction trait (0-1).
    """
    results = []

    d = detect_diplomatic_preamble_commitment(text, conviction)
    if d is not None:
        results.append(d)

    d = detect_position_collapse(text, prior_assistant_text, conviction)
    if d is not None:
        results.append(d)

    d = detect_hedging_overload(text, conviction)
    if d is not None:
        results.append(d)

    return results
