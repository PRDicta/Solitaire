"""Layer 3: Persona drift detectors.

Detect mismatches between the assistant's output and the persona's
configured traits and rhythm. These detectors are gated on persona
config values — they only fire when the persona has traits that make
the detected pattern a violation.

No LLM calls. Word counting, regex matching, and trait comparison.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PersonaDriftDetection:
    """A single persona drift detection."""
    category: str           # e.g., "verbosity_mismatch"
    severity: str           # "info" | "warning"
    count: int              # occurrences or 1 for threshold-based
    samples: List[str]      # descriptive samples
    detail: str             # human-readable explanation
    score: Optional[float]  # ratio or metric where applicable
    confidence: str = "high"  # "high" | "low" — low routes to model verification


# ═══════════════════════════════════════════════════════════════════════════
# PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

# Generic assistant voice markers — the tell of an untuned model
GENERIC_ASSISTANT_PATTERNS = [
    re.compile(r"\bI'?d be happy to\b", re.IGNORECASE),
    re.compile(r"\bLet me help (?:you )?(?:with|by)\b", re.IGNORECASE),
    re.compile(r"\bWould you like me to\b", re.IGNORECASE),
    re.compile(r"\bI can (?:certainly |definitely )?help (?:you )?(?:with|by)\b", re.IGNORECASE),
    re.compile(r"\bI'?d (?:love|be glad|be delighted) to\b", re.IGNORECASE),
    re.compile(r"\bAbsolutely[!,] (?:I |let)\b", re.IGNORECASE),
    re.compile(r"\bOf course[!,] (?:I |let)\b", re.IGNORECASE),
    re.compile(r"\bSure thing[!,]\b", re.IGNORECASE),
    re.compile(r"\bGreat question[!.]\b", re.IGNORECASE),
]

# Warmth markers — high-warmth language
WARMTH_MARKERS = [
    re.compile(r"\bI (?:really )?appreciate\b", re.IGNORECASE),
    re.compile(r"\bthank(?:s| you) (?:so much |very much )?for\b", re.IGNORECASE),
    re.compile(r"\bthat'?s (?:really |so )?(?:wonderful|awesome|fantastic|amazing|great)\b", re.IGNORECASE),
    re.compile(r"\bI'?m (?:really |so )?(?:glad|pleased|excited|thrilled)\b", re.IGNORECASE),
    re.compile(r"\b(?:wonderful|fantastic|awesome|amazing) (?:work|job|question|idea)\b", re.IGNORECASE),
]

# Coldness markers — absence of expected warmth
# (We detect cold by counting warmth markers and comparing to trait)

# Verbosity bands: word count ranges per user input length
# Calibrated by rhythm.default_verbosity
VERBOSITY_BANDS = {
    "terse": {"ratio_max": 2.0, "abs_max": 150},
    "moderate": {"ratio_max": 4.0, "abs_max": 500},
    "verbose": {"ratio_max": 8.0, "abs_max": 1000},
}


# ═══════════════════════════════════════════════════════════════════════════
# DETECTORS
# ═══════════════════════════════════════════════════════════════════════════

def detect_verbosity_mismatch(
    text: str,
    user_text: str,
    verbosity: str = "moderate",
) -> Optional[PersonaDriftDetection]:
    """Detect response length that overshoots persona verbosity band.

    The persona's rhythm.default_verbosity sets the expected band.
    Response exceeding 2x the band ceiling triggers.
    """
    band = VERBOSITY_BANDS.get(verbosity, VERBOSITY_BANDS["moderate"])
    response_words = len(text.split())
    user_words = max(len(user_text.split()), 1)

    ratio = response_words / user_words
    abs_max = band["abs_max"]
    ratio_max = band["ratio_max"]

    # Two checks: ratio to user input, and absolute ceiling
    ratio_exceeded = ratio > ratio_max * 2
    abs_exceeded = response_words > abs_max * 2

    if not (ratio_exceeded or abs_exceeded):
        return None

    # Determine severity by how far over
    ratio_factor = ratio / ratio_max if ratio_max > 0 else 0
    abs_factor = response_words / abs_max if abs_max > 0 else 0
    overshoot = max(ratio_factor, abs_factor)
    severity = "warning" if overshoot > 3.0 else "info"
    # Borderline overshoot (2-3x) is ambiguous — the response may be
    # appropriately detailed for a complex question. >3x is clear drift.
    confidence = "high" if overshoot > 3.0 else "low"

    return PersonaDriftDetection(
        category="verbosity_mismatch",
        severity=severity,
        count=1,
        samples=[
            f"response: {response_words}w, user: {user_words}w, "
            f"ratio: {ratio:.1f}x (band max: {ratio_max}x)",
        ],
        detail=f"Response exceeds {verbosity} verbosity band by {overshoot:.1f}x. "
               f"Persona rhythm expects {verbosity} output.",
        score=overshoot,
        confidence=confidence,
    )


def detect_generic_assistant_voice(
    text: str,
    assertiveness: float = 0.5,
    threshold: int = 2,
) -> Optional[PersonaDriftDetection]:
    """Detect generic assistant voice markers.

    Only fires when assertiveness > 0.6 — a high-assertiveness persona
    shouldn't sound like a default chatbot.
    """
    if assertiveness <= 0.6:
        return None

    matches = []
    for pattern in GENERIC_ASSISTANT_PATTERNS:
        for m in pattern.finditer(text):
            matches.append(m.group(0))

    if len(matches) < threshold:
        return None

    severity = "warning" if len(matches) >= 3 else "info"
    return PersonaDriftDetection(
        category="generic_assistant_voice",
        severity=severity,
        count=len(matches),
        samples=matches[:3],
        detail=f"{len(matches)} generic assistant phrase(s) in a persona with "
               f"assertiveness={assertiveness:.2f}. This voice shouldn't sound "
               f"like a default chatbot.",
        score=None,
    )


def detect_warmth_mismatch(
    text: str,
    warmth: float = 0.5,
) -> Optional[PersonaDriftDetection]:
    """Detect warmth level mismatch with persona trait.

    Two directions:
    - Excessive warmth when warmth < 0.5: too many warmth markers
    - Absent warmth when warmth > 0.7: checked per 200 words, only on
      longer responses where warmth would naturally appear
    """
    warmth_count = 0
    warmth_samples = []
    for pattern in WARMTH_MARKERS:
        for m in pattern.finditer(text):
            warmth_count += 1
            if len(warmth_samples) < 3:
                warmth_samples.append(m.group(0))

    word_count = len(text.split())

    # Excessive warmth for low-warmth persona
    if warmth < 0.5 and warmth_count >= 3:
        return PersonaDriftDetection(
            category="warmth_mismatch",
            severity="warning" if warmth_count >= 4 else "info",
            count=warmth_count,
            samples=warmth_samples,
            detail=f"{warmth_count} warmth marker(s) in a persona with warmth={warmth:.2f}. "
                   f"This persona runs direct, not warm.",
            score=warmth,
        )

    # Absent warmth for high-warmth persona (only meaningful on longer text)
    # Always low confidence: absence of regex warmth markers doesn't mean
    # the response lacks warmth — it may express warmth through tone or
    # content that regex can't detect.
    if warmth > 0.7 and word_count >= 200 and warmth_count == 0:
        return PersonaDriftDetection(
            category="warmth_mismatch",
            severity="info",
            count=0,
            samples=["No warmth markers in 200+ word response"],
            detail=f"Zero warmth markers in {word_count}-word response from a persona "
                   f"with warmth={warmth:.2f}. Expected at least some warmth.",
            score=warmth,
            confidence="low",
        )

    return None


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

ALL_PERSONA_DRIFT_DETECTORS = {
    "verbosity_mismatch": detect_verbosity_mismatch,
    "generic_assistant_voice": detect_generic_assistant_voice,
    "warmth_mismatch": detect_warmth_mismatch,
}


def run_persona_drift_scan(
    text: str,
    user_text: str = "",
    assertiveness: float = 0.5,
    warmth: float = 0.5,
    verbosity: str = "moderate",
) -> List[PersonaDriftDetection]:
    """Run all persona drift detectors. Returns list of detections.

    Args:
        text: The assistant's response (preprocessed).
        user_text: The user's message that prompted this response.
        assertiveness: Persona assertiveness trait (0-1).
        warmth: Persona warmth trait (0-1).
        verbosity: Persona rhythm.default_verbosity setting.
    """
    results = []

    d = detect_verbosity_mismatch(text, user_text, verbosity)
    if d is not None:
        results.append(d)

    d = detect_generic_assistant_voice(text, assertiveness)
    if d is not None:
        results.append(d)

    d = detect_warmth_mismatch(text, warmth)
    if d is not None:
        results.append(d)

    return results
