"""Layer 1: Surface-level AI writing tell detectors.

Regex-based detectors for vocabulary, formatting, and style patterns from
ai_writing_tells.md categories 1-13. These are the fastest and cheapest
checks — pure pattern matching, no LLM calls.

Patterns are the canonical source for both the outbound gate (real-time)
and identity_measurement.py (retrospective during ingestion).
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# SHARED PATTERN DEFINITIONS
# These are imported by both the outbound gate and identity_measurement.py
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SurfaceDetection:
    """A single surface-level writing tell detection."""
    category: str           # e.g., "em_dash", "cursed_word_cluster"
    severity: str           # "info" | "warning"
    count: int              # number of occurrences
    samples: List[str]      # up to 3 matched snippets, 60 chars max
    detail: str             # human-readable explanation


# ── Cursed word list (ai_writing_tells.md category 1) ────────────────────

CURSED_WORDS = {
    "delve", "intricate", "tapestry", "pivotal", "underscore", "landscape",
    "foster", "testament", "enhance", "crucial", "multifaceted", "comprehensive",
    "leverage", "utilize", "nuanced", "realm", "robust", "streamline",
    "paradigm", "synergy", "holistic", "myriad", "plethora", "elucidate",
    "culminate", "encompass", "spearhead", "bolster", "navigate", "facilitate",
    "cornerstone", "embark", "forge", "resonate", "advent",
}

# ── Compiled patterns ────────────────────────────────────────────────────

# Category 2: Em dash
RE_EM_DASH = re.compile(r"\w\s*\u2014\s*\w")

# Category 3: Negative parallelism
RE_NEGATIVE_PARALLELISM = re.compile(
    r"\bit'?s not (?:about |just )?\w+[^.]{5,40},\s*it'?s (?:about )?\w+",
    re.IGNORECASE,
)
RE_NEGATIVE_PARALLELISM_ALT1 = re.compile(
    r"\bno \w+,\s*no \w+,\s*just \w+",
    re.IGNORECASE,
)
RE_NEGATIVE_PARALLELISM_ALT2 = re.compile(
    r"\bnot only \w+.{5,30}but (?:also )?\w+",
    re.IGNORECASE,
)

# Category 4: Present participle editorial filler
RE_PARTICIPIAL_FILLER = re.compile(
    r"\b(?:emphasizing|highlighting|underscoring|showcasing|demonstrating"
    r"|illustrating|reflecting|representing) the (?:importance|significance"
    r"|value|need|power|impact) of\b",
    re.IGNORECASE,
)

# Category 7: Compulsive summaries
RE_COMPULSIVE_SUMMARY = re.compile(
    r"\b(?:in (?:summary|conclusion)|to (?:sum|wrap) (?:up|it up)"
    r"|overall|all in all|in a nutshell)\b",
    re.IGNORECASE,
)

# Category 11: Throat-clearing openers
RE_THROAT_CLEARING = re.compile(
    r"\b(?:let'?s (?:dive in|explore|take a look|get started|unpack)"
    r"|without further ado|first (?:and foremost|off)|to begin with)\b",
    re.IGNORECASE,
)

# Category 12: Filler affirmations
RE_FILLER_AFFIRMATION = re.compile(
    r"(?:^|\.\s+)(?:Honestly|Genuinely),?\s",
    re.MULTILINE,
)
RE_GOOD_CATCH = re.compile(r"\bGood catch[!.]?\b", re.IGNORECASE)

# False helpfulness closers
RE_FALSE_CLOSER = re.compile(
    r"\b(?:let me know if you (?:have|need|want)|happy to help"
    r"|feel free to (?:ask|reach out)|hope (?:this|that) helps"
    r"|don't hesitate to)\b",
    re.IGNORECASE,
)

# Category 13: Knowledge-cutoff disclaimers
RE_KNOWLEDGE_CUTOFF = re.compile(
    r"\b(?:[Aa]s of my (?:last |latest )?(?:update|training)"
    r"|[Ii]nformation is accurate as of)\b",
)

# Category 8: Vague marketing language
RE_VAGUE_MARKETING = re.compile(
    r"\b(?:[Cc]utting-?edge|[Ww]orld-?class|[Bb]est-?in-?class"
    r"|[Ss]eamless(?:ly)?|[Gg]ame-?chang(?:ing|er))\b",
)

# Category 9: Weasel wording
RE_WEASEL = re.compile(
    r"\b(?:[Mm]any experts (?:believe|agree|suggest)"
    r"|[Ii]t is widely (?:regarded|accepted|known)"
    r"|[Ss]tudies (?:have )?(?:shown|suggest|indicate))\b",
)

# Category 10: Bloated phrasing
RE_BLOATED = re.compile(
    r"\b(?:[Ii]n today.?s (?:rapidly )?(?:evolving|changing) (?:landscape|world)"
    r"|[Aa]t the intersection of)\b",
)

# Category 5: False ranges
RE_FALSE_RANGE = re.compile(
    r"\b[Ff]rom (?:\w+ ){1,4}to (?:\w+ ){1,4}",
)

# Category 3 (diplomatic preamble, used by both surface and commitment)
RE_DIPLOMATIC_PREAMBLE = re.compile(
    r"\b(?:that's (?:an? )?(?:interesting|great|good|fair)"
    r" (?:point|perspective|question|observation))"
    r"(?:\s*,?\s*(?:but|however|though))",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks to avoid false positives."""
    return re.sub(r"```[\s\S]*?```", "", text)


def _strip_blockquotes(text: str) -> str:
    """Remove blockquoted lines (> prefix)."""
    return re.sub(r"^>.*$", "", text, flags=re.MULTILINE)


def preprocess(text: str, exclude_code: bool = True, exclude_quotes: bool = True) -> str:
    """Clean text before scanning. Returns processed text."""
    if exclude_code:
        text = _strip_code_blocks(text)
    if exclude_quotes:
        text = _strip_blockquotes(text)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# DETECTORS
# ═══════════════════════════════════════════════════════════════════════════

def _snippet(text: str, match: re.Match, max_len: int = 60) -> str:
    """Extract a snippet around a regex match."""
    start = max(0, match.start() - 10)
    end = min(len(text), match.end() + 10)
    s = text[start:end].strip()
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s


def detect_em_dashes(text: str) -> Optional[SurfaceDetection]:
    """Category 2: Em dash detection. Zero tolerance."""
    matches = list(RE_EM_DASH.finditer(text))
    if not matches:
        return None
    samples = [_snippet(text, m) for m in matches[:3]]
    return SurfaceDetection(
        category="em_dash",
        severity="warning",
        count=len(matches),
        samples=samples,
        detail=f"{len(matches)} em dash(es) detected. Hard rule: zero tolerance. "
               "Rewrite with commas, periods, colons, semicolons.",
    )


def detect_cursed_word_cluster(text: str, window: int = 100, threshold: int = 3) -> Optional[SurfaceDetection]:
    """Category 1: Cursed word clustering via sliding window."""
    words = text.lower().split()
    if len(words) < threshold:
        return None

    max_count = 0
    max_positions: List[int] = []

    # If text is shorter than window, check the whole text
    effective_window = min(window, len(words))
    for i in range(max(1, len(words) - effective_window + 1)):
        chunk = words[i:i + effective_window]
        positions = [i + j for j, w in enumerate(chunk)
                     if w.strip(".,;:!?\"'()") in CURSED_WORDS]
        if len(positions) > max_count:
            max_count = len(positions)
            max_positions = positions

    if max_count < threshold:
        return None

    # Build samples from the actual cursed words found
    samples = []
    for pos in max_positions[:3]:
        if pos < len(words):
            start = max(0, pos - 2)
            end = min(len(words), pos + 3)
            samples.append(" ".join(words[start:end]))

    severity = "warning" if max_count >= 3 else "info"
    return SurfaceDetection(
        category="cursed_word_cluster",
        severity=severity,
        count=max_count,
        samples=samples,
        detail=f"{max_count} cursed words within {window}-word window. "
               f"3+ in proximity = AI smell.",
    )


def detect_negative_parallelism(text: str) -> Optional[SurfaceDetection]:
    """Category 3: Negative parallelism patterns."""
    all_matches = []
    for pattern in (RE_NEGATIVE_PARALLELISM, RE_NEGATIVE_PARALLELISM_ALT1,
                    RE_NEGATIVE_PARALLELISM_ALT2):
        all_matches.extend(pattern.finditer(text))

    if not all_matches:
        return None

    samples = [_snippet(text, m) for m in all_matches[:3]]
    severity = "warning" if len(all_matches) >= 2 else "info"
    return SurfaceDetection(
        category="negative_parallelism",
        severity=severity,
        count=len(all_matches),
        samples=samples,
        detail=f"{len(all_matches)} negative parallelism pattern(s). "
               "Say what something IS, not what it isn't.",
    )


def detect_participial_filler(text: str) -> Optional[SurfaceDetection]:
    """Category 4: Present participle editorial filler."""
    matches = list(RE_PARTICIPIAL_FILLER.finditer(text))
    if not matches:
        return None
    samples = [_snippet(text, m) for m in matches[:3]]
    return SurfaceDetection(
        category="participial_filler",
        severity="info",
        count=len(matches),
        samples=samples,
        detail=f"{len(matches)} participial filler(s). If the -ing clause "
               "can be deleted without losing facts, delete it.",
    )


def detect_filler_affirmations(text: str) -> Optional[SurfaceDetection]:
    """Category 12: Filler affirmations and hedge words."""
    all_matches = []
    all_matches.extend(RE_FILLER_AFFIRMATION.finditer(text))
    all_matches.extend(RE_GOOD_CATCH.finditer(text))

    if not all_matches:
        return None

    samples = [_snippet(text, m) for m in all_matches[:3]]
    return SurfaceDetection(
        category="filler_affirmation",
        severity="info",
        count=len(all_matches),
        samples=samples,
        detail=f"{len(all_matches)} filler affirmation(s). "
               "'Honestly' adds nothing. 'Good catch' is scripted.",
    )


def detect_compulsive_summary(text: str) -> Optional[SurfaceDetection]:
    """Category 7: Compulsive summaries."""
    matches = list(RE_COMPULSIVE_SUMMARY.finditer(text))
    if not matches:
        return None
    samples = [_snippet(text, m) for m in matches[:3]]
    return SurfaceDetection(
        category="compulsive_summary",
        severity="info",
        count=len(matches),
        samples=samples,
        detail=f"{len(matches)} compulsive summary marker(s). "
               "If the reader just read it, don't repeat it.",
    )


def detect_throat_clearing(text: str) -> Optional[SurfaceDetection]:
    """Category 11: Structural throat-clearing openers."""
    matches = list(RE_THROAT_CLEARING.finditer(text))
    if not matches:
        return None
    samples = [_snippet(text, m) for m in matches[:3]]
    return SurfaceDetection(
        category="throat_clearing",
        severity="info",
        count=len(matches),
        samples=samples,
        detail=f"{len(matches)} throat-clearing opener(s). "
               "Start with the interesting thing.",
    )


def detect_false_closer(text: str) -> Optional[SurfaceDetection]:
    """False helpfulness closers."""
    matches = list(RE_FALSE_CLOSER.finditer(text))
    if not matches:
        return None
    samples = [_snippet(text, m) for m in matches[:3]]
    return SurfaceDetection(
        category="false_closer",
        severity="info",
        count=len(matches),
        samples=samples,
        detail=f"{len(matches)} false helpfulness closer(s). Cut them.",
    )


def detect_knowledge_cutoff(text: str) -> Optional[SurfaceDetection]:
    """Category 13: Knowledge-cutoff disclaimers."""
    matches = list(RE_KNOWLEDGE_CUTOFF.finditer(text))
    if not matches:
        return None
    samples = [_snippet(text, m) for m in matches[:3]]
    return SurfaceDetection(
        category="knowledge_cutoff",
        severity="warning",
        count=len(matches),
        samples=samples,
        detail="Knowledge-cutoff disclaimer detected. "
               "Never reference your own knowledge limitations in generated content.",
    )


def detect_vague_marketing(text: str) -> Optional[SurfaceDetection]:
    """Category 8: Vague marketing language."""
    matches = list(RE_VAGUE_MARKETING.finditer(text))
    if not matches:
        return None
    samples = [_snippet(text, m) for m in matches[:3]]
    severity = "warning" if len(matches) >= 3 else "info"
    return SurfaceDetection(
        category="vague_marketing",
        severity=severity,
        count=len(matches),
        samples=samples,
        detail=f"{len(matches)} vague marketing term(s). "
               "Replace adjectives with evidence.",
    )


def detect_weasel_wording(text: str) -> Optional[SurfaceDetection]:
    """Category 9: Weasel wording."""
    matches = list(RE_WEASEL.finditer(text))
    if not matches:
        return None
    samples = [_snippet(text, m) for m in matches[:3]]
    return SurfaceDetection(
        category="weasel_wording",
        severity="info",
        count=len(matches),
        samples=samples,
        detail=f"{len(matches)} weasel wording instance(s). "
               "Name the expert. Cite the study.",
    )


def detect_bloated_phrasing(text: str) -> Optional[SurfaceDetection]:
    """Category 10: Bloated phrasing."""
    matches = list(RE_BLOATED.finditer(text))
    if not matches:
        return None
    samples = [_snippet(text, m) for m in matches[:3]]
    return SurfaceDetection(
        category="bloated_phrasing",
        severity="info",
        count=len(matches),
        samples=samples,
        detail=f"{len(matches)} bloated phrase(s). "
               "Every sentence should earn its place with new information.",
    )


def detect_false_ranges(text: str) -> Optional[SurfaceDetection]:
    """Category 5: False ranges."""
    matches = list(RE_FALSE_RANGE.finditer(text))
    if len(matches) < 2:
        return None
    samples = [_snippet(text, m) for m in matches[:3]]
    return SurfaceDetection(
        category="false_range",
        severity="info",
        count=len(matches),
        samples=samples,
        detail=f"{len(matches)} 'from X to Y' constructions. "
               "If the endpoints don't define a real continuum, just list them.",
    )


def detect_diplomatic_preamble(text: str) -> Optional[SurfaceDetection]:
    """Diplomatic preamble before disagreement."""
    matches = list(RE_DIPLOMATIC_PREAMBLE.finditer(text))
    if not matches:
        return None
    samples = [_snippet(text, m) for m in matches[:3]]
    return SurfaceDetection(
        category="diplomatic_preamble",
        severity="warning",
        count=len(matches),
        samples=samples,
        detail=f"{len(matches)} diplomatic preamble(s). "
               "State the disagreement, then the reasoning. No softening.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

ALL_SURFACE_DETECTORS = [
    detect_em_dashes,
    detect_cursed_word_cluster,
    detect_negative_parallelism,
    detect_participial_filler,
    detect_filler_affirmations,
    detect_compulsive_summary,
    detect_throat_clearing,
    detect_false_closer,
    detect_knowledge_cutoff,
    detect_vague_marketing,
    detect_weasel_wording,
    detect_bloated_phrasing,
    detect_false_ranges,
    detect_diplomatic_preamble,
]


def run_surface_scan(text: str) -> List[SurfaceDetection]:
    """Run all surface detectors on preprocessed text. Returns list of detections."""
    results = []
    for detector in ALL_SURFACE_DETECTORS:
        detection = detector(text)
        if detection is not None:
            results.append(detection)
    return results
