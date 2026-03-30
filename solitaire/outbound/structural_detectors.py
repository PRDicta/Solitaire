"""Layer 2: Structural shape detectors.

Heuristic detectors for paragraph and sentence shape patterns from
ai_writing_tells.md categories 14-15. These measure the geometry of
the text, not its vocabulary.

No LLM calls. Pure math on word counts and paragraph structure.
"""

import re
import statistics
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StructuralDetection:
    """A single structural writing tell detection."""
    category: str           # e.g., "paragraph_uniformity"
    severity: str           # "info" | "warning"
    count: int              # number of affected paragraphs/sentences
    samples: List[str]      # descriptive samples (word counts, etc.)
    detail: str             # human-readable explanation
    score: Optional[float]  # e.g., CV value
    confidence: str = "high"  # "high" | "low" — low routes to model verification


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _is_list_line(line: str) -> bool:
    """Check if a line is a list item."""
    stripped = line.strip()
    return bool(re.match(r"^(?:[-*+]|\d+[.)]) ", stripped))


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs by double newline, filtering non-prose."""
    raw = re.split(r"\n\s*\n", text)
    paragraphs = []
    for p in raw:
        p = p.strip()
        if not p:
            continue
        # Skip if it's entirely list items
        lines = p.split("\n")
        if all(_is_list_line(line) for line in lines if line.strip()):
            continue
        # Skip if it looks like a heading
        if p.startswith("#"):
            continue
        paragraphs.append(p)
    return paragraphs


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences. Simple heuristic."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip() and len(s.split()) >= 3]


def _word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _coefficient_of_variation(values: List[float]) -> float:
    """CV = stddev / mean. Lower = more uniform."""
    if len(values) < 2:
        return 1.0
    mean = statistics.mean(values)
    if mean == 0:
        return 1.0
    return statistics.stdev(values) / mean


# ═══════════════════════════════════════════════════════════════════════════
# DETECTORS
# ═══════════════════════════════════════════════════════════════════════════

def detect_paragraph_uniformity(
    text: str,
    min_paragraphs: int = 3,
    min_words_per_para: int = 20,
    cv_warning: float = 0.10,
    cv_info: float = 0.15,
) -> Optional[StructuralDetection]:
    """Category 14: Paragraph length uniformity.

    Checks runs of consecutive paragraphs (each 20+ words) for
    suspiciously uniform word counts via coefficient of variation.
    """
    paragraphs = _split_paragraphs(text)
    # Filter to substantial paragraphs
    para_lengths = [
        (_word_count(p), p[:60]) for p in paragraphs
        if _word_count(p) >= min_words_per_para
    ]

    if len(para_lengths) < min_paragraphs:
        return None

    # Check all consecutive runs of min_paragraphs or more
    worst_cv = 1.0
    worst_run_len = 0
    worst_lengths: List[int] = []

    for window_size in range(min_paragraphs, len(para_lengths) + 1):
        for start in range(len(para_lengths) - window_size + 1):
            run = para_lengths[start:start + window_size]
            lengths = [wc for wc, _ in run]
            cv = _coefficient_of_variation([float(x) for x in lengths])
            if cv < worst_cv:
                worst_cv = cv
                worst_run_len = window_size
                worst_lengths = lengths

    if worst_cv >= cv_info:
        return None

    severity = "warning" if worst_cv < cv_warning else "info"
    samples = [f"word counts: {worst_lengths}"]

    return StructuralDetection(
        category="paragraph_uniformity",
        severity=severity,
        count=worst_run_len,
        samples=samples,
        detail=f"{worst_run_len} consecutive paragraphs with CV={worst_cv:.2f}. "
               "Vary paragraph length. Unequal treatment for unequal ideas.",
        score=worst_cv,
    )


def detect_sentence_uniformity(
    text: str,
    min_sentences: int = 3,
    cv_threshold: float = 0.15,
    min_paragraphs_flagged: int = 2,
) -> Optional[StructuralDetection]:
    """Category 14: Sentence length uniformity within paragraphs.

    Checks whether sentences within a paragraph are suspiciously
    uniform in length. Only flags if 2+ paragraphs show the pattern.
    """
    paragraphs = _split_paragraphs(text)
    flagged_count = 0
    flagged_cvs: List[float] = []

    for para in paragraphs:
        sentences = _split_sentences(para)
        if len(sentences) < min_sentences:
            continue

        lengths = [float(_word_count(s)) for s in sentences]
        cv = _coefficient_of_variation(lengths)
        if cv < cv_threshold:
            flagged_count += 1
            flagged_cvs.append(cv)

    if flagged_count < min_paragraphs_flagged:
        return None

    avg_cv = statistics.mean(flagged_cvs)
    return StructuralDetection(
        category="sentence_uniformity",
        severity="info",
        count=flagged_count,
        samples=[f"avg CV across flagged paragraphs: {avg_cv:.2f}"],
        detail=f"{flagged_count} paragraphs have uniform sentence lengths (CV < {cv_threshold}). "
               "Vary sentence length within paragraphs.",
        score=avg_cv,
    )


def detect_repeated_openers(text: str, min_consecutive: int = 3) -> Optional[StructuralDetection]:
    """Category 14: Repeated paragraph opening words.

    Flags when 3+ consecutive paragraphs start with the same word.
    """
    paragraphs = _split_paragraphs(text)
    if len(paragraphs) < min_consecutive:
        return None

    # Get first word of each paragraph
    openers = []
    for p in paragraphs:
        words = p.split()
        if words:
            openers.append(words[0].lower().strip("*_"))

    # Check for consecutive runs of the same opener
    best_run = 0
    best_word = ""
    current_run = 1
    for i in range(1, len(openers)):
        if openers[i] == openers[i - 1]:
            current_run += 1
            if current_run > best_run:
                best_run = current_run
                best_word = openers[i]
        else:
            current_run = 1

    if best_run < min_consecutive:
        return None

    return StructuralDetection(
        category="repeated_openers",
        severity="info",
        count=best_run,
        samples=[f'"{best_word}" opens {best_run} consecutive paragraphs'],
        detail=f"{best_run} consecutive paragraphs start with '{best_word}'. "
               "Vary paragraph openings.",
        score=None,
    )


def detect_parallel_construction(text: str, min_consecutive: int = 3) -> Optional[StructuralDetection]:
    """Category 14: Parallel syntactic structure across paragraphs.

    Detects "Subject verb object" patterns repeated at paragraph starts.
    e.g., "Humans do X. LLMs do Y." three times in a row.
    """
    paragraphs = _split_paragraphs(text)
    if len(paragraphs) < min_consecutive:
        return None

    # Extract first-sentence structure pattern for each paragraph
    structures = []
    for p in paragraphs:
        sentences = _split_sentences(p)
        if not sentences:
            structures.append("")
            continue
        first = sentences[0]
        # Normalize: extract "Noun verb" pattern
        match = re.match(r"^(\w+)\s+(do|does|did|has|have|is|are|was|were|will|can|should)\s", first, re.IGNORECASE)
        if match:
            structures.append(f"{match.group(2).lower()}")
        else:
            structures.append("")

    # Check for consecutive structural matches
    best_run = 0
    current_run = 1
    for i in range(1, len(structures)):
        if structures[i] and structures[i] == structures[i - 1]:
            current_run += 1
            if current_run > best_run:
                best_run = current_run
        else:
            current_run = 1

    if best_run < min_consecutive:
        return None

    return StructuralDetection(
        category="parallel_construction",
        severity="info",
        count=best_run,
        samples=[f"{best_run} paragraphs with matching syntactic structure"],
        detail=f"{best_run} consecutive paragraphs share the same syntactic pattern. "
               "Break the symmetry.",
        score=None,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

ALL_STRUCTURAL_DETECTORS = [
    detect_paragraph_uniformity,
    detect_sentence_uniformity,
    detect_repeated_openers,
    detect_parallel_construction,
]


def run_structural_scan(text: str) -> List[StructuralDetection]:
    """Run all structural detectors. Returns list of detections."""
    results = []
    for detector in ALL_STRUCTURAL_DETECTORS:
        detection = detector(text)
        if detection is not None:
            results.append(detection)
    return results
