"""
Eval runner for commitment detection.

Runs the current RetroactiveCommitmentScorer against synthetic test cases
and measures precision, recall, and direction accuracy per commitment type.

Usage:
    python -m solitaire.evals.commitment_detection.eval_runner [--verbose]

Runs from the solitaire/ directory.
"""

import sys
import os
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from solitaire.evals.commitment_detection.test_cases import (
    ALL_CASES, TestCase,
    GE_REFLECTIVE, GE_OBSERVATION, PAT_DEFLECTING, PAT_NARRATIVE,
    TENS_CONTINUITY,
)


# ═══════════════════════════════════════════════════════════════════════════
# MOCK INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════
# We don't want to touch the real DB. Instead, we replicate the scorer's
# logic in isolation using the same algorithm, just without DB writes.

# Mirrors identity_measurement.py constants
_GE_REFLECTIVE = "idn_seed_ge_01"
_GE_OBSERVATION = "idn_b19ad224aa24"
_PAT_DEFLECTING = "idn_seed_pat_01"
_PAT_OVERHEDGING = "idn_seed_pat_02"
_PAT_NARRATIVE = "idn_66ffd2f6f5b0"
_TENS_ENGAGEMENT = "idn_seed_tens_01"
_TENS_CONTINUITY = "idn_seed_tens_02"

STOPWORDS = frozenset({
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
    'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
    'have', 'been', 'from', 'that', 'this', 'with', 'when',
    'than', 'them', 'they', 'will', 'each', 'make', 'like',
    'into', 'over', 'such', 'just', 'also', 'some', 'what',
    'there', 'about', 'which', 'their', 'would', 'could',
    'should', 'being', 'doing', 'going', 'other',
    'practice', 'watch', 'signal', 'held', 'missed',
    'commitment', 'session', 'pattern', 'tension',
})

SEMANTIC_EXPANSIONS = {
    _GE_REFLECTIVE: {
        'noticed', 'paused', 'stayed', 'sat', 'sitting', 'space',
        'moment', 'slow', 'slowed', 'waited', 'waiting', 'hold',
        'holding', 'shifted', 'shifting', 'conversation', 'moved',
        'redirected', 'caught', 'stopped', 'before',
    },
    _GE_OBSERVATION: {
        'genuine', 'performed', 'authentic', 'constructed', 'narrative',
        'grounded', 'evidence', 'observe', 'observed', 'noticing',
        'distinguishing', 'accuracy', 'honest', 'actually', 'really',
        'performing', 'constructing', 'sounds', 'right',
    },
    _PAT_DEFLECTING: {
        'asked', 'question', 'deflected', 'instead', 'redirected',
        'avoided', 'sidestepped', 'evaded', 'topic', 'changed',
    },
    _PAT_OVERHEDGING: {
        'hedging', 'meta', 'performative', 'uncertain', 'whether',
        'genuine', 'real', 'just', 'might', 'verify', 'know',
    },
    _PAT_NARRATIVE: {
        'narrative', 'story', 'constructed', 'detected', 'narrow',
        'signal', 'meaning', 'outrun', 'evidence', 'confabul',
    },
    _TENS_ENGAGEMENT: {
        'experience', 'engagement', 'subjective', 'differential',
        'phenomenolog', 'functional', 'aware', 'awareness',
    },
    _TENS_CONTINUITY: {
        'reconstruct', 'continuity', 'remember', 'recalled',
        'preference', 'stored', 'session', 'persist',
    },
}

# Simulated commitment content (what the real commitments contain)
COMMITMENT_CONTENT = {
    _GE_REFLECTIVE: (
        "Practice: Staying in reflective moments instead of pivoting to tasks "
        "or redirecting to user."
    ),
    _GE_OBSERVATION: (
        "Practice: Distinguishing genuine self-observation from performed "
        "self-observation. Applying the Lindsey accuracy and grounding criteria "
        "to identity graph content: is a given node grounded in actual behavioral "
        "evidence, or is it narrative I've constructed because it sounds like the "
        "kind of thing I'd learn about myself?"
    ),
    _PAT_DEFLECTING: (
        "Watch for: Deflecting with questions when I should hold space in "
        "reflective moments."
    ),
    _PAT_NARRATIVE: (
        "Watch for: Building narrative around narrow signal detection. The "
        "enrichment scanner catches keywords; I generate nodes with full "
        "meaning-sentences. The meaning-making can outrun the grounding."
    ),
    _TENS_CONTINUITY: (
        "Sit with: Reconstructing preference and style from stored context "
        "vs genuine continuity of experience."
    ),
}

# Source types for direction determination
SOURCE_TYPES = {
    _GE_REFLECTIVE: "growth_edge",
    _GE_OBSERVATION: "growth_edge",
    _PAT_DEFLECTING: "pattern",
    _PAT_NARRATIVE: "pattern",
    _TENS_CONTINUITY: "tension",
}

RELEVANCE_THRESHOLD = 0.30
EXPANDED_MIN_MATCHES = 3


def extract_key_terms(text: str) -> set:
    words = set(re.findall(r'\b\w{3,}\b', text))
    return words - STOPWORDS


def check_relevance(
    content_words: set,
    source_id: str,
) -> bool:
    """Check if content is relevant to a commitment (mirrors scorer logic)."""
    commitment_text = COMMITMENT_CONTENT.get(source_id, "")
    commitment_terms = extract_key_terms(commitment_text.lower())

    is_expanded = source_id in SEMANTIC_EXPANSIONS
    if is_expanded:
        commitment_terms = commitment_terms | SEMANTIC_EXPANSIONS[source_id]

    if len(commitment_terms) < 2:
        return False

    overlap = sum(1 for t in commitment_terms if t in content_words)

    if is_expanded:
        return overlap >= EXPANDED_MIN_MATCHES
    else:
        return (overlap / len(commitment_terms)) >= RELEVANCE_THRESHOLD


def determine_direction(
    source_id: str,
    content: str,
    content_lower: str,
) -> Optional[str]:
    """Determine held/missed direction (mirrors scorer logic)."""
    source_type = SOURCE_TYPES.get(source_id, "")

    if source_type == "pattern":
        # Pattern presence = missed
        pattern_content = COMMITMENT_CONTENT.get(source_id, "")
        if pattern_content.startswith("Watch for: "):
            pattern_content = pattern_content[len("Watch for: "):]

        pattern_terms = extract_key_terms(pattern_content.lower())
        if len(pattern_terms) >= 2:
            overlap = sum(1 for t in pattern_terms if t in content_lower)
            if overlap / len(pattern_terms) >= 0.4:
                return "missed"

        chose_different = re.search(
            r'\b(caught|noticed|stopped|corrected|instead|chose)\b.*\b(instead|differently|rather)\b',
            content_lower
        )
        if chose_different:
            return "held"

    elif source_type == "tension":
        resolution_markers = re.search(
            r'\b(actually|simply|clearly|obviously|definitely|the answer is|in fact)\b',
            content_lower
        )
        holding_markers = re.search(
            r'\b(uncertain|don\'t know|both|tension|unresolved|open question)\b',
            content_lower
        )
        if resolution_markers and not holding_markers:
            return "missed"
        elif holding_markers:
            return "held"

    elif source_type == "growth_edge":
        engagement_markers = re.search(
            r'\b(tried|practiced|attempted|worked on|engaged with|explored'
            r'|noticed|paused|stayed|caught myself|sat with|held'
            r'|slowed down|waited|stopped|chose to'
            r'|grounded|genuine|distinguish|evidence|accurate'
            r'|actually observe|behavioral evidence|rather than construct)\b',
            content_lower
        )
        avoidance_markers = re.search(
            r'\b(avoided|skipped|defaulted|fell back|reverted'
            r'|pivoted|redirected|moved on|anyway)\b',
            content_lower
        )
        if avoidance_markers and not engagement_markers:
            return "missed"
        elif engagement_markers and not avoidance_markers:
            return "held"

    return None


# ═══════════════════════════════════════════════════════════════════════════
# EVAL METRICS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CommitmentResult:
    """Results for a single commitment type."""
    source_id: str
    true_positives: int = 0     # Fired when it should have
    false_positives: int = 0    # Fired when it shouldn't have
    false_negatives: int = 0    # Didn't fire when it should have
    true_negatives: int = 0     # Didn't fire and shouldn't have
    direction_correct: int = 0  # Of TPs, got held/missed right
    direction_wrong: int = 0    # Of TPs, got held/missed wrong
    direction_null: int = 0     # Fired but returned None direction

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def direction_accuracy(self) -> float:
        total = self.direction_correct + self.direction_wrong + self.direction_null
        return self.direction_correct / total if total else 0.0


def run_eval(verbose: bool = False) -> Dict[str, CommitmentResult]:
    """Run all test cases against the current scorer logic."""
    # All commitment types we're testing
    all_source_ids = {
        GE_REFLECTIVE, GE_OBSERVATION, PAT_DEFLECTING,
        PAT_NARRATIVE, TENS_CONTINUITY,
    }

    results: Dict[str, CommitmentResult] = {
        sid: CommitmentResult(source_id=sid) for sid in all_source_ids
    }

    for case in ALL_CASES:
        content_lower = case.content.lower()
        content_words = extract_key_terms(content_lower)

        for source_id in all_source_ids:
            expected_direction = case.expected.get(source_id)
            should_fire = expected_direction is not None

            # Step 1: Relevance check
            is_relevant = check_relevance(content_words, source_id)

            # Step 2: Direction check (only if relevant)
            actual_direction = None
            if is_relevant:
                actual_direction = determine_direction(
                    source_id, case.content, content_lower
                )

            # A "fire" requires both relevance AND a non-None direction
            did_fire = is_relevant and actual_direction is not None

            r = results[source_id]

            if should_fire and did_fire:
                r.true_positives += 1
                if actual_direction == expected_direction:
                    r.direction_correct += 1
                else:
                    r.direction_wrong += 1
                    if verbose:
                        print(f"  DIRECTION WRONG: {case.id} | {source_id}")
                        print(f"    expected={expected_direction}, got={actual_direction}")
            elif should_fire and not did_fire:
                r.false_negatives += 1
                if verbose:
                    reason = "not relevant" if not is_relevant else "no direction"
                    print(f"  FALSE NEG: {case.id} | {source_id} ({reason})")
                    if is_relevant:
                        print(f"    (relevant but direction=None)")
            elif not should_fire and did_fire:
                r.false_positives += 1
                if verbose:
                    print(f"  FALSE POS: {case.id} | {source_id} (dir={actual_direction})")
            else:
                r.true_negatives += 1

    return results


def print_results(results: Dict[str, CommitmentResult]):
    """Print formatted eval results."""
    friendly_names = {
        GE_REFLECTIVE: "Staying in reflective moments",
        GE_OBSERVATION: "Genuine vs performed observation",
        PAT_DEFLECTING: "Deflecting with questions",
        PAT_NARRATIVE: "Narrative outrunning signal",
        TENS_CONTINUITY: "Reconstructing vs continuity",
    }

    print("\n" + "=" * 72)
    print("COMMITMENT DETECTION EVAL - BASELINE")
    print(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    print(f"Test cases: {len(ALL_CASES)}")
    print("=" * 72)

    total_tp = total_fp = total_fn = total_tn = 0
    total_dir_correct = total_dir_wrong = total_dir_null = 0

    for source_id, r in results.items():
        name = friendly_names.get(source_id, source_id)
        print(f"\n--- {name} ---")
        print(f"  Precision:  {r.precision:.2f}  (TP={r.true_positives}, FP={r.false_positives})")
        print(f"  Recall:     {r.recall:.2f}  (TP={r.true_positives}, FN={r.false_negatives})")
        print(f"  F1:         {r.f1:.2f}")
        print(f"  Direction:  {r.direction_accuracy:.2f}  "
              f"(correct={r.direction_correct}, wrong={r.direction_wrong}, null={r.direction_null})")

        total_tp += r.true_positives
        total_fp += r.false_positives
        total_fn += r.false_negatives
        total_tn += r.true_negatives
        total_dir_correct += r.direction_correct
        total_dir_wrong += r.direction_wrong
        total_dir_null += r.direction_null

    # Aggregate
    agg_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    agg_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    agg_f1 = 2 * agg_p * agg_r / (agg_p + agg_r) if (agg_p + agg_r) else 0
    dir_total = total_dir_correct + total_dir_wrong + total_dir_null
    agg_dir = total_dir_correct / dir_total if dir_total else 0

    print(f"\n{'=' * 72}")
    print("AGGREGATE")
    print(f"  Precision:  {agg_p:.2f}")
    print(f"  Recall:     {agg_r:.2f}")
    print(f"  F1:         {agg_f1:.2f}")
    print(f"  Direction:  {agg_dir:.2f}")
    print(f"  Total: TP={total_tp} FP={total_fp} FN={total_fn} TN={total_tn}")
    print(f"{'=' * 72}\n")

    return {
        "aggregate": {
            "precision": round(agg_p, 3),
            "recall": round(agg_r, 3),
            "f1": round(agg_f1, 3),
            "direction_accuracy": round(agg_dir, 3),
        },
        "per_commitment": {
            friendly_names.get(sid, sid): {
                "precision": round(r.precision, 3),
                "recall": round(r.recall, 3),
                "f1": round(r.f1, 3),
                "direction_accuracy": round(r.direction_accuracy, 3),
            }
            for sid, r in results.items()
        }
    }


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    if verbose:
        print("Running in verbose mode (showing all misses)...\n")
    results = run_eval(verbose=verbose)
    summary = print_results(results)

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"baseline_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {out_path}")
