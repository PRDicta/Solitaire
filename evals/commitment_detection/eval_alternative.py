"""
Eval runner for alternative commitment detection scorers.

Runs the BehavioralSignatureScorer against the same test cases
and compares with the baseline.

Usage:
    python -m solitaire.evals.commitment_detection.eval_alternative [--verbose]
"""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from solitaire.evals.commitment_detection.test_cases import (
    ALL_CASES, TestCase,
    GE_REFLECTIVE, GE_OBSERVATION, PAT_DEFLECTING, PAT_NARRATIVE,
    TENS_CONTINUITY,
)
from solitaire.evals.commitment_detection.alternative_scorers import (
    BehavioralSignatureScorer,
    StructuralAnalyzer,
)
from solitaire.evals.commitment_detection.eval_runner import (
    CommitmentResult, print_results,
)


def run_alternative_eval(verbose: bool = False) -> Dict[str, CommitmentResult]:
    """Run all test cases against the behavioral signature scorer."""
    scorer = BehavioralSignatureScorer()
    structural = StructuralAnalyzer()

    all_source_ids = {
        GE_REFLECTIVE, GE_OBSERVATION, PAT_DEFLECTING,
        PAT_NARRATIVE, TENS_CONTINUITY,
    }

    results: Dict[str, CommitmentResult] = {
        sid: CommitmentResult(source_id=sid) for sid in all_source_ids
    }

    for case in ALL_CASES:
        for source_id in all_source_ids:
            expected_direction = case.expected.get(source_id)
            should_fire = expected_direction is not None

            # Primary: behavioral signature
            actual_direction = scorer.score(case.content, source_id)

            # Supplementary: structural analysis for reflective moments
            if source_id == GE_REFLECTIVE and actual_direction is None:
                if structural.has_reflective_to_task_pivot(case.content):
                    actual_direction = "missed"

            did_fire = actual_direction is not None
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
                    print(f"  FALSE NEG: {case.id} | {source_id}")
            elif not should_fire and did_fire:
                r.false_positives += 1
                if verbose:
                    print(f"  FALSE POS: {case.id} | {source_id} (dir={actual_direction})")
            else:
                r.true_negatives += 1

    return results


def compare_results(baseline_path: str, alternative: Dict):
    """Print comparison between baseline and alternative."""
    with open(baseline_path) as f:
        baseline = json.load(f)

    print("\n" + "=" * 72)
    print("COMPARISON: BASELINE vs BEHAVIORAL SIGNATURES")
    print("=" * 72)

    ba = baseline["aggregate"]
    print(f"\n{'Metric':<20} {'Baseline':>10} {'Signatures':>12} {'Delta':>10}")
    print("-" * 52)
    for metric in ["precision", "recall", "f1", "direction_accuracy"]:
        bv = ba[metric]
        av = alternative["aggregate"][metric]
        delta = av - bv
        sign = "+" if delta > 0 else ""
        print(f"{metric:<20} {bv:>10.3f} {av:>12.3f} {sign}{delta:>9.3f}")

    print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    if verbose:
        print("Running alternative scorer in verbose mode...\n")

    results = run_alternative_eval(verbose=verbose)
    summary = print_results(results)

    # Save
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"behavioral_signatures_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {out_path}")

    # Compare with most recent baseline
    baselines = sorted(results_dir.glob("baseline_*.json"))
    if baselines:
        compare_results(str(baselines[-1]), summary)
