"""Outbound Writing Quality Gate — orchestrator.

Scans assistant-generated text for AI writing tells across multiple layers.
Called by the Stop hook after each response. Results are written as a marker
file that the evaluation gate picks up on the next turn.

Design principle: same as the evaluation gate — structural enforcement,
not behavioral instruction. The ops block says "no em dashes." This gate
catches the ones that slip through generation momentum.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from solitaire.outbound.surface_detectors import (
    preprocess,
    run_surface_scan,
    SurfaceDetection,
)
from solitaire.outbound.structural_detectors import (
    run_structural_scan,
    StructuralDetection,
)
from solitaire.outbound.persona_detectors import (
    run_persona_drift_scan,
    PersonaDriftDetection,
)
from solitaire.outbound.commitment_detectors import (
    run_commitment_scan,
    CommitmentDetection,
)
from solitaire.outbound.context_detectors import (
    run_context_scan,
    ContextDetection,
)
from solitaire.outbound.config import WritingGateConfig, PersonaTraits, TranscriptContext


@dataclass
class WritingViolation:
    """A single writing quality violation."""
    layer: int              # 1=surface, 2=structural, 3=persona, 4=commitment, 5=context
    category: str           # e.g., "em_dash", "paragraph_uniformity"
    severity: str           # "info" | "warning"
    count: int              # occurrences
    samples: List[str]      # up to 3 matched snippets
    detail: str             # human-readable explanation
    score: Optional[float]  # numeric value where applicable
    confidence: str = "high"  # "high" | "low" — low-confidence routes to model verification


@dataclass
class WritingGateResult:
    """Result of a full writing quality scan."""
    violations: List[WritingViolation]
    layer_scores: Dict[str, Optional[float]]
    summary: str
    scan_version: str = "1.0"

    def has_violations(self) -> bool:
        return len(self.violations) > 0

    def to_marker_dict(self) -> dict:
        """Convert to the dict format expected by marker.write_marker()."""
        return {
            "violations": [
                {
                    "layer": v.layer,
                    "category": v.category,
                    "severity": v.severity,
                    "count": v.count,
                    "samples": v.samples,
                    "detail": v.detail,
                    "score": v.score,
                    "confidence": v.confidence,
                }
                for v in self.violations
            ],
        }


def scan(
    text: str,
    config: Optional[WritingGateConfig] = None,
    persona_traits: Optional[PersonaTraits] = None,
    transcript: Optional[TranscriptContext] = None,
) -> WritingGateResult:
    """Run the full writing quality scan on assistant output.

    Args:
        text: The assistant's response text.
        config: Per-persona configuration. Uses defaults if None.
        persona_traits: Persona trait values for Layers 3-4. Uses neutral defaults if None.
        transcript: Transcript context for Layers 4-5. Skips context-dependent checks if None.

    Returns:
        WritingGateResult with all violations found.
    """
    if config is None:
        config = WritingGateConfig()
    if persona_traits is None:
        persona_traits = PersonaTraits()
    if transcript is None:
        transcript = TranscriptContext()

    violations: List[WritingViolation] = []

    # Skip short responses
    word_count = len(text.split())
    if word_count < config.min_response_length:
        return WritingGateResult(
            violations=[],
            layer_scores={"surface": None, "structural": None,
                          "persona_drift": None, "commitment": None, "context": None},
            summary="",
        )

    # Preprocess: strip code blocks and blockquotes
    cleaned = preprocess(
        text,
        exclude_code=config.exclude_code_blocks,
        exclude_quotes=config.exclude_quoted_text,
    )

    # Layer 1: Surface tells
    if config.surface.enabled:
        surface_hits = run_surface_scan(cleaned)
        for hit in surface_hits:
            violations.append(WritingViolation(
                layer=1,
                category=hit.category,
                severity=hit.severity,
                count=hit.count,
                samples=hit.samples,
                detail=hit.detail,
                score=None,
                confidence=hit.confidence,
            ))

    # Layer 2: Structural shape
    if config.structural.enabled:
        structural_hits = run_structural_scan(cleaned)
        for hit in structural_hits:
            violations.append(WritingViolation(
                layer=2,
                category=hit.category,
                severity=hit.severity,
                count=hit.count,
                samples=hit.samples,
                detail=hit.detail,
                score=hit.score,
                confidence=hit.confidence,
            ))

    # Layer 3: Persona drift
    if config.persona_drift.enabled:
        persona_hits = run_persona_drift_scan(
            cleaned,
            user_text=transcript.user_text,
            assertiveness=persona_traits.assertiveness,
            warmth=persona_traits.warmth,
            verbosity=persona_traits.verbosity,
        )
        for hit in persona_hits:
            violations.append(WritingViolation(
                layer=3,
                category=hit.category,
                severity=hit.severity,
                count=hit.count,
                samples=hit.samples,
                detail=hit.detail,
                score=hit.score,
                confidence=hit.confidence,
            ))

    # Layer 4: Commitment adherence
    if config.commitment.enabled:
        commitment_hits = run_commitment_scan(
            cleaned,
            prior_assistant_text=transcript.prior_assistant_text,
            conviction=persona_traits.conviction,
        )
        for hit in commitment_hits:
            violations.append(WritingViolation(
                layer=4,
                category=hit.category,
                severity=hit.severity,
                count=hit.count,
                samples=hit.samples,
                detail=hit.detail,
                score=hit.score,
                confidence=hit.confidence,
            ))

    # Layer 5: Context coherence
    if config.context.enabled:
        context_hits = run_context_scan(
            cleaned,
            user_text=transcript.user_text,
            prior_turns=transcript.prior_turns_text,
        )
        for hit in context_hits:
            violations.append(WritingViolation(
                layer=5,
                category=hit.category,
                severity=hit.severity,
                count=hit.count,
                samples=hit.samples,
                detail=hit.detail,
                score=hit.score,
                confidence=hit.confidence,
            ))

    # Build layer scores
    layer_scores: Dict[str, Optional[float]] = {
        "surface": None,
        "structural": None,
        "persona_drift": None,
        "commitment": None,
        "context": None,
    }
    for layer_name, layer_num in [("surface", 1), ("structural", 2),
                                   ("persona_drift", 3), ("commitment", 4),
                                   ("context", 5)]:
        layer_v = [v for v in violations if v.layer == layer_num]
        if layer_v:
            warning_count = sum(1 for v in layer_v if v.severity == "warning")
            info_count = sum(1 for v in layer_v if v.severity == "info")
            layer_scores[layer_name] = min(1.0, (warning_count * 0.3 + info_count * 0.1))

    # Build summary
    sorted_v = sorted(violations, key=lambda v: {"warning": 0, "info": 1}.get(v.severity, 2))
    summary_parts = []
    for v in sorted_v[:3]:
        if v.score is not None:
            summary_parts.append(f"{v.category} (CV={v.score:.2f})")
        elif v.count > 1:
            summary_parts.append(f"{v.count} {v.category}")
        else:
            summary_parts.append(v.category)
    summary = ", ".join(summary_parts)

    return WritingGateResult(
        violations=violations,
        layer_scores=layer_scores,
        summary=summary,
    )
