"""
Solitaire — Adaptation Engine (Thermostat)

Closes the behavioral feedback loop. The disposition filter detects signals
in real time, behavioral_diff summarizes each session, and trend_analyzer
identifies cross-session patterns. This module sits above all three and
proposes concrete lifecycle changes:

Tier A: Commitment lifecycle management
  - Retire commitments that have been consistently honored (integrated behavior)
  - Escalate commitments that are stuck (mostly missed or partial)
  - Sharpen commitments that are invisible to the scorer (fuzzy wording)

Tier B: Signal weight graduation
  - Identify signals reinforced enough to become permanent baseline shifts
  - Propose ratchets with conservative deltas and confidence scores

The engine produces recommendations only. It never mutates state directly.
The caller (engine.py or CLI) decides whether to act.
"""
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .trend_analyzer import (
    CommitmentArc,
    SignalTrend,
    TraitTrajectory,
    TrendReport,
)


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class CommitmentRecommendation:
    """Enriched recommendation for a single commitment."""
    commitment_id: str
    commitment_content: str
    action: str               # "retire", "escalate", "sharpen"
    confidence: float         # 0.0-1.0
    reasoning: str
    evidence: Dict[str, Any]
    suggested_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commitment_id": self.commitment_id,
            "commitment_content": self.commitment_content,
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "suggested_actions": self.suggested_actions,
        }


@dataclass
class GraduationProposal:
    """Proposal to shift a trait's baseline based on sustained signal reinforcement."""
    signal_key: str
    trait: str
    direction: str            # "up" or "down"
    current_baseline: float
    proposed_baseline: float
    delta: float
    confidence: float
    reasoning: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_key": self.signal_key,
            "trait": self.trait,
            "direction": self.direction,
            "current_baseline": round(self.current_baseline, 4),
            "proposed_baseline": round(self.proposed_baseline, 4),
            "delta": round(self.delta, 4),
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "evidence": self.evidence,
        }


@dataclass
class AdaptationResult:
    """Complete output from the adaptation engine."""
    commitment_recommendations: List[CommitmentRecommendation] = field(default_factory=list)
    graduation_proposals: List[GraduationProposal] = field(default_factory=list)
    analyzed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commitment_recommendations": [r.to_dict() for r in self.commitment_recommendations],
            "graduation_proposals": [p.to_dict() for p in self.graduation_proposals],
            "analyzed_at": self.analyzed_at,
        }

    def format_readable(self) -> str:
        lines = []
        if self.commitment_recommendations:
            lines.append("Commitment Lifecycle:")
            for r in self.commitment_recommendations:
                label = r.action.upper()
                content = r.commitment_content[:60]
                lines.append(f"  [{label:>8}] {content} (confidence: {r.confidence:.0%})")
                lines.append(f"             {r.reasoning}")
        if self.graduation_proposals:
            if lines:
                lines.append("")
            lines.append("Graduation Proposals:")
            for p in self.graduation_proposals:
                lines.append(
                    f"  {p.signal_key} -> {p.trait} baseline: "
                    f"{p.current_baseline:.3f} -> {p.proposed_baseline:.3f} "
                    f"(confidence: {p.confidence:.0%})"
                )
                lines.append(f"             {p.reasoning}")
        if not lines:
            lines.append("No actionable recommendations.")
        return "\n".join(lines)

    @property
    def has_recommendations(self) -> bool:
        return bool(self.commitment_recommendations) or bool(self.graduation_proposals)


# ── Adaptation Engine ────────────────────────────────────────────────────

class AdaptationEngine:
    """Thermostat layer: proposes lifecycle changes based on trend data.

    Consumes TrendReport + optional DriftAnalytics output.
    Produces recommendations only; never mutates state directly.
    """

    # Graduation thresholds
    GRAD_MIN_REINFORCEMENT = 8        # near-permanent floor
    GRAD_MIN_SESSIONS_ACTIVE = 15     # must be active across this many sessions
    GRAD_MAX_VOLATILITY = 0.02        # trait volatility ceiling
    GRAD_MIN_DRIFT_MAGNITUDE = 0.03   # minimum net drift to justify a ratchet
    GRAD_BASELINE_FACTOR = 0.75       # bake 75% of observed drift into baseline

    # Sharpen: independent low-evaluation-rate trigger
    SHARPEN_MIN_SESSIONS = 5          # commitment must exist for at least this many
    SHARPEN_EVAL_RATE_CEILING = 0.4   # below this = fuzzy

    def __init__(
        self,
        conn: sqlite3.Connection,
        persona=None,
    ):
        self.conn = conn
        self.persona = persona

    def evaluate(
        self,
        trend_report: TrendReport,
        drift_analytics=None,
    ) -> AdaptationResult:
        """Run full adaptation analysis.

        Args:
            trend_report: Output from TrendAnalyzer.analyze()
            drift_analytics: Optional AnalyticsReport from DriftAnalytics

        Returns:
            AdaptationResult with recommendations and proposals.
        """
        result = AdaptationResult(
            analyzed_at=datetime.now(timezone.utc).isoformat(),
        )

        # Tier A: commitment lifecycle
        result.commitment_recommendations = self._evaluate_commitments(
            trend_report.commitment_arcs,
            trend_report.session_window,
        )

        # Tier B: signal graduation
        near_permanent = []
        signal_stats = {}
        if drift_analytics is not None:
            near_permanent = getattr(drift_analytics, "near_permanent_signals", [])
            signal_stats = getattr(drift_analytics, "signal_stats", {})

        result.graduation_proposals = self._evaluate_graduation_candidates(
            signal_trends=trend_report.signal_trends,
            trait_trajectories=trend_report.trait_trajectories,
            near_permanent_signals=near_permanent,
            signal_stats=signal_stats,
        )

        return result

    # ── Tier A: Commitment Lifecycle ─────────────────────────────────────

    def _evaluate_commitments(
        self,
        commitment_arcs: List[CommitmentArc],
        session_window: int,
    ) -> List[CommitmentRecommendation]:
        """Enrich TrendAnalyzer's commitment recommendations with confidence and actions."""
        recommendations = []

        for arc in commitment_arcs:
            if arc.recommendation == "monitor":
                # Check for independent sharpen trigger (low evaluation rate)
                rec = self._check_low_evaluation_sharpen(arc, session_window)
                if rec:
                    recommendations.append(rec)
                continue

            if arc.recommendation == "retire":
                recommendations.append(self._build_retire(arc))
            elif arc.recommendation == "escalate":
                recommendations.append(self._build_escalate(arc))
            elif arc.recommendation == "sharpen":
                recommendations.append(self._build_sharpen(arc, session_window))

        return recommendations

    def _build_retire(self, arc: CommitmentArc) -> CommitmentRecommendation:
        streak = arc.recent_streak
        consecutive_h = 0
        for c in reversed(streak):
            if c == "H":
                consecutive_h += 1
            else:
                break

        honor_rate = arc.honored_count / arc.sessions_evaluated if arc.sessions_evaluated else 0
        confidence = min(1.0, (consecutive_h / 8) * 0.7 + honor_rate * 0.3)

        return CommitmentRecommendation(
            commitment_id=arc.commitment_id,
            commitment_content=arc.commitment_content,
            action="retire",
            confidence=confidence,
            reasoning=(
                f"Honored in {arc.honored_count} of {arc.sessions_evaluated} "
                f"evaluated sessions (streak: {streak}). "
                f"The behavior appears integrated."
            ),
            evidence={
                "honored_count": arc.honored_count,
                "sessions_evaluated": arc.sessions_evaluated,
                "recent_streak": streak,
                "honor_rate": round(honor_rate, 3),
                "consecutive_honored": consecutive_h,
            },
            suggested_actions=[
                "Mark commitment as 'retired_honored'",
                "Record in identity graph as integrated behavior",
            ],
        )

    def _build_escalate(self, arc: CommitmentArc) -> CommitmentRecommendation:
        streak = arc.recent_streak
        consecutive_non_h = 0
        for c in reversed(streak):
            if c != "H":
                consecutive_non_h += 1
            else:
                break

        miss_rate = (arc.missed_count + arc.partial_count) / arc.sessions_evaluated if arc.sessions_evaluated else 0
        confidence = min(1.0, (consecutive_non_h / 5) * 0.6 + miss_rate * 0.4)

        # Vary suggestions based on pattern
        miss_only = all(c == "M" for c in streak[-3:]) if len(streak) >= 3 else False
        if miss_only:
            suggestions = [
                "Reword to be more specific and actionable",
                "Break into smaller behavioral steps",
                "Pair with a concrete session-start trigger",
            ]
        else:
            suggestions = [
                "Add a behavioral trigger or cue",
                "Consider whether the commitment scope is too broad",
                "Review if external factors are blocking adherence",
            ]

        return CommitmentRecommendation(
            commitment_id=arc.commitment_id,
            commitment_content=arc.commitment_content,
            action="escalate",
            confidence=confidence,
            reasoning=(
                f"Non-honored in {arc.missed_count + arc.partial_count} of "
                f"{arc.sessions_evaluated} evaluated sessions "
                f"(streak: {streak}). Needs attention."
            ),
            evidence={
                "missed_count": arc.missed_count,
                "partial_count": arc.partial_count,
                "sessions_evaluated": arc.sessions_evaluated,
                "recent_streak": streak,
                "miss_rate": round(miss_rate, 3),
                "consecutive_non_honored": consecutive_non_h,
            },
            suggested_actions=suggestions,
        )

    def _build_sharpen(self, arc: CommitmentArc, session_window: int) -> CommitmentRecommendation:
        eval_rate = arc.sessions_evaluated / session_window if session_window else 0
        partial_ratio = arc.partial_count / arc.sessions_evaluated if arc.sessions_evaluated else 0
        confidence = min(1.0, max(partial_ratio, 1.0 - eval_rate) * 0.8)

        return CommitmentRecommendation(
            commitment_id=arc.commitment_id,
            commitment_content=arc.commitment_content,
            action="sharpen",
            confidence=confidence,
            reasoning=(
                f"Partial in {arc.partial_count} of {arc.sessions_evaluated} "
                f"evaluated sessions (eval rate: {eval_rate:.0%}). "
                f"Detection may be too coarse."
            ),
            evidence={
                "partial_count": arc.partial_count,
                "sessions_evaluated": arc.sessions_evaluated,
                "session_window": session_window,
                "eval_rate": round(eval_rate, 3),
                "partial_ratio": round(partial_ratio, 3),
            },
            suggested_actions=[
                "Make the commitment more behaviorally specific",
                "Define observable markers for this commitment",
                "Consider whether this commitment is still relevant",
            ],
        )

    def _check_low_evaluation_sharpen(
        self, arc: CommitmentArc, session_window: int
    ) -> Optional[CommitmentRecommendation]:
        """Independent sharpen trigger for commitments with very low evaluation rates.

        Catches fuzzy commitments that TrendAnalyzer classified as 'monitor'
        because they have too few evaluations to hit the partial ratio threshold.
        """
        if session_window < self.SHARPEN_MIN_SESSIONS:
            return None

        # How many sessions has this commitment existed for?
        sessions_existed = self._get_commitment_age_in_sessions(
            arc.commitment_id, session_window
        )
        if sessions_existed < self.SHARPEN_MIN_SESSIONS:
            return None

        eval_rate = arc.sessions_evaluated / sessions_existed if sessions_existed else 1.0
        if eval_rate >= self.SHARPEN_EVAL_RATE_CEILING:
            return None

        confidence = min(1.0, (1.0 - eval_rate) * 0.8)

        return CommitmentRecommendation(
            commitment_id=arc.commitment_id,
            commitment_content=arc.commitment_content,
            action="sharpen",
            confidence=confidence,
            reasoning=(
                f"Only evaluated in {arc.sessions_evaluated} of {sessions_existed} "
                f"sessions it existed for (eval rate: {eval_rate:.0%}). "
                f"The scorer may not be detecting this commitment."
            ),
            evidence={
                "sessions_evaluated": arc.sessions_evaluated,
                "sessions_existed": sessions_existed,
                "eval_rate": round(eval_rate, 3),
            },
            suggested_actions=[
                "Make the commitment more behaviorally specific",
                "Define observable markers for this commitment",
                "Consider whether this commitment is still relevant",
            ],
        )

    def _get_commitment_age_in_sessions(self, commitment_id: str, max_window: int) -> int:
        """Count sessions since a commitment was created, capped at max_window."""
        try:
            row = self.conn.execute(
                "SELECT metadata FROM identity_nodes WHERE id = ?",
                (commitment_id,),
            ).fetchone()
            if not row:
                return max_window  # assume it's been around the whole window

            meta = json.loads(row[0]) if row[0] else {}
            first_seen = meta.get("first_seen", "")
            if not first_seen:
                return max_window

            # Count sessions after first_seen
            count = self.conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE created_at >= ? AND status = 'ended'",
                (first_seen,),
            ).fetchone()[0]
            return min(count, max_window)
        except Exception:
            return max_window

    # ── Tier B: Signal Graduation ────────────────────────────────────────

    def _evaluate_graduation_candidates(
        self,
        signal_trends: List[SignalTrend],
        trait_trajectories: List[TraitTrajectory],
        near_permanent_signals: List[Tuple[str, int]],
        signal_stats: Dict,
    ) -> List[GraduationProposal]:
        """Identify signals ready to graduate into permanent baseline shifts."""
        if not near_permanent_signals:
            return []

        proposals = []
        near_perm_map = {key: count for key, count in near_permanent_signals}
        trend_map = {s.key: s for s in signal_trends}
        trait_map = {t.trait: t for t in trait_trajectories}

        # Existing ratchet candidates to avoid duplicates
        existing_ratchets = set()
        if self.persona and hasattr(self.persona, '_state') and self.persona._state:
            existing_ratchets = set(self.persona._state.ratchet_candidates.keys())

        for signal_key, reinforcement_count in near_permanent_signals:
            if reinforcement_count < self.GRAD_MIN_REINFORCEMENT:
                continue

            trend = trend_map.get(signal_key)
            if not trend:
                continue

            # Gate: must be active across enough sessions
            if trend.sessions_active < self.GRAD_MIN_SESSIONS_ACTIVE:
                continue

            # Gate: must not be decelerating or extinct
            if trend.trend in ("decelerating", "extinct"):
                continue

            # Find the dominant trait this signal pushes
            dominant_trait, dominant_direction = self._get_dominant_trait(
                signal_key, signal_stats
            )
            if not dominant_trait:
                continue

            # Gate: trait must already be in ratchet tracking, skip if already there
            if dominant_trait in existing_ratchets:
                continue

            trajectory = trait_map.get(dominant_trait)
            if not trajectory:
                continue

            # Gate: trait must not be volatile
            if trajectory.direction == "volatile" or trajectory.volatility > self.GRAD_MAX_VOLATILITY:
                continue

            # Gate: meaningful drift magnitude
            if abs(trajectory.net_drift) < self.GRAD_MIN_DRIFT_MAGNITUDE:
                continue

            # All gates passed: propose graduation
            delta = trajectory.net_drift * self.GRAD_BASELINE_FACTOR
            proposed = trajectory.baseline_value + delta

            # Clamp to [0, 1]
            proposed = max(0.0, min(1.0, proposed))

            confidence = self._graduation_confidence(
                reinforcement_count, trend.sessions_active,
                trajectory.volatility, trajectory.net_drift,
            )

            proposals.append(GraduationProposal(
                signal_key=signal_key,
                trait=dominant_trait,
                direction=dominant_direction,
                current_baseline=trajectory.baseline_value,
                proposed_baseline=proposed,
                delta=delta,
                confidence=confidence,
                reasoning=(
                    f"Signal '{signal_key}' has reinforced {reinforcement_count} times "
                    f"across {trend.sessions_active} sessions, pushing {dominant_trait} "
                    f"{dominant_direction}. Trait drift is {trajectory.net_drift:+.3f} "
                    f"with volatility {trajectory.volatility:.3f}. "
                    f"Proposing {self.GRAD_BASELINE_FACTOR:.0%} of drift as baseline shift."
                ),
                evidence={
                    "reinforcement_count": reinforcement_count,
                    "sessions_active": trend.sessions_active,
                    "signal_trend": trend.trend,
                    "trait_direction": trajectory.direction,
                    "net_drift": round(trajectory.net_drift, 4),
                    "volatility": round(trajectory.volatility, 4),
                },
            ))

        return proposals

    def _get_dominant_trait(
        self, signal_key: str, signal_stats: Dict
    ) -> Tuple[Optional[str], str]:
        """Find the trait most impacted by a signal and its direction.

        Returns (trait_name, "up"/"down") or (None, "") if no data.
        """
        stats = signal_stats.get(signal_key)
        if not stats:
            return None, ""

        traits = getattr(stats, "traits_impacted", {})
        if not traits:
            return None, ""

        dominant = max(traits, key=lambda t: abs(traits[t]))
        direction = "up" if traits[dominant] > 0 else "down"
        return dominant, direction

    def _graduation_confidence(
        self,
        reinforcement_count: int,
        sessions_active: int,
        volatility: float,
        net_drift: float,
    ) -> float:
        """Compute confidence score for a graduation proposal."""
        reinforcement_score = min(1.0, reinforcement_count / 20) * 0.3
        longevity_score = min(1.0, sessions_active / 30) * 0.3
        stability_score = max(0.0, min(1.0, 1.0 - volatility / 0.05)) * 0.2
        magnitude_score = min(1.0, abs(net_drift) / 0.1) * 0.2
        return min(1.0, reinforcement_score + longevity_score + stability_score + magnitude_score)


# ── Pipeline Entry Point ─────────────────────────────────────────────────

def run_adaptation_analysis(
    conn: sqlite3.Connection,
    persona=None,
    trend_report: Optional[TrendReport] = None,
    drift_analytics=None,
) -> Optional[Dict]:
    """Run adaptation analysis. Called from engine.py end().

    Returns result dict or None on error/empty.
    """
    try:
        if trend_report is None:
            return None

        engine = AdaptationEngine(conn, persona)
        result = engine.evaluate(trend_report, drift_analytics)

        if not result.has_recommendations:
            return None

        return result.to_dict()
    except Exception:
        return None
