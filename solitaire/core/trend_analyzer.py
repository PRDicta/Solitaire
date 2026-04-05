"""
Solitaire — Cross-Session Trend Analyzer

Detects behavioral trends that span session boundaries. The disposition
filter and identity reflection operate within single sessions. This module
looks across sessions to surface patterns invisible from inside any one.

Three analysis axes:
1. Signal trends: firing rate acceleration/deceleration over session windows
2. Trait trajectories: moving averages and drift direction over time
3. Commitment arcs: outcome distributions that reveal stuck or graduating commitments

Runs at boot (lightweight, cached) or on demand via CLI. No LLM call.
All data comes from disposition_drift entries and identity_signals tables.
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .persona import VALID_TRAIT_NAMES


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class SignalTrend:
    """Firing rate trend for a single disposition signal."""
    key: str
    total_fires: int = 0
    sessions_active: int = 0
    # Rates: fires per session over windows
    rate_5: float = 0.0      # last 5 sessions
    rate_10: float = 0.0     # last 10 sessions
    rate_20: float = 0.0     # last 20 sessions
    trend: str = "stable"    # accelerating, decelerating, stable, extinct, new
    last_fired_session: str = ""


@dataclass
class TraitTrajectory:
    """Moving average and drift direction for a single trait."""
    trait: str
    current_value: float = 0.5
    baseline_value: float = 0.5
    avg_5: float = 0.5       # mean over last 5 sessions
    avg_10: float = 0.5      # mean over last 10 sessions
    direction: str = "stable"  # rising, falling, stable, volatile
    net_drift: float = 0.0   # current - baseline
    volatility: float = 0.0  # std dev of session-end values over last 10


@dataclass
class CommitmentArc:
    """Outcome distribution for a single commitment across sessions."""
    commitment_id: str
    commitment_content: str
    sessions_evaluated: int = 0
    honored_count: int = 0
    missed_count: int = 0
    partial_count: int = 0
    recent_streak: str = ""      # last 5 outcomes as string, e.g. "HMPPH"
    recommendation: str = "monitor"  # escalate, retire, sharpen, monitor


@dataclass
class TrendReport:
    """Complete cross-session trend analysis."""
    session_window: int           # how many sessions were analyzed
    signal_trends: List[SignalTrend] = field(default_factory=list)
    trait_trajectories: List[TraitTrajectory] = field(default_factory=list)
    commitment_arcs: List[CommitmentArc] = field(default_factory=list)
    # Surfaced highlights for boot injection
    alerts: List[str] = field(default_factory=list)

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_window": self.session_window,
            "signal_trends": [
                {
                    "key": s.key,
                    "total_fires": s.total_fires,
                    "sessions_active": s.sessions_active,
                    "rate_5": round(s.rate_5, 3),
                    "rate_10": round(s.rate_10, 3),
                    "rate_20": round(s.rate_20, 3),
                    "trend": s.trend,
                }
                for s in self.signal_trends
            ],
            "trait_trajectories": [
                {
                    "trait": t.trait,
                    "current": round(t.current_value, 4),
                    "baseline": round(t.baseline_value, 4),
                    "avg_5": round(t.avg_5, 4),
                    "avg_10": round(t.avg_10, 4),
                    "direction": t.direction,
                    "net_drift": round(t.net_drift, 4),
                    "volatility": round(t.volatility, 4),
                }
                for t in self.trait_trajectories
            ],
            "commitment_arcs": [
                {
                    "id": c.commitment_id,
                    "content": c.commitment_content[:80],
                    "sessions": c.sessions_evaluated,
                    "honored": c.honored_count,
                    "missed": c.missed_count,
                    "partial": c.partial_count,
                    "streak": c.recent_streak,
                    "recommendation": c.recommendation,
                }
                for c in self.commitment_arcs
            ],
            "alerts": self.alerts,
        }

    def format_boot_summary(self) -> str:
        """Compact summary suitable for boot context injection."""
        if not self.alerts and not self.signal_trends:
            return ""

        lines = ["[TREND ANALYSIS]"]

        # Trait drift summary (only traits that have moved)
        moving_traits = [
            t for t in self.trait_trajectories
            if t.direction != "stable"
        ]
        if moving_traits:
            parts = []
            for t in moving_traits:
                arrow = "+" if t.net_drift > 0 else ""
                parts.append(f"{t.trait} {arrow}{t.net_drift:.3f} ({t.direction})")
            lines.append(f"Trait drift: {', '.join(parts)}")

        # Signal acceleration/deceleration (only notable ones)
        notable_signals = [
            s for s in self.signal_trends
            if s.trend in ("accelerating", "decelerating", "extinct")
        ]
        if notable_signals:
            for s in notable_signals[:3]:
                lines.append(f"Signal '{s.key}': {s.trend} (rate: {s.rate_5:.2f}/session)")

        # Commitment recommendations (only actionable)
        actionable = [
            c for c in self.commitment_arcs
            if c.recommendation != "monitor"
        ]
        if actionable:
            for c in actionable:
                lines.append(
                    f"Commitment '{c.commitment_content[:50]}': {c.recommendation} "
                    f"(streak: {c.recent_streak})"
                )

        # Alerts
        for alert in self.alerts:
            lines.append(f"! {alert}")

        return "\n".join(lines)


# ── Trend Analyzer ───────────────────────────────────────────────────────

class TrendAnalyzer:
    """Cross-session trend detection from disposition drift and identity signals.

    Designed to run at boot or on demand. Reads directly from SQLite.
    No LLM call, no network dependency.
    """

    # Trend classification thresholds
    ACCELERATION_RATIO = 1.5   # rate_5 / rate_10 > this = accelerating
    DECELERATION_RATIO = 0.5   # rate_5 / rate_10 < this = decelerating
    EXTINCT_SESSIONS = 10      # no fires in last N sessions = extinct
    VOLATILE_THRESHOLD = 0.03  # std dev above this = volatile trait
    DRIFT_THRESHOLD = 0.02     # net drift above this = directional

    # Commitment recommendation thresholds
    RETIRE_HONORED_STREAK = 8  # consecutive honored = ready to retire
    ESCALATE_PARTIAL_STREAK = 5  # consecutive partial/missed = needs attention
    SHARPEN_MIXED_RATIO = 0.4  # partial > this fraction = detection is fuzzy

    def __init__(self, conn, persona=None):
        """
        Args:
            conn: SQLite connection (persona-scoped rolodex.db)
            persona: Optional PersonaProfile for baseline values
        """
        self.conn = conn
        self.persona = persona

    def analyze(self, max_sessions: int = 30) -> TrendReport:
        """Run full cross-session trend analysis.

        Args:
            max_sessions: Maximum number of recent sessions to analyze.

        Returns:
            TrendReport with all three axes populated.
        """
        # Get ordered session list from conversations table
        sessions = self._get_recent_sessions(max_sessions)
        if not sessions:
            return TrendReport(session_window=0)

        session_window = len(sessions)
        session_set = set(sessions)

        # Load disposition drift data
        drift_rows = self._get_drift_entries(sessions)

        # Axis 1: Signal trends
        signal_trends = self._analyze_signal_trends(drift_rows, sessions)

        # Axis 2: Trait trajectories
        trait_trajectories = self._analyze_trait_trajectories(drift_rows, sessions)

        # Axis 3: Commitment arcs
        commitment_arcs = self._analyze_commitment_arcs(sessions)

        # Generate alerts from findings
        alerts = self._generate_alerts(signal_trends, trait_trajectories, commitment_arcs)

        return TrendReport(
            session_window=session_window,
            signal_trends=signal_trends,
            trait_trajectories=trait_trajectories,
            commitment_arcs=commitment_arcs,
            alerts=alerts,
        )

    # ── Axis 1: Signal Trends ────────────────────────────────────────────

    def _analyze_signal_trends(
        self,
        drift_rows: List[Dict],
        sessions: List[str],
    ) -> List[SignalTrend]:
        """Compute firing rate trends per signal across session windows."""
        # Build per-signal per-session fire counts
        signal_fires: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        signal_total: Dict[str, int] = defaultdict(int)
        signal_last_session: Dict[str, str] = {}

        for row in drift_rows:
            try:
                content = json.loads(row["content"]) if isinstance(row["content"], str) else row["content"]
            except (json.JSONDecodeError, TypeError):
                continue

            signal_key = content.get("signal", "unknown")
            session_id = row.get("session_id", "")

            signal_fires[signal_key][session_id] += 1
            signal_total[signal_key] += 1
            signal_last_session[signal_key] = session_id

        results = []
        session_count = len(sessions)

        for signal_key, per_session in signal_fires.items():
            trend = SignalTrend(key=signal_key)
            trend.total_fires = signal_total[signal_key]
            trend.sessions_active = len(per_session)
            trend.last_fired_session = signal_last_session.get(signal_key, "")

            # Compute windowed rates
            trend.rate_5 = self._windowed_rate(per_session, sessions, 5)
            trend.rate_10 = self._windowed_rate(per_session, sessions, 10)
            trend.rate_20 = self._windowed_rate(per_session, sessions, 20)

            # Classify trend
            trend.trend = self._classify_signal_trend(
                trend, sessions, per_session
            )

            results.append(trend)

        # Sort by total fires descending
        results.sort(key=lambda s: s.total_fires, reverse=True)
        return results

    def _windowed_rate(
        self,
        per_session: Dict[str, int],
        sessions: List[str],
        window: int,
    ) -> float:
        """Compute fires-per-session rate over the last N sessions."""
        if not sessions:
            return 0.0
        window_sessions = sessions[-window:]
        total_fires = sum(per_session.get(s, 0) for s in window_sessions)
        return total_fires / len(window_sessions)

    def _classify_signal_trend(
        self,
        trend: SignalTrend,
        sessions: List[str],
        per_session: Dict[str, int],
    ) -> str:
        """Classify a signal's trend based on windowed rates."""
        # Check for extinction: no fires in last EXTINCT_SESSIONS sessions
        recent_window = sessions[-self.EXTINCT_SESSIONS:] if len(sessions) >= self.EXTINCT_SESSIONS else sessions
        recent_fires = sum(per_session.get(s, 0) for s in recent_window)
        if recent_fires == 0 and trend.total_fires > 0:
            return "extinct"

        # Check for new signal (only appeared in last 5 sessions)
        all_session_indices = {s: i for i, s in enumerate(sessions)}
        first_session = min(
            (all_session_indices.get(s, len(sessions)) for s in per_session),
            default=0,
        )
        if first_session >= len(sessions) - 5:
            return "new"

        # Compare short vs medium window rates
        if trend.rate_10 > 0:
            ratio = trend.rate_5 / trend.rate_10
            if ratio > self.ACCELERATION_RATIO:
                return "accelerating"
            elif ratio < self.DECELERATION_RATIO:
                return "decelerating"

        return "stable"

    # ── Axis 2: Trait Trajectories ───────────────────────────────────────

    def _analyze_trait_trajectories(
        self,
        drift_rows: List[Dict],
        sessions: List[str],
    ) -> List[TraitTrajectory]:
        """Compute trait moving averages and drift direction."""
        # Build per-session trait snapshots (last snapshot value per session)
        trait_sessions: Dict[str, Dict[str, float]] = {
            t: {} for t in VALID_TRAIT_NAMES
        }

        for row in drift_rows:
            try:
                content = json.loads(row["content"]) if isinstance(row["content"], str) else row["content"]
            except (json.JSONDecodeError, TypeError):
                continue

            snapshot = content.get("active_profile_snapshot", {})
            session_id = row.get("session_id", "")

            for trait, value in snapshot.items():
                if trait in VALID_TRAIT_NAMES:
                    # Keep the last value per session (chronological order)
                    trait_sessions[trait][session_id] = value

        results = []
        for trait in VALID_TRAIT_NAMES:
            traj = TraitTrajectory(trait=trait)

            # Get baseline from persona
            if self.persona:
                traj.baseline_value = self.persona.baseline.get(trait, 0.5)
                traj.current_value = self.persona.traits.get(trait, 0.5)
            else:
                traj.baseline_value = 0.5
                traj.current_value = 0.5

            traj.net_drift = traj.current_value - traj.baseline_value

            # Compute session-end values in session order
            session_values = []
            for s in sessions:
                if s in trait_sessions[trait]:
                    session_values.append(trait_sessions[trait][s])

            if not session_values:
                traj.direction = "stable"
                results.append(traj)
                continue

            # Moving averages
            traj.avg_5 = self._mean(session_values[-5:])
            traj.avg_10 = self._mean(session_values[-10:])

            # Volatility (std dev over last 10 session-end values)
            recent_values = session_values[-10:]
            traj.volatility = self._std_dev(recent_values)

            # Direction classification
            traj.direction = self._classify_trait_direction(traj, session_values)

            results.append(traj)

        return results

    def _classify_trait_direction(
        self,
        traj: TraitTrajectory,
        session_values: List[float],
    ) -> str:
        """Classify trait trajectory direction."""
        if traj.volatility > self.VOLATILE_THRESHOLD:
            # High variance could mask a trend. Check if there's
            # still a consistent direction underneath the noise.
            if abs(traj.net_drift) > self.DRIFT_THRESHOLD * 2:
                return "rising" if traj.net_drift > 0 else "falling"
            return "volatile"

        if abs(traj.net_drift) < self.DRIFT_THRESHOLD:
            return "stable"

        # Check short vs medium average for recency
        if len(session_values) >= 5:
            recent_trend = traj.avg_5 - traj.avg_10
            if abs(recent_trend) > self.DRIFT_THRESHOLD:
                return "rising" if recent_trend > 0 else "falling"

        return "rising" if traj.net_drift > 0 else "falling"

    # ── Axis 3: Commitment Arcs ──────────────────────────────────────────

    def _analyze_commitment_arcs(
        self,
        sessions: List[str],
    ) -> List[CommitmentArc]:
        """Analyze commitment outcome distributions across sessions."""
        # Get active commitments from identity graph
        try:
            commitments = self.conn.execute(
                """SELECT id, content, metadata FROM identity_nodes
                   WHERE node_type = 'commitment'
                   AND (status IS NULL OR status = 'active')""",
            ).fetchall()
        except Exception:
            return []

        if not commitments:
            return []

        session_set = set(sessions)
        results = []

        for row in commitments:
            c_id, c_content, c_meta_raw = row

            arc = CommitmentArc(
                commitment_id=c_id,
                commitment_content=c_content,
            )

            # Get all signals for this commitment
            try:
                signals = self.conn.execute(
                    """SELECT session_id, signal_type, source, confidence
                       FROM identity_signals
                       WHERE commitment_id = ?
                       ORDER BY created_at ASC""",
                    (c_id,),
                ).fetchall()
            except Exception:
                continue

            if not signals:
                results.append(arc)
                continue

            # Aggregate per-session outcomes
            # Within a session, the outcome is determined by the highest-weight signals
            session_outcomes: Dict[str, str] = {}
            session_signal_weights: Dict[str, Dict[str, float]] = defaultdict(
                lambda: defaultdict(float)
            )

            for sig_session, sig_type, sig_source, sig_conf in signals:
                if sig_session not in session_set:
                    continue

                # Weight by source reliability (Vazire SOKA)
                weight = {
                    "user_correction": 1.0,
                    "enrichment_scanner": 0.6,
                    "self_report": 0.3,
                }.get(sig_source, 0.5)

                session_signal_weights[sig_session][sig_type] += weight * sig_conf

            # Determine per-session outcome from weighted signal aggregation
            for session_id, type_weights in session_signal_weights.items():
                held_w = type_weights.get("held", 0.0)
                missed_w = type_weights.get("missed", 0.0)

                if held_w > missed_w and held_w > 0:
                    session_outcomes[session_id] = "honored"
                elif missed_w > held_w and missed_w > 0:
                    session_outcomes[session_id] = "missed"
                elif held_w > 0 or missed_w > 0:
                    session_outcomes[session_id] = "partial"

            # Count outcomes
            for outcome in session_outcomes.values():
                if outcome == "honored":
                    arc.honored_count += 1
                elif outcome == "missed":
                    arc.missed_count += 1
                elif outcome == "partial":
                    arc.partial_count += 1

            arc.sessions_evaluated = len(session_outcomes)

            # Build recent streak (last 5 sessions with outcomes, in order)
            streak_chars = []
            for s in sessions:
                if s in session_outcomes:
                    outcome = session_outcomes[s]
                    streak_chars.append(
                        {"honored": "H", "missed": "M", "partial": "P"}[outcome]
                    )
            arc.recent_streak = "".join(streak_chars[-5:])

            # Determine recommendation
            arc.recommendation = self._recommend_commitment_action(arc)

            results.append(arc)

        return results

    def _recommend_commitment_action(self, arc: CommitmentArc) -> str:
        """Recommend an action for a commitment based on its outcome arc."""
        if arc.sessions_evaluated < 3:
            return "monitor"  # not enough data

        streak = arc.recent_streak

        # Retirement: consistent honored streak
        if len(streak) >= self.RETIRE_HONORED_STREAK:
            if all(c == "H" for c in streak[-self.RETIRE_HONORED_STREAK:]):
                return "retire"
        # Shorter retirement for very strong signals
        if len(streak) >= 5 and all(c == "H" for c in streak[-5:]):
            if arc.honored_count > arc.sessions_evaluated * 0.8:
                return "retire"

        # Escalation: consistent partial or missed
        recent = streak[-self.ESCALATE_PARTIAL_STREAK:]
        if len(recent) >= self.ESCALATE_PARTIAL_STREAK:
            non_honored = sum(1 for c in recent if c != "H")
            if non_honored >= self.ESCALATE_PARTIAL_STREAK:
                return "escalate"

        # Sharpening: high partial ratio suggests fuzzy detection
        if arc.sessions_evaluated >= 5:
            partial_ratio = arc.partial_count / arc.sessions_evaluated
            if partial_ratio > self.SHARPEN_MIXED_RATIO:
                return "sharpen"

        return "monitor"

    # ── Alert Generation ─────────────────────────────────────────────────

    def _generate_alerts(
        self,
        signal_trends: List[SignalTrend],
        trait_trajectories: List[TraitTrajectory],
        commitment_arcs: List[CommitmentArc],
    ) -> List[str]:
        """Generate human-readable alerts from trend data."""
        alerts = []

        # Signal alerts
        for s in signal_trends:
            if s.trend == "accelerating" and s.rate_5 > 1.0:
                alerts.append(
                    f"Signal '{s.key}' is accelerating: "
                    f"{s.rate_5:.1f}/session (was {s.rate_10:.1f})"
                )
            elif s.trend == "extinct" and s.total_fires > 10:
                alerts.append(
                    f"Signal '{s.key}' has gone extinct "
                    f"({s.total_fires} lifetime fires, none recently)"
                )

        # Trait alerts
        for t in trait_trajectories:
            if t.direction == "volatile":
                alerts.append(
                    f"Trait '{t.trait}' is volatile "
                    f"(std dev {t.volatility:.3f} over last 10 sessions)"
                )
            elif t.direction in ("rising", "falling") and abs(t.net_drift) > 0.05:
                alerts.append(
                    f"Trait '{t.trait}' has drifted {t.net_drift:+.3f} from baseline "
                    f"({t.direction}, now {t.current_value:.3f})"
                )

        # Commitment alerts
        for c in commitment_arcs:
            if c.recommendation == "retire":
                alerts.append(
                    f"Commitment '{c.commitment_content[:40]}' ready to retire "
                    f"(streak: {c.recent_streak})"
                )
            elif c.recommendation == "escalate":
                alerts.append(
                    f"Commitment '{c.commitment_content[:40]}' needs attention "
                    f"(streak: {c.recent_streak})"
                )
            elif c.recommendation == "sharpen":
                alerts.append(
                    f"Commitment '{c.commitment_content[:40]}' detection may be fuzzy "
                    f"({c.partial_count}/{c.sessions_evaluated} partial)"
                )

        return alerts

    # ── Data Loading ─────────────────────────────────────────────────────

    def _get_recent_sessions(self, max_sessions: int) -> List[str]:
        """Get the most recent session IDs in chronological order."""
        try:
            rows = self.conn.execute(
                """SELECT id FROM conversations
                   WHERE status = 'ended' OR status = 'active'
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (max_sessions,),
            ).fetchall()
            # Reverse to chronological order
            return [r[0] for r in reversed(rows)]
        except Exception:
            return []

    def _get_drift_entries(self, sessions: List[str]) -> List[Dict]:
        """Load disposition drift entries for the given sessions."""
        if not sessions:
            return []

        try:
            placeholders = ",".join("?" * len(sessions))
            rows = self.conn.execute(
                f"""SELECT content, created_at, conversation_id as session_id
                    FROM rolodex_entries
                    WHERE category = 'disposition_drift'
                    AND conversation_id IN ({placeholders})
                    ORDER BY created_at ASC""",
                sessions,
            ).fetchall()
            return [
                {"content": r[0], "created_at": r[1], "session_id": r[2]}
                for r in rows
            ]
        except Exception:
            return []

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _std_dev(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5


# ── Pipeline Entry Point ─────────────────────────────────────────────────

def run_trend_analysis(
    conn,
    persona=None,
    max_sessions: int = 30,
) -> Optional[Dict]:
    """Run trend analysis. Called from boot or CLI.

    Returns report dict or None on error.
    """
    try:
        analyzer = TrendAnalyzer(conn, persona)
        report = analyzer.analyze(max_sessions)
        if report.session_window > 0:
            return report.to_dict()
        return None
    except Exception:
        return None  # Trend analysis is additive; never block boot or session


def format_trend_report(report_dict: Dict) -> str:
    """Format a trend report dict for CLI display."""
    lines = []
    lines.append("╔══ Cross-Session Trends ══╗")
    lines.append(f"  Analyzed {report_dict['session_window']} sessions")
    lines.append("")

    # Signal trends
    trends = report_dict.get("signal_trends", [])
    notable = [s for s in trends if s["trend"] != "stable"]
    if notable:
        lines.append("  Signal Trends:")
        for s in notable[:8]:
            lines.append(
                f"    {s['key']:<30} {s['trend']:<14} "
                f"rate: {s['rate_5']:.2f}/sess (5) {s['rate_10']:.2f}/sess (10)"
            )
        lines.append("")

    # Trait trajectories
    traits = report_dict.get("trait_trajectories", [])
    moving = [t for t in traits if t["direction"] != "stable"]
    if moving:
        lines.append("  Trait Movement:")
        for t in moving:
            arrow = "+" if t["net_drift"] > 0 else ""
            lines.append(
                f"    {t['trait']:<16} {arrow}{t['net_drift']:.4f} ({t['direction']})"
                f"  avg5={t['avg_5']:.3f} avg10={t['avg_10']:.3f}"
            )
        lines.append("")

    # Commitment arcs
    arcs = report_dict.get("commitment_arcs", [])
    actionable = [c for c in arcs if c["recommendation"] != "monitor"]
    if actionable:
        lines.append("  Commitment Recommendations:")
        for c in actionable:
            lines.append(
                f"    [{c['recommendation'].upper():>8}] {c['content']:<50} "
                f"streak: {c['streak']}"
            )
        lines.append("")

    # Alerts
    alerts = report_dict.get("alerts", [])
    if alerts:
        lines.append("  Alerts:")
        for a in alerts:
            lines.append(f"    ! {a}")

    return "\n".join(lines)
