"""
The Librarian — Drift Analytics
Surfaces signal hit rates, trait trajectories, and persona health metrics
from accumulated disposition_drift entries in the rolodex.

This is the diagnostic dashboard for the persona system. It answers:
- Which signals fire most often? (are patterns too sensitive or dead?)
- Which signals have the most cumulative impact on traits?
- How have traits moved over time? (trajectory)
- Are any signals reinforcing enough to become near-permanent?
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .persona import VALID_TRAIT_NAMES, PersonaProfile


# ─── Data Structures ──────────────────────────────────────────────────────

@dataclass
class SignalStats:
    """Aggregated statistics for a single signal."""
    key: str
    total_fires: int = 0
    total_confidence: float = 0.0
    traits_impacted: Dict[str, float] = field(default_factory=dict)  # trait → net nudge
    sessions_active: int = 0  # how many distinct sessions it appeared in
    last_fired_at: Optional[str] = None
    first_fired_at: Optional[str] = None
    max_reinforcement: int = 0

    @property
    def avg_confidence(self) -> float:
        return self.total_confidence / self.total_fires if self.total_fires else 0.0

    @property
    def net_impact(self) -> float:
        """Total absolute impact across all traits."""
        return sum(abs(v) for v in self.traits_impacted.values())


@dataclass
class TraitSnapshot:
    """A point-in-time snapshot of a trait's value."""
    session_id: str
    value: float
    timestamp: str


@dataclass
class AnalyticsReport:
    """Complete analytics output for CLI display."""
    signal_stats: Dict[str, SignalStats]
    trait_trajectories: Dict[str, List[TraitSnapshot]]
    total_drift_entries: int
    total_sessions_with_drift: int
    most_influential_signals: List[Tuple[str, float]]  # (key, net_impact)
    hottest_traits: List[Tuple[str, float]]  # (trait, total_absolute_movement)
    stale_signals: List[str]  # signals that haven't fired recently
    near_permanent_signals: List[Tuple[str, int]]  # (key, reinforcement_count)


# ─── Analytics Engine ─────────────────────────────────────────────────────

class DriftAnalytics:
    """Computes analytics from disposition_drift entries in the rolodex."""

    def __init__(self, persona: PersonaProfile):
        self.persona = persona

    def analyze(self, drift_rows: List[Dict[str, Any]], session_history: Optional[List[str]] = None) -> AnalyticsReport:
        """Run full analytics on drift entries.

        Args:
            drift_rows: List of dicts with keys: content (JSON string), created_at, session_id
            session_history: Optional ordered list of session_ids for timeline context

        Returns:
            Complete AnalyticsReport
        """
        signal_stats: Dict[str, SignalStats] = {}
        trait_trajectories: Dict[str, List[TraitSnapshot]] = {t: [] for t in VALID_TRAIT_NAMES}
        sessions_seen: set = set()
        signal_sessions: Dict[str, set] = defaultdict(set)

        for row in drift_rows:
            try:
                content = json.loads(row["content"]) if isinstance(row["content"], str) else row["content"]
            except (json.JSONDecodeError, TypeError):
                continue

            signal_key = content.get("signal", "unknown")
            traits_affected = content.get("traits_affected", {})
            snapshot = content.get("active_profile_snapshot", {})
            confidence = content.get("confidence", 0.0)
            reinforcement = content.get("reinforcement_count", 0)
            created_at = row.get("created_at", "")
            session_id = row.get("session_id", content.get("session_id", ""))

            sessions_seen.add(session_id)
            signal_sessions[signal_key].add(session_id)

            # Build or update signal stats
            if signal_key not in signal_stats:
                signal_stats[signal_key] = SignalStats(key=signal_key, first_fired_at=created_at)

            stats = signal_stats[signal_key]
            stats.total_fires += 1
            stats.total_confidence += confidence
            stats.last_fired_at = created_at  # overwrites — rows are chronological
            stats.max_reinforcement = max(stats.max_reinforcement, reinforcement)

            for trait, nudge in traits_affected.items():
                stats.traits_impacted[trait] = stats.traits_impacted.get(trait, 0.0) + nudge

            # Record trait snapshots for trajectory
            for trait, value in snapshot.items():
                if trait in VALID_TRAIT_NAMES:
                    trait_trajectories[trait].append(TraitSnapshot(
                        session_id=session_id,
                        value=value,
                        timestamp=created_at,
                    ))

        # Compute session counts per signal
        for key, sessions in signal_sessions.items():
            if key in signal_stats:
                signal_stats[key].sessions_active = len(sessions)

        # Rank signals by impact
        most_influential = sorted(
            [(k, s.net_impact) for k, s in signal_stats.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Rank traits by total movement
        trait_movement: Dict[str, float] = defaultdict(float)
        for stats in signal_stats.values():
            for trait, nudge in stats.traits_impacted.items():
                trait_movement[trait] += abs(nudge)
        hottest_traits = sorted(
            [(t, v) for t, v in trait_movement.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Identify stale signals (not fired in most recent 25% of entries)
        stale_signals = []
        if drift_rows:
            recent_cutoff = len(drift_rows) * 3 // 4
            recent_signals = set()
            for row in drift_rows[recent_cutoff:]:
                try:
                    c = json.loads(row["content"]) if isinstance(row["content"], str) else row["content"]
                    recent_signals.add(c.get("signal"))
                except (json.JSONDecodeError, TypeError):
                    continue
            stale_signals = [k for k in signal_stats if k not in recent_signals]

        # Near-permanent signals (reinforcement > 8)
        near_permanent = [
            (k, s.max_reinforcement)
            for k, s in signal_stats.items()
            if s.max_reinforcement >= 8
        ]
        near_permanent.sort(key=lambda x: x[1], reverse=True)

        return AnalyticsReport(
            signal_stats=signal_stats,
            trait_trajectories=trait_trajectories,
            total_drift_entries=len(drift_rows),
            total_sessions_with_drift=len(sessions_seen),
            most_influential_signals=most_influential,
            hottest_traits=hottest_traits,
            stale_signals=stale_signals,
            near_permanent_signals=near_permanent,
        )

    # ─── ASCII Rendering ─────────────────────────────────────────────

    def format_signal_table(self, report: AnalyticsReport, top_n: int = 10) -> str:
        """Render signal hit rates as an ASCII table."""
        lines = []
        lines.append("╔══ Signal Hit Rates ══╗")
        lines.append("")
        lines.append(f"  {'Signal':<35} {'Fires':>6} {'Avg Conf':>9} {'Sessions':>9} {'Impact':>8}")
        lines.append(f"  {'─' * 35} {'─' * 6} {'─' * 9} {'─' * 9} {'─' * 8}")

        sorted_signals = sorted(
            report.signal_stats.values(),
            key=lambda s: s.total_fires,
            reverse=True,
        )

        for stats in sorted_signals[:top_n]:
            name = stats.key[:35]
            lines.append(
                f"  {name:<35} {stats.total_fires:>6} "
                f"{stats.avg_confidence:>8.2f} "
                f"{stats.sessions_active:>9} "
                f"{stats.net_impact:>+7.4f}"
            )

        if len(sorted_signals) > top_n:
            lines.append(f"  ... and {len(sorted_signals) - top_n} more signals")

        return "\n".join(lines)

    def format_trait_trajectory(self, report: AnalyticsReport) -> str:
        """Render trait trajectories as ASCII sparklines."""
        lines = []
        lines.append("")
        lines.append("╔══ Trait Trajectories ══╗")
        lines.append("")

        spark_width = 30
        spark_chars = " ▁▂▃▄▅▆▇█"

        for trait in VALID_TRAIT_NAMES:
            snapshots = report.trait_trajectories.get(trait, [])
            if not snapshots:
                continue

            values = [s.value for s in snapshots]

            # Downsample to spark_width points
            if len(values) > spark_width:
                step = len(values) / spark_width
                sampled = [values[int(i * step)] for i in range(spark_width)]
            else:
                sampled = values

            # Build sparkline
            if sampled:
                v_min = min(sampled)
                v_max = max(sampled)
                v_range = v_max - v_min if v_max > v_min else 1.0
                spark = ""
                for v in sampled:
                    idx = int(((v - v_min) / v_range) * (len(spark_chars) - 1))
                    spark += spark_chars[idx]
            else:
                spark = "─" * spark_width

            baseline = self.persona.baseline.get(trait)
            current = self.persona.traits.get(trait)
            diff = current - baseline

            if abs(diff) > 0.001:
                direction = "↑" if diff > 0 else "↓"
                drift_str = f" {direction}{abs(diff):.3f}"
            else:
                drift_str = ""

            lines.append(f"  {trait:<16} {spark}  {baseline:.2f} → {current:.2f}{drift_str}")

        return "\n".join(lines)

    def format_influence_ranking(self, report: AnalyticsReport, top_n: int = 5) -> str:
        """Render the most influential signals with impact bars."""
        lines = []
        lines.append("")
        lines.append("╔══ Most Influential Signals ══╗")
        lines.append("")

        if not report.most_influential_signals:
            lines.append("  No signals have fired yet.")
            return "\n".join(lines)

        max_impact = max(imp for _, imp in report.most_influential_signals[:top_n]) if report.most_influential_signals else 1.0
        bar_width = 20

        for key, impact in report.most_influential_signals[:top_n]:
            if max_impact > 0:
                filled = int((impact / max_impact) * bar_width)
            else:
                filled = 0
            bar = "█" * filled + "░" * (bar_width - filled)
            # Show which traits are affected
            stats = report.signal_stats[key]
            trait_summary = ", ".join(
                f"{t}{'↑' if v > 0 else '↓'}"
                for t, v in sorted(stats.traits_impacted.items(), key=lambda x: abs(x[1]), reverse=True)
            )
            lines.append(f"  {bar} {impact:>+.4f}  {key}")
            lines.append(f"  {'':>22}         → {trait_summary}")

        return "\n".join(lines)

    def format_health_summary(self, report: AnalyticsReport) -> str:
        """Render overall persona health indicators."""
        lines = []
        lines.append("")
        lines.append("╔══ Persona Health ══╗")
        lines.append("")
        lines.append(f"  Total drift entries:     {report.total_drift_entries}")
        lines.append(f"  Sessions with drift:     {report.total_sessions_with_drift}")

        if report.total_drift_entries > 0:
            avg_per_session = report.total_drift_entries / max(report.total_sessions_with_drift, 1)
            lines.append(f"  Avg entries/session:     {avg_per_session:.1f}")

        if report.hottest_traits:
            top_trait, top_movement = report.hottest_traits[0]
            lines.append(f"  Most active trait:       {top_trait} ({top_movement:+.4f} total movement)")

        if report.stale_signals:
            lines.append(f"  Stale signals:           {', '.join(report.stale_signals[:5])}")
            if len(report.stale_signals) > 5:
                lines.append(f"                           ... and {len(report.stale_signals) - 5} more")

        if report.near_permanent_signals:
            lines.append(f"  Near-permanent signals:")
            for key, count in report.near_permanent_signals[:3]:
                effective_decay = self.persona.drift.effective_decay(count)
                lines.append(f"    {key} (reinforced {count}x, decay: {effective_decay:.3f})")

        return "\n".join(lines)

    def format_full_report(self, report: AnalyticsReport, top_n: int = 10) -> str:
        """Render the complete analytics dashboard."""
        sections = [
            self.format_signal_table(report, top_n),
            self.format_influence_ranking(report),
            self.format_trait_trajectory(report),
            self.format_health_summary(report),
        ]
        return "\n".join(sections)

    def to_json(self, report: AnalyticsReport) -> Dict[str, Any]:
        """Serialize analytics report to JSON-safe dict."""
        return {
            "total_drift_entries": report.total_drift_entries,
            "total_sessions_with_drift": report.total_sessions_with_drift,
            "signals": {
                key: {
                    "fires": s.total_fires,
                    "avg_confidence": round(s.avg_confidence, 3),
                    "sessions_active": s.sessions_active,
                    "net_impact": round(s.net_impact, 6),
                    "traits_impacted": {t: round(v, 6) for t, v in s.traits_impacted.items()},
                    "max_reinforcement": s.max_reinforcement,
                    "first_fired_at": s.first_fired_at,
                    "last_fired_at": s.last_fired_at,
                }
                for key, s in report.signal_stats.items()
            },
            "most_influential": [
                {"signal": k, "impact": round(v, 6)}
                for k, v in report.most_influential_signals[:10]
            ],
            "hottest_traits": [
                {"trait": t, "movement": round(v, 6)}
                for t, v in report.hottest_traits
            ],
            "stale_signals": report.stale_signals,
            "near_permanent": [
                {"signal": k, "reinforcement_count": c}
                for k, c in report.near_permanent_signals
            ],
        }


# ─── Query Helper ─────────────────────────────────────────────────────────

def get_drift_entries_query(limit: int = 500) -> str:
    """SQL query to fetch drift entries for analytics."""
    return f"""
        SELECT content, created_at, conversation_id as session_id
        FROM rolodex_entries
        WHERE category = 'disposition_drift'
        ORDER BY created_at ASC
        LIMIT {limit}
    """
