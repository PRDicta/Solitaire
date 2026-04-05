"""
Solitaire — Behavioral Diff

Generates a concrete picture of how the current session's behavioral profile
compares to recent history. Instead of just HONORED/MISSED/PARTIAL outcomes,
this produces a readable diff: which traits moved, which signals fired,
which commitments were active, and how this session compares to the rolling
average.

Designed to run at session end (after identity reflection) and produce
output suitable for the trace log, residue, or boot injection.

No LLM call. All data from disposition_drift entries and identity_signals.
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .persona import VALID_TRAIT_NAMES, PersonaProfile


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class TraitDelta:
    """How a single trait moved this session vs. recent average."""
    trait: str
    session_start: float = 0.5
    session_end: float = 0.5
    session_delta: float = 0.0       # end - start within this session
    rolling_avg: float = 0.5         # avg session-end value over last 5
    deviation: float = 0.0           # session_end - rolling_avg
    direction: str = ""              # "higher", "lower", "unchanged"
    signal_count: int = 0            # how many drift events touched this trait


@dataclass
class SignalActivity:
    """Signal firing activity for this session."""
    key: str
    fires: int = 0
    avg_fires_recent: float = 0.0    # avg fires per session over last 5
    deviation: str = ""              # "above_avg", "below_avg", "normal", "first_time"


@dataclass
class CommitmentStatus:
    """Commitment status for this session."""
    commitment_id: str
    content: str
    outcome: str = "not_evaluated"   # honored, missed, partial, not_evaluated
    signal_count: int = 0
    sources: List[str] = field(default_factory=list)  # signal sources


@dataclass
class BehavioralDiff:
    """Complete behavioral diff for a session."""
    session_id: str
    trait_deltas: List[TraitDelta] = field(default_factory=list)
    signal_activity: List[SignalActivity] = field(default_factory=list)
    commitment_statuses: List[CommitmentStatus] = field(default_factory=list)
    # Summary stats
    total_drift_events: int = 0
    unique_signals_fired: int = 0
    traits_moved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_drift_events": self.total_drift_events,
            "unique_signals_fired": self.unique_signals_fired,
            "traits_moved": self.traits_moved,
            "trait_deltas": [
                {
                    "trait": t.trait,
                    "session_delta": round(t.session_delta, 4),
                    "deviation": round(t.deviation, 4),
                    "direction": t.direction,
                    "signal_count": t.signal_count,
                }
                for t in self.trait_deltas
                if t.direction != "unchanged"
            ],
            "signal_activity": [
                {
                    "key": s.key,
                    "fires": s.fires,
                    "avg_recent": round(s.avg_fires_recent, 2),
                    "deviation": s.deviation,
                }
                for s in self.signal_activity
                if s.fires > 0
            ],
            "commitment_statuses": [
                {
                    "id": c.commitment_id,
                    "content": c.content[:60],
                    "outcome": c.outcome,
                    "signals": c.signal_count,
                }
                for c in self.commitment_statuses
            ],
        }

    def format_readable(self) -> str:
        """Human-readable diff for trace log or residue."""
        lines = []

        # Trait movement
        moved = [t for t in self.trait_deltas if t.direction != "unchanged"]
        if moved:
            parts = []
            for t in moved:
                arrow = "+" if t.session_delta > 0 else ""
                vs = ""
                if t.deviation != 0.0:
                    vs_dir = "above" if t.deviation > 0 else "below"
                    vs = f", {vs_dir} avg by {abs(t.deviation):.3f}"
                parts.append(f"{t.trait} {arrow}{t.session_delta:.3f}{vs}")
            lines.append(f"Traits: {', '.join(parts)}")

        # Signal activity (deviations from normal only)
        notable = [
            s for s in self.signal_activity
            if s.deviation in ("above_avg", "first_time")
        ]
        if notable:
            parts = [
                f"{s.key} ({s.fires}x, {s.deviation})"
                for s in notable[:5]
            ]
            lines.append(f"Signals: {', '.join(parts)}")

        # Commitments
        active = [c for c in self.commitment_statuses if c.outcome != "not_evaluated"]
        if active:
            parts = [
                f"{c.content[:30]}={c.outcome}"
                for c in active
            ]
            lines.append(f"Commitments: {', '.join(parts)}")

        if not lines:
            lines.append("No notable behavioral changes this session.")

        return " | ".join(lines)


# ── Diff Generator ───────────────────────────────────────────────────────

class BehavioralDiffGenerator:
    """Generates behavioral diffs by comparing session data to recent history.

    Reads from the rolodex (disposition_drift entries) and identity graph
    (identity_signals). No LLM call.
    """

    MOVEMENT_THRESHOLD = 0.002  # ignore trait movements smaller than this

    def __init__(self, conn, persona: Optional[PersonaProfile] = None):
        self.conn = conn
        self.persona = persona

    def generate(
        self,
        session_id: str,
        pattern_snapshot: Optional[Dict[str, float]] = None,
        recent_window: int = 5,
    ) -> BehavioralDiff:
        """Generate a behavioral diff for a session.

        Args:
            session_id: Current session ID.
            pattern_snapshot: Trait values captured at session start.
                Format: {trait_name: value} from persona.traits at boot.
            recent_window: Number of prior sessions for rolling average.

        Returns:
            BehavioralDiff with all axes populated.
        """
        diff = BehavioralDiff(session_id=session_id)

        # Get prior sessions for comparison
        prior_sessions = self._get_prior_sessions(session_id, recent_window)

        # Load drift data for current session
        current_drift = self._get_session_drift(session_id)
        diff.total_drift_events = len(current_drift)

        # Load drift data for prior sessions
        prior_drift = self._get_sessions_drift(prior_sessions) if prior_sessions else []

        # Axis 1: Trait deltas
        diff.trait_deltas = self._compute_trait_deltas(
            current_drift, prior_drift, prior_sessions, pattern_snapshot
        )
        diff.traits_moved = sum(
            1 for t in diff.trait_deltas if t.direction != "unchanged"
        )

        # Axis 2: Signal activity
        diff.signal_activity = self._compute_signal_activity(
            current_drift, prior_drift, prior_sessions
        )
        diff.unique_signals_fired = sum(
            1 for s in diff.signal_activity if s.fires > 0
        )

        # Axis 3: Commitment statuses
        diff.commitment_statuses = self._compute_commitment_statuses(session_id)

        return diff

    # ── Trait Deltas ─────────────────────────────────────────────────────

    def _compute_trait_deltas(
        self,
        current_drift: List[Dict],
        prior_drift: List[Dict],
        prior_sessions: List[str],
        snapshot: Optional[Dict[str, float]],
    ) -> List[TraitDelta]:
        """Compute how each trait moved this session vs. rolling average."""
        # Get current trait values from persona
        current_values = {}
        start_values = {}
        if self.persona:
            current_values = dict(self.persona.traits._values)
            start_values = dict(snapshot) if snapshot else dict(self.persona.baseline)
        else:
            for t in VALID_TRAIT_NAMES:
                current_values[t] = 0.5
                start_values[t] = 0.5

        # Compute rolling average from prior sessions
        prior_session_ends = self._extract_session_end_values(prior_drift, prior_sessions)

        # Count signals per trait in current session
        trait_signal_counts = defaultdict(int)
        for row in current_drift:
            content = self._parse_content(row)
            if not content:
                continue
            for trait in content.get("traits_affected", {}):
                trait_signal_counts[trait] += 1

        results = []
        for trait in VALID_TRAIT_NAMES:
            delta = TraitDelta(trait=trait)
            delta.session_start = start_values.get(trait, 0.5)
            delta.session_end = current_values.get(trait, 0.5)
            delta.session_delta = delta.session_end - delta.session_start
            delta.signal_count = trait_signal_counts.get(trait, 0)

            # Rolling average from prior sessions
            prior_values = prior_session_ends.get(trait, [])
            if prior_values:
                delta.rolling_avg = sum(prior_values) / len(prior_values)
            else:
                delta.rolling_avg = delta.session_start

            delta.deviation = delta.session_end - delta.rolling_avg

            # Classify direction
            if abs(delta.session_delta) < self.MOVEMENT_THRESHOLD:
                delta.direction = "unchanged"
            elif delta.session_delta > 0:
                delta.direction = "higher"
            else:
                delta.direction = "lower"

            results.append(delta)

        return results

    def _extract_session_end_values(
        self,
        drift_rows: List[Dict],
        sessions: List[str],
    ) -> Dict[str, List[float]]:
        """Extract the last trait snapshot value per session."""
        # Per session, keep the last snapshot for each trait
        session_trait_last: Dict[str, Dict[str, float]] = defaultdict(dict)

        for row in drift_rows:
            content = self._parse_content(row)
            if not content:
                continue
            session_id = row.get("session_id", "")
            snapshot = content.get("active_profile_snapshot", {})
            for trait, value in snapshot.items():
                if trait in VALID_TRAIT_NAMES:
                    session_trait_last[session_id][trait] = value

        # Collect per-trait values across sessions (in order)
        result: Dict[str, List[float]] = defaultdict(list)
        for s in sessions:
            if s in session_trait_last:
                for trait, value in session_trait_last[s].items():
                    result[trait].append(value)

        return result

    # ── Signal Activity ──────────────────────────────────────────────────

    def _compute_signal_activity(
        self,
        current_drift: List[Dict],
        prior_drift: List[Dict],
        prior_sessions: List[str],
    ) -> List[SignalActivity]:
        """Compute signal firing rates and compare to recent average."""
        # Current session counts
        current_counts: Dict[str, int] = defaultdict(int)
        for row in current_drift:
            content = self._parse_content(row)
            if not content:
                continue
            key = content.get("signal", "unknown")
            current_counts[key] += 1

        # Prior session counts (per signal per session)
        prior_per_session: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in prior_drift:
            content = self._parse_content(row)
            if not content:
                continue
            key = content.get("signal", "unknown")
            session_id = row.get("session_id", "")
            prior_per_session[key][session_id] += 1

        # All signal keys seen
        all_keys = set(current_counts.keys()) | set(prior_per_session.keys())

        results = []
        num_prior = max(len(prior_sessions), 1)

        for key in sorted(all_keys):
            activity = SignalActivity(key=key)
            activity.fires = current_counts.get(key, 0)

            # Average over prior sessions
            prior_counts = prior_per_session.get(key, {})
            total_prior = sum(prior_counts.values())
            activity.avg_fires_recent = total_prior / num_prior

            # Classify deviation
            if activity.fires > 0 and activity.avg_fires_recent == 0:
                activity.deviation = "first_time"
            elif activity.fires > activity.avg_fires_recent * 1.5 and activity.fires >= 2:
                activity.deviation = "above_avg"
            elif activity.fires < activity.avg_fires_recent * 0.5 and activity.avg_fires_recent >= 1:
                activity.deviation = "below_avg"
            else:
                activity.deviation = "normal"

            results.append(activity)

        # Sort: fires descending
        results.sort(key=lambda s: s.fires, reverse=True)
        return results

    # ── Commitment Statuses ──────────────────────────────────────────────

    def _compute_commitment_statuses(
        self,
        session_id: str,
    ) -> List[CommitmentStatus]:
        """Get commitment evaluation outcomes for this session."""
        try:
            # Get active commitments
            commitments = self.conn.execute(
                """SELECT id, content FROM identity_nodes
                   WHERE node_type = 'commitment'
                   AND (status IS NULL OR status = 'active')""",
            ).fetchall()
        except Exception:
            return []

        if not commitments:
            return []

        results = []
        for c_id, c_content in commitments:
            status = CommitmentStatus(
                commitment_id=c_id,
                content=c_content,
            )

            # Get signals for this commitment in this session
            try:
                signals = self.conn.execute(
                    """SELECT signal_type, source, confidence
                       FROM identity_signals
                       WHERE commitment_id = ?
                       AND session_id = ?""",
                    (c_id, session_id),
                ).fetchall()
            except Exception:
                signals = []

            status.signal_count = len(signals)
            status.sources = list(set(s[1] for s in signals))

            if not signals:
                status.outcome = "not_evaluated"
            else:
                # Weighted outcome (same logic as identity_reflection)
                weights = {
                    "user_correction": 1.0,
                    "enrichment_scanner": 0.6,
                    "self_report": 0.3,
                }
                held_w = sum(
                    weights.get(s[1], 0.5) * s[2]
                    for s in signals if s[0] == "held"
                )
                missed_w = sum(
                    weights.get(s[1], 0.5) * s[2]
                    for s in signals if s[0] == "missed"
                )

                if held_w > missed_w and held_w > 0:
                    status.outcome = "honored"
                elif missed_w > held_w and missed_w > 0:
                    status.outcome = "missed"
                elif held_w > 0 or missed_w > 0:
                    status.outcome = "partial"
                else:
                    status.outcome = "not_evaluated"

            results.append(status)

        return results

    # ── Data Loading ─────────────────────────────────────────────────────

    def _get_prior_sessions(self, current_session: str, window: int) -> List[str]:
        """Get the N sessions before the current one, in chronological order."""
        try:
            rows = self.conn.execute(
                """SELECT id FROM conversations
                   WHERE id != ?
                   AND (status = 'ended' OR status = 'active')
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (current_session, window),
            ).fetchall()
            return [r[0] for r in reversed(rows)]
        except Exception:
            return []

    def _get_session_drift(self, session_id: str) -> List[Dict]:
        """Load drift entries for a single session."""
        try:
            rows = self.conn.execute(
                """SELECT content, created_at, conversation_id as session_id
                   FROM rolodex_entries
                   WHERE category = 'disposition_drift'
                   AND conversation_id = ?
                   ORDER BY created_at ASC""",
                (session_id,),
            ).fetchall()
            return [
                {"content": r[0], "created_at": r[1], "session_id": r[2]}
                for r in rows
            ]
        except Exception:
            return []

    def _get_sessions_drift(self, sessions: List[str]) -> List[Dict]:
        """Load drift entries for multiple sessions."""
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

    @staticmethod
    def _parse_content(row: Dict) -> Optional[Dict]:
        """Parse JSON content from a drift row."""
        try:
            c = row.get("content", "")
            if isinstance(c, str):
                return json.loads(c)
            return c
        except (json.JSONDecodeError, TypeError):
            return None


# ── Pipeline Entry Point ─────────────────────────────────────────────────

def generate_behavioral_diff(
    conn,
    session_id: str,
    persona: Optional[PersonaProfile] = None,
    pattern_snapshot: Optional[Dict[str, float]] = None,
) -> Optional[Dict]:
    """Generate behavioral diff for a session. Called from cmd_end.

    Returns report dict or None on error.
    """
    try:
        generator = BehavioralDiffGenerator(conn, persona)
        diff = generator.generate(session_id, pattern_snapshot)
        if diff.total_drift_events > 0 or diff.commitment_statuses:
            return diff.to_dict()
        return None
    except Exception:
        return None  # Behavioral diff is additive; never block session end


def format_behavioral_diff(diff: BehavioralDiff) -> str:
    """Format for trace log or CLI display."""
    return diff.format_readable()
