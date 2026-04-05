"""Tests for the adaptation engine (thermostat)."""
import json
import sqlite3
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from solitaire.core.adaptation_engine import (
    AdaptationEngine,
    AdaptationResult,
    CommitmentRecommendation,
    GraduationProposal,
    run_adaptation_analysis,
)
from solitaire.core.trend_analyzer import (
    CommitmentArc,
    SignalTrend,
    TraitTrajectory,
    TrendReport,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    """In-memory SQLite database with schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            status TEXT DEFAULT 'ended'
        )
    """)
    conn.execute("""
        CREATE TABLE identity_nodes (
            id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT,
            confidence REAL,
            observation_count INTEGER DEFAULT 1,
            metadata TEXT DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE TABLE identity_signals (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            commitment_id TEXT,
            signal_type TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            created_at DATETIME NOT NULL
        )
    """)
    conn.commit()
    return conn


def _make_arc(
    commitment_id="c1",
    content="Test commitment",
    sessions_evaluated=10,
    honored=8,
    missed=1,
    partial=1,
    streak="HHHPH",
    recommendation="monitor",
) -> CommitmentArc:
    arc = CommitmentArc(
        commitment_id=commitment_id,
        commitment_content=content,
    )
    arc.sessions_evaluated = sessions_evaluated
    arc.honored_count = honored
    arc.missed_count = missed
    arc.partial_count = partial
    arc.recent_streak = streak
    arc.recommendation = recommendation
    return arc


def _make_signal_trend(
    key="warmth_appreciated",
    total_fires=20,
    sessions_active=18,
    trend="stable",
) -> SignalTrend:
    return SignalTrend(
        key=key,
        total_fires=total_fires,
        sessions_active=sessions_active,
        trend=trend,
    )


def _make_trajectory(
    trait="warmth",
    current=0.6,
    baseline=0.55,
    direction="rising",
    net_drift=0.05,
    volatility=0.01,
) -> TraitTrajectory:
    return TraitTrajectory(
        trait=trait,
        current_value=current,
        baseline_value=baseline,
        direction=direction,
        net_drift=net_drift,
        volatility=volatility,
    )


def _make_report(
    session_window=30,
    arcs=None,
    signal_trends=None,
    trait_trajectories=None,
) -> TrendReport:
    return TrendReport(
        session_window=session_window,
        commitment_arcs=arcs or [],
        signal_trends=signal_trends or [],
        trait_trajectories=trait_trajectories or [],
    )


def _make_signal_stats(key, traits_impacted):
    """Build a mock signal stats object with traits_impacted."""
    stats = MagicMock()
    stats.traits_impacted = traits_impacted
    return {key: stats}


# ── Tests: Commitment Retirement ─────────────────────────────────────────

class TestCommitmentRetirement:

    def test_retire_with_strong_honored_streak(self, db):
        """Commitment with long honored streak should get retire recommendation."""
        arc = _make_arc(
            sessions_evaluated=10, honored=10, missed=0, partial=0,
            streak="HHHHH", recommendation="retire",
        )
        report = _make_report(arcs=[arc])

        engine = AdaptationEngine(db)
        result = engine.evaluate(report)

        assert len(result.commitment_recommendations) == 1
        rec = result.commitment_recommendations[0]
        assert rec.action == "retire"
        assert rec.confidence > 0.5

    def test_no_retire_when_monitor(self, db):
        """Commitment classified as 'monitor' should not get retire recommendation."""
        arc = _make_arc(
            sessions_evaluated=10, honored=6, missed=2, partial=2,
            streak="HPMHH", recommendation="monitor",
        )
        report = _make_report(session_window=30, arcs=[arc])

        engine = AdaptationEngine(db)
        result = engine.evaluate(report)

        assert not any(r.action == "retire" for r in result.commitment_recommendations)

    def test_retire_confidence_scales_with_streak(self, db):
        """Longer honored streak should yield higher confidence."""
        arc_short = _make_arc(streak="HMHHH", honored=8, sessions_evaluated=10, recommendation="retire")
        arc_long = _make_arc(
            commitment_id="c2", streak="HHHHH", honored=10,
            sessions_evaluated=10, recommendation="retire",
        )
        report = _make_report(arcs=[arc_short, arc_long])

        engine = AdaptationEngine(db)
        result = engine.evaluate(report)

        recs = {r.commitment_id: r for r in result.commitment_recommendations}
        assert recs["c2"].confidence > recs["c1"].confidence

    def test_retire_includes_reasoning(self, db):
        """Retirement recommendation should include human-readable reasoning."""
        arc = _make_arc(
            honored=10, sessions_evaluated=10, streak="HHHHH",
            recommendation="retire",
        )
        report = _make_report(arcs=[arc])

        engine = AdaptationEngine(db)
        result = engine.evaluate(report)

        rec = result.commitment_recommendations[0]
        assert "Honored" in rec.reasoning
        assert "10" in rec.reasoning
        assert len(rec.suggested_actions) >= 1


# ── Tests: Commitment Escalation ─────────────────────────────────────────

class TestCommitmentEscalation:

    def test_escalate_with_consecutive_misses(self, db):
        """Commitment with consecutive misses should get escalate recommendation."""
        arc = _make_arc(
            sessions_evaluated=10, honored=2, missed=6, partial=2,
            streak="MMMMP", recommendation="escalate",
        )
        report = _make_report(arcs=[arc])

        engine = AdaptationEngine(db)
        result = engine.evaluate(report)

        assert len(result.commitment_recommendations) == 1
        rec = result.commitment_recommendations[0]
        assert rec.action == "escalate"
        assert rec.confidence > 0.4

    def test_escalate_includes_reword_suggestion_for_all_misses(self, db):
        """All-miss streak should suggest rewording."""
        arc = _make_arc(
            sessions_evaluated=8, honored=1, missed=7, partial=0,
            streak="MMMMM", recommendation="escalate",
        )
        report = _make_report(arcs=[arc])

        engine = AdaptationEngine(db)
        result = engine.evaluate(report)

        rec = result.commitment_recommendations[0]
        assert any("specific" in s.lower() or "reword" in s.lower() for s in rec.suggested_actions)

    def test_escalate_mixed_pattern_different_suggestions(self, db):
        """Mixed miss/partial pattern should get different suggestions than pure miss."""
        arc = _make_arc(
            sessions_evaluated=10, honored=2, missed=4, partial=4,
            streak="MPMPM", recommendation="escalate",
        )
        report = _make_report(arcs=[arc])

        engine = AdaptationEngine(db)
        result = engine.evaluate(report)

        rec = result.commitment_recommendations[0]
        assert any("trigger" in s.lower() or "scope" in s.lower() for s in rec.suggested_actions)

    def test_escalate_confidence_scales_with_miss_rate(self, db):
        """Higher miss rate should yield higher confidence."""
        arc_low = _make_arc(
            commitment_id="c1", sessions_evaluated=10,
            honored=3, missed=4, partial=3,
            streak="MPMHM", recommendation="escalate",
        )
        arc_high = _make_arc(
            commitment_id="c2", sessions_evaluated=10,
            honored=1, missed=8, partial=1,
            streak="MMMMM", recommendation="escalate",
        )
        report = _make_report(arcs=[arc_low, arc_high])

        engine = AdaptationEngine(db)
        result = engine.evaluate(report)

        recs = {r.commitment_id: r for r in result.commitment_recommendations}
        assert recs["c2"].confidence > recs["c1"].confidence


# ── Tests: Commitment Sharpening ─────────────────────────────────────────

class TestCommitmentSharpening:

    def test_sharpen_high_partial_ratio(self, db):
        """High partial ratio from TrendAnalyzer should produce sharpen recommendation."""
        arc = _make_arc(
            sessions_evaluated=10, honored=2, missed=1, partial=7,
            streak="PPPHP", recommendation="sharpen",
        )
        report = _make_report(session_window=30, arcs=[arc])

        engine = AdaptationEngine(db)
        result = engine.evaluate(report)

        assert len(result.commitment_recommendations) == 1
        rec = result.commitment_recommendations[0]
        assert rec.action == "sharpen"

    def test_sharpen_low_evaluation_rate(self, db):
        """Commitment with very low evaluation rate should trigger independent sharpen."""
        # This commitment exists for 20 sessions but only evaluated in 3
        now = datetime.now(timezone.utc)
        first_seen = (now - timedelta(days=30)).isoformat()

        # Add 20 sessions
        for i in range(20):
            db.execute(
                "INSERT INTO conversations (id, created_at, status) VALUES (?, ?, 'ended')",
                (f"sess_{i}", (now - timedelta(days=20 - i)).isoformat()),
            )

        # Add commitment with first_seen in metadata
        db.execute(
            "INSERT INTO identity_nodes (id, node_type, content, status, metadata) VALUES (?, 'commitment', ?, 'active', ?)",
            ("c_fuzzy", "Be more mindful", json.dumps({"first_seen": first_seen})),
        )
        db.commit()

        # Arc: monitor recommendation, but only 3 evaluations
        arc = _make_arc(
            commitment_id="c_fuzzy", content="Be more mindful",
            sessions_evaluated=3, honored=1, missed=1, partial=1,
            streak="HMP", recommendation="monitor",
        )
        report = _make_report(session_window=30, arcs=[arc])

        engine = AdaptationEngine(db)
        result = engine.evaluate(report)

        sharpen_recs = [r for r in result.commitment_recommendations if r.action == "sharpen"]
        assert len(sharpen_recs) == 1
        assert "eval rate" in sharpen_recs[0].reasoning.lower() or "evaluated" in sharpen_recs[0].reasoning.lower()

    def test_no_sharpen_for_new_commitment(self, db):
        """Commitment that hasn't existed long enough should not trigger sharpen."""
        now = datetime.now(timezone.utc)
        first_seen = (now - timedelta(days=2)).isoformat()

        for i in range(3):
            db.execute(
                "INSERT INTO conversations (id, created_at, status) VALUES (?, ?, 'ended')",
                (f"sess_{i}", (now - timedelta(days=2 - i)).isoformat()),
            )
        db.execute(
            "INSERT INTO identity_nodes (id, node_type, content, status, metadata) VALUES (?, 'commitment', ?, 'active', ?)",
            ("c_new", "New commitment", json.dumps({"first_seen": first_seen})),
        )
        db.commit()

        arc = _make_arc(
            commitment_id="c_new", content="New commitment",
            sessions_evaluated=1, honored=0, missed=0, partial=1,
            streak="P", recommendation="monitor",
        )
        report = _make_report(session_window=30, arcs=[arc])

        engine = AdaptationEngine(db)
        result = engine.evaluate(report)

        assert not any(r.action == "sharpen" for r in result.commitment_recommendations)


# ── Tests: Graduation Candidates ─────────────────────────────────────────

class TestGraduationCandidates:

    def test_identify_near_permanent_signal(self, db):
        """Signal meeting all graduation criteria should produce a proposal."""
        signal_trends = [_make_signal_trend("warmth_appreciated", sessions_active=20, trend="stable")]
        trajectories = [_make_trajectory("warmth", baseline=0.55, net_drift=0.05, volatility=0.01)]
        near_perm = [("warmth_appreciated", 12)]
        stats = _make_signal_stats("warmth_appreciated", {"warmth": 0.02})

        report = _make_report(signal_trends=signal_trends, trait_trajectories=trajectories)

        engine = AdaptationEngine(db)
        result = engine.evaluate(report, drift_analytics=MagicMock(
            near_permanent_signals=near_perm,
            signal_stats=stats,
        ))

        assert len(result.graduation_proposals) == 1
        prop = result.graduation_proposals[0]
        assert prop.signal_key == "warmth_appreciated"
        assert prop.trait == "warmth"
        assert prop.direction == "up"
        assert prop.proposed_baseline > prop.current_baseline

    def test_reject_volatile_trait(self, db):
        """Volatile trait should block graduation."""
        signal_trends = [_make_signal_trend("warmth_appreciated", sessions_active=20)]
        trajectories = [_make_trajectory("warmth", volatility=0.04, direction="volatile")]
        near_perm = [("warmth_appreciated", 12)]
        stats = _make_signal_stats("warmth_appreciated", {"warmth": 0.02})

        report = _make_report(signal_trends=signal_trends, trait_trajectories=trajectories)

        engine = AdaptationEngine(db)
        result = engine.evaluate(report, drift_analytics=MagicMock(
            near_permanent_signals=near_perm,
            signal_stats=stats,
        ))

        assert len(result.graduation_proposals) == 0

    def test_reject_decelerating_signal(self, db):
        """Decelerating signal should not graduate."""
        signal_trends = [_make_signal_trend("warmth_appreciated", sessions_active=20, trend="decelerating")]
        trajectories = [_make_trajectory("warmth", volatility=0.01)]
        near_perm = [("warmth_appreciated", 12)]
        stats = _make_signal_stats("warmth_appreciated", {"warmth": 0.02})

        report = _make_report(signal_trends=signal_trends, trait_trajectories=trajectories)

        engine = AdaptationEngine(db)
        result = engine.evaluate(report, drift_analytics=MagicMock(
            near_permanent_signals=near_perm,
            signal_stats=stats,
        ))

        assert len(result.graduation_proposals) == 0

    def test_reject_low_session_count(self, db):
        """Signal active in too few sessions should not graduate."""
        signal_trends = [_make_signal_trend("warmth_appreciated", sessions_active=8, trend="stable")]
        trajectories = [_make_trajectory("warmth", volatility=0.01)]
        near_perm = [("warmth_appreciated", 12)]
        stats = _make_signal_stats("warmth_appreciated", {"warmth": 0.02})

        report = _make_report(signal_trends=signal_trends, trait_trajectories=trajectories)

        engine = AdaptationEngine(db)
        result = engine.evaluate(report, drift_analytics=MagicMock(
            near_permanent_signals=near_perm,
            signal_stats=stats,
        ))

        assert len(result.graduation_proposals) == 0

    def test_reject_tiny_drift(self, db):
        """Drift below minimum magnitude should not graduate."""
        signal_trends = [_make_signal_trend("warmth_appreciated", sessions_active=20)]
        trajectories = [_make_trajectory("warmth", net_drift=0.01, volatility=0.005)]
        near_perm = [("warmth_appreciated", 12)]
        stats = _make_signal_stats("warmth_appreciated", {"warmth": 0.001})

        report = _make_report(signal_trends=signal_trends, trait_trajectories=trajectories)

        engine = AdaptationEngine(db)
        result = engine.evaluate(report, drift_analytics=MagicMock(
            near_permanent_signals=near_perm,
            signal_stats=stats,
        ))

        assert len(result.graduation_proposals) == 0

    def test_propose_baseline_shift_conservative(self, db):
        """Proposed baseline should be 75% of observed drift, not 100%."""
        signal_trends = [_make_signal_trend("warmth_appreciated", sessions_active=20)]
        trajectories = [_make_trajectory("warmth", baseline=0.50, net_drift=0.10, volatility=0.01)]
        near_perm = [("warmth_appreciated", 15)]
        stats = _make_signal_stats("warmth_appreciated", {"warmth": 0.02})

        report = _make_report(signal_trends=signal_trends, trait_trajectories=trajectories)

        engine = AdaptationEngine(db)
        result = engine.evaluate(report, drift_analytics=MagicMock(
            near_permanent_signals=near_perm,
            signal_stats=stats,
        ))

        assert len(result.graduation_proposals) == 1
        prop = result.graduation_proposals[0]
        expected_delta = 0.10 * 0.75
        assert abs(prop.delta - expected_delta) < 0.001
        assert abs(prop.proposed_baseline - (0.50 + expected_delta)) < 0.001

    def test_skip_already_ratcheting_trait(self, db):
        """Trait already in ratchet_candidates should be skipped."""
        signal_trends = [_make_signal_trend("warmth_appreciated", sessions_active=20)]
        trajectories = [_make_trajectory("warmth", baseline=0.55, net_drift=0.05, volatility=0.01)]
        near_perm = [("warmth_appreciated", 12)]
        stats = _make_signal_stats("warmth_appreciated", {"warmth": 0.02})

        report = _make_report(signal_trends=signal_trends, trait_trajectories=trajectories)

        # Mock persona with existing ratchet candidate for warmth
        persona = MagicMock()
        persona._state.ratchet_candidates = {"warmth": MagicMock()}

        engine = AdaptationEngine(db, persona=persona)
        result = engine.evaluate(report, drift_analytics=MagicMock(
            near_permanent_signals=near_perm,
            signal_stats=stats,
        ))

        assert len(result.graduation_proposals) == 0

    def test_graduation_confidence_scoring(self, db):
        """Confidence should scale with reinforcement, sessions, stability, and drift."""
        signal_trends = [_make_signal_trend("warmth_appreciated", sessions_active=25)]
        trajectories = [_make_trajectory("warmth", baseline=0.50, net_drift=0.08, volatility=0.005)]
        near_perm = [("warmth_appreciated", 18)]
        stats = _make_signal_stats("warmth_appreciated", {"warmth": 0.02})

        report = _make_report(signal_trends=signal_trends, trait_trajectories=trajectories)

        engine = AdaptationEngine(db)
        result = engine.evaluate(report, drift_analytics=MagicMock(
            near_permanent_signals=near_perm,
            signal_stats=stats,
        ))

        assert len(result.graduation_proposals) == 1
        prop = result.graduation_proposals[0]
        # Strong signal: high reinforcement, many sessions, low volatility, good drift
        assert prop.confidence > 0.7

    def test_no_proposals_without_drift_analytics(self, db):
        """Without drift analytics, no graduation proposals should be generated."""
        signal_trends = [_make_signal_trend("warmth_appreciated", sessions_active=20)]
        trajectories = [_make_trajectory("warmth")]
        report = _make_report(signal_trends=signal_trends, trait_trajectories=trajectories)

        engine = AdaptationEngine(db)
        result = engine.evaluate(report, drift_analytics=None)

        assert len(result.graduation_proposals) == 0


# ── Tests: AdaptationResult ──────────────────────────────────────────────

class TestAdaptationResult:

    def test_to_dict_serializable(self):
        """Result should be JSON-serializable via to_dict."""
        result = AdaptationResult(
            commitment_recommendations=[
                CommitmentRecommendation(
                    commitment_id="c1", commitment_content="Test",
                    action="retire", confidence=0.85,
                    reasoning="Consistently honored",
                    evidence={"streak": "HHHHH"},
                    suggested_actions=["Mark retired"],
                ),
            ],
            graduation_proposals=[
                GraduationProposal(
                    signal_key="warmth_appreciated", trait="warmth",
                    direction="up", current_baseline=0.55,
                    proposed_baseline=0.5875, delta=0.0375,
                    confidence=0.72, reasoning="Sustained reinforcement",
                    evidence={"reinforcement_count": 15},
                ),
            ],
            analyzed_at="2026-04-05T00:00:00+00:00",
        )
        d = result.to_dict()
        serialized = json.dumps(d)
        assert serialized  # no exception
        assert d["commitment_recommendations"][0]["action"] == "retire"
        assert d["graduation_proposals"][0]["trait"] == "warmth"

    def test_format_readable_includes_recommendations(self):
        """Readable format should mention the action and commitment."""
        result = AdaptationResult(
            commitment_recommendations=[
                CommitmentRecommendation(
                    commitment_id="c1", commitment_content="Stay in reflective moments",
                    action="escalate", confidence=0.6,
                    reasoning="Mostly missed",
                    evidence={}, suggested_actions=[],
                ),
            ],
            analyzed_at="2026-04-05T00:00:00+00:00",
        )
        text = result.format_readable()
        assert "ESCALATE" in text
        assert "Stay in reflective" in text

    def test_format_readable_empty(self):
        """Empty result should say no actionable recommendations."""
        result = AdaptationResult(analyzed_at="2026-04-05T00:00:00+00:00")
        text = result.format_readable()
        assert "No actionable" in text

    def test_has_recommendations_property(self):
        """has_recommendations should be False when empty."""
        empty = AdaptationResult()
        assert not empty.has_recommendations

        with_rec = AdaptationResult(
            commitment_recommendations=[
                CommitmentRecommendation(
                    commitment_id="c1", commitment_content="x",
                    action="retire", confidence=0.8,
                    reasoning="y", evidence={},
                ),
            ],
        )
        assert with_rec.has_recommendations


# ── Tests: Pipeline Entry Point ──────────────────────────────────────────

class TestPipelineEntryPoint:

    def test_returns_none_on_empty(self, db):
        """No trend report should return None."""
        result = run_adaptation_analysis(db, trend_report=None)
        assert result is None

    def test_returns_none_when_nothing_actionable(self, db):
        """All-monitor commitments with no graduation candidates should return None."""
        arc = _make_arc(recommendation="monitor")
        report = _make_report(session_window=2, arcs=[arc])  # too short for sharpen

        result = run_adaptation_analysis(db, trend_report=report)
        assert result is None

    def test_returns_dict_on_data(self, db):
        """Valid recommendations should return a dict."""
        arc = _make_arc(recommendation="retire", streak="HHHHH", honored=10, sessions_evaluated=10)
        report = _make_report(arcs=[arc])

        result = run_adaptation_analysis(db, trend_report=report)
        assert isinstance(result, dict)
        assert "commitment_recommendations" in result

    def test_never_raises_on_error(self):
        """Pipeline entry should catch all exceptions and return None."""
        # Pass garbage that would cause an AttributeError
        result = run_adaptation_analysis(
            conn="not_a_connection",
            trend_report="not_a_report",
        )
        assert result is None
