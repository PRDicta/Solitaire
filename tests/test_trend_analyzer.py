"""Tests for the cross-session trend analyzer."""
import json
import sqlite3
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone, timedelta

from solitaire.core.trend_analyzer import (
    TrendAnalyzer,
    TrendReport,
    SignalTrend,
    TraitTrajectory,
    CommitmentArc,
    run_trend_analysis,
    format_trend_report,
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
        CREATE TABLE rolodex_entries (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            content TEXT NOT NULL,
            content_type TEXT NOT NULL DEFAULT 'prose',
            category TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]',
            source_range TEXT NOT NULL DEFAULT '{}',
            access_count INTEGER DEFAULT 0,
            created_at DATETIME NOT NULL,
            tier TEXT DEFAULT 'cold',
            metadata TEXT DEFAULT '{}'
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


def _add_session(conn, session_id, created_at):
    conn.execute(
        "INSERT INTO conversations (id, created_at, status) VALUES (?, ?, 'ended')",
        (session_id, created_at.isoformat()),
    )


def _add_drift(conn, session_id, signal_key, traits_affected, snapshot, created_at, confidence=0.8):
    content = json.dumps({
        "signal": signal_key,
        "traits_affected": traits_affected,
        "active_profile_snapshot": snapshot,
        "confidence": confidence,
        "reinforcement_count": 1,
    })
    import uuid
    conn.execute(
        """INSERT INTO rolodex_entries
           (id, conversation_id, content, content_type, category, created_at)
           VALUES (?, ?, ?, 'prose', 'disposition_drift', ?)""",
        (str(uuid.uuid4()), session_id, content, created_at.isoformat()),
    )


def _add_commitment(conn, commitment_id, content):
    conn.execute(
        """INSERT INTO identity_nodes (id, node_type, content, status, metadata)
           VALUES (?, 'commitment', ?, 'active', '{}')""",
        (commitment_id, content),
    )


def _add_signal(conn, session_id, commitment_id, signal_type, source="enrichment_scanner", confidence=0.6):
    import uuid
    conn.execute(
        """INSERT INTO identity_signals
           (id, session_id, commitment_id, signal_type, content, source, confidence, created_at)
           VALUES (?, ?, ?, ?, 'test signal', ?, ?, ?)""",
        (str(uuid.uuid4()), session_id, commitment_id, signal_type,
         source, confidence, datetime.now(timezone.utc).isoformat()),
    )


# ── Tests: Signal Trends ─────────────────────────────────────────────────

class TestSignalTrends:

    def test_stable_signal(self, db):
        """Signal firing at consistent rate should be classified as stable."""
        now = datetime.now(timezone.utc)
        for i in range(10):
            sid = f"sess_{i}"
            _add_session(db, sid, now - timedelta(days=10 - i))
            _add_drift(db, sid, "warmth_appreciated", {"warmth": 0.01},
                       {"warmth": 0.55}, now - timedelta(days=10 - i))
        db.commit()

        analyzer = TrendAnalyzer(db)
        report = analyzer.analyze(max_sessions=10)

        warmth_trend = next(s for s in report.signal_trends if s.key == "warmth_appreciated")
        assert warmth_trend.trend == "stable"
        assert warmth_trend.total_fires == 10

    def test_accelerating_signal(self, db):
        """Signal firing much more in recent sessions = accelerating."""
        now = datetime.now(timezone.utc)
        # 20 sessions: signal fires 0 in first 10, 1x in sessions 10-14, 4x in last 5
        for i in range(20):
            sid = f"sess_{i}"
            _add_session(db, sid, now - timedelta(days=20 - i))
            if i >= 15:
                for _ in range(4):
                    _add_drift(db, sid, "ai_tone_flagged", {"assertiveness": 0.02},
                               {"assertiveness": 0.65}, now - timedelta(days=20 - i))
            elif i == 12:
                _add_drift(db, sid, "ai_tone_flagged", {"assertiveness": 0.01},
                           {"assertiveness": 0.6}, now - timedelta(days=20 - i))
        db.commit()

        analyzer = TrendAnalyzer(db)
        report = analyzer.analyze(max_sessions=20)

        tone_trend = next(s for s in report.signal_trends if s.key == "ai_tone_flagged")
        # rate_5 = 20/5 = 4.0, rate_10 = 21/10 = 2.1, ratio = 1.9 > 1.5
        assert tone_trend.trend == "accelerating"

    def test_extinct_signal(self, db):
        """Signal that hasn't fired in last 10 sessions = extinct."""
        now = datetime.now(timezone.utc)
        for i in range(20):
            sid = f"sess_{i}"
            _add_session(db, sid, now - timedelta(days=20 - i))
            # Only fire in first 5 sessions
            if i < 5:
                _add_drift(db, sid, "pushback_rejected", {"conviction": -0.02},
                           {"conviction": 0.7}, now - timedelta(days=20 - i))
        db.commit()

        analyzer = TrendAnalyzer(db)
        report = analyzer.analyze(max_sessions=20)

        pb_trend = next(s for s in report.signal_trends if s.key == "pushback_rejected")
        assert pb_trend.trend == "extinct"

    def test_new_signal(self, db):
        """Signal appearing only in last few sessions = new."""
        now = datetime.now(timezone.utc)
        for i in range(10):
            sid = f"sess_{i}"
            _add_session(db, sid, now - timedelta(days=10 - i))
        # Signal only in last 3 sessions
        for i in range(7, 10):
            _add_drift(db, f"sess_{i}", "humor_landed", {"humor": 0.01},
                       {"humor": 0.5}, now - timedelta(days=10 - i))
        db.commit()

        analyzer = TrendAnalyzer(db)
        report = analyzer.analyze(max_sessions=10)

        humor_trend = next(s for s in report.signal_trends if s.key == "humor_landed")
        assert humor_trend.trend == "new"

    def test_empty_database(self, db):
        """No sessions should return empty report."""
        analyzer = TrendAnalyzer(db)
        report = analyzer.analyze()
        assert report.session_window == 0
        assert report.signal_trends == []


# ── Tests: Trait Trajectories ────────────────────────────────────────────

class TestTraitTrajectories:

    def test_rising_trait(self, db):
        """Trait consistently increasing = rising."""
        now = datetime.now(timezone.utc)
        for i in range(10):
            sid = f"sess_{i}"
            _add_session(db, sid, now - timedelta(days=10 - i))
            warmth_val = 0.50 + i * 0.01
            _add_drift(db, sid, "warmth_appreciated", {"warmth": 0.01},
                       {"warmth": warmth_val}, now - timedelta(days=10 - i))
        db.commit()

        # Mock persona
        persona = MagicMock()
        persona.baseline = {"warmth": 0.50}
        persona.traits = MagicMock()
        persona.traits._values = {"warmth": 0.59}
        persona.traits.get = lambda t, d=0.5: 0.59 if t == "warmth" else d

        analyzer = TrendAnalyzer(db, persona)
        report = analyzer.analyze(max_sessions=10)

        warmth = next(t for t in report.trait_trajectories if t.trait == "warmth")
        assert warmth.direction in ("rising",)
        assert warmth.net_drift > 0

    def test_stable_trait(self, db):
        """Trait barely moving = stable."""
        now = datetime.now(timezone.utc)
        for i in range(10):
            sid = f"sess_{i}"
            _add_session(db, sid, now - timedelta(days=10 - i))
            _add_drift(db, sid, "positive_acknowledgment", {"conviction": 0.001},
                       {"conviction": 0.700}, now - timedelta(days=10 - i))
        db.commit()

        persona = MagicMock()
        persona.baseline = {"conviction": 0.700}
        persona.traits = MagicMock()
        persona.traits._values = {"conviction": 0.701}
        persona.traits.get = lambda t, d=0.5: 0.701 if t == "conviction" else d

        analyzer = TrendAnalyzer(db, persona)
        report = analyzer.analyze(max_sessions=10)

        conviction = next(t for t in report.trait_trajectories if t.trait == "conviction")
        assert conviction.direction == "stable"


# ── Tests: Commitment Arcs ───────────────────────────────────────────────

class TestCommitmentArcs:

    def test_retire_recommendation(self, db):
        """Consistently honored commitment should get retire recommendation."""
        now = datetime.now(timezone.utc)
        _add_commitment(db, "c1", "Practice staying in reflective moments")

        for i in range(10):
            sid = f"sess_{i}"
            _add_session(db, sid, now - timedelta(days=10 - i))
            _add_signal(db, sid, "c1", "held")
        db.commit()

        analyzer = TrendAnalyzer(db)
        report = analyzer.analyze(max_sessions=10)

        assert len(report.commitment_arcs) == 1
        arc = report.commitment_arcs[0]
        assert arc.recommendation == "retire"
        assert arc.honored_count >= 5

    def test_escalate_recommendation(self, db):
        """Consistently missed commitment should get escalate recommendation."""
        now = datetime.now(timezone.utc)
        _add_commitment(db, "c2", "Watch for deflecting with questions")

        for i in range(10):
            sid = f"sess_{i}"
            _add_session(db, sid, now - timedelta(days=10 - i))
            _add_signal(db, sid, "c2", "missed")
        db.commit()

        analyzer = TrendAnalyzer(db)
        report = analyzer.analyze(max_sessions=10)

        arc = report.commitment_arcs[0]
        assert arc.recommendation == "escalate"

    def test_sharpen_recommendation(self, db):
        """Mostly partial outcomes = fuzzy detection = sharpen."""
        now = datetime.now(timezone.utc)
        _add_commitment(db, "c3", "Distinguish genuine from performed observation")

        for i in range(10):
            sid = f"sess_{i}"
            _add_session(db, sid, now - timedelta(days=10 - i))
            if i % 3 == 0:
                # Some honored sessions to break escalation streak
                _add_signal(db, sid, "c3", "held", confidence=0.8)
            else:
                # Most sessions partial (equal held + missed)
                _add_signal(db, sid, "c3", "held", confidence=0.3)
                _add_signal(db, sid, "c3", "missed", confidence=0.3)
        db.commit()

        analyzer = TrendAnalyzer(db)
        report = analyzer.analyze(max_sessions=10)

        arc = report.commitment_arcs[0]
        assert arc.recommendation == "sharpen"

    def test_no_commitments(self, db):
        """No active commitments should produce empty arcs."""
        now = datetime.now(timezone.utc)
        for i in range(5):
            _add_session(db, f"sess_{i}", now - timedelta(days=5 - i))
        db.commit()

        analyzer = TrendAnalyzer(db)
        report = analyzer.analyze(max_sessions=5)
        assert report.commitment_arcs == []


# ── Tests: Alerts ────────────────────────────────────────────────────────

class TestAlerts:

    def test_alerts_generated_for_drifting_trait(self, db):
        """Large trait drift should generate an alert."""
        now = datetime.now(timezone.utc)
        for i in range(10):
            sid = f"sess_{i}"
            _add_session(db, sid, now - timedelta(days=10 - i))
            _add_drift(db, sid, "warmth_appreciated", {"warmth": 0.01},
                       {"warmth": 0.50 + i * 0.01}, now - timedelta(days=10 - i))
        db.commit()

        persona = MagicMock()
        persona.baseline = {"warmth": 0.50}
        persona.traits = MagicMock()
        persona.traits._values = {"warmth": 0.60}
        persona.traits.get = lambda t, d=0.5: 0.60 if t == "warmth" else d

        analyzer = TrendAnalyzer(db, persona)
        report = analyzer.analyze(max_sessions=10)

        warmth_alerts = [a for a in report.alerts if "warmth" in a]
        assert len(warmth_alerts) >= 1


# ── Tests: Report Formatting ────────────────────────────────────────────

class TestReportFormatting:

    def test_boot_summary_empty_when_no_data(self):
        report = TrendReport(session_window=0)
        assert report.format_boot_summary() == ""

    def test_to_dict_serializable(self, db):
        """Report dict should be JSON-serializable."""
        now = datetime.now(timezone.utc)
        _add_session(db, "sess_1", now)
        _add_drift(db, "sess_1", "warmth_appreciated", {"warmth": 0.01},
                   {"warmth": 0.55}, now)
        db.commit()

        analyzer = TrendAnalyzer(db)
        report = analyzer.analyze()
        d = report.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_format_trend_report(self, db):
        """format_trend_report should produce string output."""
        now = datetime.now(timezone.utc)
        _add_session(db, "sess_1", now)
        _add_drift(db, "sess_1", "warmth_appreciated", {"warmth": 0.01},
                   {"warmth": 0.55}, now)
        db.commit()

        result = run_trend_analysis(db)
        assert result is not None
        formatted = format_trend_report(result)
        assert "Cross-Session Trends" in formatted


# ── Tests: Pipeline Entry Point ──────────────────────────────────────────

class TestPipelineEntryPoint:

    def test_run_trend_analysis_returns_none_on_empty_db(self, db):
        result = run_trend_analysis(db)
        assert result is None

    def test_run_trend_analysis_returns_dict(self, db):
        now = datetime.now(timezone.utc)
        _add_session(db, "sess_1", now)
        _add_drift(db, "sess_1", "warmth_appreciated", {"warmth": 0.01},
                   {"warmth": 0.55}, now)
        db.commit()

        result = run_trend_analysis(db)
        assert result is not None
        assert "session_window" in result
        assert result["session_window"] == 1
