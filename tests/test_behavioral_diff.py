"""Tests for the behavioral diff generator."""
import json
import sqlite3
import uuid
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone, timedelta

from solitaire.core.behavioral_diff import (
    BehavioralDiffGenerator,
    BehavioralDiff,
    TraitDelta,
    SignalActivity,
    CommitmentStatus,
    generate_behavioral_diff,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    """In-memory SQLite database with required tables."""
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


def _add_session(conn, session_id, created_at, status="ended"):
    conn.execute(
        "INSERT INTO conversations (id, created_at, status) VALUES (?, ?, ?)",
        (session_id, created_at.isoformat(), status),
    )


def _add_drift(conn, session_id, signal_key, traits_affected, snapshot, created_at):
    content = json.dumps({
        "signal": signal_key,
        "traits_affected": traits_affected,
        "active_profile_snapshot": snapshot,
        "confidence": 0.8,
        "reinforcement_count": 1,
    })
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
    conn.execute(
        """INSERT INTO identity_signals
           (id, session_id, commitment_id, signal_type, content, source, confidence, created_at)
           VALUES (?, ?, ?, ?, 'test', ?, ?, ?)""",
        (str(uuid.uuid4()), session_id, commitment_id, signal_type,
         source, confidence, datetime.now(timezone.utc).isoformat()),
    )


def _mock_persona(trait_values, baseline=None):
    persona = MagicMock()
    persona.traits = MagicMock()
    persona.traits._values = dict(trait_values)
    persona.traits.get = lambda t, d=0.5: trait_values.get(t, d)
    persona.baseline = baseline or dict(trait_values)
    return persona


# ── Tests: Trait Deltas ──────────────────────────────────────────────────

class TestTraitDeltas:

    def test_trait_movement_detected(self, db):
        """Traits that moved in the session should show direction."""
        now = datetime.now(timezone.utc)
        current = "current_session"
        _add_session(db, current, now, status="active")

        # Two drift events pushing warmth up
        _add_drift(db, current, "warmth_appreciated", {"warmth": 0.02},
                   {"warmth": 0.53}, now)
        _add_drift(db, current, "warmth_appreciated", {"warmth": 0.02},
                   {"warmth": 0.55}, now + timedelta(minutes=5))
        db.commit()

        persona = _mock_persona(
            {"warmth": 0.55},
            baseline={"warmth": 0.50},
        )
        snapshot = {"warmth": 0.51}

        gen = BehavioralDiffGenerator(db, persona)
        diff = gen.generate(current, pattern_snapshot=snapshot)

        warmth = next(t for t in diff.trait_deltas if t.trait == "warmth")
        assert warmth.direction == "higher"
        assert warmth.session_delta > 0
        assert warmth.signal_count == 2

    def test_unchanged_trait(self, db):
        """Traits with no drift events stay unchanged."""
        now = datetime.now(timezone.utc)
        current = "current_session"
        _add_session(db, current, now, status="active")
        db.commit()

        persona = _mock_persona({"warmth": 0.50}, baseline={"warmth": 0.50})
        snapshot = {"warmth": 0.50}

        gen = BehavioralDiffGenerator(db, persona)
        diff = gen.generate(current, pattern_snapshot=snapshot)

        warmth = next(t for t in diff.trait_deltas if t.trait == "warmth")
        assert warmth.direction == "unchanged"

    def test_deviation_from_rolling_average(self, db):
        """Deviation should compare session-end to rolling average of prior sessions."""
        now = datetime.now(timezone.utc)

        # 5 prior sessions all ending at warmth=0.55
        for i in range(5):
            sid = f"prior_{i}"
            _add_session(db, sid, now - timedelta(days=5 - i))
            _add_drift(db, sid, "warmth_appreciated", {"warmth": 0.01},
                       {"warmth": 0.55}, now - timedelta(days=5 - i))

        # Current session ends at warmth=0.65 (10 points above average)
        current = "current_session"
        _add_session(db, current, now, status="active")
        _add_drift(db, current, "warmth_appreciated", {"warmth": 0.05},
                   {"warmth": 0.65}, now)
        db.commit()

        persona = _mock_persona({"warmth": 0.65}, baseline={"warmth": 0.50})
        gen = BehavioralDiffGenerator(db, persona)
        diff = gen.generate(current)

        warmth = next(t for t in diff.trait_deltas if t.trait == "warmth")
        assert warmth.deviation > 0.05  # significantly above prior average


# ── Tests: Signal Activity ───────────────────────────────────────────────

class TestSignalActivity:

    def test_first_time_signal(self, db):
        """Signal never seen in prior sessions = first_time."""
        now = datetime.now(timezone.utc)

        # 3 prior sessions with only warmth signals
        for i in range(3):
            sid = f"prior_{i}"
            _add_session(db, sid, now - timedelta(days=3 - i))
            _add_drift(db, sid, "warmth_appreciated", {"warmth": 0.01},
                       {"warmth": 0.55}, now - timedelta(days=3 - i))

        # Current session fires a brand new signal
        current = "current_session"
        _add_session(db, current, now, status="active")
        _add_drift(db, current, "ai_tone_flagged", {"assertiveness": 0.02},
                   {"assertiveness": 0.65}, now)
        db.commit()

        gen = BehavioralDiffGenerator(db)
        diff = gen.generate(current)

        tone = next(s for s in diff.signal_activity if s.key == "ai_tone_flagged")
        assert tone.deviation == "first_time"
        assert tone.fires == 1

    def test_above_average_signal(self, db):
        """Signal firing much more than average = above_avg."""
        now = datetime.now(timezone.utc)

        # 5 prior sessions with 1 fire each
        for i in range(5):
            sid = f"prior_{i}"
            _add_session(db, sid, now - timedelta(days=5 - i))
            _add_drift(db, sid, "positive_acknowledgment", {"conviction": 0.01},
                       {"conviction": 0.7}, now - timedelta(days=5 - i))

        # Current session fires 5 times
        current = "current_session"
        _add_session(db, current, now, status="active")
        for _ in range(5):
            _add_drift(db, current, "positive_acknowledgment", {"conviction": 0.01},
                       {"conviction": 0.72}, now)
        db.commit()

        gen = BehavioralDiffGenerator(db)
        diff = gen.generate(current)

        ack = next(s for s in diff.signal_activity if s.key == "positive_acknowledgment")
        assert ack.deviation == "above_avg"
        assert ack.fires == 5


# ── Tests: Commitment Statuses ───────────────────────────────────────────

class TestCommitmentStatuses:

    def test_honored_commitment(self, db):
        """Commitment with held signals should show honored."""
        now = datetime.now(timezone.utc)
        current = "current_session"
        _add_session(db, current, now, status="active")
        _add_commitment(db, "c1", "Stay in reflective moments")
        _add_signal(db, current, "c1", "held")
        _add_signal(db, current, "c1", "held")
        db.commit()

        gen = BehavioralDiffGenerator(db)
        diff = gen.generate(current)

        assert len(diff.commitment_statuses) == 1
        assert diff.commitment_statuses[0].outcome == "honored"
        assert diff.commitment_statuses[0].signal_count == 2

    def test_missed_commitment(self, db):
        """Commitment with missed signals should show missed."""
        now = datetime.now(timezone.utc)
        current = "current_session"
        _add_session(db, current, now, status="active")
        _add_commitment(db, "c2", "Watch for deflecting")
        _add_signal(db, current, "c2", "missed", source="user_correction", confidence=1.0)
        db.commit()

        gen = BehavioralDiffGenerator(db)
        diff = gen.generate(current)

        assert diff.commitment_statuses[0].outcome == "missed"

    def test_unevaluated_commitment(self, db):
        """Commitment with no signals this session = not_evaluated."""
        now = datetime.now(timezone.utc)
        current = "current_session"
        _add_session(db, current, now, status="active")
        _add_commitment(db, "c3", "Distinguish genuine from performed")
        db.commit()

        gen = BehavioralDiffGenerator(db)
        diff = gen.generate(current)

        assert diff.commitment_statuses[0].outcome == "not_evaluated"


# ── Tests: Summary Stats ────────────────────────────────────────────────

class TestSummaryStats:

    def test_stats_counted_correctly(self, db):
        """Summary stats should reflect actual data."""
        now = datetime.now(timezone.utc)
        current = "current_session"
        _add_session(db, current, now, status="active")
        _add_drift(db, current, "warmth_appreciated", {"warmth": 0.02},
                   {"warmth": 0.55}, now)
        _add_drift(db, current, "pushback_accepted", {"conviction": 0.03},
                   {"conviction": 0.75}, now)
        db.commit()

        persona = _mock_persona(
            {"warmth": 0.55, "conviction": 0.75},
            baseline={"warmth": 0.50, "conviction": 0.70},
        )
        snapshot = {"warmth": 0.51, "conviction": 0.72}

        gen = BehavioralDiffGenerator(db, persona)
        diff = gen.generate(current, pattern_snapshot=snapshot)

        assert diff.total_drift_events == 2
        assert diff.unique_signals_fired == 2


# ── Tests: Output Formats ────────────────────────────────────────────────

class TestOutputFormats:

    def test_to_dict_serializable(self, db):
        """to_dict output should be JSON-serializable."""
        now = datetime.now(timezone.utc)
        current = "current_session"
        _add_session(db, current, now, status="active")
        _add_drift(db, current, "warmth_appreciated", {"warmth": 0.02},
                   {"warmth": 0.55}, now)
        db.commit()

        gen = BehavioralDiffGenerator(db)
        diff = gen.generate(current)
        d = diff.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_format_readable(self, db):
        """format_readable should return a non-empty string."""
        now = datetime.now(timezone.utc)
        current = "current_session"
        _add_session(db, current, now, status="active")
        _add_drift(db, current, "warmth_appreciated", {"warmth": 0.03},
                   {"warmth": 0.55}, now)
        db.commit()

        persona = _mock_persona({"warmth": 0.55}, baseline={"warmth": 0.50})
        snapshot = {"warmth": 0.52}

        gen = BehavioralDiffGenerator(db, persona)
        diff = gen.generate(current, pattern_snapshot=snapshot)
        readable = diff.format_readable()
        assert len(readable) > 0

    def test_empty_diff_readable(self, db):
        """Empty diff should produce 'no notable changes' message."""
        now = datetime.now(timezone.utc)
        current = "current_session"
        _add_session(db, current, now, status="active")
        db.commit()

        gen = BehavioralDiffGenerator(db)
        diff = gen.generate(current)
        readable = diff.format_readable()
        assert "No notable" in readable


# ── Tests: Pipeline Entry Point ──────────────────────────────────────────

class TestPipelineEntryPoint:

    def test_generate_returns_none_on_empty_session(self, db):
        """No drift events and no commitments should return None."""
        now = datetime.now(timezone.utc)
        current = "current_session"
        _add_session(db, current, now, status="active")
        db.commit()

        result = generate_behavioral_diff(db, current)
        assert result is None

    def test_generate_returns_dict_with_data(self, db):
        """With drift data, should return a dict."""
        now = datetime.now(timezone.utc)
        current = "current_session"
        _add_session(db, current, now, status="active")
        _add_drift(db, current, "warmth_appreciated", {"warmth": 0.02},
                   {"warmth": 0.55}, now)
        db.commit()

        result = generate_behavioral_diff(db, current)
        assert result is not None
        assert "session_id" in result
        assert result["total_drift_events"] == 1
