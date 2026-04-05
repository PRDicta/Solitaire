"""Tests for the batch review engine."""
import json
import sqlite3
import uuid
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from solitaire.core.batch_review import (
    BATCH_REVIEW_SOURCE,
    BATCH_REVIEW_WEIGHT,
    CATEGORY_ROTATION,
    BatchReviewEngine,
    ReviewDecision,
    ReviewItem,
    ReviewRunResult,
    ReviewVerdict,
    ensure_review_schema,
    get_review_status,
    run_review_apply,
    run_review_gather,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    """In-memory SQLite database with full schema."""
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
            strength REAL,
            valence TEXT,
            observation_count INTEGER DEFAULT 1,
            trajectory TEXT,
            first_seen TEXT,
            last_seen TEXT,
            discovery_session TEXT,
            metadata TEXT DEFAULT '{}',
            created_at DATETIME,
            updated_at DATETIME
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
    conn.execute("""
        CREATE TABLE identity_candidates (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            node_type TEXT NOT NULL,
            content TEXT NOT NULL,
            signal_source TEXT,
            promoted INTEGER DEFAULT 0,
            dismissed INTEGER DEFAULT 0,
            created_at DATETIME NOT NULL
        )
    """)
    ensure_review_schema(conn)
    conn.commit()
    return conn


def _add_signal(conn, signal_id, session_id, commitment_id, signal_type,
                source="enrichment_scanner", confidence=0.6, created_at=None):
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO identity_signals
           (id, session_id, commitment_id, signal_type, content, source, confidence, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (signal_id, session_id, commitment_id, signal_type,
         f"Retroactive score: test content for {signal_type}", source, confidence, created_at),
    )


def _add_commitment(conn, commitment_id, content):
    conn.execute(
        """INSERT INTO identity_nodes (id, node_type, content, status, metadata, created_at, updated_at)
           VALUES (?, 'commitment', ?, 'active', '{}', ?, ?)""",
        (commitment_id, content,
         datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat()),
    )


def _add_candidate(conn, cand_id, content, node_type="realization", session_id="sess_1"):
    conn.execute(
        """INSERT INTO identity_candidates
           (id, session_id, node_type, content, signal_source, promoted, dismissed, created_at)
           VALUES (?, ?, ?, ?, 'enrichment_scanner', 0, 0, ?)""",
        (cand_id, session_id, node_type, content, datetime.now(timezone.utc).isoformat()),
    )


def _add_drift(conn, entry_id, signal_key, confidence=0.5, session_id="sess_1"):
    content = json.dumps({
        "signal": signal_key,
        "traits_affected": {"warmth": 0.01},
        "active_profile_snapshot": {"warmth": 0.55},
        "confidence": confidence,
        "reinforcement_count": 1,
    })
    conn.execute(
        """INSERT INTO rolodex_entries
           (id, conversation_id, content, content_type, category, created_at)
           VALUES (?, ?, ?, 'prose', 'disposition_drift', ?)""",
        (entry_id, session_id, content, datetime.now(timezone.utc).isoformat()),
    )


def _add_growth_edge(conn, node_id, content, status="identified"):
    conn.execute(
        """INSERT INTO identity_nodes
           (id, node_type, content, status, observation_count, metadata, last_seen, created_at, updated_at)
           VALUES (?, 'growth_edge', ?, ?, 3, '{}', ?, ?, ?)""",
        (node_id, content, status,
         datetime.now(timezone.utc).isoformat(),
         datetime.now(timezone.utc).isoformat(),
         datetime.now(timezone.utc).isoformat()),
    )


# ── Tests: Review Gathering ──────────────────────────────────────────────

class TestReviewGathering:

    def test_gather_commitment_signals(self, db):
        """Should return enrichment_scanner signals with commitment context."""
        _add_commitment(db, "c1", "Stay grounded in reflective moments")
        _add_signal(db, "sig_1", "sess_1", "c1", "held")
        _add_signal(db, "sig_2", "sess_1", "c1", "missed")
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        result = engine.gather("commitment_signals")

        assert result["item_count"] == 2
        assert result["category"] == "commitment_signals"
        assert len(result["guiding_questions"]) > 0
        item = result["items"][0]
        assert "Stay grounded" in item["context"]

    def test_gather_identity_candidates(self, db):
        """Should return pending candidates."""
        _add_candidate(db, "cand_1", "I realized I was rushing past the details")
        _add_candidate(db, "cand_2", "This pattern keeps repeating")
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        result = engine.gather("identity_candidates")

        assert result["item_count"] == 2

    def test_gather_disposition_drift_filters_high_confidence(self, db):
        """Should only return low-confidence drift entries."""
        _add_drift(db, "drift_1", "warmth_appreciated", confidence=0.5)
        _add_drift(db, "drift_2", "pushback_accepted", confidence=0.9)
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        result = engine.gather("disposition_drift")

        assert result["item_count"] == 1
        assert result["items"][0]["item_id"] == "drift_1"

    def test_gather_growth_edges(self, db):
        """Should return active growth edges."""
        _add_growth_edge(db, "ge_1", "Staying in reflective moments")
        _add_growth_edge(db, "ge_2", "Integrated behavior", status="integrated")
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        result = engine.gather("growth_edge_evolution")

        assert result["item_count"] == 1
        assert result["items"][0]["item_id"] == "ge_1"

    def test_gather_respects_limit(self, db):
        """Should not return more than limit items."""
        for i in range(10):
            _add_candidate(db, f"cand_{i}", f"Candidate {i}")
        db.commit()

        engine = BatchReviewEngine(db, limit=3)
        result = engine.gather("identity_candidates")

        assert result["item_count"] == 3

    def test_watermark_filtering(self, db):
        """Should only return items after the watermark."""
        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=2)).isoformat()
        new = now.isoformat()

        _add_commitment(db, "c1", "Test commitment")
        _add_signal(db, "sig_old", "sess_1", "c1", "held", created_at=old)
        _add_signal(db, "sig_new", "sess_2", "c1", "held", created_at=new)

        # Set watermark by inserting a review_log entry
        db.execute(
            """INSERT INTO review_log
               (id, category, started_at, completed_at, watermark, rotation_index)
               VALUES (?, 'commitment_signals', ?, ?, ?, 0)""",
            ("rev_1", old, old, old),
        )
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        result = engine.gather("commitment_signals")

        assert result["item_count"] == 1
        assert result["items"][0]["item_id"] == "sig_new"

    def test_auto_rotation(self, db):
        """Auto gather should pick the next category in rotation."""
        engine = BatchReviewEngine(db, limit=10)

        # No prior runs: should pick first in rotation
        result = engine.gather_auto()
        assert result["category"] == CATEGORY_ROTATION[0]

    def test_auto_rotation_advances(self, db):
        """After a run, auto should advance to next category."""
        db.execute(
            """INSERT INTO review_log
               (id, category, started_at, completed_at, rotation_index)
               VALUES (?, 'commitment_signals', ?, ?, 0)""",
            ("rev_1", datetime.now(timezone.utc).isoformat(),
             datetime.now(timezone.utc).isoformat()),
        )
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        result = engine.gather_auto()
        assert result["category"] == CATEGORY_ROTATION[1]

    def test_unknown_category_returns_error(self, db):
        """Unknown category should return error dict."""
        engine = BatchReviewEngine(db)
        result = engine.gather("nonexistent_category")
        assert "error" in result


# ── Tests: Commitment Signal Review ──────────────────────────────────────

class TestCommitmentSignalReview:

    def test_confirm_creates_new_signal(self, db):
        """Confirming a signal should create a new batch_review signal."""
        _add_commitment(db, "c1", "Test commitment")
        _add_signal(db, "sig_1", "sess_1", "c1", "held")
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        decisions = [{
            "item_id": "sig_1",
            "category": "commitment_signals",
            "verdict": "confirmed",
            "reasoning": "Signal correctly identifies honored commitment",
        }]
        result = engine.apply_decisions(decisions)

        assert result.confirmed == 1
        # Check new signal was created
        rows = db.execute(
            "SELECT source, confidence FROM identity_signals WHERE source = ?",
            (BATCH_REVIEW_SOURCE,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == BATCH_REVIEW_WEIGHT

    def test_correct_flips_signal_type(self, db):
        """Correcting a signal should create a signal with opposite type."""
        _add_commitment(db, "c1", "Test commitment")
        _add_signal(db, "sig_1", "sess_1", "c1", "held")
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        decisions = [{
            "item_id": "sig_1",
            "category": "commitment_signals",
            "verdict": "corrected",
            "reasoning": "Keyword match was misleading; commitment was actually missed",
        }]
        result = engine.apply_decisions(decisions)

        assert result.corrected == 1
        row = db.execute(
            "SELECT signal_type FROM identity_signals WHERE source = ?",
            (BATCH_REVIEW_SOURCE,),
        ).fetchone()
        assert row[0] == "missed"  # flipped from "held"

    def test_original_signal_preserved(self, db):
        """Original heuristic signal should not be modified."""
        _add_commitment(db, "c1", "Test commitment")
        _add_signal(db, "sig_1", "sess_1", "c1", "held")
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        decisions = [{
            "item_id": "sig_1",
            "category": "commitment_signals",
            "verdict": "corrected",
            "reasoning": "Wrong direction",
        }]
        engine.apply_decisions(decisions)

        # Original still exists unchanged
        row = db.execute(
            "SELECT signal_type, source FROM identity_signals WHERE id = 'sig_1'",
        ).fetchone()
        assert row[0] == "held"
        assert row[1] == "enrichment_scanner"


# ── Tests: Identity Candidate Review ─────────────────────────────────────

class TestIdentityCandidateReview:

    def test_confirm_promotes(self, db):
        """Confirming a candidate should set promoted flag."""
        _add_candidate(db, "cand_1", "I realized I was avoiding the hard conversations")
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        decisions = [{
            "item_id": "cand_1",
            "category": "identity_candidates",
            "verdict": "confirmed",
            "reasoning": "Genuine realization with substantive content",
        }]
        result = engine.apply_decisions(decisions)

        assert result.confirmed == 1
        row = db.execute("SELECT promoted FROM identity_candidates WHERE id = 'cand_1'").fetchone()
        assert row[0] == 1

    def test_dismiss_dismisses(self, db):
        """Dismissing a candidate should set dismissed flag."""
        _add_candidate(db, "cand_1", "I just realized the time")
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        decisions = [{
            "item_id": "cand_1",
            "category": "identity_candidates",
            "verdict": "dismissed",
            "reasoning": "Casual usage of 'realized', not a genuine identity insight",
        }]
        result = engine.apply_decisions(decisions)

        assert result.dismissed == 1
        row = db.execute("SELECT dismissed FROM identity_candidates WHERE id = 'cand_1'").fetchone()
        assert row[0] == 1


# ── Tests: Disposition Drift Review ──────────────────────────────────────

class TestDispositionDriftReview:

    def test_confirm_updates_metadata(self, db):
        """Confirming drift should mark as reviewed with upgraded confidence."""
        _add_drift(db, "drift_1", "warmth_appreciated", confidence=0.5)
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        decisions = [{
            "item_id": "drift_1",
            "category": "disposition_drift",
            "verdict": "confirmed",
            "reasoning": "Signal correctly detected warmth appreciation",
        }]
        result = engine.apply_decisions(decisions)

        assert result.confirmed == 1
        row = db.execute("SELECT content FROM rolodex_entries WHERE id = 'drift_1'").fetchone()
        data = json.loads(row[0])
        assert data["reviewed"] is True
        assert data["review_confidence"] == 0.85

    def test_correct_flags_incorrect(self, db):
        """Correcting drift should mark as incorrect."""
        _add_drift(db, "drift_1", "pushback_accepted", confidence=0.5)
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        decisions = [{
            "item_id": "drift_1",
            "category": "disposition_drift",
            "verdict": "corrected",
            "reasoning": "User was being polite, not accepting pushback",
        }]
        result = engine.apply_decisions(decisions)

        assert result.corrected == 1
        row = db.execute("SELECT content FROM rolodex_entries WHERE id = 'drift_1'").fetchone()
        data = json.loads(row[0])
        assert data["review_result"] == "incorrect"


# ── Tests: Growth Edge Review ────────────────────────────────────────────

class TestGrowthEdgeReview:

    def test_upgrade_advances_status(self, db):
        """Upgrading a growth edge should advance its status."""
        _add_growth_edge(db, "ge_1", "Staying present in reflective moments", status="identified")
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        decisions = [{
            "item_id": "ge_1",
            "category": "growth_edge_evolution",
            "verdict": "upgraded",
            "reasoning": "Evidence of active practice across recent sessions",
            "corrections": {"new_status": "practicing"},
        }]
        result = engine.apply_decisions(decisions)

        assert result.upgraded == 1
        row = db.execute("SELECT status FROM identity_nodes WHERE id = 'ge_1'").fetchone()
        assert row[0] == "practicing"

    def test_confirm_leaves_unchanged(self, db):
        """Confirming a growth edge should not change its status."""
        _add_growth_edge(db, "ge_1", "Processing with engagement", status="practicing")
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        decisions = [{
            "item_id": "ge_1",
            "category": "growth_edge_evolution",
            "verdict": "confirmed",
            "reasoning": "Still actively being practiced, not yet improving",
        }]
        result = engine.apply_decisions(decisions)

        assert result.confirmed == 1
        row = db.execute("SELECT status FROM identity_nodes WHERE id = 'ge_1'").fetchone()
        assert row[0] == "practicing"


# ── Tests: Override Rate ─────────────────────────────────────────────────

class TestOverrideRate:

    def test_correct_calculation(self, db):
        """Override rate should be corrected / (confirmed + corrected + upgraded)."""
        _add_commitment(db, "c1", "Test")
        for i in range(5):
            _add_signal(db, f"sig_{i}", "sess_1", "c1", "held")
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        decisions = [
            {"item_id": "sig_0", "category": "commitment_signals", "verdict": "confirmed", "reasoning": "ok"},
            {"item_id": "sig_1", "category": "commitment_signals", "verdict": "confirmed", "reasoning": "ok"},
            {"item_id": "sig_2", "category": "commitment_signals", "verdict": "corrected", "reasoning": "wrong"},
            {"item_id": "sig_3", "category": "commitment_signals", "verdict": "upgraded", "reasoning": "better"},
            {"item_id": "sig_4", "category": "commitment_signals", "verdict": "deferred", "reasoning": "unsure"},
        ]
        result = engine.apply_decisions(decisions)

        # corrected=1, confirmed=2, upgraded=1 -> 1/4 = 0.25
        assert abs(result.override_rate - 0.25) < 0.001

    def test_zero_decisive_produces_zero_rate(self, db):
        """All deferred should give 0 override rate."""
        result = ReviewRunResult()
        result.deferred = 5
        result.compute_override_rate()
        assert result.override_rate == 0.0


# ── Tests: Checkpoint ────────────────────────────────────────────────────

class TestCheckpoint:

    def test_watermark_persists(self, db):
        """After a run, watermark should be stored in review_log."""
        _add_commitment(db, "c1", "Test")
        now = datetime.now(timezone.utc).isoformat()
        _add_signal(db, "sig_1", "sess_1", "c1", "held", created_at=now)
        db.commit()

        engine = BatchReviewEngine(db, limit=10)
        decisions = [{
            "item_id": "sig_1",
            "category": "commitment_signals",
            "verdict": "confirmed",
            "reasoning": "ok",
            "metadata": {"created_at": now},
        }]
        engine.apply_decisions(decisions)

        # Check watermark was written
        row = db.execute(
            "SELECT watermark FROM review_log WHERE category = 'commitment_signals'"
        ).fetchone()
        assert row is not None
        assert row[0] == now

    def test_rotation_advances_after_run(self, db):
        """Rotation index should advance after each run."""
        engine = BatchReviewEngine(db, limit=10)
        decisions = [{
            "item_id": "test_1",
            "category": "commitment_signals",
            "verdict": "deferred",
            "reasoning": "test",
        }]
        engine.apply_decisions(decisions)

        row = db.execute("SELECT rotation_index FROM review_log ORDER BY completed_at DESC LIMIT 1").fetchone()
        assert row is not None


# ── Tests: ReviewRunResult ───────────────────────────────────────────────

class TestReviewRunResult:

    def test_to_dict_serializable(self):
        """Result should be JSON-serializable."""
        result = ReviewRunResult(
            category="commitment_signals",
            items_reviewed=5,
            confirmed=3,
            corrected=1,
            upgraded=1,
            override_rate=0.2,
            actions=[{"item_id": "sig_1", "verdict": "confirmed"}],
            analyzed_at="2026-04-05T00:00:00+00:00",
        )
        serialized = json.dumps(result.to_dict())
        assert serialized
        d = json.loads(serialized)
        assert d["category"] == "commitment_signals"
        assert d["override_rate"] == 0.2

    def test_format_readable(self):
        """Readable format should include key counts."""
        result = ReviewRunResult(
            category="commitment_signals",
            items_reviewed=5,
            confirmed=3,
            corrected=1,
            upgraded=1,
            override_rate=0.2,
        )
        text = result.format_readable()
        assert "commitment_signals" in text
        assert "Corrected: 1" in text

    def test_has_activity(self):
        """has_activity should reflect items_reviewed."""
        empty = ReviewRunResult()
        assert not empty.has_activity

        active = ReviewRunResult(items_reviewed=1)
        assert active.has_activity


# ── Tests: Pipeline Entry Points ─────────────────────────────────────────

class TestPipelineEntryPoints:

    def test_gather_returns_dict(self, db):
        """run_review_gather should return a dict."""
        result = run_review_gather(db, category="commitment_signals", limit=5)
        assert isinstance(result, dict)
        assert result["item_count"] == 0

    def test_gather_auto_works(self, db):
        """run_review_gather with auto should pick a category."""
        result = run_review_gather(db, category="auto", limit=5)
        assert isinstance(result, dict)
        assert "category" in result

    def test_apply_returns_none_on_empty(self, db):
        """Empty decisions should return None."""
        result = run_review_apply(db, decisions=[])
        assert result is None

    def test_apply_returns_dict_on_data(self, db):
        """Valid decisions should return a result dict."""
        _add_commitment(db, "c1", "Test")
        _add_signal(db, "sig_1", "sess_1", "c1", "held")
        db.commit()

        result = run_review_apply(db, decisions=[{
            "item_id": "sig_1",
            "category": "commitment_signals",
            "verdict": "confirmed",
            "reasoning": "ok",
        }])
        assert isinstance(result, dict)
        assert result["confirmed"] == 1

    def test_status_returns_dict(self, db):
        """get_review_status should return a dict."""
        result = get_review_status(db)
        assert isinstance(result, dict)
        assert "runs" in result

    def test_never_raises(self):
        """Pipeline entry should catch exceptions."""
        result = run_review_gather("not_a_connection", category="auto")
        assert result is None

    def test_dry_run_no_writes(self, db):
        """Dry run should not write any signals or review logs."""
        _add_commitment(db, "c1", "Test")
        _add_signal(db, "sig_1", "sess_1", "c1", "held")
        db.commit()

        result = run_review_apply(db, decisions=[{
            "item_id": "sig_1",
            "category": "commitment_signals",
            "verdict": "confirmed",
            "reasoning": "ok",
        }], dry_run=True)

        assert isinstance(result, dict)
        # No new signals created
        rows = db.execute("SELECT COUNT(*) FROM identity_signals WHERE source = ?",
                          (BATCH_REVIEW_SOURCE,)).fetchone()
        assert rows[0] == 0
        # No review log created
        rows = db.execute("SELECT COUNT(*) FROM review_log").fetchone()
        assert rows[0] == 0
