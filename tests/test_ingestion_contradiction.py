"""
Tests for Phase 4: Ingestion-Time Contradiction Detection.

Tests the IngestionContradictionDetector that checks newly ingested entries
against existing entries for conflicts, and the gap-fill coverage for
conflict_utils functions not covered by test_contradiction_resolution.py.
"""
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

import pytest

from solitaire.indexing.contradiction_detector import (
    IngestionContradictionDetector,
    PendingContradiction,
)
from solitaire.retrieval.conflict_utils import (
    extract_claim_entities,
    has_temporal_markers,
    detect_claim_conflict,
)


# ─── Minimal RolodexEntry stand-in ────────────────────────────────────────────

@dataclass
class FakeEntry:
    """Mirrors RolodexEntry fields used by IngestionContradictionDetector."""
    id: str = "new-1"
    content: str = ""
    tags: List[str] = field(default_factory=list)
    category: str = "note"
    provenance: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _make_test_db() -> sqlite3.Connection:
    """Create in-memory DB with minimal schema for contradiction detection."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE rolodex_entries (
        id TEXT PRIMARY KEY,
        content TEXT,
        tags TEXT DEFAULT '[]',
        category TEXT DEFAULT 'note',
        provenance TEXT DEFAULT 'unknown',
        metadata TEXT DEFAULT '{}',
        created_at TEXT,
        archived_at TEXT,
        superseded_by TEXT,
        last_accessed TEXT,
        linked_ids TEXT DEFAULT '[]',
        access_count INTEGER DEFAULT 0
    )""")
    conn.execute("""CREATE VIRTUAL TABLE rolodex_fts USING fts5(
        entry_id, content, tags
    )""")
    conn.execute("""CREATE TABLE pending_contradictions (
        id TEXT PRIMARY KEY,
        entry_id_old TEXT NOT NULL,
        entry_id_new TEXT NOT NULL,
        conflict_type TEXT NOT NULL,
        description TEXT NOT NULL,
        detected_at TEXT NOT NULL,
        resolved_at TEXT,
        resolution TEXT,
        resolved_by TEXT
    )""")
    return conn


def _seed_entry(conn: sqlite3.Connection, entry_id: str, content: str,
                tags: Optional[List[str]] = None, metadata: Optional[str] = None):
    """Insert a rolodex entry and its FTS record."""
    tags_json = json.dumps(tags or [])
    meta = metadata or "{}"
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO rolodex_entries (id, content, tags, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        (entry_id, content, tags_json, meta, now),
    )
    conn.execute(
        "INSERT INTO rolodex_fts (entry_id, content, tags) VALUES (?, ?, ?)",
        (entry_id, content, tags_json),
    )
    conn.commit()


# ─── Contradiction Check Tests ────────────────────────────────────────────────

class TestContradictionCheck:

    def test_numeric_conflict_detected_and_stored(self):
        """Numeric conflict between new and existing entry is detected and persisted."""
        conn = _make_test_db()
        _seed_entry(conn, "old-1",
                     "Solitaire core memory system rated at 95% of original vision.")
        detector = IngestionContradictionDetector(conn)

        new_entry = FakeEntry(
            id="new-1",
            content="Solitaire core memory system revised to 75% of original vision after audit.",
            tags=["solitaire", "rating"],
        )
        results = detector.check(new_entry)

        assert len(results) >= 1
        assert any(c.conflict_type == "numeric" for c in results)
        # Verify persisted to DB
        row = conn.execute(
            "SELECT conflict_type FROM pending_contradictions WHERE entry_id_new = ?",
            ("new-1",),
        ).fetchone()
        assert row is not None
        assert row[0] == "numeric"

    def test_preference_conflict_detected(self):
        """Preference conflict between new and existing entry is detected."""
        conn = _make_test_db()
        _seed_entry(conn, "old-1",
                     "The client prefers dark mode for all dashboard interfaces.")
        detector = IngestionContradictionDetector(conn)

        new_entry = FakeEntry(
            id="new-1",
            content="The client switched to light mode for all dashboard interfaces.",
            tags=["client", "preferences"],
        )
        results = detector.check(new_entry)

        assert len(results) >= 1
        assert any(c.conflict_type == "preference" for c in results)

    def test_negation_conflict_detected(self):
        """Negation conflict between new and existing entry is detected."""
        conn = _make_test_db()
        _seed_entry(conn, "old-1",
                     "Team uses weekly standup reports for project tracking.")
        detector = IngestionContradictionDetector(conn)

        new_entry = FakeEntry(
            id="new-1",
            content="Team no longer uses weekly standup reports for project tracking.",
            tags=["team", "process"],
        )
        results = detector.check(new_entry)

        assert len(results) >= 1
        assert any(c.conflict_type == "negation" for c in results)

    def test_temporal_supersession_detected(self):
        """Temporal markers with 3+ shared entities trigger temporal conflict."""
        conn = _make_test_db()
        _seed_entry(conn, "old-1",
                     "Philip uses Python and FastAPI for backend development at Dicta.")
        detector = IngestionContradictionDetector(conn)

        new_entry = FakeEntry(
            id="new-1",
            content="Philip recently switched to using Rust for backend development at Dicta.",
            tags=["philip", "backend", "dicta"],
        )
        results = detector.check(new_entry)

        # Should detect either a preference/negation conflict or temporal supersession
        assert len(results) >= 1

    def test_short_content_returns_empty(self):
        """Content shorter than 30 chars is skipped."""
        conn = _make_test_db()
        _seed_entry(conn, "old-1", "Some existing long content about a topic.")
        detector = IngestionContradictionDetector(conn)

        new_entry = FakeEntry(id="new-1", content="Short text.")
        results = detector.check(new_entry)

        assert results == []

    def test_too_few_entities_returns_empty(self):
        """Content with fewer than 2 extractable entities is skipped."""
        conn = _make_test_db()
        _seed_entry(conn, "old-1", "Some existing content about things.")
        detector = IngestionContradictionDetector(conn)

        # "ok" is < 4 chars, "yes" is < 4 chars; only long filler words remain
        new_entry = FakeEntry(
            id="new-1",
            content="The the the the the the the the the the ok yes.",
        )
        results = detector.check(new_entry)

        assert results == []

    def test_self_comparison_skipped(self):
        """New entry should not be compared against itself if it appears in candidates."""
        conn = _make_test_db()
        # Seed with same ID as the new entry (simulating edge case)
        _seed_entry(conn, "entry-1",
                     "Solitaire core memory system rated at 95% of original vision.")
        detector = IngestionContradictionDetector(conn)

        new_entry = FakeEntry(
            id="entry-1",
            content="Solitaire core memory system revised to 75% of original vision after audit.",
            tags=["solitaire", "rating"],
        )
        results = detector.check(new_entry)

        # Should not produce a self-contradiction
        assert all(c.entry_id_old != c.entry_id_new for c in results)

    def test_no_candidates_returns_empty(self):
        """When no FTS candidates match, returns empty."""
        conn = _make_test_db()
        # Seed with completely unrelated content
        _seed_entry(conn, "old-1",
                     "Recipe for chocolate cake requires flour sugar eggs butter.")
        detector = IngestionContradictionDetector(conn)

        new_entry = FakeEntry(
            id="new-1",
            content="Quantum computing research breakthroughs in superconducting qubits.",
            tags=["quantum", "research"],
        )
        results = detector.check(new_entry)

        assert results == []


# ─── Contradiction Resolution Tests ──────────────────────────────────────────

class TestContradictionResolution:

    def _seed_contradiction(self, conn: sqlite3.Connection) -> str:
        """Seed a contradiction and return its ID."""
        _seed_entry(conn, "old-1", "Old content about a topic with many words for testing.",
                     metadata='{"confidence": {"effective": 0.8}}')
        _seed_entry(conn, "new-1", "New content about a topic with many words for testing.",
                     metadata='{"confidence": {"effective": 0.7}}')
        cid = "test-contra-1"
        conn.execute(
            """INSERT INTO pending_contradictions
               (id, entry_id_old, entry_id_new, conflict_type, description, detected_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cid, "old-1", "new-1", "numeric", "Test contradiction", "2026-03-29T00:00:00"),
        )
        conn.commit()
        return cid

    def test_resolve_new_wins_supersedes_old(self):
        """new_wins resolution marks old entry as superseded."""
        conn = _make_test_db()
        cid = self._seed_contradiction(conn)
        detector = IngestionContradictionDetector(conn)

        result = detector.resolve(cid, "new_wins")

        assert result is True
        row = conn.execute(
            "SELECT superseded_by FROM rolodex_entries WHERE id = 'old-1'"
        ).fetchone()
        assert row[0] == "new-1"

    def test_resolve_old_wins_reduces_confidence(self):
        """old_wins resolution reduces new entry's confidence."""
        conn = _make_test_db()
        cid = self._seed_contradiction(conn)
        detector = IngestionContradictionDetector(conn)

        result = detector.resolve(cid, "old_wins")

        assert result is True
        row = conn.execute(
            "SELECT metadata FROM rolodex_entries WHERE id = 'new-1'"
        ).fetchone()
        meta = json.loads(row[0])
        assert meta["confidence"]["effective"] == pytest.approx(0.5, abs=0.01)

    def test_resolve_both_valid_no_side_effects(self):
        """both_valid marks resolved but doesn't modify either entry."""
        conn = _make_test_db()
        cid = self._seed_contradiction(conn)
        detector = IngestionContradictionDetector(conn)

        result = detector.resolve(cid, "both_valid")

        assert result is True
        # Old entry not superseded
        old_row = conn.execute(
            "SELECT superseded_by FROM rolodex_entries WHERE id = 'old-1'"
        ).fetchone()
        assert old_row[0] is None
        # New entry confidence unchanged
        new_row = conn.execute(
            "SELECT metadata FROM rolodex_entries WHERE id = 'new-1'"
        ).fetchone()
        meta = json.loads(new_row[0])
        assert meta["confidence"]["effective"] == pytest.approx(0.7, abs=0.01)
        # Contradiction marked resolved
        c_row = conn.execute(
            "SELECT resolved_at, resolution FROM pending_contradictions WHERE id = ?",
            (cid,),
        ).fetchone()
        assert c_row[0] is not None
        assert c_row[1] == "both_valid"

    def test_resolve_nonexistent_returns_false(self):
        """Resolving a contradiction that doesn't exist returns False."""
        conn = _make_test_db()
        detector = IngestionContradictionDetector(conn)

        result = detector.resolve("nonexistent-id", "new_wins")

        assert result is False


# ─── Get Pending Tests ───────────────────────────────────────────────────────

class TestGetPending:

    def test_returns_unresolved_only(self):
        """Only returns contradictions where resolved_at IS NULL."""
        conn = _make_test_db()
        # One unresolved
        conn.execute(
            """INSERT INTO pending_contradictions
               (id, entry_id_old, entry_id_new, conflict_type, description, detected_at)
               VALUES ('c1', 'a', 'b', 'numeric', 'desc1', '2026-03-29T01:00:00')""",
        )
        # One resolved
        conn.execute(
            """INSERT INTO pending_contradictions
               (id, entry_id_old, entry_id_new, conflict_type, description, detected_at,
                resolved_at, resolution)
               VALUES ('c2', 'c', 'd', 'negation', 'desc2', '2026-03-29T00:00:00',
                       '2026-03-29T02:00:00', 'new_wins')""",
        )
        conn.commit()

        detector = IngestionContradictionDetector(conn)
        pending = detector.get_pending()

        assert len(pending) == 1
        assert pending[0].id == "c1"

    def test_respects_limit(self):
        """Limit parameter constrains result count."""
        conn = _make_test_db()
        for i in range(5):
            conn.execute(
                """INSERT INTO pending_contradictions
                   (id, entry_id_old, entry_id_new, conflict_type, description, detected_at)
                   VALUES (?, 'a', 'b', 'numeric', 'desc', ?)""",
                (f"c{i}", f"2026-03-29T0{i}:00:00"),
            )
        conn.commit()

        detector = IngestionContradictionDetector(conn)
        pending = detector.get_pending(limit=2)

        assert len(pending) == 2

    def test_ordered_by_detected_at_desc(self):
        """Most recent contradictions come first."""
        conn = _make_test_db()
        conn.execute(
            """INSERT INTO pending_contradictions
               (id, entry_id_old, entry_id_new, conflict_type, description, detected_at)
               VALUES ('older', 'a', 'b', 'numeric', 'desc', '2026-03-28T00:00:00')""",
        )
        conn.execute(
            """INSERT INTO pending_contradictions
               (id, entry_id_old, entry_id_new, conflict_type, description, detected_at)
               VALUES ('newer', 'c', 'd', 'numeric', 'desc', '2026-03-29T00:00:00')""",
        )
        conn.commit()

        detector = IngestionContradictionDetector(conn)
        pending = detector.get_pending()

        assert pending[0].id == "newer"
        assert pending[1].id == "older"


# ─── Conflict Utils Gap Coverage ─────────────────────────────────────────────

class TestExtractClaimEntities:

    def test_returns_4plus_char_words(self):
        """Only words with 4+ characters are returned."""
        entities = extract_claim_entities("The big cat ran fast today")
        # "big", "cat", "ran" are < 4 chars; "fast" = 4 chars, "today" = 5 chars
        assert "fast" in entities
        assert "today" in entities
        assert "big" not in entities
        assert "cat" not in entities
        assert "ran" not in entities

    def test_excludes_stopwords(self):
        """Stop words are filtered out even if 4+ chars."""
        entities = extract_claim_entities("This being about those other through")
        assert "this" not in entities
        assert "being" not in entities
        assert "about" not in entities
        assert "those" not in entities
        assert "other" not in entities
        assert "through" not in entities

    def test_tags_included_in_extraction(self):
        """Tags are appended to content for entity extraction."""
        entities = extract_claim_entities("Some content here.", tags=["solitaire", "testing"])
        assert "solitaire" in entities
        assert "testing" in entities

    def test_returns_lowercased(self):
        """All returned entities are lowercase."""
        entities = extract_claim_entities("Solitaire and FastAPI integration")
        for entity in entities:
            assert entity == entity.lower()


class TestHasTemporalMarkers:

    def test_detects_now(self):
        assert has_temporal_markers("Philip now uses Rust for backend work.")

    def test_detects_currently(self):
        assert has_temporal_markers("Currently using dark mode for all interfaces.")

    def test_detects_switched_to(self):
        assert has_temporal_markers("The team switched to weekly sprints.")

    def test_detects_no_longer(self):
        assert has_temporal_markers("Client no longer prefers email updates.")

    def test_detects_recently(self):
        assert has_temporal_markers("Recently started using the new API.")

    def test_returns_false_for_neutral_text(self):
        assert not has_temporal_markers("The standard configuration uses default settings.")

    def test_case_insensitive(self):
        assert has_temporal_markers("CURRENTLY using the new pipeline.")
