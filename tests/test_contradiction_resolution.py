"""
Tests for Phase 2: Contradiction Resolution in the retrieval pipeline.

Tests the reranker's post-filter that detects and suppresses stale entries
when newer entries contradict them.
"""
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import pytest

from solitaire.retrieval.reranker import Reranker, RerankerConfig, ScoredCandidate
from solitaire.retrieval.conflict_utils import (
    numeric_conflict, negation_conflict, preference_conflict,
)


# ─── Minimal RolodexEntry stand-in for tests ────────────────────────────────

@dataclass
class FakeEntry:
    """Minimal entry for reranker testing. Mirrors RolodexEntry fields used."""
    id: str = "test-1"
    content: str = ""
    tags: List[str] = field(default_factory=list)
    category: str = "note"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 5
    verbatim_source: bool = True
    provenance: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_accessed: Optional[datetime] = None
    embedding: Optional[list] = None


def _make_scored(entry: FakeEntry, score: float = 0.5) -> ScoredCandidate:
    """Wrap a FakeEntry in a ScoredCandidate with a given composite score."""
    sc = ScoredCandidate(entry=entry)
    sc.composite_score = score
    return sc


# ─── Numeric Contradiction Tests ────────────────────────────────────────────

class TestNumericContradiction:

    def test_conflicting_percentages_suppresses_older(self):
        """Two entries about the same subject with different % values."""
        reranker = Reranker()

        older = FakeEntry(
            id="old-1",
            content="Solitaire core memory system rated at 95% of original vision.",
            created_at=datetime(2026, 2, 16, tzinfo=timezone.utc),
        )
        newer = FakeEntry(
            id="new-1",
            content="Solitaire core memory system revised to 85% of original vision after audit.",
            created_at=datetime(2026, 3, 20, tzinfo=timezone.utc),
        )

        scored = [
            _make_scored(newer, 0.8),
            _make_scored(older, 0.7),
        ]

        result = reranker._resolve_contradictions(scored, limit=5)
        ids = [sc.entry.id for sc in result]

        assert "new-1" in ids
        assert "old-1" not in ids, "Older contradicted entry should be suppressed"

    def test_no_suppression_when_values_close(self):
        """Values within 5% threshold should not trigger suppression."""
        reranker = Reranker()

        entry_a = FakeEntry(
            id="a",
            content="Productization wrapper at 72% complete.",
            created_at=datetime(2026, 2, 16, tzinfo=timezone.utc),
        )
        entry_b = FakeEntry(
            id="b",
            content="Productization wrapper at 75% complete.",
            created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )

        scored = [_make_scored(entry_b, 0.8), _make_scored(entry_a, 0.7)]
        result = reranker._resolve_contradictions(scored, limit=5)

        assert len(result) == 2, "Close values should not trigger suppression"

    def test_no_suppression_different_subjects(self):
        """Different percentage claims about different subjects."""
        reranker = Reranker()

        entry_a = FakeEntry(
            id="a",
            content="Authentication module at 40% progress.",
            created_at=datetime(2026, 2, 16, tzinfo=timezone.utc),
        )
        entry_b = FakeEntry(
            id="b",
            content="Database migration at 90% progress.",
            created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )

        scored = [_make_scored(entry_b, 0.8), _make_scored(entry_a, 0.7)]
        result = reranker._resolve_contradictions(scored, limit=5)

        assert len(result) == 2, "Different subjects should not trigger suppression"


# ─── Preference Contradiction Tests ─────────────────────────────────────────

class TestPreferenceContradiction:

    def test_preference_change_suppresses_older(self):
        """'prefers weekly' vs 'switched to monthly' should suppress older."""
        reranker = Reranker()

        older = FakeEntry(
            id="old-pref",
            content="Client prefers weekly reports for all project updates.",
            created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
        )
        newer = FakeEntry(
            id="new-pref",
            content="Client switched to monthly reports starting March.",
            created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )

        scored = [_make_scored(newer, 0.8), _make_scored(older, 0.75)]
        result = reranker._resolve_contradictions(scored, limit=5)
        ids = [sc.entry.id for sc in result]

        assert "new-pref" in ids
        assert "old-pref" not in ids


# ─── Negation Contradiction Tests ───────────────────────────────────────────

class TestNegationContradiction:

    def test_negation_suppresses_older(self):
        """'uses dark mode' vs 'no longer uses dark mode' should suppress older."""
        reranker = Reranker()

        older = FakeEntry(
            id="old-neg",
            content="Philip uses dark mode for all editors and terminals.",
            created_at=datetime(2026, 1, 10, tzinfo=timezone.utc),
        )
        newer = FakeEntry(
            id="new-neg",
            content="Philip no longer uses dark mode after the display upgrade.",
            created_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        )

        scored = [_make_scored(newer, 0.8), _make_scored(older, 0.7)]
        result = reranker._resolve_contradictions(scored, limit=5)
        ids = [sc.entry.id for sc in result]

        assert "new-neg" in ids
        assert "old-neg" not in ids

    def test_stopped_pattern(self):
        """'stopped using' negation should suppress the affirmative entry."""
        reranker = Reranker()

        older = FakeEntry(
            id="old-stop",
            content="Team uses Slack for all internal communication.",
            created_at=datetime(2026, 1, 10, tzinfo=timezone.utc),
        )
        newer = FakeEntry(
            id="new-stop",
            content="Team stopped using Slack, migrated to Discord.",
            created_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        )

        scored = [_make_scored(newer, 0.8), _make_scored(older, 0.7)]
        result = reranker._resolve_contradictions(scored, limit=5)
        ids = [sc.entry.id for sc in result]

        assert "new-stop" in ids
        assert "old-stop" not in ids


# ─── Edge Cases ─────────────────────────────────────────────────────────────

class TestContradictionEdgeCases:

    def test_single_entry_passes_through(self):
        """Single entry should pass through unchanged."""
        reranker = Reranker()
        entry = FakeEntry(id="solo", content="Just one entry.")
        scored = [_make_scored(entry, 0.9)]

        result = reranker._resolve_contradictions(scored, limit=5)
        assert len(result) == 1
        assert result[0].entry.id == "solo"

    def test_empty_list(self):
        """Empty scored list should return empty."""
        reranker = Reranker()
        result = reranker._resolve_contradictions([], limit=5)
        assert result == []

    def test_suppression_backfills_results(self):
        """When a top entry is suppressed, lower entries should fill in."""
        reranker = Reranker()

        old_contradicted = FakeEntry(
            id="old-c",
            content="Project completion at 40% done.",
            created_at=datetime(2026, 1, 10, tzinfo=timezone.utc),
        )
        new_correction = FakeEntry(
            id="new-c",
            content="Project completion revised to 80% done.",
            created_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        )
        unrelated = FakeEntry(
            id="unrelated",
            content="Meeting notes from the design review.",
            created_at=datetime(2026, 3, 10, tzinfo=timezone.utc),
        )

        scored = [
            _make_scored(new_correction, 0.9),
            _make_scored(old_contradicted, 0.85),
            _make_scored(unrelated, 0.6),
        ]

        result = reranker._resolve_contradictions(scored, limit=3)
        ids = [sc.entry.id for sc in result]

        assert "new-c" in ids
        assert "unrelated" in ids
        assert "old-c" not in ids

    def test_three_entries_only_oldest_suppressed(self):
        """With three conflicting entries, only the two oldest should be suppressed
        against the newest (pairwise, not chain)."""
        reranker = Reranker()

        oldest = FakeEntry(
            id="v1",
            content="Pipeline throughput at 20% efficiency.",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        middle = FakeEntry(
            id="v2",
            content="Pipeline throughput at 50% efficiency.",
            created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )
        newest = FakeEntry(
            id="v3",
            content="Pipeline throughput at 80% efficiency.",
            created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )

        scored = [
            _make_scored(newest, 0.9),
            _make_scored(middle, 0.8),
            _make_scored(oldest, 0.7),
        ]

        result = reranker._resolve_contradictions(scored, limit=5)
        ids = [sc.entry.id for sc in result]

        assert "v3" in ids, "Newest should survive"
        # v1 and v2 each conflict with v3 (>5% delta, shared context),
        # so both should be suppressed
        assert "v1" not in ids
        assert "v2" not in ids


# ─── Integration: Full Rerank with Contradiction Resolution ─────────────────

class TestRerankerIntegration:

    def test_rerank_resolves_contradictions_in_results(self):
        """Full rerank call should suppress contradicted entries."""
        reranker = Reranker()

        older = FakeEntry(
            id="old-int",
            content="Solitaire productization at 40% complete across all platforms.",
            tags=["solitaire", "productization"],
            created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
        )
        newer = FakeEntry(
            id="new-int",
            content="Solitaire productization revised to 75% complete after CI pipeline shipped.",
            tags=["solitaire", "productization"],
            created_at=datetime(2026, 3, 20, tzinfo=timezone.utc),
        )

        candidates = [
            (older, 0.85),   # High FTS score (keyword match)
            (newer, 0.80),   # Slightly lower FTS but newer
        ]

        results = reranker.rerank(
            candidates=candidates,
            query="solitaire productization status",
            limit=5,
        )
        ids = [sc.entry.id for sc in results]

        assert "new-int" in ids
        assert "old-int" not in ids, "Older contradicted entry should be filtered from results"


# ─── Static Method Unit Tests ───────────────────────────────────────────────

class TestConflictDetection:

    def test_numeric_conflict_detected(self):
        assert numeric_conflict(
            "completion at 40% done",
            "completion revised to 80% done",
        )

    def test_numeric_no_conflict_close_values(self):
        assert not numeric_conflict(
            "completion at 73% done",
            "completion at 75% done",
        )

    def test_negation_conflict_detected(self):
        assert negation_conflict(
            "team uses slack for communication",
            "team no longer uses slack",
        )

    def test_negation_no_conflict_different_subjects(self):
        assert not negation_conflict(
            "team uses slack for communication",
            "client no longer uses email",
        )

    def test_preference_conflict_detected(self):
        assert preference_conflict(
            "client prefers weekly reports",
            "client switched to monthly summaries",
        )
