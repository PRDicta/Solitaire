"""
Solitaire -- Ingestion-Time Contradiction Detector

Runs during ingestion to detect contradictions between a newly ingested
entry and high-confidence existing entries. Detected contradictions are
persisted to `pending_contradictions` and surfaced in the next session
briefing for user resolution.

Uses the shared conflict detection from retrieval/conflict_utils.py
(same heuristics as the reranker post-filter). Adds temporal supersession
detection for ingestion-specific use.
"""
import sqlite3
import uuid
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from ..core.types import RolodexEntry
from ..retrieval.conflict_utils import (
    extract_claim_entities,
    detect_claim_conflict,
    has_temporal_markers,
)


@dataclass
class PendingContradiction:
    """A detected contradiction awaiting resolution."""
    id: str
    entry_id_old: str
    entry_id_new: str
    conflict_type: str     # numeric | preference | negation | temporal
    description: str
    detected_at: str


class IngestionContradictionDetector:
    """
    Checks newly ingested entries against high-confidence existing entries
    for contradictions. Lightweight: FTS-based candidate retrieval, no
    embedding calls.
    """

    # Only compare against entries with significance >= this threshold
    SIGNIFICANCE_THRESHOLD = 0.5
    # Max candidates to compare against (keep it fast)
    MAX_CANDIDATES = 20

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def check(self, new_entry: RolodexEntry) -> List[PendingContradiction]:
        """
        Check a newly ingested entry against existing entries for contradictions.
        Stores any detected contradictions in pending_contradictions table.
        Returns the list of detected contradictions.
        """
        if not new_entry.content or len(new_entry.content.strip()) < 30:
            return []

        # Extract entities from the new entry for candidate matching
        new_entities = extract_claim_entities(new_entry.content, new_entry.tags)
        if len(new_entities) < 2:
            # Too few entities to make a meaningful comparison
            return []

        # Find candidate entries with overlapping entities via FTS
        candidates = self._find_candidates(new_entry, new_entities)
        if not candidates:
            return []

        contradictions = []
        new_text = new_entry.content
        new_has_temporal = has_temporal_markers(new_text)

        for old_id, old_content, old_tags_json in candidates:
            if old_id == new_entry.id:
                continue

            # Standard conflict detection (numeric, preference, negation)
            conflict_type = detect_claim_conflict(new_text, old_content)

            # Temporal supersession: new entry has temporal markers suggesting
            # it replaces an older claim about the same entities
            if not conflict_type and new_has_temporal:
                old_entities = extract_claim_entities(
                    old_content,
                    json.loads(old_tags_json) if old_tags_json else None,
                )
                overlap = new_entities & old_entities
                # Need significant overlap (3+ shared entities) to suggest
                # these entries are about the same subject
                if len(overlap) >= 3:
                    conflict_type = "temporal"

            if conflict_type:
                # Build human-readable description
                old_snippet = old_content[:120].replace("\n", " ")
                new_snippet = new_text[:120].replace("\n", " ")
                description = (
                    f"{conflict_type} conflict: "
                    f'old: "{old_snippet}..." vs '
                    f'new: "{new_snippet}..."'
                )

                contradiction = PendingContradiction(
                    id=str(uuid.uuid4())[:12],
                    entry_id_old=old_id,
                    entry_id_new=new_entry.id,
                    conflict_type=conflict_type,
                    description=description,
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
                contradictions.append(contradiction)

        # Persist to DB
        for c in contradictions:
            self._store(c)

        return contradictions

    def _find_candidates(
        self,
        new_entry: RolodexEntry,
        entities: set,
    ) -> List[tuple]:
        """
        Find existing entries with overlapping entities using FTS.
        Returns list of (id, content, tags) tuples.
        """
        # Build FTS query from top entities (limit to 5 for speed)
        query_terms = sorted(entities, key=len, reverse=True)[:5]
        fts_query = " OR ".join(query_terms)

        try:
            rows = self.conn.execute(
                """
                SELECT e.id, e.content, e.tags
                FROM rolodex_fts f
                JOIN rolodex_entries e ON e.id = f.entry_id
                WHERE rolodex_fts MATCH ?
                  AND e.archived_at IS NULL
                  AND e.id != ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, new_entry.id, self.MAX_CANDIDATES),
            ).fetchall()
            return rows
        except Exception:
            return []

    def _store(self, contradiction: PendingContradiction) -> None:
        """Persist a contradiction to the pending_contradictions table."""
        try:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO pending_contradictions
                (id, entry_id_old, entry_id_new, conflict_type, description,
                 detected_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    contradiction.id,
                    contradiction.entry_id_old,
                    contradiction.entry_id_new,
                    contradiction.conflict_type,
                    contradiction.description,
                    contradiction.detected_at,
                ),
            )
            self.conn.commit()
        except Exception:
            pass  # Non-critical; don't block ingestion

    def resolve(
        self,
        contradiction_id: str,
        resolution: str,
        resolved_by: str = "user",
    ) -> bool:
        """
        Mark a contradiction as resolved.

        Args:
            contradiction_id: The contradiction ID.
            resolution: One of 'new_wins', 'old_wins', 'both_valid'.
            resolved_by: 'user' or 'auto'.

        Returns True if the contradiction was found and resolved.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Get the contradiction details
        row = self.conn.execute(
            "SELECT entry_id_old, entry_id_new FROM pending_contradictions WHERE id = ?",
            (contradiction_id,),
        ).fetchone()
        if not row:
            return False

        old_id, new_id = row[0], row[1]

        # Mark resolved
        self.conn.execute(
            """
            UPDATE pending_contradictions
            SET resolved_at = ?, resolution = ?, resolved_by = ?
            WHERE id = ?
            """,
            (now, resolution, resolved_by, contradiction_id),
        )

        # Apply resolution effects
        if resolution == "new_wins":
            # Old entry is superseded
            self.conn.execute(
                "UPDATE rolodex_entries SET superseded_by = ? WHERE id = ?",
                (new_id, old_id),
            )
        elif resolution == "old_wins":
            # New entry gets reduced confidence via metadata
            self._reduce_confidence(new_id)

        self.conn.commit()
        return True

    def _reduce_confidence(self, entry_id: str) -> None:
        """Reduce an entry's confidence in metadata after losing a contradiction."""
        row = self.conn.execute(
            "SELECT metadata FROM rolodex_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        if not row:
            return
        meta = json.loads(row[0] or "{}")
        conf = meta.get("confidence", {})
        if isinstance(conf, dict):
            conf["effective"] = max(0.1, conf.get("effective", 0.5) - 0.2)
            meta["confidence"] = conf
        self.conn.execute(
            "UPDATE rolodex_entries SET metadata = ? WHERE id = ?",
            (json.dumps(meta), entry_id),
        )

    def get_pending(self, limit: int = 10) -> List[PendingContradiction]:
        """Get unresolved contradictions, most recent first."""
        rows = self.conn.execute(
            """
            SELECT id, entry_id_old, entry_id_new, conflict_type,
                   description, detected_at
            FROM pending_contradictions
            WHERE resolved_at IS NULL
            ORDER BY detected_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            PendingContradiction(
                id=r[0], entry_id_old=r[1], entry_id_new=r[2],
                conflict_type=r[3], description=r[4], detected_at=r[5],
            )
            for r in rows
        ]
