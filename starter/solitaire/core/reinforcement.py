"""
Solitaire -- Reinforcement Pipeline (Hindsight-inspired)

Detects when newly ingested content overlaps with existing entries and
strengthens their confidence scores. This is the "repeated observation"
mechanism: if the same fact surfaces across multiple sessions, its
confidence increases.

Two detection modes:
1. Token-level Jaccard similarity (fast, no embeddings needed)
2. Exact fact matching for UserFact entries (subject+predicate overlap)

The pipeline runs at ingestion time, after a new entry is stored. It
scans recent entries in the same category for overlap. If a match is
found above the reinforcement threshold, the existing entry's confidence
is reinforced via the confidence module.

This does NOT prevent the new entry from being stored. Both entries
persist -- the older one just gets a confidence bump. The maintenance
engine's near-duplicate pass may later merge them if they're close enough.
"""

import json
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

from .confidence import (
    ConfidenceScore,
    extract_confidence_from_metadata,
    merge_confidence_into_metadata,
    initial_confidence,
    reinforce,
)


# -- Configuration --------------------------------------------------------

# Jaccard threshold for reinforcement. Lower than the duplicate-merge
# threshold (0.85) because we want to catch paraphrased restatements,
# not just near-copies.
REINFORCEMENT_THRESHOLD = 0.55

# Maximum number of existing entries to compare against per ingestion.
# Keeps the O(n) scan bounded.
MAX_CANDIDATES = 100

# Categories exempt from reinforcement (they have their own lifecycle).
EXEMPT_CATEGORIES = {"disposition_drift", "behavioral"}


# -- Core pipeline --------------------------------------------------------

def find_reinforcement_targets(
    conn: sqlite3.Connection,
    new_content: str,
    new_category: str,
    exclude_id: Optional[str] = None,
    threshold: float = REINFORCEMENT_THRESHOLD,
    max_candidates: int = MAX_CANDIDATES,
) -> List[Tuple[str, float]]:
    """
    Find existing entries that the new content reinforces.

    Returns a list of (entry_id, similarity_score) tuples for entries
    above the reinforcement threshold. Sorted by similarity descending.

    Uses token-level Jaccard similarity, same as the maintenance engine's
    near-duplicate detection but with a lower threshold.
    """
    if new_category in EXEMPT_CATEGORIES:
        return []

    new_tokens = set(new_content.lower().split())
    if len(new_tokens) < 5:
        return []

    # Fetch recent active entries in the same category
    sql = """
        SELECT id, content, metadata FROM rolodex_entries
        WHERE superseded_by IS NULL
        AND category = ?
        ORDER BY created_at DESC
        LIMIT ?
    """
    params: list = [new_category, max_candidates]

    rows = conn.execute(sql, params).fetchall()

    targets = []
    for row in rows:
        entry_id = row["id"] if isinstance(row, sqlite3.Row) else row[0]
        if entry_id == exclude_id:
            continue

        content = row["content"] if isinstance(row, sqlite3.Row) else row[1]
        existing_tokens = set(content.lower().split())
        if len(existing_tokens) < 5:
            continue

        intersection = len(new_tokens & existing_tokens)
        union = len(new_tokens | existing_tokens)
        if union == 0:
            continue

        similarity = intersection / union
        if similarity >= threshold:
            targets.append((entry_id, similarity))

    targets.sort(key=lambda t: t[1], reverse=True)
    return targets


def reinforce_entries(
    conn: sqlite3.Connection,
    targets: List[Tuple[str, float]],
    now: Optional[datetime] = None,
) -> int:
    """
    Apply reinforcement to the given entries.

    For each target, loads its current confidence score from metadata,
    applies reinforcement, and writes it back. If the entry has no
    confidence score yet (pre-Hindsight entry), initializes one from
    its provenance before reinforcing.

    Returns the number of entries reinforced.
    """
    if now is None:
        now = datetime.utcnow()

    reinforced_count = 0

    for entry_id, similarity in targets:
        row = conn.execute(
            "SELECT metadata, category, provenance FROM rolodex_entries WHERE id = ?",
            (entry_id,)
        ).fetchone()
        if not row:
            continue

        metadata_raw = row["metadata"] if isinstance(row, sqlite3.Row) else row[0]
        category = row["category"] if isinstance(row, sqlite3.Row) else row[1]

        # Handle provenance column which may not exist in older schemas
        try:
            provenance = row["provenance"] if isinstance(row, sqlite3.Row) else row[2]
        except (IndexError, KeyError):
            provenance = "unknown"
        provenance = provenance or "unknown"

        metadata = json.loads(metadata_raw or "{}")

        # Load or initialize confidence score
        score = extract_confidence_from_metadata(metadata)
        if score is None:
            score = initial_confidence(provenance=provenance, category=category)

        # Apply reinforcement
        score = reinforce(score, now=now)

        # Write back
        metadata = merge_confidence_into_metadata(metadata, score)
        conn.execute(
            "UPDATE rolodex_entries SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), entry_id)
        )
        reinforced_count += 1

    if reinforced_count > 0:
        conn.commit()

    return reinforced_count


def on_entry_created(
    conn: sqlite3.Connection,
    entry_id: str,
    content: str,
    category: str,
    provenance: str = "unknown",
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Hook called after a new entry is stored. Performs two operations:

    1. Stamps the new entry with an initial confidence score.
    2. Finds and reinforces existing entries that overlap with this content.

    Returns a summary dict with counts and target IDs.
    """
    if now is None:
        now = datetime.utcnow()

    result = {
        "entry_id": entry_id,
        "initial_confidence": None,
        "reinforcement_targets": 0,
        "reinforced_ids": [],
    }

    # 1. Stamp initial confidence on the new entry
    init_score = initial_confidence(provenance=provenance, category=category)
    row = conn.execute(
        "SELECT metadata FROM rolodex_entries WHERE id = ?", (entry_id,)
    ).fetchone()
    if row:
        metadata_raw = row["metadata"] if isinstance(row, sqlite3.Row) else row[0]
        metadata = json.loads(metadata_raw or "{}")
        metadata = merge_confidence_into_metadata(metadata, init_score)
        conn.execute(
            "UPDATE rolodex_entries SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), entry_id)
        )
        conn.commit()
        result["initial_confidence"] = init_score.effective

    # 2. Find and reinforce overlapping entries
    if category not in EXEMPT_CATEGORIES:
        targets = find_reinforcement_targets(
            conn, content, category, exclude_id=entry_id
        )
        if targets:
            count = reinforce_entries(conn, targets, now=now)
            result["reinforcement_targets"] = count
            result["reinforced_ids"] = [t[0] for t in targets[:count]]

    return result
