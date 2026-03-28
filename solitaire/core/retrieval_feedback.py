"""
Solitaire — Retrieval Feedback (Phase 19: Self-Learning Layer 1)

Closes the feedback loop between retrieval and entry quality.
When recall returns entries, we track which ones the model actually
used in its response vs. which were returned but ignored. Over time,
this adjusts entry significance weights so the most useful entries
float to the top.

Signal flow:
    1. recall() returns entry IDs → logged as retrieval_outcomes (used=0)
    2. ingest-turn() fires → compare recalled IDs against assistant text
    3. Entries whose content appears in the response → mark used=1
    4. End-of-session or maintain: run weight adjustment
       - Used entries: significance += USED_BOOST (capped at 1.0)
       - Ignored 3+ times: significance -= IGNORED_PENALTY (floored at 0.1)
       - Never recalled in 30+ days: confidence decay kicks in
"""

import json
import re
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from .memory_weight import (
    MemoryWeight,
    apply_confidence_decay,
    extract_weight_from_metadata,
    merge_weight_into_metadata,
)


# ─── Constants ──────────────────────────────────────────────────────────────

USED_BOOST = 0.05          # Significance bump per confirmed use
IGNORED_PENALTY = 0.1      # Significance penalty after 3+ ignores
IGNORED_THRESHOLD = 3      # Consecutive ignores before penalty applies
STALE_DAYS = 30            # Days without recall before confidence decay
SIGNIFICANCE_CAP = 1.0
SIGNIFICANCE_FLOOR = 0.1
FUZZY_MATCH_MIN_WORDS = 3  # Minimum consecutive content words to count as "used"


# ─── Outcome Recording ─────────────────────────────────────────────────────

def record_recall_outcomes(
    conn: sqlite3.Connection,
    session_id: str,
    query_text: str,
    entry_ids: List[str],
    timestamp: Optional[datetime] = None,
) -> int:
    """
    Record that these entries were returned by a recall query.
    Initially all marked used=0. The ingest step updates them.

    Returns the count of outcomes recorded.
    """
    if not entry_ids:
        return 0

    now = timestamp or datetime.now(timezone.utc)
    count = 0
    for eid in entry_ids:
        outcome_id = str(uuid.uuid4())
        try:
            conn.execute(
                """INSERT INTO retrieval_outcomes
                   (id, entry_id, session_id, query_text, recalled_at, used, confidence, context)
                   VALUES (?, ?, ?, ?, ?, 0, 0.0, '')""",
                (outcome_id, eid, session_id, query_text, now.isoformat()),
            )
            count += 1
        except sqlite3.IntegrityError:
            pass  # Duplicate ID, skip
    conn.commit()
    return count


def evaluate_usage(
    conn: sqlite3.Connection,
    session_id: str,
    assistant_response: str,
    recalled_entry_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare recalled entries against the assistant's response to determine
    which entries were actually used.

    Strategy: extract key terms from each recalled entry's content, then
    check if enough of those terms appear in the assistant response.
    This is fuzzy matching, not exact substring -- the model paraphrases.

    Args:
        conn: Database connection.
        session_id: Current session ID.
        assistant_response: The assistant's response text.
        recalled_entry_ids: If provided, only evaluate these entries.
                           Otherwise, evaluate all unresolved outcomes for this session.

    Returns:
        Dict with: evaluated count, used count, ignored count, entry details.
    """
    # Get pending outcomes for this session
    if recalled_entry_ids:
        placeholders = ",".join("?" for _ in recalled_entry_ids)
        rows = conn.execute(
            f"""SELECT ro.id AS outcome_id, ro.entry_id, ro.query_text
                FROM retrieval_outcomes ro
                WHERE ro.session_id = ?
                AND ro.entry_id IN ({placeholders})
                AND ro.used = 0""",
            [session_id] + list(recalled_entry_ids),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT ro.id AS outcome_id, ro.entry_id, ro.query_text
               FROM retrieval_outcomes ro
               WHERE ro.session_id = ?
               AND ro.used = 0""",
            (session_id,),
        ).fetchall()

    if not rows:
        return {"evaluated": 0, "used": 0, "ignored": 0, "details": []}

    # Fetch entry content for matching
    entry_ids = list({r["entry_id"] for r in rows})
    entry_content = {}
    for eid in entry_ids:
        content_row = conn.execute(
            "SELECT content FROM rolodex_entries WHERE id = ?", (eid,)
        ).fetchone()
        if content_row:
            entry_content[eid] = content_row["content"]

    # Normalize response for matching
    response_lower = assistant_response.lower()
    response_words = set(re.findall(r'\b\w{4,}\b', response_lower))

    results = {"evaluated": 0, "used": 0, "ignored": 0, "details": []}

    for row in rows:
        eid = row["entry_id"]
        content = entry_content.get(eid, "")
        if not content:
            continue

        used = _fuzzy_match(content, response_lower, response_words)
        confidence = _compute_match_confidence(content, response_lower, response_words)

        conn.execute(
            """UPDATE retrieval_outcomes
               SET used = ?, confidence = ?, context = ?
               WHERE id = ?""",
            (
                1 if used else 0,
                confidence,
                "fuzzy_match" if used else "no_match",
                row["outcome_id"],
            ),
        )

        results["evaluated"] += 1
        if used:
            results["used"] += 1
        else:
            results["ignored"] += 1
        results["details"].append({
            "entry_id": eid,
            "used": used,
            "confidence": round(confidence, 3),
        })

    conn.commit()
    return results


# ─── Weight Adjustment ──────────────────────────────────────────────────────

def adjust_weights(
    conn: sqlite3.Connection,
    session_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Adjust entry significance weights based on accumulated retrieval outcomes.

    Rules:
        - Entry recalled AND used: significance += USED_BOOST (cap 1.0)
        - Entry recalled AND ignored IGNORED_THRESHOLD+ times: significance -= IGNORED_PENALTY (floor 0.1)
        - Entry never recalled in STALE_DAYS: apply confidence decay

    Can be scoped to a single session or run across all data.

    Returns:
        Dict with: boosted count, penalized count, decayed count, details.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    stats = {"boosted": 0, "penalized": 0, "decayed": 0, "unchanged": 0}

    # 1. Process used entries (boost significance)
    if session_id:
        used_rows = conn.execute(
            """SELECT DISTINCT entry_id FROM retrieval_outcomes
               WHERE session_id = ? AND used = 1""",
            (session_id,),
        ).fetchall()
    else:
        used_rows = conn.execute(
            "SELECT DISTINCT entry_id FROM retrieval_outcomes WHERE used = 1"
        ).fetchall()

    for row in used_rows:
        eid = row["entry_id"]
        updated = _boost_significance(conn, eid, USED_BOOST)
        if updated:
            stats["boosted"] += 1

    # 2. Process ignored entries (penalize if threshold met)
    if session_id:
        # Count ignores per entry in this session
        ignored_rows = conn.execute(
            """SELECT entry_id, COUNT(*) as ignore_count
               FROM retrieval_outcomes
               WHERE session_id = ? AND used = 0
               GROUP BY entry_id
               HAVING COUNT(*) >= ?""",
            (session_id, IGNORED_THRESHOLD),
        ).fetchall()
    else:
        # Count total consecutive ignores across all sessions
        ignored_rows = conn.execute(
            """SELECT entry_id, COUNT(*) as ignore_count
               FROM retrieval_outcomes
               WHERE used = 0
               AND entry_id NOT IN (
                   SELECT DISTINCT entry_id FROM retrieval_outcomes WHERE used = 1
               )
               GROUP BY entry_id
               HAVING COUNT(*) >= ?""",
            (IGNORED_THRESHOLD,),
        ).fetchall()

    for row in ignored_rows:
        eid = row["entry_id"]
        updated = _penalize_significance(conn, eid, IGNORED_PENALTY)
        if updated:
            stats["penalized"] += 1

    # 3. Process stale entries (confidence decay for entries never recalled)
    stale_cutoff = now - timedelta(days=STALE_DAYS)
    stale_rows = conn.execute(
        """SELECT id, metadata, category, created_at
           FROM rolodex_entries
           WHERE (last_accessed IS NULL OR last_accessed < ?)
           AND created_at < ?
           AND (superseded_by IS NULL OR superseded_by = '')
           AND id NOT IN (
               SELECT DISTINCT entry_id FROM retrieval_outcomes
               WHERE recalled_at > ?
           )""",
        (
            stale_cutoff.isoformat(),
            stale_cutoff.isoformat(),
            stale_cutoff.isoformat(),
        ),
    ).fetchall()

    for row in stale_rows:
        eid = row["id"]
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        weight = extract_weight_from_metadata(metadata)
        if not weight:
            continue

        is_uk = row["category"] == "user_knowledge"
        age_days = (now - datetime.fromisoformat(
            row["created_at"].replace('Z', '+00:00')
        )).total_seconds() / 86400.0

        new_weight = apply_confidence_decay(weight, age_days, is_user_knowledge=is_uk)
        if new_weight.confidence < weight.confidence:
            new_metadata = merge_weight_into_metadata(metadata, new_weight)
            conn.execute(
                "UPDATE rolodex_entries SET metadata = ? WHERE id = ?",
                (json.dumps(new_metadata), eid),
            )
            stats["decayed"] += 1
        else:
            stats["unchanged"] += 1

    conn.commit()
    return stats


def get_retrieval_stats(
    conn: sqlite3.Connection,
    session_id: Optional[str] = None,
    window_sessions: int = 5,
) -> Dict[str, Any]:
    """
    Aggregate retrieval outcome statistics.

    Returns:
        Dict with: total_recalls, total_used, total_ignored,
        use_rate, per_entry breakdown for top entries.
    """
    if session_id:
        base_query = "SELECT * FROM retrieval_outcomes WHERE session_id = ?"
        params: list = [session_id]
    else:
        base_query = "SELECT * FROM retrieval_outcomes"
        params = []

    rows = conn.execute(base_query, params).fetchall()

    if not rows:
        return {
            "total_recalls": 0,
            "total_used": 0,
            "total_ignored": 0,
            "use_rate": 0.0,
            "top_used": [],
            "top_ignored": [],
        }

    total = len(rows)
    used = sum(1 for r in rows if r["used"])
    ignored = total - used

    # Per-entry aggregation
    entry_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"recalled": 0, "used": 0})
    for r in rows:
        entry_stats[r["entry_id"]]["recalled"] += 1
        if r["used"]:
            entry_stats[r["entry_id"]]["used"] += 1

    # Top used and top ignored
    top_used = sorted(
        [(eid, s["used"], s["recalled"]) for eid, s in entry_stats.items() if s["used"] > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    top_ignored = sorted(
        [(eid, s["recalled"] - s["used"], s["recalled"])
         for eid, s in entry_stats.items() if s["recalled"] > s["used"]],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    return {
        "total_recalls": total,
        "total_used": used,
        "total_ignored": ignored,
        "use_rate": round(used / max(1, total), 3),
        "top_used": [
            {"entry_id": eid, "used_count": u, "recall_count": r}
            for eid, u, r in top_used
        ],
        "top_ignored": [
            {"entry_id": eid, "ignored_count": i, "recall_count": r}
            for eid, i, r in top_ignored
        ],
    }


# ─── Internal Helpers ───────────────────────────────────────────────────────

def _fuzzy_match(
    entry_content: str,
    response_lower: str,
    response_words: Set[str],
) -> bool:
    """
    Determine if an entry's content was likely used in the response.

    Strategy: extract significant words (4+ chars) from the entry content,
    check how many appear in the response. If the overlap ratio exceeds
    a threshold, the entry was probably used.
    """
    content_words = set(re.findall(r'\b\w{4,}\b', entry_content.lower()))
    if len(content_words) < FUZZY_MATCH_MIN_WORDS:
        # Very short entry: check if any key terms appear
        return bool(content_words & response_words)

    overlap = content_words & response_words
    ratio = len(overlap) / len(content_words) if content_words else 0

    # Threshold: at least 25% of the entry's significant words appear
    # in the response. This is intentionally loose because the model
    # paraphrases heavily.
    return ratio >= 0.25


def _compute_match_confidence(
    entry_content: str,
    response_lower: str,
    response_words: Set[str],
) -> float:
    """
    Compute a 0.0-1.0 confidence score for how strongly the entry
    influenced the response. Higher overlap = higher confidence.
    """
    content_words = set(re.findall(r'\b\w{4,}\b', entry_content.lower()))
    if not content_words:
        return 0.0

    overlap = content_words & response_words
    ratio = len(overlap) / len(content_words)

    # Scale to 0.0-1.0 with diminishing returns past 50% overlap
    if ratio >= 0.5:
        return min(1.0, 0.7 + (ratio - 0.5) * 0.6)
    return ratio * 1.4  # Linear up to 0.7 at 50%


def _boost_significance(
    conn: sqlite3.Connection,
    entry_id: str,
    boost: float,
) -> bool:
    """Boost an entry's significance weight. Returns True if updated."""
    row = conn.execute(
        "SELECT metadata FROM rolodex_entries WHERE id = ?", (entry_id,)
    ).fetchone()
    if not row:
        return False

    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    weight = extract_weight_from_metadata(metadata)

    if weight:
        new_sig = min(SIGNIFICANCE_CAP, weight.significance + boost)
        if new_sig == weight.significance:
            return False
        weight.significance = new_sig
    else:
        # No weight yet: create one with boosted default
        weight = MemoryWeight(significance=min(SIGNIFICANCE_CAP, 0.3 + boost))

    new_metadata = merge_weight_into_metadata(metadata, weight)
    conn.execute(
        "UPDATE rolodex_entries SET metadata = ? WHERE id = ?",
        (json.dumps(new_metadata), entry_id),
    )
    return True


def _penalize_significance(
    conn: sqlite3.Connection,
    entry_id: str,
    penalty: float,
) -> bool:
    """Penalize an entry's significance weight. Returns True if updated."""
    row = conn.execute(
        "SELECT metadata FROM rolodex_entries WHERE id = ?", (entry_id,)
    ).fetchone()
    if not row:
        return False

    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    weight = extract_weight_from_metadata(metadata)

    if weight:
        new_sig = max(SIGNIFICANCE_FLOOR, weight.significance - penalty)
        weight.significance = new_sig
        metadata["memory_weight"] = weight.__dict__
        conn.execute(
            "UPDATE rolodex_entries SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), entry_id),
        )
        return True
    return False