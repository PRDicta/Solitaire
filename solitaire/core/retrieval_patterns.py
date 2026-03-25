"""
Solitaire — Retrieval Pattern Detection (Phase 19: Self-Learning Layer 2)

Analyzes retrieval_outcomes over a rolling window to identify:
    1. Hot topics: Subjects recalled in 3+ of last 5 sessions (active work streams)
    2. Dead zones: Topics with entries not recalled in 30+ days (archival candidates)
    3. Gap signals: Queries that return 0 or low-relevance results repeatedly
       (the system is being asked about something it doesn't know)

These patterns feed into:
    - Boot context (hot topics get priority in manifest)
    - Auto-recall preflight (gap signals surface as notes)
    - Maintenance reports (dead zones flagged for review)
    - Proactive tool finding (2c, future) via gap-to-search pipeline
"""

import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple


# ─── Constants ──────────────────────────────────────────────────────────────

HOT_TOPIC_THRESHOLD = 3       # Recalled in N+ of last window_sessions
HOT_TOPIC_WINDOW = 5          # Rolling window in sessions
DEAD_ZONE_DAYS = 30           # No recall in N days = dead zone
GAP_SIGNAL_THRESHOLD = 2      # Same topic queried N+ times with no/low results
GAP_SIGNAL_WINDOW_DAYS = 14   # Look for gap signals within this window


# ─── Hot Topics ─────────────────────────────────────────────────────────────

def detect_hot_topics(
    conn: sqlite3.Connection,
    window_sessions: int = HOT_TOPIC_WINDOW,
    threshold: int = HOT_TOPIC_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Identify topics that are actively being recalled across recent sessions.

    Strategy: join retrieval_outcomes with topic assignments to find which
    topics appear in recall results across multiple sessions. Topics recalled
    in threshold+ of the last window_sessions are "hot".

    Returns:
        List of dicts: [{"topic": label, "topic_id": id, "recall_count": N,
                         "sessions": M, "last_recalled": timestamp}]
    """
    # Get the last N distinct session IDs from retrieval_outcomes
    session_rows = conn.execute(
        """SELECT DISTINCT session_id FROM retrieval_outcomes
           ORDER BY recalled_at DESC"""
    ).fetchall()

    if not session_rows:
        return []

    recent_sessions = [r["session_id"] for r in session_rows[:window_sessions]]
    placeholders = ",".join("?" for _ in recent_sessions)

    # Join outcomes with entries to get topic assignments
    rows = conn.execute(
        f"""SELECT ro.entry_id, ro.session_id, ro.recalled_at,
                   re.topic_id, t.label AS topic_label
            FROM retrieval_outcomes ro
            JOIN rolodex_entries re ON ro.entry_id = re.id
            LEFT JOIN topics t ON re.topic_id = t.id
            WHERE ro.session_id IN ({placeholders})
            AND re.topic_id IS NOT NULL""",
        recent_sessions,
    ).fetchall()

    if not rows:
        return []

    # Aggregate: per topic, count distinct sessions and total recalls
    topic_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"topic_id": None, "sessions": set(), "recall_count": 0, "last_recalled": None}
    )

    for r in rows:
        label = r["topic_label"] or "unlabeled"
        td = topic_data[label]
        td["topic_id"] = r["topic_id"]
        td["sessions"].add(r["session_id"])
        td["recall_count"] += 1
        recalled_at = r["recalled_at"]
        if td["last_recalled"] is None or recalled_at > td["last_recalled"]:
            td["last_recalled"] = recalled_at

    # Filter by threshold
    hot = []
    for label, data in topic_data.items():
        session_count = len(data["sessions"])
        if session_count >= threshold:
            hot.append({
                "topic": label,
                "topic_id": data["topic_id"],
                "recall_count": data["recall_count"],
                "sessions": session_count,
                "last_recalled": data["last_recalled"],
            })

    # Sort by session count desc, then recall count desc
    hot.sort(key=lambda x: (x["sessions"], x["recall_count"]), reverse=True)
    return hot


# ─── Dead Zones ─────────────────────────────────────────────────────────────

def detect_dead_zones(
    conn: sqlite3.Connection,
    stale_days: int = DEAD_ZONE_DAYS,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Identify topics with entries that haven't been recalled in stale_days.

    Strategy: find topics where NO entries have appeared in retrieval_outcomes
    within the stale window. These are knowledge areas that may be obsolete
    or poorly indexed.

    Returns:
        List of dicts: [{"topic": label, "topic_id": id, "entry_count": N,
                         "last_recalled": timestamp or None,
                         "days_stale": float}]
    """
    if now is None:
        now = datetime.utcnow()

    cutoff = (now - timedelta(days=stale_days)).isoformat()

    # Get all topics with their entry counts
    topic_rows = conn.execute(
        """SELECT t.id, t.label, COUNT(re.id) AS entry_count
           FROM topics t
           JOIN rolodex_entries re ON re.topic_id = t.id
           WHERE (re.superseded_by IS NULL OR re.superseded_by = '')
           GROUP BY t.id
           HAVING COUNT(re.id) > 0
           ORDER BY entry_count DESC"""
    ).fetchall()

    if not topic_rows:
        return []

    dead_zones = []
    for tr in topic_rows:
        topic_id = tr["id"]

        # Check if any entries from this topic were recalled recently
        recent_recall = conn.execute(
            """SELECT MAX(ro.recalled_at) AS last_recall
               FROM retrieval_outcomes ro
               JOIN rolodex_entries re ON ro.entry_id = re.id
               WHERE re.topic_id = ?
               AND ro.recalled_at > ?""",
            (topic_id, cutoff),
        ).fetchone()

        if recent_recall and recent_recall["last_recall"]:
            continue  # This topic has recent recall activity, not dead

        # Find when this topic was last recalled (ever)
        last_ever = conn.execute(
            """SELECT MAX(ro.recalled_at) AS last_recall
               FROM retrieval_outcomes ro
               JOIN rolodex_entries re ON ro.entry_id = re.id
               WHERE re.topic_id = ?""",
            (topic_id,),
        ).fetchone()

        last_recalled = last_ever["last_recall"] if last_ever else None

        # Also check last_accessed on entries as a fallback for pre-feedback data
        if not last_recalled:
            fallback = conn.execute(
                """SELECT MAX(last_accessed) AS la
                   FROM rolodex_entries WHERE topic_id = ?""",
                (topic_id,),
            ).fetchone()
            last_recalled = fallback["la"] if fallback else None

        # Calculate staleness
        if last_recalled:
            try:
                lr_dt = datetime.fromisoformat(
                    last_recalled.replace('Z', '+00:00')
                ).replace(tzinfo=None)
                days_stale = (now - lr_dt).total_seconds() / 86400.0
            except (ValueError, AttributeError):
                days_stale = stale_days + 1  # Assume stale if parsing fails
        else:
            days_stale = stale_days + 1  # Never recalled = definitely stale

        if days_stale >= stale_days:
            dead_zones.append({
                "topic": tr["label"],
                "topic_id": topic_id,
                "entry_count": tr["entry_count"],
                "last_recalled": last_recalled,
                "days_stale": round(days_stale, 1),
            })

    # Sort by entry count desc (biggest dead zones first)
    dead_zones.sort(key=lambda x: x["entry_count"], reverse=True)
    return dead_zones


# ─── Gap Signals ────────────────────────────────────────────────────────────

def detect_gap_signals(
    conn: sqlite3.Connection,
    threshold: int = GAP_SIGNAL_THRESHOLD,
    window_days: int = GAP_SIGNAL_WINDOW_DAYS,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Identify recurring queries that consistently return no or low results.

    Strategy: examine query_log for queries where found=0 or the returned
    entries were not used (from retrieval_outcomes). Cluster similar queries
    by keyword overlap to detect recurring gaps.

    Returns:
        List of dicts: [{"query_pattern": str, "occurrences": int,
                         "last_asked": timestamp, "sample_queries": [str]}]
    """
    if now is None:
        now = datetime.utcnow()

    cutoff = (now - timedelta(days=window_days)).isoformat()

    # Get failed queries (found=0)
    miss_rows = conn.execute(
        """SELECT query_text, timestamp FROM query_log
           WHERE found = 0 AND timestamp > ?
           ORDER BY timestamp DESC""",
        (cutoff,),
    ).fetchall()

    # Also get queries where entries were returned but none were used
    low_quality_rows = conn.execute(
        """SELECT ql.query_text, ql.timestamp
           FROM query_log ql
           WHERE ql.found = 1 AND ql.timestamp > ?
           AND NOT EXISTS (
               SELECT 1 FROM retrieval_outcomes ro
               WHERE ro.query_text = ql.query_text
               AND ro.used = 1
               AND ro.recalled_at > ?
           )""",
        (cutoff, cutoff),
    ).fetchall()

    # Combine all gap queries
    all_gaps = []
    for r in miss_rows:
        all_gaps.append({"query": r["query_text"], "timestamp": r["timestamp"], "type": "miss"})
    for r in low_quality_rows:
        all_gaps.append({"query": r["query_text"], "timestamp": r["timestamp"], "type": "low_quality"})

    if not all_gaps:
        return []

    # Cluster similar queries by keyword overlap
    clusters = _cluster_queries(all_gaps)

    # Filter by threshold
    signals = []
    for pattern, cluster in clusters.items():
        if len(cluster["queries"]) >= threshold:
            signals.append({
                "query_pattern": pattern,
                "occurrences": len(cluster["queries"]),
                "last_asked": cluster["last_asked"],
                "sample_queries": cluster["queries"][:5],
                "gap_types": list(cluster["types"]),
            })

    # Sort by occurrences desc
    signals.sort(key=lambda x: x["occurrences"], reverse=True)
    return signals


def check_gap_for_query(
    conn: sqlite3.Connection,
    query: str,
    window_days: int = GAP_SIGNAL_WINDOW_DAYS,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """
    Check if a specific query matches a known gap signal.

    This is the function called during auto-recall preflight. If the user's
    message matches a recurring gap, we surface a note: "This topic has been
    asked about N times with no good results. Consider ingesting reference material."

    Returns:
        Dict with gap info if a match is found, None otherwise.
    """
    if now is None:
        now = datetime.utcnow()

    # Extract key terms from the query
    query_terms = _extract_key_terms(query)
    if len(query_terms) < 2:
        return None  # Too few terms to match meaningfully

    cutoff = (now - timedelta(days=window_days)).isoformat()

    # Check query_log for similar failed queries
    miss_count = 0
    low_quality_count = 0

    # Misses: found=0
    miss_rows = conn.execute(
        """SELECT query_text FROM query_log
           WHERE found = 0 AND timestamp > ?""",
        (cutoff,),
    ).fetchall()

    for r in miss_rows:
        other_terms = _extract_key_terms(r["query_text"])
        overlap = query_terms & other_terms
        if len(overlap) >= 2 or (len(overlap) >= 1 and len(query_terms) <= 3):
            miss_count += 1

    # Low quality: found=1 but nothing used
    low_rows = conn.execute(
        """SELECT ql.query_text FROM query_log ql
           WHERE ql.found = 1 AND ql.timestamp > ?
           AND NOT EXISTS (
               SELECT 1 FROM retrieval_outcomes ro
               WHERE ro.query_text = ql.query_text
               AND ro.used = 1
               AND ro.recalled_at > ?
           )""",
        (cutoff, cutoff),
    ).fetchall()

    for r in low_rows:
        other_terms = _extract_key_terms(r["query_text"])
        overlap = query_terms & other_terms
        if len(overlap) >= 2 or (len(overlap) >= 1 and len(query_terms) <= 3):
            low_quality_count += 1

    total = miss_count + low_quality_count
    if total >= GAP_SIGNAL_THRESHOLD:
        return {
            "query_pattern": " ".join(sorted(query_terms)),
            "occurrences": total,
            "miss_count": miss_count,
            "low_quality_count": low_quality_count,
            "note": (
                f"This topic has been queried {total} times recently "
                f"with no good recall results. Consider ingesting reference "
                f"material or creating entries about this subject."
            ),
        }

    return None


# ─── Full Pattern Report ────────────────────────────────────────────────────

def get_pattern_report(
    conn: sqlite3.Connection,
    window_sessions: int = HOT_TOPIC_WINDOW,
    stale_days: int = DEAD_ZONE_DAYS,
    gap_window_days: int = GAP_SIGNAL_WINDOW_DAYS,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Generate the full pattern detection report.

    Returns:
        Dict with: hot_topics, dead_zones, gaps, generated_at timestamp.
    """
    if now is None:
        now = datetime.utcnow()

    return {
        "hot_topics": detect_hot_topics(conn, window_sessions=window_sessions),
        "dead_zones": detect_dead_zones(conn, stale_days=stale_days, now=now),
        "gaps": detect_gap_signals(conn, window_days=gap_window_days, now=now),
        "generated_at": now.isoformat(),
        "config": {
            "window_sessions": window_sessions,
            "stale_days": stale_days,
            "gap_window_days": gap_window_days,
        },
    }


# ─── Internal Helpers ───────────────────────────────────────────────────────

def _extract_key_terms(text: str) -> Set[str]:
    """Extract significant terms (4+ chars) from a query string."""
    stop_words = {
        'what', 'when', 'where', 'which', 'about', 'their', 'there',
        'this', 'that', 'with', 'from', 'have', 'been', 'will', 'would',
        'could', 'should', 'more', 'some', 'your', 'also', 'very',
        'just', 'like', 'does', 'tell', 'show', 'find', 'give',
        'much', 'many', 'such', 'than', 'then', 'them', 'they',
        'these', 'those', 'here', 'only', 'most', 'other', 'into',
        'over', 'same', 'back', 'each', 'even', 'still', 'after',
    }
    words = set(re.findall(r'\b\w{4,}\b', text.lower()))
    return words - stop_words


def _cluster_queries(
    gap_queries: List[Dict[str, Any]],
    min_overlap: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """
    Cluster similar queries by keyword overlap.

    Uses a greedy approach: for each query, check if it overlaps with
    any existing cluster. If so, merge. If not, start a new cluster.

    Returns:
        Dict mapping cluster pattern (sorted key terms) to cluster data.
    """
    clusters: Dict[str, Dict[str, Any]] = {}

    for item in gap_queries:
        terms = _extract_key_terms(item["query"])
        if len(terms) < 1:
            continue

        # Find best matching cluster
        best_cluster = None
        best_overlap = 0

        for pattern, cluster in clusters.items():
            cluster_terms = set(pattern.split())
            overlap = len(terms & cluster_terms)
            if overlap >= min_overlap or (overlap >= 1 and len(terms) <= 2):
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cluster = pattern

        if best_cluster:
            # Merge into existing cluster
            clusters[best_cluster]["queries"].append(item["query"])
            clusters[best_cluster]["types"].add(item["type"])
            if item["timestamp"] > clusters[best_cluster]["last_asked"]:
                clusters[best_cluster]["last_asked"] = item["timestamp"]
        else:
            # New cluster
            pattern = " ".join(sorted(terms))
            if pattern in clusters:
                clusters[pattern]["queries"].append(item["query"])
                clusters[pattern]["types"].add(item["type"])
                if item["timestamp"] > clusters[pattern]["last_asked"]:
                    clusters[pattern]["last_asked"] = item["timestamp"]
            else:
                clusters[pattern] = {
                    "queries": [item["query"]],
                    "types": {item["type"]},
                    "last_asked": item["timestamp"],
                }

    return clusters
