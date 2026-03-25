"""
The Librarian — System Change Propagator

Scans a persona's recent Rolodex entries for system-level content
(new CLI commands, pipeline changes, architectural decisions, new modules)
and publishes summaries to the SharedKnowledgeStore so that other personas
gain awareness of infrastructure changes.

Designed to run as a step in the harvest-full pipeline.

Heuristic-only (no LLM calls). Operates in verbatim mode.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path


# ─── Detection Signals ───────────────────────────────────────────────────────
# Each signal is a (keyword_group, weight) pair.
# An entry must accumulate enough weight to cross the threshold.

SYSTEM_SIGNAL_GROUPS = [
    # CLI / command-level changes
    ({"cli", "command", "cmd_", "subcommand", "argparse", "sys.argv"}, 2),
    # Named commands (high signal when combined with action verbs)
    ({"harvest", "harvest-full", "ingest", "recall", "maintain", "boot",
      "integrity", "chain", "build-chains", "batch_ingest", "correct",
      "end", "profile", "shared", "briefs", "timeline", "graph",
      "identity", "onboard"}, 1),
    # Pipeline / architectural terms
    ({"pipeline", "ingestion", "extraction", "enrichment", "scanner",
      "propagation", "schema", "migration", "module", "wired"}, 2),
    # Action verbs indicating change
    ({"implemented", "added", "created", "built", "introduced",
      "replaced", "rewrote", "refactored", "removed", "deprecated",
      "integrated", "completed"}, 1),
    # Code structure signals
    ({"src/core/", "src/storage/", "src/indexing/", "src/retrieval/",
      "librarian_cli.py", "rolodex.py", "schema.py", ".py"}, 1),
    # Roadmap / milestone language
    ({"roadmap", "phase", "milestone", "capability", "gap",
      "item #", "spec", "v0.", "v1.", "v2."}, 1),
    # Database / storage changes
    ({"table", "column", "index", "fts5", "sqlite", "CREATE TABLE",
      "ALTER TABLE", "rolodex_entries", "shared_knowledge"}, 2),
    # New feature signals
    ({"new module", "new command", "new table", "new field",
      "new pipeline", "new step", "new category"}, 3),
]

# Minimum score to classify an entry as system-level
# Calibrated against Chief's Rolodex: genuine system changes score 7.7+,
# conversational entries with incidental keyword overlap score 4-7.
SYSTEM_SIGNAL_THRESHOLD = 7

# How far back to scan (entries created within this window)
DEFAULT_SCAN_WINDOW_HOURS = 48

# Categories that are more likely to contain system info
SYSTEM_RELEVANT_CATEGORIES = {
    "note", "breakthrough", "decision", "correction",
    "user_knowledge", "project_status",
}

# Fingerprint prefix for dedup in shared_knowledge metadata
PROPAGATION_FINGERPRINT_KEY = "system_propagation_fp"


@dataclass
class SystemChangeCandidate:
    """A Rolodex entry identified as describing a system-level change."""
    entry_id: str
    content: str
    score: float
    matched_signals: List[str]
    category: str
    created_at: str
    source_persona: str


def _score_entry(content: str, category: str) -> Tuple[float, List[str]]:
    """Score a Rolodex entry for system-level content.

    Returns (score, list_of_matched_signal_descriptions).
    """
    lower = content.lower()
    total_score = 0.0
    matched = []

    for keywords, weight in SYSTEM_SIGNAL_GROUPS:
        hits = [kw for kw in keywords if kw.lower() in lower]
        if hits:
            # Score: weight * log-ish scaling (diminishing returns for many hits)
            group_score = weight * min(len(hits), 3)
            total_score += group_score
            matched.extend(hits)

    # Category bonus: system info in decision/breakthrough categories is higher signal
    if category in ("decision", "breakthrough"):
        total_score *= 1.3
    elif category in ("correction",):
        total_score *= 1.2

    # Length penalty: very short entries are less likely to be meaningful system docs
    word_count = len(content.split())
    if word_count < 15:
        total_score *= 0.5
    elif word_count > 100:
        total_score *= 1.1  # Longer technical descriptions are higher signal

    return total_score, matched


def _content_fingerprint(content: str) -> str:
    """Create a stable fingerprint for dedup."""
    normalized = content.lower().strip()[:500]
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def scan_for_system_changes(
    rolodex_conn: sqlite3.Connection,
    persona_key: str,
    session_id: Optional[str] = None,
    scan_window_hours: int = DEFAULT_SCAN_WINDOW_HOURS,
) -> List[SystemChangeCandidate]:
    """Scan recent Rolodex entries for system-level content.

    Args:
        rolodex_conn: Connection to the persona's Rolodex DB
        persona_key: Which persona's entries we're scanning
        session_id: If provided, only scan entries from this session
        scan_window_hours: How far back to look

    Returns:
        List of SystemChangeCandidate objects above the threshold
    """
    cutoff = (datetime.utcnow() - timedelta(hours=scan_window_hours)).isoformat()

    if session_id:
        query = """
            SELECT id, content, category, created_at, conversation_id
            FROM rolodex_entries
            WHERE conversation_id = ?
            AND superseded_by IS NULL
            AND content IS NOT NULL
            AND length(content) > 30
            ORDER BY created_at DESC
        """
        params = [session_id]
    else:
        query = """
            SELECT id, content, category, created_at, conversation_id
            FROM rolodex_entries
            WHERE created_at > ?
            AND superseded_by IS NULL
            AND content IS NOT NULL
            AND length(content) > 30
            ORDER BY created_at DESC
            LIMIT 200
        """
        params = [cutoff]

    candidates = []
    for row in rolodex_conn.execute(query, params).fetchall():
        content = row[1] if isinstance(row, tuple) else row["content"]
        category = row[2] if isinstance(row, tuple) else row["category"]
        entry_id = row[0] if isinstance(row, tuple) else row["id"]
        created_at = row[3] if isinstance(row, tuple) else row["created_at"]

        if category not in SYSTEM_RELEVANT_CATEGORIES:
            continue

        score, matched = _score_entry(content, category)

        if score >= SYSTEM_SIGNAL_THRESHOLD:
            candidates.append(SystemChangeCandidate(
                entry_id=entry_id,
                content=content,
                score=score,
                matched_signals=matched,
                category=category,
                created_at=created_at,
                source_persona=persona_key,
            ))

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def propagate_to_shared(
    candidates: List[SystemChangeCandidate],
    shared_store,  # SharedKnowledgeStore instance
    source_persona: str,
    dry_run: bool = False,
    max_publish: int = 10,
) -> Dict[str, Any]:
    """Publish system-level entries to SharedKnowledgeStore.

    Handles dedup: entries already published (by fingerprint) are skipped.

    Args:
        candidates: SystemChangeCandidate objects to consider
        shared_store: SharedKnowledgeStore instance
        source_persona: Persona key doing the publishing
        dry_run: If True, report what would be published without writing
        max_publish: Cap on entries per harvest run (prevent flooding)

    Returns:
        Stats dict with counts and details
    """
    stats = {
        "candidates_scanned": len(candidates),
        "already_published": 0,
        "published": 0,
        "skipped_cap": 0,
        "entries": [],
    }

    # Build set of existing fingerprints from shared knowledge
    existing_fps = set()
    try:
        rows = shared_store.conn.execute(
            "SELECT metadata FROM shared_knowledge WHERE category = 'system_change'"
        ).fetchall()
        for row in rows:
            meta_str = row[0] if isinstance(row, tuple) else row["metadata"]
            if meta_str:
                try:
                    meta = json.loads(meta_str)
                    fp = meta.get(PROPAGATION_FINGERPRINT_KEY)
                    if fp:
                        existing_fps.add(fp)
                except (json.JSONDecodeError, TypeError):
                    pass
    except Exception:
        pass  # Table might not exist yet; will be created on first publish

    published_count = 0
    for candidate in candidates:
        fp = _content_fingerprint(candidate.content)

        if fp in existing_fps:
            stats["already_published"] += 1
            continue

        if published_count >= max_publish:
            stats["skipped_cap"] += 1
            continue

        if not dry_run:
            shared_store.publish(
                content=candidate.content,
                category="system_change",
                source_persona=source_persona,
                tags=["system", "auto-propagated", f"source:{source_persona}"],
                visibility="all",
                metadata={
                    PROPAGATION_FINGERPRINT_KEY: fp,
                    "source_entry_id": candidate.entry_id,
                    "detection_score": round(candidate.score, 1),
                    "matched_signals": candidate.matched_signals[:10],
                },
            )

        existing_fps.add(fp)  # Prevent intra-batch dupes
        published_count += 1
        stats["entries"].append({
            "entry_id": candidate.entry_id,
            "score": round(candidate.score, 1),
            "content_preview": candidate.content[:120],
            "fingerprint": fp,
        })

    stats["published"] = published_count
    return stats


def run_system_propagation(
    rolodex_conn: sqlite3.Connection,
    shared_store,
    persona_key: str,
    session_id: Optional[str] = None,
    scan_window_hours: int = DEFAULT_SCAN_WINDOW_HOURS,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Full propagation pass: scan + publish.

    This is the entry point called from cmd_harvest_full.

    Args:
        rolodex_conn: Persona's Rolodex DB connection
        shared_store: SharedKnowledgeStore instance
        persona_key: Active persona key
        session_id: Optional session to scope the scan
        scan_window_hours: Lookback window
        dry_run: Preview mode

    Returns:
        Combined stats from scan and publish
    """
    candidates = scan_for_system_changes(
        rolodex_conn=rolodex_conn,
        persona_key=persona_key,
        session_id=session_id,
        scan_window_hours=scan_window_hours,
    )

    if not candidates:
        return {
            "candidates_found": 0,
            "published": 0,
            "status": "no_system_changes_detected",
        }

    publish_stats = propagate_to_shared(
        candidates=candidates,
        shared_store=shared_store,
        source_persona=persona_key,
        dry_run=dry_run,
    )

    return {
        "candidates_found": len(candidates),
        **publish_stats,
        "status": "dry_run" if dry_run else "propagated",
    }
