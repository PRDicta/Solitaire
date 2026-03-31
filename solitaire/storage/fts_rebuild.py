"""
Solitaire -- FTS Index Rebuild

Rebuilds FTS5 virtual tables from their source tables.
Used after restore operations or when FTS indexes become
corrupted or out of sync.

Integrity chain:
    1. SQLite backup (last known good state)
    2. JSONL audit trail (forensic rebuild if SQLite is damaged)
    3. FTS rebuilt from whichever source is intact

This module handles step 3: rebuilding FTS from SQLite rows.
For a full rebuild from JSONL, use the JSONL store's replay
mechanism to repopulate SQLite first, then call rebuild_all_fts.
"""

import json
import logging
import sqlite3
from typing import Any, Dict

logger = logging.getLogger(__name__)


def rebuild_rolodex_fts(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Rebuild the rolodex_fts index from rolodex_entries.

    Drops and recreates the FTS5 virtual table, then repopulates
    from all active (non-superseded, non-archived) entries.

    Returns:
        Dict with status, entry counts, and any mismatches.
    """
    try:
        # Count active entries before rebuild
        active_count = conn.execute(
            "SELECT COUNT(*) FROM rolodex_entries "
            "WHERE superseded_by IS NULL AND archived_at IS NULL"
        ).fetchone()[0]

        # Drop and recreate FTS table
        conn.execute("DROP TABLE IF EXISTS rolodex_fts")
        conn.execute("""
            CREATE VIRTUAL TABLE rolodex_fts USING fts5(
                entry_id,
                content,
                tags,
                category,
                tokenize='porter unicode61'
            )
        """)

        # Repopulate from active entries
        rows = conn.execute(
            "SELECT id, content, tags, category FROM rolodex_entries "
            "WHERE superseded_by IS NULL AND archived_at IS NULL"
        ).fetchall()

        for row in rows:
            conn.execute(
                "INSERT INTO rolodex_fts (entry_id, content, tags, category) "
                "VALUES (?, ?, ?, ?)",
                (row["id"], row["content"], row["tags"], row["category"]),
            )

        conn.commit()

        # Optimize the FTS index
        conn.execute(
            "INSERT INTO rolodex_fts(rolodex_fts) VALUES ('optimize')"
        )
        conn.commit()

        # Verify
        fts_count = conn.execute(
            "SELECT COUNT(*) FROM rolodex_fts"
        ).fetchone()[0]

        status = "ok" if fts_count == active_count else "mismatch"
        if status == "mismatch":
            logger.warning(
                "FTS count mismatch: %d FTS rows vs %d active entries",
                fts_count, active_count,
            )

        return {
            "table": "rolodex_fts",
            "status": status,
            "active_entries": active_count,
            "fts_entries": fts_count,
        }
    except Exception as e:
        logger.error("rolodex_fts rebuild failed: %s", e)
        return {"table": "rolodex_fts", "status": "error", "reason": str(e)}


def rebuild_chains_fts(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Rebuild the chains_fts index from chains table."""
    try:
        chain_count = conn.execute("SELECT COUNT(*) FROM chains").fetchone()[0]

        conn.execute("DROP TABLE IF EXISTS chains_fts")
        conn.execute("""
            CREATE VIRTUAL TABLE chains_fts USING fts5(
                chain_id,
                summary,
                topics,
                tokenize='porter unicode61'
            )
        """)

        rows = conn.execute("SELECT id, summary, topics FROM chains").fetchall()
        for row in rows:
            conn.execute(
                "INSERT INTO chains_fts (chain_id, summary, topics) "
                "VALUES (?, ?, ?)",
                (row["id"], row["summary"], row["topics"]),
            )

        conn.commit()
        conn.execute(
            "INSERT INTO chains_fts(chains_fts) VALUES ('optimize')"
        )
        conn.commit()

        fts_count = conn.execute(
            "SELECT COUNT(*) FROM chains_fts"
        ).fetchone()[0]

        status = "ok" if fts_count == chain_count else "mismatch"
        return {
            "table": "chains_fts",
            "status": status,
            "source_rows": chain_count,
            "fts_entries": fts_count,
        }
    except Exception as e:
        logger.error("chains_fts rebuild failed: %s", e)
        return {"table": "chains_fts", "status": "error", "reason": str(e)}


def rebuild_topics_fts(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Rebuild the topics_fts index from topics table."""
    try:
        topic_count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]

        conn.execute("DROP TABLE IF EXISTS topics_fts")
        conn.execute("""
            CREATE VIRTUAL TABLE topics_fts USING fts5(
                topic_id,
                label,
                description,
                tokenize='porter unicode61'
            )
        """)

        rows = conn.execute(
            "SELECT id, label, description FROM topics"
        ).fetchall()
        for row in rows:
            conn.execute(
                "INSERT INTO topics_fts (topic_id, label, description) "
                "VALUES (?, ?, ?)",
                (row["id"], row["label"], row["description"]),
            )

        conn.commit()
        conn.execute(
            "INSERT INTO topics_fts(topics_fts) VALUES ('optimize')"
        )
        conn.commit()

        fts_count = conn.execute(
            "SELECT COUNT(*) FROM topics_fts"
        ).fetchone()[0]

        status = "ok" if fts_count == topic_count else "mismatch"
        return {
            "table": "topics_fts",
            "status": status,
            "source_rows": topic_count,
            "fts_entries": fts_count,
        }
    except Exception as e:
        logger.error("topics_fts rebuild failed: %s", e)
        return {"table": "topics_fts", "status": "error", "reason": str(e)}


def rebuild_all_fts(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Rebuild all FTS indexes from their source tables.

    Returns:
        Dict with overall status and per-table results.
    """
    results = {
        "rolodex": rebuild_rolodex_fts(conn),
        "chains": rebuild_chains_fts(conn),
        "topics": rebuild_topics_fts(conn),
    }

    all_ok = all(r["status"] == "ok" for r in results.values())
    any_error = any(r["status"] == "error" for r in results.values())

    if any_error:
        overall = "error"
    elif all_ok:
        overall = "ok"
    else:
        overall = "mismatch"

    return {"status": overall, "tables": results}
