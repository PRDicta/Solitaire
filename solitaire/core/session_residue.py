"""
Session Residue: Compressed poetic encoding of a session's texture.

Written by the active Claude instance at session end (the only moment
with full context), stored per-persona, loaded at next boot as priming.

One paragraph, 40-80 tokens. Never longer. Not information — orientation.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from solitaire.core.types import estimate_tokens


# ─── Storage ────────────────────────────────────────────────────────────────

def write_residue(
    conn: sqlite3.Connection,
    session_id: str,
    residue_text: str,
    persona_key: str = "",
    persona_dir: Optional[str] = None,
    jsonl_store=None,
) -> dict:
    """
    Store a session residue.

    Args:
        conn: SQLite connection to the rolodex database
        session_id: Current session ID
        residue_text: The poetic residue paragraph (40-80 tokens ideal)
        persona_key: Active persona key (for file-based fallback)
        persona_dir: Mounted persona directory (persistent). If provided,
                     latest_residue.json is written here instead of the
                     ephemeral session-local path.
        jsonl_store: Optional PersonaJsonlStore for JSONL append.

    Returns:
        Status dict with token count and any warnings.
    """
    residue_text = residue_text.strip()
    if not residue_text:
        return {"status": "error", "detail": "Empty residue text"}

    tokens = estimate_tokens(residue_text)
    warnings = []

    if tokens > 120:
        warnings.append(f"Residue is {tokens} tokens — target is 40-80. Consider trimming.")
    if tokens < 15:
        warnings.append(f"Residue is only {tokens} tokens — may be too thin to prime effectively.")

    # Store in DB (rolodex_entries with a special source_type)
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute("""
            INSERT INTO rolodex_entries (
                id, content, content_type, source_type, category,
                conversation_id, created_at, tags
            ) VALUES (
                hex(randomblob(16)), ?, 'session_residue', 'session_residue',
                'session_residue', ?, ?, ?
            )
        """, (
            residue_text,
            session_id,
            now,
            json.dumps(["session_residue", persona_key, now[:10]]),
        ))
        conn.commit()
    except Exception as e:
        return {"status": "error", "detail": f"DB write failed: {e}"}

    # JSONL append (canonical persistence)
    if jsonl_store:
        try:
            jsonl_store.metadata.append(
                record_type="session_residue",
                op="create",
                data={
                    "content": residue_text,
                    "session_id": session_id,
                    "persona_key": persona_key,
                    "timestamp": now,
                    "tokens": tokens,
                },
                session_id=session_id,
                record_id=f"residue_{session_id[:8]}",
            )
        except Exception:
            warnings.append("JSONL append failed; DB entry saved.")

    # Write file to persistent mounted dir (if available), else ephemeral fallback
    residue_dir = None
    if persona_dir:
        residue_dir = os.path.join(persona_dir, "residue")
    else:
        residue_dir = _residue_dir(conn, persona_key)

    if residue_dir:
        try:
            os.makedirs(residue_dir, exist_ok=True)
            residue_file = os.path.join(residue_dir, "latest_residue.json")
            with open(residue_file, "w") as f:
                json.dump({
                    "session_id": session_id,
                    "timestamp": now,
                    "residue": residue_text,
                    "tokens": tokens,
                }, f, indent=2)
        except Exception:
            warnings.append("File-based fallback write failed; DB entry saved.")

    result = {
        "status": "ok",
        "tokens": tokens,
        "session_id": session_id,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def load_latest_residue(
    conn: sqlite3.Connection,
    current_session_id: str,
    persona_key: str = "",
    persona_dir: Optional[str] = None,
    jsonl_store=None,
) -> dict:
    """
    Load the most recent session residue (from a prior session).

    Resolution order:
    1. File-based lookup (latest_residue.json in persona dir, fast)
    2. DB query (most recent session_residue row)
    3. JSONL scan (most recent session_residue record in metadata.jsonl)

    Excludes residues from the current session.

    Returns:
        Dict with keys: text (str), timestamp (str|None), session_id (str|None).
        Empty text if none found. Timestamp is ISO format when available.
    """
    empty = {"text": "", "timestamp": None, "session_id": None}

    # Try file first (fast path), preferring mounted dir
    residue_dir = None
    if persona_dir:
        residue_dir = os.path.join(persona_dir, "residue")
    else:
        residue_dir = _residue_dir(conn, persona_key)

    if residue_dir:
        residue_file = os.path.join(residue_dir, "latest_residue.json")
        try:
            if os.path.exists(residue_file):
                with open(residue_file) as f:
                    data = json.load(f)
                # Don't load our own session's residue
                if data.get("session_id") != current_session_id:
                    return {
                        "text": data.get("residue", ""),
                        "timestamp": data.get("timestamp"),
                        "session_id": data.get("session_id"),
                    }
        except Exception:
            pass

    # DB fallback: find most recent residue not from current session
    try:
        row = conn.execute("""
            SELECT content, created_at, conversation_id FROM rolodex_entries
            WHERE source_type = 'session_residue'
              AND conversation_id != ?
              AND superseded_by IS NULL
            ORDER BY created_at DESC
            LIMIT 1
        """, (current_session_id,)).fetchone()
        if row:
            return {
                "text": row[0],
                "timestamp": row[1] if len(row) > 1 else None,
                "session_id": row[2] if len(row) > 2 else None,
            }
    except Exception:
        pass

    # JSONL fallback: scan metadata store for session_residue records
    if jsonl_store:
        try:
            result = _load_residue_from_jsonl(jsonl_store, current_session_id)
            if result and result.get("text"):
                return result
        except Exception:
            pass

    return empty


def _load_residue_from_jsonl(jsonl_store, current_session_id: str) -> dict:
    """Scan JSONL metadata store for the most recent session_residue record."""
    empty = {"text": "", "timestamp": None, "session_id": None}
    try:
        # Scan metadata store for session_residue records
        results = list(jsonl_store.metadata.scan(
            record_type="session_residue",
        ))
        if not results:
            return empty

        # Sort by timestamp (most recent first), exclude current session
        candidates = []
        for rec in results:
            rec_session = rec.get("session_id", "")
            if rec_session != current_session_id and rec.get("content"):
                candidates.append(rec)

        if not candidates:
            return empty

        # Sort by timestamp descending
        candidates.sort(
            key=lambda r: r.get("timestamp", ""),
            reverse=True,
        )
        best = candidates[0]
        return {
            "text": best.get("content", ""),
            "timestamp": best.get("timestamp"),
            "session_id": best.get("session_id"),
        }
    except Exception:
        return empty


def build_residue_block(residue_text: str, timestamp: Optional[str] = None, session_id: Optional[str] = None) -> str:
    """
    Format a residue for injection into the context block.

    Args:
        residue_text: The residue paragraph.
        timestamp: ISO timestamp of when the residue was written (displayed as date hint).
        session_id: Session ID the residue came from (short prefix shown for traceability).

    Returns empty string if no residue available.
    """
    if not residue_text or not residue_text.strip():
        return ""

    # Build a concise metadata line so the model can tell how fresh this is
    meta_parts = []
    if timestamp:
        # Show date + time (trim to minute precision)
        ts_display = timestamp[:16].replace("T", " ")
        meta_parts.append(f"written: {ts_display}")
    if session_id:
        meta_parts.append(f"session: {session_id[:8]}")

    header = "═══ SESSION RESIDUE (prior session)"
    if meta_parts:
        header += f" [{', '.join(meta_parts)}]"
    header += " ═══"

    return f"{header}\n\n{residue_text.strip()}\n\n═══ END RESIDUE ═══"


# ─── Internal ───────────────────────────────────────────────────────────────

def _residue_dir(conn: sqlite3.Connection, persona_key: str) -> Optional[str]:
    """Derive the residue storage directory from the DB path and persona key."""
    if not persona_key:
        return None
    try:
        # The DB is typically at personas/{key}/rolodex_{key}.db or similar
        # We want personas/{key}/residue/
        db_path = conn.execute("PRAGMA database_list").fetchone()
        if db_path and len(db_path) > 2:
            db_file = db_path[2]
            if db_file and os.path.exists(db_file):
                db_dir = os.path.dirname(db_file)
                # If we're in the session-local copy, use the persona dir from the mount
                # Look for personas/{key}/ in the path
                if f"personas/{persona_key}" in db_dir:
                    return os.path.join(db_dir, "residue")
                # Otherwise construct from the DB directory
                return os.path.join(db_dir, "residue")
    except Exception:
        pass
    return None
