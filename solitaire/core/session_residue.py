"""
Session Residue + Session Tail: Context handoff across session boundaries.

Residue: Compressed encoding of a session's texture, written by the active
Claude instance. Dynamic budget (40-250+ tokens).

Session Tail: Rolling capture of the last N turns of conversation, written
automatically on every ingest-turn. Loaded at next boot as "red-hot" context
so the model knows exactly where the prior session ended. No retrieval needed.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from solitaire.core.types import estimate_tokens

# Max tokens per turn when truncating for the tail file
_TAIL_TURN_MAX_TOKENS = 80
_TAIL_DEFAULT_TURNS = 10
_RESIDUE_HISTORY_MAX = 3


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
        residue_text: The session residue (dynamic length, as needed)
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

    if tokens < 15:
        warnings.append(f"Residue is only {tokens} tokens — may be too thin to prime effectively.")

    # Store in DB (rolodex_entries + rolodex_fts in same transaction)
    now = datetime.now(timezone.utc).isoformat()
    entry_id = uuid.uuid4().hex.upper()
    tags_json = json.dumps(["session_residue", persona_key, now[:10]])
    try:
        conn.execute("""
            INSERT INTO rolodex_entries (
                id, content, content_type, source_type, category,
                conversation_id, created_at, tags
            ) VALUES (
                ?, ?, 'session_residue', 'session_residue',
                'session_residue', ?, ?, ?
            )
        """, (
            entry_id,
            residue_text,
            session_id,
            now,
            tags_json,
        ))
        conn.execute(
            "INSERT INTO rolodex_fts (entry_id, content, tags, category) VALUES (?, ?, ?, ?)",
            (entry_id, residue_text, tags_json, "session_residue")
        )
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
            new_entry = {
                "session_id": session_id,
                "timestamp": now,
                "residue": residue_text,
                "tokens": tokens,
            }
            # Build history from existing file (keep last N prior entries)
            history: List[dict] = []
            try:
                if os.path.exists(residue_file):
                    with open(residue_file) as f:
                        existing = json.load(f)
                    # Support both old flat format and new latest+history format
                    if "latest" in existing:
                        prev = existing["latest"]
                        history = existing.get("history", [])
                    else:
                        prev = existing
                    if prev.get("residue"):
                        history.insert(0, prev)
                    history = history[:_RESIDUE_HISTORY_MAX]
            except Exception:
                pass  # Corrupted file -- start fresh history
            with open(residue_file, "w") as f:
                json.dump({
                    "latest": new_entry,
                    "history": history,
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
                # Support both old flat format and new latest+history format
                entry = data.get("latest", data) if "latest" in data else data
                # Don't load our own session's residue
                if entry.get("session_id") != current_session_id:
                    return {
                        "text": entry.get("residue", ""),
                        "timestamp": entry.get("timestamp"),
                        "session_id": entry.get("session_id"),
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


# ─── Session Tail ──────────────────────────────────────────────────────────

def _truncate_content(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens, preserving sentence boundaries."""
    tokens = estimate_tokens(text)
    if tokens <= max_tokens:
        return text
    # Rough char-per-token ratio, then trim to last sentence boundary
    target_chars = int(max_tokens * 3.5)
    truncated = text[:target_chars]
    # Try to end at a sentence boundary
    for sep in (". ", ".\n", "? ", "! "):
        last = truncated.rfind(sep)
        if last > target_chars * 0.5:
            return truncated[:last + 1] + " [...]"
    return truncated + " [...]"


def write_session_tail(
    conn: sqlite3.Connection,
    session_id: str,
    persona_key: str = "",
    persona_dir: Optional[str] = None,
    max_turns: int = _TAIL_DEFAULT_TURNS,
) -> dict:
    """
    Write a rolling session tail file with the last N turns of conversation.

    Called on every ingest-turn to maintain a rolling window. Survives crashes
    because it's written to disk after each turn pair ingestion.

    Args:
        conn: SQLite connection (must have messages table)
        session_id: Current session ID
        persona_key: Active persona key
        persona_dir: Mounted persona directory (persistent)
        max_turns: Number of message rows to capture (default 10)

    Returns:
        Status dict.
    """
    if not session_id:
        return {"status": "error", "detail": "No session ID"}

    # Query the messages table for the last N rows of this session
    try:
        rows = conn.execute("""
            SELECT role, content, turn_number, token_count, timestamp
            FROM messages
            WHERE conversation_id = ?
            ORDER BY turn_number DESC, timestamp DESC
            LIMIT ?
        """, (session_id, max_turns)).fetchall()
    except Exception as e:
        return {"status": "error", "detail": f"Messages query failed: {e}"}

    if not rows:
        return {"status": "ok", "detail": "No messages to capture"}

    # Reverse to chronological order and truncate each turn
    rows.reverse()
    turns = []
    for role, content, turn_number, token_count, timestamp in rows:
        truncated = _truncate_content(content, _TAIL_TURN_MAX_TOKENS)
        turns.append({
            "role": role,
            "content": truncated,
            "turn_number": turn_number,
            "timestamp": timestamp,
        })

    now = datetime.now(timezone.utc).isoformat()
    tail_data = {
        "session_id": session_id,
        "timestamp": now,
        "turn_count": len(turns),
        "turns": turns,
    }

    # Write to file
    tail_dir = None
    if persona_dir:
        tail_dir = os.path.join(persona_dir, "residue")
    else:
        tail_dir = _residue_dir(conn, persona_key)

    if tail_dir:
        try:
            os.makedirs(tail_dir, exist_ok=True)
            tail_file = os.path.join(tail_dir, "latest_tail.json")
            with open(tail_file, "w") as f:
                json.dump(tail_data, f, indent=2)
        except Exception as e:
            return {"status": "error", "detail": f"Tail file write failed: {e}"}

    return {"status": "ok", "turns_captured": len(turns), "session_id": session_id}


def load_session_tail(
    conn: sqlite3.Connection,
    current_session_id: str,
    persona_key: str = "",
    persona_dir: Optional[str] = None,
) -> dict:
    """
    Load the session tail from the prior session.

    Returns:
        Dict with keys: turns (list), timestamp (str|None), session_id (str|None).
        Empty turns list if none found.
    """
    empty = {"turns": [], "timestamp": None, "session_id": None}

    tail_dir = None
    if persona_dir:
        tail_dir = os.path.join(persona_dir, "residue")
    else:
        tail_dir = _residue_dir(conn, persona_key)

    if not tail_dir:
        return empty

    tail_file = os.path.join(tail_dir, "latest_tail.json")
    try:
        if not os.path.exists(tail_file):
            return empty
        with open(tail_file) as f:
            data = json.load(f)
        # Don't load our own session's tail
        if data.get("session_id") == current_session_id:
            return empty
        return {
            "turns": data.get("turns", []),
            "timestamp": data.get("timestamp"),
            "session_id": data.get("session_id"),
        }
    except Exception:
        return empty


def build_tail_block(tail_data: dict) -> str:
    """
    Format a session tail for injection into the context block.

    Args:
        tail_data: Dict from load_session_tail with turns, timestamp, session_id.

    Returns:
        Formatted block string, or empty string if no turns.
    """
    turns = tail_data.get("turns", [])
    if not turns:
        return ""

    meta_parts = []
    timestamp = tail_data.get("timestamp")
    session_id = tail_data.get("session_id")
    if timestamp:
        ts_display = timestamp[:16].replace("T", " ")
        meta_parts.append(f"written: {ts_display}")
    if session_id:
        meta_parts.append(f"session: {session_id[:8]}")

    header = "═══ SESSION TAIL (prior session)"
    if meta_parts:
        header += f" [{', '.join(meta_parts)}]"
    header += " ═══"

    lines = [header, ""]
    for turn in turns:
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "").strip()
        if content:
            lines.append(f"[{role}] {content}")
            lines.append("")

    lines.append("═══ END TAIL ═══")
    return "\n".join(lines)


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
