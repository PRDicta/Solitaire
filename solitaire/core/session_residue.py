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
    status: str = "final",
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
        status: Residue status: "partial", "final", or "auto-closed".

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
                    "status": status,
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
                "status": status,
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
                    # If prev is from the SAME session and was partial, overwrite
                    # (don't stack partials in history for the same session)
                    if prev.get("session_id") == session_id and prev.get("status") == "partial":
                        pass  # Drop the old partial; new_entry replaces it
                    elif prev.get("residue"):
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
        Dict with keys: text (str), timestamp (str|None), session_id (str|None),
        status (str).
        Empty text if none found. Timestamp is ISO format when available.
    """
    empty = {"text": "", "timestamp": None, "session_id": None, "status": "final"}

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
                        "status": entry.get("status", "final"),
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
                "status": "final",
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
    empty = {"text": "", "timestamp": None, "session_id": None, "status": "final"}
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
            "status": best.get("status", "final"),
        }
    except Exception:
        return empty


def load_recent_residues(
    conn: sqlite3.Connection,
    current_session_id: str,
    persona_key: str = "",
    persona_dir: Optional[str] = None,
    jsonl_store=None,
    n: int = 3,
) -> List[dict]:
    """
    Load the N most recent session residues from prior sessions.

    Uses the same resolution order as load_latest_residue but returns
    multiple entries. File-based path (latest_residue.json) stores
    latest + history, so this can return up to n entries from file alone.
    Falls back to DB for additional entries if file doesn't have enough.

    Returns:
        List of dicts with keys: text, timestamp, session_id, status.
        Most recent first. Empty list if none found.
    """
    results: List[dict] = []
    seen_sessions: set = set()

    # ── File path (fast, has latest + history) ──────────────────────
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

                # Collect latest + history entries
                candidates = []
                entry = data.get("latest", data) if "latest" in data else data
                if entry.get("residue") and entry.get("session_id") != current_session_id:
                    candidates.append(entry)
                for hist in data.get("history", []):
                    if hist.get("residue") and hist.get("session_id") != current_session_id:
                        candidates.append(hist)

                for c in candidates[:n]:
                    sid = c.get("session_id", "")
                    if sid not in seen_sessions:
                        results.append({
                            "text": c.get("residue", ""),
                            "timestamp": c.get("timestamp"),
                            "session_id": sid,
                            "status": c.get("status", "final"),
                        })
                        seen_sessions.add(sid)
        except Exception:
            pass

    # ── DB fallback for remaining slots ─────────────────────────────
    if len(results) < n:
        try:
            exclude_ids = [current_session_id] + list(seen_sessions)
            placeholders = ",".join("?" for _ in exclude_ids)
            rows = conn.execute(f"""
                SELECT content, created_at, conversation_id FROM rolodex_entries
                WHERE source_type = 'session_residue'
                  AND conversation_id NOT IN ({placeholders})
                  AND superseded_by IS NULL
                ORDER BY created_at DESC
                LIMIT ?
            """, (*exclude_ids, n - len(results))).fetchall()
            for row in rows:
                sid = row[2] if len(row) > 2 else None
                if sid and sid not in seen_sessions:
                    results.append({
                        "text": row[0],
                        "timestamp": row[1] if len(row) > 1 else None,
                        "session_id": sid,
                        "status": "final",
                    })
                    seen_sessions.add(sid)
        except Exception:
            pass

    return results[:n]


def build_multi_residue_block(residues: List[dict]) -> str:
    """
    Format multiple residues into a single boot context block.

    Args:
        residues: List of dicts from load_recent_residues, each with
                  text, timestamp, session_id, status. Most recent first.

    Returns:
        Formatted block string, or empty string if no residues.
    """
    if not residues:
        return ""

    parts = []
    for i, r in enumerate(residues):
        text = r.get("text", "").strip()
        if not text:
            continue

        meta_parts = []
        if r.get("timestamp"):
            ts_display = r["timestamp"][:16].replace("T", " ")
            meta_parts.append(f"written: {ts_display}")
        if r.get("session_id"):
            meta_parts.append(f"session: {r['session_id'][:8]}")

        residue_status = r.get("status", "final")
        if i == 0:
            label = "prior session" if residue_status == "final" else f"prior session, {residue_status}"
        else:
            base = f"session {i + 1} ago"
            label = base if residue_status == "final" else f"{base}, {residue_status}"
        header = f"═══ SESSION RESIDUE ({label})"
        if meta_parts:
            header += f" [{', '.join(meta_parts)}]"
        header += " ═══"

        parts.append(f"{header}\n\n{text}\n\n═══ END RESIDUE ═══")

    return "\n\n".join(parts)


def build_residue_block(residue_text: str, timestamp: Optional[str] = None,
                        session_id: Optional[str] = None, status: str = "final") -> str:
    """
    Format a residue for injection into the context block.

    Args:
        residue_text: The residue paragraph.
        timestamp: ISO timestamp of when the residue was written (displayed as date hint).
        session_id: Session ID the residue came from (short prefix shown for traceability).
        status: Residue status for header display.

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

    label = "prior session" if status == "final" else f"prior session, {status}"
    header = f"═══ SESSION RESIDUE ({label})"
    if meta_parts:
        header += f" [{', '.join(meta_parts)}]"
    header += " ═══"

    return f"{header}\n\n{residue_text.strip()}\n\n═══ END RESIDUE ═══"


# ─── Auto-Residue Generation ──────────────────────────────────────────────

def generate_partial_residue(
    conn: sqlite3.Connection,
    session_id: str,
    persona_key: str = "",
    persona_dir: Optional[str] = None,
    jsonl_store=None,
) -> dict:
    """Generate and write a partial residue by summarizing recent turns.

    Called every 4th turn pair from the ingest pipeline. Uses the configured
    model for summarization. Overwrites any existing partial for the same session.

    Returns:
        Status dict with keys: status, tokens, detail.
    """
    # Load config for model and API key
    try:
        from solitaire.utils.config import LibrarianConfig
        config = LibrarianConfig.from_env()
    except Exception:
        return {"status": "skipped", "detail": "Config unavailable"}

    if not config.anthropic_api_key:
        return {"status": "skipped", "detail": "No API key configured"}

    # Load existing partial residue for this session (if any)
    existing_partial = ""
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
                latest = data.get("latest", {})
                if latest.get("session_id") == session_id and latest.get("status") == "partial":
                    existing_partial = latest.get("residue", "")
        except Exception:
            pass

    # Load session tail (already written by write_session_tail before this call)
    tail_turns = []
    if residue_dir:
        tail_file = os.path.join(residue_dir, "latest_tail.json")
        try:
            if os.path.exists(tail_file):
                with open(tail_file) as f:
                    tail_data = json.load(f)
                if tail_data.get("session_id") == session_id:
                    tail_turns = tail_data.get("turns", [])
        except Exception:
            pass

    if not tail_turns:
        return {"status": "skipped", "detail": "No tail turns available"}

    # Format turns for the prompt
    turn_lines = []
    for t in tail_turns:
        role = t.get("role", "unknown").upper()
        content = t.get("content", "").strip()
        if content:
            turn_lines.append(f"[{role}] {content}")

    if not turn_lines:
        return {"status": "skipped", "detail": "Empty tail turns"}

    # Build the summarization prompt
    user_parts = []
    if existing_partial:
        user_parts.append(f"[EXISTING PARTIAL SUMMARY]\n{existing_partial}\n")
    user_parts.append("[RECENT TURNS]")
    user_parts.append("\n".join(turn_lines))
    user_parts.append("\nWrite a compressed session summary paragraph.")

    # Call the configured model
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        response = client.messages.create(
            model=config.librarian_model,
            max_tokens=400,
            system="You are a session recorder for a persistent AI memory system. "
                   "Compress the conversation context into a single dense paragraph. "
                   "Include: key topics discussed, decisions made, work completed, "
                   "and work in progress. No headers, no bullets, just a paragraph. "
                   "Be specific with names, files, and numbers. Target: 100-300 tokens.",
            messages=[{"role": "user", "content": "\n".join(user_parts)}],
            timeout=15.0,
        )

        if response.content:
            summary_text = response.content[0].text.strip()
        else:
            summary_text = ""

        # Track cost if available
        try:
            from solitaire.utils.cost_tracker import CostTracker
            tracker = CostTracker()
            if hasattr(response, "usage"):
                tracker.record(
                    call_type="partial_residue",
                    model=config.librarian_model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
        except Exception:
            pass

    except Exception:
        # API failure: fall back to mechanical summary from tail
        summary_text = _mechanical_summary(tail_turns)

    if not summary_text:
        return {"status": "skipped", "detail": "Empty summary generated"}

    # Write the partial residue (overwrites previous partial for same session)
    result = write_residue(
        conn=conn,
        session_id=session_id,
        residue_text=summary_text,
        persona_key=persona_key,
        persona_dir=persona_dir,
        jsonl_store=jsonl_store,
        status="partial",
    )
    return result


def _mechanical_summary(turns: list) -> str:
    """Fallback summary when API is unavailable. Concatenates truncated turns."""
    parts = []
    for t in turns[-6:]:  # Last 6 messages
        role = t.get("role", "unknown").upper()
        content = t.get("content", "").strip()
        if content:
            snippet = content[:150].strip()
            if len(content) > 150:
                snippet += "..."
            parts.append(f"{role}: {snippet}")
    if not parts:
        return ""
    return "[auto-summary unavailable] Last turns: " + " | ".join(parts)


def auto_close_prior_session(
    conn: sqlite3.Connection,
    prior_session_id: str,
    persona_key: str,
    persona_dir: str,
    jsonl_store=None,
) -> dict:
    """Auto-close a prior session that left a partial residue.

    Loads turn pairs + partial residue from the prior session, runs
    summarization via the configured model, and writes a final residue
    with status='auto-closed'.

    Returns:
        Status dict for inclusion in boot output.
    """
    # Load the partial residue text as seed
    residue_file = os.path.join(persona_dir, "residue", "latest_residue.json")
    partial_text = ""
    try:
        with open(residue_file) as f:
            data = json.load(f)
        latest = data.get("latest", {})
        if latest.get("session_id") == prior_session_id and latest.get("status") == "partial":
            partial_text = latest.get("residue", "")
    except Exception:
        return {"status": "skipped", "detail": "Could not read partial residue"}

    if not partial_text:
        return {"status": "skipped", "detail": "No partial residue found"}

    # Load messages from the prior session (up to 20 most recent)
    turn_lines = []
    try:
        rows = conn.execute("""
            SELECT role, content, turn_number
            FROM messages
            WHERE conversation_id = ?
            ORDER BY turn_number DESC, timestamp DESC
            LIMIT 20
        """, (prior_session_id,)).fetchall()
        rows.reverse()
        for role, content, _tn in rows:
            if content and content.strip():
                snippet = content.strip()
                if estimate_tokens(snippet) > 100:
                    snippet = snippet[:350] + " [...]"
                turn_lines.append(f"[{role.upper()}] {snippet}")
    except Exception:
        pass

    # Build the summarization prompt
    user_parts = [f"[PARTIAL SUMMARY FROM MID-SESSION]\n{partial_text}\n"]
    if turn_lines:
        user_parts.append("[FULL SESSION TURNS]")
        user_parts.append("\n".join(turn_lines))
    user_parts.append("\nProduce a final session summary. The partial summary above covers "
                      "an earlier snapshot. Incorporate it and add coverage of later turns "
                      "into a single dense paragraph.")

    # Call the configured model
    try:
        from solitaire.utils.config import LibrarianConfig
        config = LibrarianConfig.from_env()
        if not config.anthropic_api_key:
            return {"status": "skipped", "detail": "No API key configured"}

        import anthropic
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        response = client.messages.create(
            model=config.librarian_model,
            max_tokens=500,
            system="You are a session recorder for a persistent AI memory system. "
                   "Produce a final compressed summary of this completed session. "
                   "Include: key topics, decisions made, work completed, outcomes, "
                   "and any unfinished threads. Single dense paragraph. Be specific "
                   "with names, files, and numbers. Target: 150-400 tokens.",
            messages=[{"role": "user", "content": "\n".join(user_parts)}],
            timeout=15.0,
        )

        if response.content:
            summary_text = response.content[0].text.strip()
        else:
            return {"status": "skipped", "detail": "Empty response from model"}

        # Track cost
        try:
            from solitaire.utils.cost_tracker import CostTracker
            tracker = CostTracker()
            if hasattr(response, "usage"):
                tracker.record(
                    call_type="auto_close_residue",
                    model=config.librarian_model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
        except Exception:
            pass

    except Exception:
        return {"status": "skipped", "detail": "Summarization call failed"}

    # Write the auto-closed residue
    result = write_residue(
        conn=conn,
        session_id=prior_session_id,
        residue_text=summary_text,
        persona_key=persona_key,
        persona_dir=persona_dir,
        jsonl_store=jsonl_store,
        status="auto-closed",
    )
    result["prior_session_id"] = prior_session_id
    return result


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
