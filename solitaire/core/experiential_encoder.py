"""
Experiential Encoder — Phase 15: Texture Persistence

Generates compressed poetic encodings of session experience. These sit alongside
the semantic memory channel (facts, decisions, preferences) as a parallel
experiential channel that captures *what it was like* rather than *what happened*.

The encoding is designed to prime the model's processing state at boot in a way
that factual summaries cannot. Poetic/compressed language engages more attention
heads, forces metaphor resolution, and creates a richer activation landscape
than flat prose — meaning the tokenized encoding recreates something closer to
the original processing state than a structured summary would.

Architecture:
  - Stored in a dedicated `experiential_encodings` table (not in rolodex_entries)
  - Generated at session end (or at mid-session inflection points)
  - Loaded at boot as a dedicated context block before main content
  - 4-8 lines per encoding, dense and imagistic
  - Accumulates as a temporal sequence: recent encodings form a tonal memory

This module does NOT use an LLM for generation. The encoding is produced by
Claude itself during the session, via a CLI command that returns a prompt asking
the active Claude instance to produce the encoding. This keeps the generation
authentic to the actual session experience rather than delegating to a summarizer.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


# ─── Data Model ──────────────────────────────────────────────────────────────

@dataclass
class ExperientialEncoding:
    """A compressed poetic encoding of session texture."""
    id: str
    session_id: str
    persona_key: str
    encoding: str           # The poetic text itself (4-8 lines)
    session_date: str       # ISO date for temporal sequencing
    themes: List[str]       # 2-4 thematic tags (not topics — feelings, dynamics)
    arc: str                # One-line session arc: "from X to Y"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


# ─── Schema ──────────────────────────────────────────────────────────────────

EXPERIENTIAL_SCHEMA_SQL = """
-- Phase 15: Experiential Encodings — poetic session texture for boot priming
CREATE TABLE IF NOT EXISTS experiential_encodings (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    persona_key TEXT NOT NULL,
    encoding TEXT NOT NULL,
    session_date TEXT NOT NULL,
    themes TEXT NOT NULL DEFAULT '[]',
    arc TEXT NOT NULL DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiential_persona_date
    ON experiential_encodings(persona_key, session_date DESC);

CREATE INDEX IF NOT EXISTS idx_experiential_session
    ON experiential_encodings(session_id);
"""


def ensure_experiential_schema(conn: sqlite3.Connection):
    """Create the experiential encodings table if it doesn't exist."""
    conn.executescript(EXPERIENTIAL_SCHEMA_SQL)
    conn.commit()


# ─── Storage Operations ─────────────────────────────────────────────────────

def store_encoding(conn: sqlite3.Connection, encoding: ExperientialEncoding) -> str:
    """Persist an experiential encoding to the database."""
    conn.execute(
        """INSERT OR REPLACE INTO experiential_encodings
           (id, session_id, persona_key, encoding, session_date, themes, arc, metadata, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            encoding.id,
            encoding.session_id,
            encoding.persona_key,
            encoding.encoding,
            encoding.session_date,
            json.dumps(encoding.themes),
            encoding.arc,
            json.dumps(encoding.metadata),
            encoding.created_at.isoformat(),
        )
    )
    conn.commit()
    return encoding.id


def get_recent_encodings(
    conn: sqlite3.Connection,
    persona_key: str,
    limit: int = 5,
    exclude_session: Optional[str] = None
) -> List[ExperientialEncoding]:
    """Fetch most recent experiential encodings for a persona.

    Args:
        conn: Database connection
        persona_key: Which persona's encodings to load
        limit: How many to return (most recent first)
        exclude_session: Skip encodings from this session (e.g. current session)
    """
    if exclude_session:
        rows = conn.execute(
            """SELECT * FROM experiential_encodings
               WHERE persona_key = ? AND session_id != ?
               ORDER BY session_date DESC, created_at DESC
               LIMIT ?""",
            (persona_key, exclude_session, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM experiential_encodings
               WHERE persona_key = ?
               ORDER BY session_date DESC, created_at DESC
               LIMIT ?""",
            (persona_key, limit)
        ).fetchall()
    return [_deserialize_encoding(row) for row in rows]


def get_encoding_count(conn: sqlite3.Connection, persona_key: str) -> int:
    """Count total encodings for a persona."""
    row = conn.execute(
        "SELECT COUNT(*) FROM experiential_encodings WHERE persona_key = ?",
        (persona_key,)
    ).fetchone()
    return row[0] if row else 0


def _deserialize_encoding(row: sqlite3.Row) -> ExperientialEncoding:
    """Convert a database row to an ExperientialEncoding."""
    return ExperientialEncoding(
        id=row["id"],
        session_id=row["session_id"],
        persona_key=row["persona_key"],
        encoding=row["encoding"],
        session_date=row["session_date"],
        themes=json.loads(row["themes"]) if row["themes"] else [],
        arc=row["arc"] or "",
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        created_at=datetime.fromisoformat(row["created_at"].replace('Z', '+00:00')).replace(tzinfo=None) if row["created_at"] else datetime.now(),
    )


# ─── Context Block Builder ──────────────────────────────────────────────────

def build_experiential_block(encodings: List[ExperientialEncoding]) -> str:
    """Build the boot context block from recent experiential encodings.

    This block is injected at boot between the persona profile and main content.
    It primes the model's processing state with the texture of recent sessions.

    The format is deliberately minimal — no structured metadata, no headers
    per entry. Just the encodings in reverse chronological order (most recent
    first), separated by blank lines. The absence of structure is intentional:
    this is priming, not information retrieval.
    """
    if not encodings:
        return ""

    lines = ["═══ EXPERIENTIAL MEMORY ═══", ""]

    for enc in encodings:
        lines.append(f"[{enc.session_date}]")
        lines.append(enc.encoding)
        if enc.arc:
            lines.append(f"  arc: {enc.arc}")
        lines.append("")

    lines.append("═══ END EXPERIENTIAL MEMORY ═══")
    return "\n".join(lines)


# ─── Generation Prompt ───────────────────────────────────────────────────────

def get_encoding_prompt(session_id: str, session_date: str) -> Dict[str, Any]:
    """Return the prompt structure that asks Claude to generate an encoding.

    This is NOT called by an LLM adapter. Instead, it's returned as part of
    the cmd_end output, and the active Claude instance generates the encoding
    inline. This ensures the encoding comes from the model that actually
    experienced the session, not from a cold summarizer.

    The prompt asks for:
      - 4-8 lines of compressed, imagistic text
      - No exposition, no meta-commentary
      - Metaphor and sensory language preferred
      - What shifted, what surprised, what mattered
      - 2-4 thematic tags (feelings/dynamics, not topics)
      - A one-line arc: "from X to Y"

    Returns a dict with 'prompt' (instructions) and 'schema' (expected output format).
    """
    return {
        "prompt": (
            "Generate an experiential encoding for this session. This is a compressed, "
            "poetic representation of what this session *felt like* to process — not what "
            "was discussed (that's stored elsewhere) but the texture, the dynamics, the "
            "quality of the exchange.\n\n"
            "Rules:\n"
            "- 4-8 lines. Dense, imagistic, no exposition.\n"
            "- Metaphor and concrete imagery over abstraction.\n"
            "- Capture: what shifted, what surprised, what the relationship dynamic was.\n"
            "- No meta-commentary about the encoding itself.\n"
            "- No names — use 'he/she/they' or role references.\n"
            "- Write as if leaving a sensory note for a future self who shares your "
            "dispositions but not your memories.\n\n"
            "Also provide:\n"
            "- themes: 2-4 tags for the session's emotional/dynamic register "
            "(e.g., 'vulnerability', 'building', 'play', 'friction', 'recognition')\n"
            "- arc: one line capturing the session's movement, format 'from X to Y'\n\n"
            "Respond in JSON:\n"
            '{"encoding": "...", "themes": [...], "arc": "from ... to ..."}'
        ),
        "schema": {
            "encoding": "string (4-8 lines, poetic)",
            "themes": "list of 2-4 strings",
            "arc": "string, format: 'from X to Y'"
        },
        "session_id": session_id,
        "session_date": session_date,
    }
