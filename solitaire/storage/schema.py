"""
The Librarian — Database Schema
SQLite table definitions, initialization, and serialization helpers.
All conversation content is stored verbatim — no compression, ever.
"""
import sqlite3
import json
import struct
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path


def _safe_fromisoformat(s):
    """Parse ISO datetime, handling 'Z' suffix (Python <3.11 compat).
    Returns naive datetime to match codebase convention (utcnow everywhere)."""
    if s is None:
        return None
    dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
    return dt.replace(tzinfo=None) if dt.tzinfo else dt
from ..core.types import (
    RolodexEntry, ContentModality, EntryCategory, Tier
)
# ─── Schema SQL ──────────────────────────────────────────────────────────────
SCHEMA_SQL = """
-- Core rolodex entries: every piece of indexed content
CREATE TABLE IF NOT EXISTS rolodex_entries (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL,
    category TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    source_range TEXT NOT NULL DEFAULT '{}',
    access_count INTEGER DEFAULT 0,
    last_accessed DATETIME,
    created_at DATETIME NOT NULL,
    tier TEXT DEFAULT 'cold',
    embedding BLOB,
    linked_ids TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    archived_at DATETIME DEFAULT NULL
);
-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_entries_conversation
    ON rolodex_entries(conversation_id);
CREATE INDEX IF NOT EXISTS idx_entries_category
    ON rolodex_entries(category);
CREATE INDEX IF NOT EXISTS idx_entries_tier
    ON rolodex_entries(tier);
CREATE INDEX IF NOT EXISTS idx_entries_access_count
    ON rolodex_entries(access_count DESC);
CREATE INDEX IF NOT EXISTS idx_entries_created_at
    ON rolodex_entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_entries_disposition_drift
    ON rolodex_entries(category, created_at DESC)
    WHERE category = 'disposition_drift';
-- Full-text search index for keyword queries
CREATE VIRTUAL TABLE IF NOT EXISTS rolodex_fts USING fts5(
    entry_id,
    content,
    tags,
    category,
    tokenize='porter unicode61'
);
-- Conversation tracking
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at DATETIME NOT NULL,
    ended_at DATETIME,
    total_tokens INTEGER DEFAULT 0,
    entry_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active'
);
-- Query log for analytics
CREATE TABLE IF NOT EXISTS query_log (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    found BOOLEAN NOT NULL,
    entry_ids TEXT DEFAULT '[]',
    search_time_ms REAL,
    search_type TEXT,
    timestamp DATETIME NOT NULL
);
-- Phase 4: Persistent message storage for session resume/cross-session
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    token_count INTEGER DEFAULT 0,
    timestamp DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_turn
    ON messages(conversation_id, turn_number);

-- Phase 3: Preload log for tracking prediction accuracy
CREATE TABLE IF NOT EXISTS preload_log (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    pressure REAL NOT NULL,
    predicted_entry_ids TEXT DEFAULT '[]',
    injected_entry_ids TEXT DEFAULT '[]',
    cache_warmed_entry_ids TEXT DEFAULT '[]',
    hit_entry_ids TEXT DEFAULT '[]',
    timestamp DATETIME NOT NULL
);

-- Phase 7: Reasoning Chains — narrative indexes capturing the "why"
CREATE TABLE IF NOT EXISTS chains (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    chain_index INTEGER NOT NULL,
    turn_range_start INTEGER NOT NULL,
    turn_range_end INTEGER NOT NULL,
    summary TEXT NOT NULL,
    topics TEXT NOT NULL DEFAULT '[]',
    related_entries TEXT NOT NULL DEFAULT '[]',
    embedding BLOB,
    created_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chains_session_index
    ON chains(session_id, chain_index);
CREATE INDEX IF NOT EXISTS idx_chains_created_at
    ON chains(created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS chains_fts USING fts5(
    chain_id,
    summary,
    topics,
    tokenize='porter unicode61'
);

-- Phase 8: Topics — emergent categories of knowledge
CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    label TEXT UNIQUE NOT NULL,
    description TEXT DEFAULT '',
    parent_topic_id TEXT,
    created_at DATETIME NOT NULL,
    last_updated DATETIME,
    entry_count INTEGER DEFAULT 0,
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_topics_label
    ON topics(label);
CREATE INDEX IF NOT EXISTS idx_topics_parent
    ON topics(parent_topic_id);

CREATE VIRTUAL TABLE IF NOT EXISTS topics_fts USING fts5(
    topic_id,
    label,
    description,
    tokenize='porter unicode61'
);

-- Phase 8: Topic assignment log (traces topic evolution)
CREATE TABLE IF NOT EXISTS topic_assignments (
    id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    source TEXT NOT NULL,
    assigned_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_topic_assignments_entry
    ON topic_assignments(entry_id);
CREATE INDEX IF NOT EXISTS idx_topic_assignments_topic
    ON topic_assignments(topic_id);

-- User profile: structured key-value store for user preferences
CREATE TABLE IF NOT EXISTS user_profile (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    source_session TEXT,
    updated_at DATETIME NOT NULL
);

-- Phase 12: Document registry — tracks registered documents for read-on-demand
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_hash TEXT,
    title TEXT,
    page_count INTEGER,
    summary TEXT DEFAULT '',
    registered_at DATETIME NOT NULL,
    last_read_at DATETIME,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_documents_file_type
    ON documents(file_type);
CREATE INDEX IF NOT EXISTS idx_documents_registered
    ON documents(registered_at DESC);

-- Phase 13: Project clusters — emergent groupings of related topics
-- Auto-inferred from topic co-occurrence; optionally user-named (Option C hybrid)
CREATE TABLE IF NOT EXISTS project_clusters (
    id TEXT PRIMARY KEY,
    label TEXT,
    description TEXT DEFAULT '',
    topic_ids TEXT NOT NULL DEFAULT '[]',
    is_user_named INTEGER DEFAULT 0,
    created_at DATETIME NOT NULL,
    last_active DATETIME,
    session_count INTEGER DEFAULT 0,
    entry_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_project_clusters_active
    ON project_clusters(last_active DESC);

-- Phase 10: Boot manifest — pre-computed context plan, refined each session
CREATE TABLE IF NOT EXISTS boot_manifest (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    source_session_id TEXT,
    manifest_type TEXT NOT NULL,
    total_token_cost INTEGER NOT NULL,
    entry_count INTEGER NOT NULL,
    topic_summary TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_manifest_updated
    ON boot_manifest(updated_at DESC);

-- Phase 16: Process Trace — in-the-moment phenomenological markers
-- Captures process markers ([pm: ...]) written during response generation.
-- These are first-person observations of processing state, recorded live.
CREATE TABLE IF NOT EXISTS process_trace (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    response_number INTEGER NOT NULL,
    marker_type TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_process_trace_session
    ON process_trace(session_id, response_number);
CREATE INDEX IF NOT EXISTS idx_process_trace_type
    ON process_trace(marker_type);
CREATE INDEX IF NOT EXISTS idx_process_trace_created
    ON process_trace(created_at DESC);

-- Phase 10: Manifest entries — ranked content selections for boot context
CREATE TABLE IF NOT EXISTS manifest_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manifest_id INTEGER NOT NULL,
    entry_id TEXT NOT NULL,
    composite_score REAL NOT NULL,
    token_cost INTEGER NOT NULL,
    topic_label TEXT,
    selection_reason TEXT NOT NULL,
    was_accessed INTEGER DEFAULT 0,
    slot_rank INTEGER NOT NULL,
    FOREIGN KEY (manifest_id) REFERENCES boot_manifest(id)
);

CREATE INDEX IF NOT EXISTS idx_manifest_entries_manifest
    ON manifest_entries(manifest_id);
CREATE INDEX IF NOT EXISTS idx_manifest_entries_entry
    ON manifest_entries(entry_id);
"""
# ─── Database Initialization ─────────────────────────────────────────────────
def _is_fuse_mount(path: Path) -> bool:
    """Detect if a path is on a FUSE filesystem.

    Cowork mounts the user's folder into its Linux VM via FUSE.
    SQLite WAL mode relies on shared-memory locking (mmap) which
    FUSE doesn't support, causing the DB to fall back to a temp
    path that gets wiped between sessions.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["stat", "-f", "-c", "%T", str(path.parent)],
            capture_output=True, text=True, timeout=2
        )
        fs_type = result.stdout.strip().lower()
        return "fuse" in fs_type
    except Exception:
        # Fallback: check /proc/mounts on Linux
        try:
            mounts = Path("/proc/mounts").read_text()
            path_str = str(path.resolve())
            for line in mounts.splitlines():
                parts = line.split()
                if len(parts) >= 3 and path_str.startswith(parts[1]):
                    if "fuse" in parts[2].lower():
                        return True
        except Exception:
            pass
    return False


def init_database(db_path: str) -> sqlite3.Connection:
    """
    Create database and tables if they don't exist.
    Returns an open connection.

    On FUSE filesystems (e.g. Cowork's VM mount), uses journal_mode=OFF
    instead of WAL because FUSE doesn't support the shared-memory locking
    that WAL requires. This trades crash-recovery journaling for actually
    persisting data on disk across sessions.
    """
    # Ensure directory exists
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # FUSE doesn't support WAL's shared-memory locking — use OFF instead
    if _is_fuse_mount(path):
        conn.execute("PRAGMA journal_mode=OFF")
    else:
        conn.execute("PRAGMA journal_mode=WAL")      # Better concurrent read/write
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    # Phase 4: safely extend conversations table for existing databases
    _safe_add_columns(conn)
    # Phase 15: Experiential encodings table
    try:
        from ..core.experiential_encoder import ensure_experiential_schema
        ensure_experiential_schema(conn)
    except Exception:
        pass  # Non-fatal — table will be created on first use
    # User facts store (provenance-tagged structured facts)
    try:
        from ..core.user_facts import UserFactsStore
        UserFactsStore(conn)  # Creates table if missing
    except Exception:
        pass  # Non-fatal — table will be created on first use
    # Hindsight: temporal reasoning timeline table
    try:
        from ..core.temporal_reasoning import ensure_temporal_schema
        ensure_temporal_schema(conn)
    except Exception:
        pass  # Non-fatal — table will be created on first use
    return conn


def _safe_add_columns(conn: sqlite3.Connection):
    """Add new columns to existing tables (idempotent)."""
    additions = [
        # Phase 4: conversation metadata
        ("conversations", "summary", "TEXT DEFAULT ''"),
        ("conversations", "last_active", "DATETIME"),
        ("conversations", "message_count", "INTEGER DEFAULT 0"),
        # Phase 8: topic assignment on entries
        ("rolodex_entries", "topic_id", "TEXT"),
        # Corrections: track superseded entries
        ("rolodex_entries", "superseded_by", "TEXT"),
        # Verbatim source flag: TRUE = original user/assistant text, FALSE = assistant summary/paraphrase
        ("rolodex_entries", "verbatim_source", "INTEGER DEFAULT 1"),
        # Phase 12: Document source tracking
        ("rolodex_entries", "source_type", "TEXT DEFAULT 'conversation'"),
        ("rolodex_entries", "document_id", "TEXT"),
        ("rolodex_entries", "source_location", "TEXT DEFAULT ''"),
        # Bi-temporal: when the content describes (NULL = contemporaneous with created_at)
        ("rolodex_entries", "event_time", "DATETIME"),
        # Provenance: who asserted this content (user-stated | assistant-inferred | system | unknown)
        ("rolodex_entries", "provenance", "TEXT DEFAULT 'unknown'"),
        # Archive: entries excluded from search but preserved for timeline reconstruction.
        # NULL = active (searchable). Non-NULL = archived (invisible to FTS/semantic).
        ("rolodex_entries", "archived_at", "DATETIME DEFAULT NULL"),
    ]
    for table, column, col_type in additions:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists — that's fine

    # Phase 12: Index for source_type filtering
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_source_type ON rolodex_entries(source_type)")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Archive: partial index for efficient filtering of active-only entries
    try:
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entries_archived
            ON rolodex_entries(archived_at)
            WHERE archived_at IS NOT NULL
        """)
        conn.commit()
    except sqlite3.OperationalError:
        pass
# ─── Serialization ───────────────────────────────────────────────────────────
def serialize_entry(entry: RolodexEntry) -> tuple:
    """Convert RolodexEntry to a tuple for SQL INSERT."""
    return (
        entry.id,
        entry.conversation_id,
        entry.content,
        entry.content_type.value,
        entry.category.value,
        json.dumps(entry.tags),
        json.dumps(entry.source_range),
        entry.access_count,
        entry.last_accessed.isoformat() if entry.last_accessed else None,
        entry.created_at.isoformat(),
        entry.tier.value,
        serialize_embedding(entry.embedding) if entry.embedding else None,
        json.dumps(entry.linked_ids),
        json.dumps(entry.metadata),
    )
def _safe_tier(val):
    """Parse Tier safely, returning COLD for unknown/empty values."""
    if not val:
        return Tier.COLD
    try:
        return Tier(val)
    except ValueError:
        return Tier.COLD


def _safe_json_loads(val, default):
    """Parse JSON safely, returning default for empty/None values."""
    if not val:
        return default
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return default


def deserialize_entry(row: sqlite3.Row) -> RolodexEntry:
    """Convert a database row back to a RolodexEntry.

    Handles legacy/imported entries that may have empty or malformed
    JSON fields by falling back to safe defaults.
    """
    return RolodexEntry(
        id=row["id"],
        conversation_id=row["conversation_id"],
        content=row["content"],
        content_type=ContentModality(row["content_type"]),
        category=EntryCategory(row["category"]),
        tags=_safe_json_loads(row["tags"], []),
        source_range=_safe_json_loads(row["source_range"], {}),
        access_count=row["access_count"] or 0,
        last_accessed=(
            _safe_fromisoformat(row["last_accessed"])
            if row["last_accessed"] else None
        ),
        created_at=_safe_fromisoformat(row["created_at"]),
        tier=_safe_tier(row["tier"]),
        embedding=(
            deserialize_embedding(row["embedding"])
            if row["embedding"] else None
        ),
        linked_ids=_safe_json_loads(row["linked_ids"], []),
        metadata=_safe_json_loads(row["metadata"], {}),
        verbatim_source=bool(row["verbatim_source"]) if "verbatim_source" in row.keys() else True,
        source_type=row["source_type"] if "source_type" in row.keys() else "conversation",
        document_id=row["document_id"] if "document_id" in row.keys() else None,
        source_location=row["source_location"] if "source_location" in row.keys() else "",
        event_time=(
            _safe_fromisoformat(row["event_time"])
            if "event_time" in row.keys() and row["event_time"] else None
        ),
        provenance=row["provenance"] if "provenance" in row.keys() else "unknown",
    )
# ─── Compression Codebook Schema ────────────────────────────────────────────
CODEBOOK_SCHEMA_SQL = """
-- Compression codebook: learned compression patterns with progressive promotion
CREATE TABLE IF NOT EXISTS compression_codebook (
    id TEXT PRIMARY KEY,
    pattern_text TEXT NOT NULL,          -- original verbose phrase
    warm_form TEXT NOT NULL,             -- stage 1: emoji-anchored / abbreviated
    hot_form TEXT,                       -- stage 2: single emoji/token (NULL until promoted)
    stage INTEGER NOT NULL DEFAULT 0,    -- CompressionStage enum value (0=COLD, 1=WARM, 2=HOT)
    token_cost_original INTEGER,         -- tokens in verbose form
    token_cost_warm INTEGER,             -- tokens in warm form
    token_cost_hot INTEGER,              -- tokens in hot form (NULL until promoted)
    times_seen INTEGER DEFAULT 1,        -- compile/boot cycles this appeared in
    times_decoded_ok INTEGER DEFAULT 0,  -- successful cold-context transfers
    confidence REAL DEFAULT 0.0,         -- 0.0-1.0 promotion readiness
    first_seen_at DATETIME NOT NULL,
    last_seen_at DATETIME NOT NULL,
    promoted_at DATETIME,               -- when promoted to current stage
    source_entry_ids TEXT DEFAULT '[]',  -- JSON: behavioral entry IDs using this pattern
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_codebook_stage
    ON compression_codebook(stage);
CREATE INDEX IF NOT EXISTS idx_codebook_confidence
    ON compression_codebook(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_codebook_times_seen
    ON compression_codebook(times_seen DESC);
"""


def ensure_codebook_schema(conn: sqlite3.Connection):
    """Create the compression codebook table if it doesn't exist."""
    conn.executescript(CODEBOOK_SCHEMA_SQL)
    conn.commit()


# ─── Shared Database Schema ──────────────────────────────────────────────────
# The shared DB holds user profile data that is persona-independent.
# Each persona has its own rolodex DB, but profile lives here.

SHARED_SCHEMA_SQL = """
-- User profile: structured key-value store for user preferences (shared across personas)
CREATE TABLE IF NOT EXISTS user_profile (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    source_session TEXT,
    updated_at DATETIME NOT NULL
);
"""


def init_shared_database(db_path: str) -> sqlite3.Connection:
    """Create/open the shared profile database.

    This is a lightweight DB holding only the user_profile table,
    shared across all persona rolodex databases.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if _is_fuse_mount(path):
        conn.execute("PRAGMA journal_mode=OFF")
    else:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SHARED_SCHEMA_SQL)
    conn.commit()
    return conn


def serialize_embedding(embedding: List[float]) -> bytes:
    """Pack a float list into compact binary (float32)."""
    return struct.pack(f"{len(embedding)}f", *embedding)
def deserialize_embedding(blob: bytes) -> List[float]:
    """Unpack binary back to float list."""
    count = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{count}f", blob))
