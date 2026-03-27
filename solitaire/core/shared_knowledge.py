"""
The Librarian — Multi-Persona Shared Knowledge Module
Enables personas to publish and query shared knowledge without context bleed.
Manages visibility, TTL, and supersession for cross-persona information.
"""
import sqlite3
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path


# ─── Shared Knowledge Schema ──────────────────────────────────────────────────
# Extends the shared database (shared.db) beyond profile to include
# structured cross-persona knowledge with visibility and TTL controls.

SHARED_KNOWLEDGE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS shared_knowledge (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    source_persona TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    visibility TEXT NOT NULL DEFAULT 'all',
    created_at DATETIME NOT NULL,
    expires_at DATETIME,
    superseded_by TEXT,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_shared_category ON shared_knowledge(category);
CREATE INDEX IF NOT EXISTS idx_shared_source ON shared_knowledge(source_persona);
CREATE INDEX IF NOT EXISTS idx_shared_created ON shared_knowledge(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_shared_expires ON shared_knowledge(expires_at);

CREATE VIRTUAL TABLE IF NOT EXISTS shared_knowledge_fts USING fts5(
    entry_id,
    content,
    tags,
    category,
    tokenize='porter unicode61'
);
"""


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class SharedEntry:
    """
    A single shared knowledge entry visible to selected personas.

    Attributes:
        id: UUID for the entry
        content: The knowledge content
        category: One of: fact, decision, preference, definition, project_status
        source_persona: Which persona created it (e.g., "agent_a", "agent_b")
        tags: List of tags for organization
        visibility: "all" (every persona sees it) or comma-separated persona keys
        created_at: ISO timestamp of creation
        expires_at: Optional ISO timestamp for time-sensitive knowledge
        superseded_by: Optional UUID of entry that replaces this one
        metadata: Extra context (dict, stored as JSON)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    category: str = "fact"
    source_persona: str = ""
    tags: List[str] = field(default_factory=list)
    visibility: str = "all"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = None
    superseded_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SharedEntry":
        """Construct from dict."""
        return cls(**data)

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Check if entry has passed its expiration time."""
        if not self.expires_at:
            return False
        if now is None:
            now = datetime.now(timezone.utc)
        try:
            expires = datetime.fromisoformat(self.expires_at)
            return now > expires
        except (ValueError, TypeError):
            return False

    def is_superseded(self) -> bool:
        """Check if entry has been superseded."""
        return self.superseded_by is not None

    def is_visible_to(self, persona_key: str) -> bool:
        """Check if this entry is visible to a given persona."""
        if self.visibility == "all":
            return True
        # visibility is comma-separated persona keys
        persona_list = [p.strip() for p in self.visibility.split(",")]
        return persona_key in persona_list


# ─── SharedKnowledgeStore ──────────────────────────────────────────────────────

class SharedKnowledgeStore:
    """
    Manages shared knowledge across personas.

    Implements:
    - publish() — add knowledge to shared layer
    - query() — search with FTS and visibility filtering
    - get_recent() — fetch recent entries visible to a persona
    - supersede() — correct/replace an entry
    - prune_expired() — clean up expired entries
    - get_shared_context() — format knowledge for boot injection
    - should_share() — heuristic to determine if content should be published
    """

    def __init__(self, shared_db_path: str):
        """Initialize the store with the shared database path."""
        self.db_path = shared_db_path
        self.conn = None
        self._connect()
        self.ensure_schema()

    def _connect(self):
        """Open database connection with FUSE awareness."""
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # FUSE detection (same pattern as schema.py)
        if self._is_fuse_mount(path):
            self.conn.execute("PRAGMA journal_mode=OFF")
        else:
            self.conn.execute("PRAGMA journal_mode=WAL")

        self.conn.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _is_fuse_mount(path: Path) -> bool:
        """Detect if path is on FUSE filesystem."""
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

    def ensure_schema(self):
        """Create schema tables if they don't exist."""
        self.conn.executescript(SHARED_KNOWLEDGE_SCHEMA_SQL)
        self.conn.commit()

    def publish(
        self,
        content: str,
        category: str,
        source_persona: str,
        tags: Optional[List[str]] = None,
        visibility: str = "all",
        expires_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SharedEntry:
        """
        Publish knowledge to the shared layer.

        Args:
            content: The knowledge content
            category: One of: fact, decision, preference, definition, project_status
            source_persona: Which persona is publishing (e.g., "agent_a", "agent_b")
            tags: Optional tags for organization
            visibility: "all" or comma-separated persona keys (e.g., "agent_a,agent_b")
            expires_at: Optional ISO timestamp for TTL
            metadata: Optional extra context

        Returns:
            The created SharedEntry
        """
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}

        entry = SharedEntry(
            id=str(uuid.uuid4()),
            content=content,
            category=category,
            source_persona=source_persona,
            tags=tags,
            visibility=visibility,
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=expires_at,
            metadata=metadata,
        )

        # Insert into main table
        self.conn.execute(
            """
            INSERT INTO shared_knowledge
            (id, content, category, source_persona, tags, visibility, created_at, expires_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.content,
                entry.category,
                entry.source_persona,
                json.dumps(entry.tags),
                entry.visibility,
                entry.created_at,
                entry.expires_at,
                json.dumps(entry.metadata),
            ),
        )

        # Insert into FTS index
        self.conn.execute(
            """
            INSERT INTO shared_knowledge_fts (entry_id, content, tags, category)
            VALUES (?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.content,
                " ".join(entry.tags),
                entry.category,
            ),
        )

        self.conn.commit()
        return entry

    def query(
        self,
        search_text: str,
        persona_key: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[SharedEntry]:
        """
        Search shared knowledge using FTS with visibility filtering.

        Args:
            search_text: Full-text search query
            persona_key: If provided, filters to entries visible to this persona
            category: If provided, filters to entries in this category
            limit: Maximum results to return

        Returns:
            List of SharedEntry objects visible to the persona (or all if no persona given)
        """
        query = """
            SELECT sk.id, sk.content, sk.category, sk.source_persona,
                   sk.tags, sk.visibility, sk.created_at, sk.expires_at,
                   sk.superseded_by, sk.metadata
            FROM shared_knowledge sk
            INNER JOIN shared_knowledge_fts fts ON sk.id = fts.entry_id
            WHERE fts.content MATCH ?
            AND sk.superseded_by IS NULL
            AND (sk.expires_at IS NULL OR sk.expires_at > datetime('now'))
        """
        params: List[Any] = [search_text]

        if category:
            query += " AND sk.category = ?"
            params.append(category)

        query += " ORDER BY sk.created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        entries = [self._deserialize_entry(row) for row in rows]

        # Apply persona visibility filter
        if persona_key:
            entries = [e for e in entries if e.is_visible_to(persona_key)]

        return entries

    def get_recent(
        self,
        persona_key: Optional[str] = None,
        limit: int = 20,
    ) -> List[SharedEntry]:
        """
        Get the most recent shared knowledge entries.

        Args:
            persona_key: If provided, filters to entries visible to this persona
            limit: Maximum results to return

        Returns:
            List of SharedEntry objects (most recent first)
        """
        query = """
            SELECT id, content, category, source_persona, tags, visibility,
                   created_at, expires_at, superseded_by, metadata
            FROM shared_knowledge
            WHERE superseded_by IS NULL
            AND (expires_at IS NULL OR expires_at > datetime('now'))
            ORDER BY created_at DESC
            LIMIT ?
        """

        rows = self.conn.execute(query, [limit]).fetchall()
        entries = [self._deserialize_entry(row) for row in rows]

        # Apply persona visibility filter
        if persona_key:
            entries = [e for e in entries if e.is_visible_to(persona_key)]

        return entries

    def supersede(
        self,
        old_id: str,
        new_content: str,
        source_persona: str,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        visibility: Optional[str] = None,
    ) -> SharedEntry:
        """
        Correct or replace an existing shared entry.

        Creates a new entry and marks the old one as superseded.

        Args:
            old_id: UUID of the entry to replace
            new_content: The corrected content
            source_persona: Which persona is publishing the correction
            tags: Optional tags (defaults to old entry's tags)
            category: Optional category (defaults to old entry's category)
            visibility: Optional visibility (defaults to old entry's visibility)

        Returns:
            The new SharedEntry
        """
        # Fetch old entry to inherit properties
        old_row = self.conn.execute(
            "SELECT category, tags, visibility FROM shared_knowledge WHERE id = ?",
            [old_id],
        ).fetchone()

        if not old_row:
            raise ValueError(f"Entry {old_id} not found")

        old_category = old_row["category"]
        old_tags = json.loads(old_row["tags"]) if old_row["tags"] else []
        old_visibility = old_row["visibility"]

        # Create new entry
        new_entry = self.publish(
            content=new_content,
            category=category or old_category,
            source_persona=source_persona,
            tags=tags or old_tags,
            visibility=visibility or old_visibility,
        )

        # Mark old entry as superseded
        self.conn.execute(
            "UPDATE shared_knowledge SET superseded_by = ? WHERE id = ?",
            [new_entry.id, old_id],
        )
        self.conn.commit()

        return new_entry

    def prune_expired(self) -> int:
        """
        Remove entries past their expiration time.

        Returns:
            Number of entries deleted
        """
        cursor = self.conn.execute(
            """
            DELETE FROM shared_knowledge
            WHERE expires_at IS NOT NULL
            AND expires_at < datetime('now')
            """
        )
        self.conn.commit()
        return cursor.rowcount

    def get_shared_context(
        self,
        persona_key: str,
        limit: int = 10,
    ) -> str:
        """
        Format recent shared knowledge for boot context injection.

        Returns a plain-text summary of recent shared knowledge visible
        to the given persona, suitable for inclusion in the boot prompt.

        Args:
            persona_key: The persona requesting context
            limit: Maximum entries to include

        Returns:
            Formatted string ready for boot context
        """
        entries = self.get_recent(persona_key=persona_key, limit=limit)

        if not entries:
            return ""

        lines = ["## Shared Knowledge (Cross-Persona)"]

        for entry in entries:
            # Format: [category] source: content #tag1 #tag2 ...
            tag_str = " " + " ".join(f"#{tag}" for tag in entry.tags) if entry.tags else ""
            lines.append(f"- [{entry.category}] {entry.source_persona}: {entry.content}{tag_str}")

        return "\n".join(lines)

    def should_share(self, content: str, category: str) -> bool:
        """
        Heuristic to determine if an entry should be shared across personas.

        Returns True for categories that should propagate:
        - user_knowledge: Always share (user facts are universal)
        - decision: Always share (decisions affect all contexts)
        - project_status: Always share (project state is cross-cutting)
        - fact: Share if tagged with user-related tags
        - preference: Always share (user preferences are universal)
        - definition: Always share

        Returns False for persona-specific observations:
        - notes, disposition_drift, and persona-specific observations

        Args:
            content: The knowledge content
            category: The category of the entry

        Returns:
            True if the entry should be published to shared layer
        """
        # Categories that always share
        always_share = {"user_knowledge", "decision", "project_status", "preference", "definition", "system_change"}
        if category in always_share:
            return True

        # Fact: share if tagged with user-related tags
        if category == "fact":
            user_tags = {"user", "user_fact", "user_knowledge", "universal"}
            # Check if content mentions user-relevant keywords
            return any(tag in user_tags for tag in content.lower().split())

        # Everything else: don't share by default
        return False

    def _deserialize_entry(self, row: sqlite3.Row) -> SharedEntry:
        """Convert database row to SharedEntry."""
        return SharedEntry(
            id=row["id"],
            content=row["content"],
            category=row["category"],
            source_persona=row["source_persona"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            visibility=row["visibility"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            superseded_by=row["superseded_by"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
