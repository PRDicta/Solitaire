"""
The Librarian — Active Summarization (Module 3)

Living project briefs that auto-update as context accumulates.

ProjectBrief dataclass represents a living summary for any entity/project:
- entity_name: the subject (e.g., "MyProject", "Example Corp", "Sample App")
- brief: current summary (2-5 sentences, auto-built from key facts)
- key_facts: bullet-point facts (max 10, auto-trimmed, auto-deduped)
- status: current status/phase (e.g., "active", "completed", "in progress")
- last_updated: ISO timestamp
- entry_count: how many entries contributed
- confidence: 0-1, based on entry count / baseline threshold

ActiveSummarizer class incrementally updates briefs using heuristic keyword extraction.
No LLM calls — pure pattern matching and fact aggregation.

Schema: project_briefs table with entity_name as primary key.
"""

import json
import re
import sqlite3
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from datetime import datetime


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ProjectBrief:
    """
    A living summary for an entity or project.

    Auto-updates as new content about the entity is ingested.
    Confidence scales with entry count (entry_count / 20 = confidence).

    Attributes:
        entity_name: The subject (e.g., "MyProject", "Example project").
        brief: Current summary (2-5 sentences, auto-built from key_facts).
        key_facts: Bullet-point facts (max 10, auto-trimmed, auto-deduped).
        status: Current status/phase (e.g., "active", "completed", "in progress", "blocked").
        last_updated: ISO timestamp (UTC).
        entry_count: How many entries have contributed to this brief.
        confidence: 0.0-1.0. Scales with entry count (min(1.0, entry_count / 20)).
    """
    entity_name: str
    brief: str = ""
    key_facts: List[str] = field(default_factory=list)
    status: str = "active"
    last_updated: str = ""
    entry_count: int = 0
    confidence: float = 0.0

    def __post_init__(self):
        """Validate bounds."""
        if not self.last_updated:
            self.last_updated = datetime.utcnow().isoformat()
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")
        if len(self.key_facts) > 10:
            self.key_facts = self.key_facts[-10:]


# ─── Status Detection Patterns ───────────────────────────────────────────────

# Words that indicate a status change
STATUS_KEYWORDS = {
    "completed": ["completed", "done", "finished", "shipped", "released", "launched", "published"],
    "in progress": ["in progress", "ongoing", "in progress", "underway", "active", "working on"],
    "blocked": ["blocked", "blocked on", "stuck", "halted", "paused", "waiting on"],
    "planned": ["planned", "scheduled", "upcoming", "will", "going to", "plan to"],
    "started": ["started", "began", "initiated", "launched", "kicked off"],
}

_COMPILED_STATUS_PATTERNS = {
    status: [re.compile(kw, re.IGNORECASE) for kw in keywords]
    for status, keywords in STATUS_KEYWORDS.items()
}


# ─── Entity Detection Patterns ───────────────────────────────────────────────

# Common project/entity names (case-insensitive)
KNOWN_ENTITIES = {
    "myproject",
    "example corp",
    "example project",
    "sample app",
    "demo tool",
    "test system",
}


# ─── Schema ──────────────────────────────────────────────────────────────────

PROJECT_BRIEFS_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_briefs (
    entity_name TEXT PRIMARY KEY,
    brief TEXT NOT NULL DEFAULT '',
    key_facts TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    last_updated DATETIME NOT NULL,
    entry_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.0,
    contributing_entry_ids TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_project_briefs_updated
    ON project_briefs(last_updated DESC);
"""


# ─── Core Class ──────────────────────────────────────────────────────────────

class ActiveSummarizer:
    """
    Incrementally builds and maintains living project briefs.

    No LLM dependency. Uses heuristic keyword extraction and status detection.
    Pure SQLite, no external dependencies beyond Python standard library.
    """

    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize the Active Summarizer.

        Args:
            conn: SQLite database connection.
        """
        self.conn = conn
        self.ensure_schema()

    def ensure_schema(self):
        """Create the project_briefs table if it doesn't exist."""
        self.conn.executescript(PROJECT_BRIEFS_SCHEMA)
        self.conn.commit()

    def _extract_key_phrases(self, content: str, entity_name: str) -> List[str]:
        """
        Extract key phrases/sentences from content that mention the entity.

        Simple heuristic: split into sentences, keep those containing entity_name
        or key signal words, deduplicate.

        Args:
            content: Text to extract from.
            entity_name: The entity to search for.

        Returns:
            List of extracted phrases (max 5 per call).
        """
        entity_lower = entity_name.lower()
        content_lower = content.lower()

        # Skip if entity doesn't appear in content
        if entity_lower not in content_lower:
            return []

        # Split into sentences
        sentences = re.split(r'[.!?\n]+', content)
        extracted = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue

            # Keep sentences mentioning the entity or containing key signal words
            if entity_lower in sentence.lower():
                # Trim to reasonable length and deduplicate
                phrase = sentence[:100]
                if phrase not in extracted:
                    extracted.append(phrase)

        return extracted[:5]  # Max 5 phrases per content block

    def _detect_status(self, content: str) -> Optional[str]:
        """
        Detect status keywords in content.

        Returns the status if found (completed, in progress, blocked, planned, started).
        If multiple statuses found, returns the first match.

        Args:
            content: Text to analyze.

        Returns:
            Status string or None.
        """
        for status, patterns in _COMPILED_STATUS_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(content):
                    return status
        return None

    def _rebuild_brief_from_facts(self, key_facts: List[str]) -> str:
        """
        Rebuild brief from top key facts.

        Joins top 3-5 facts into a paragraph (2-5 sentences).

        Args:
            key_facts: List of fact strings.

        Returns:
            Brief string (2-5 sentences).
        """
        if not key_facts:
            return ""

        # Take top 3-5 facts
        top_facts = key_facts[-5:] if len(key_facts) > 5 else key_facts

        # Join into a paragraph
        brief = " ".join(f.strip().rstrip('.') for f in top_facts)
        brief = brief[:300]  # Cap at 300 chars

        # Ensure it ends with a period
        if brief and not brief.endswith('.'):
            brief += '.'

        return brief

    def update_brief(
        self,
        entity_name: str,
        new_content: str,
        entry_id: Optional[str] = None,
    ) -> ProjectBrief:
        """
        Incrementally update a brief when new content about the entity arrives.

        Heuristic-based (no LLM):
        1. Extract key phrases from new_content that mention entity_name
        2. Append to key_facts (deduplicate, keep max 10)
        3. Detect status words and update status if found
        4. Rebuild brief from top 5 facts
        5. Bump entry_count and update confidence

        Args:
            entity_name: The entity to update.
            new_content: New content/text about the entity.
            entry_id: Optional ID of the source entry (for tracking).

        Returns:
            Updated ProjectBrief.
        """
        # Get current brief or create new
        current = self.get_brief(entity_name)
        if current is None:
            current = ProjectBrief(
                entity_name=entity_name,
                brief="",
                key_facts=[],
                status="active",
                last_updated=datetime.utcnow().isoformat(),
                entry_count=0,
                confidence=0.0,
            )

        # Extract key phrases from new content
        new_phrases = self._extract_key_phrases(new_content, entity_name)

        # Append to key_facts and deduplicate (keep most recent)
        existing_facts = list(current.key_facts)
        for phrase in new_phrases:
            if phrase not in existing_facts:
                existing_facts.append(phrase)

        # Trim to max 10 facts
        if len(existing_facts) > 10:
            existing_facts = existing_facts[-10:]

        # Detect status and update if found
        detected_status = self._detect_status(new_content)
        if detected_status:
            current.status = detected_status

        # Rebuild brief from facts
        current.brief = self._rebuild_brief_from_facts(existing_facts)
        current.key_facts = existing_facts
        current.entry_count += 1
        current.confidence = min(1.0, current.entry_count / 20.0)
        current.last_updated = datetime.utcnow().isoformat()

        # Store in database
        self._store_brief(current, entry_id)

        return current

    def _store_brief(self, brief: ProjectBrief, contributing_entry_id: Optional[str] = None):
        """
        Store or update a brief in the database.

        Args:
            brief: ProjectBrief to store.
            contributing_entry_id: Optional entry ID to track.
        """
        # Retrieve existing contributing_entry_ids
        row = self.conn.execute(
            "SELECT contributing_entry_ids FROM project_briefs WHERE entity_name = ?",
            (brief.entity_name,)
        ).fetchone()

        contributing_ids = []
        if row:
            try:
                contributing_ids = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                contributing_ids = []

        # Append new entry ID if provided
        if contributing_entry_id and contributing_entry_id not in contributing_ids:
            contributing_ids.append(contributing_entry_id)

        # Upsert into database
        self.conn.execute(
            """
            INSERT INTO project_briefs (
                entity_name, brief, key_facts, status, last_updated,
                entry_count, confidence, contributing_entry_ids, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_name) DO UPDATE SET
                brief = excluded.brief,
                key_facts = excluded.key_facts,
                status = excluded.status,
                last_updated = excluded.last_updated,
                entry_count = excluded.entry_count,
                confidence = excluded.confidence,
                contributing_entry_ids = excluded.contributing_entry_ids
            """,
            (
                brief.entity_name,
                brief.brief,
                json.dumps(brief.key_facts),
                brief.status,
                brief.last_updated,
                brief.entry_count,
                brief.confidence,
                json.dumps(contributing_ids),
                "{}",
            ),
        )
        self.conn.commit()

    def get_brief(self, entity_name: str) -> Optional[ProjectBrief]:
        """
        Retrieve the current brief for an entity.

        Args:
            entity_name: The entity name.

        Returns:
            ProjectBrief if exists, None otherwise.
        """
        row = self.conn.execute(
            "SELECT brief, key_facts, status, last_updated, entry_count, confidence FROM project_briefs WHERE entity_name = ?",
            (entity_name,)
        ).fetchone()

        if not row:
            return None

        try:
            key_facts = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            key_facts = []

        return ProjectBrief(
            entity_name=entity_name,
            brief=row[0],
            key_facts=key_facts,
            status=row[2],
            last_updated=row[3],
            entry_count=row[4],
            confidence=row[5],
        )

    def get_all_briefs(self, limit: int = 20) -> List[ProjectBrief]:
        """
        Retrieve all briefs, sorted by last_updated (newest first).

        Args:
            limit: Maximum number of briefs to return.

        Returns:
            List of ProjectBrief objects.
        """
        rows = self.conn.execute(
            """
            SELECT entity_name, brief, key_facts, status, last_updated, entry_count, confidence
            FROM project_briefs
            ORDER BY last_updated DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

        briefs = []
        for row in rows:
            try:
                key_facts = json.loads(row[2])
            except (json.JSONDecodeError, TypeError):
                key_facts = []

            briefs.append(ProjectBrief(
                entity_name=row[0],
                brief=row[1],
                key_facts=key_facts,
                status=row[3],
                last_updated=row[4],
                entry_count=row[5],
                confidence=row[6],
            ))

        return briefs

    def process_entry(
        self,
        content: str,
        entry_id: Optional[str] = None,
    ) -> List[str]:
        """
        Called during ingestion. Detect which entities are mentioned and update their briefs.

        Uses entity detection heuristics:
        1. Check for known entity names (case-insensitive)
        2. Check for capitalized multi-word phrases
        3. Return list of updated entity names

        Args:
            content: Entry content to analyze.
            entry_id: Optional entry ID for tracking.

        Returns:
            List of entity names that were updated.
        """
        content_lower = content.lower()
        detected_entities = []
        seen_lower = set()

        # Check for known entities
        for entity in KNOWN_ENTITIES:
            if entity in content_lower:
                detected_entities.append(entity)
                seen_lower.add(entity.lower())

        # Check for capitalized multi-word phrases (basic NER)
        # Pattern: 2+ capitalized words in sequence
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        for match in re.finditer(pattern, content):
            phrase = match.group(0)
            phrase_lower = phrase.lower()
            if phrase_lower not in seen_lower and len(phrase) > 4:
                detected_entities.append(phrase_lower)
                seen_lower.add(phrase_lower)

        # Update brief for each detected entity
        updated = []
        for entity in detected_entities:
            self.update_brief(entity, content, entry_id)
            updated.append(entity)

        return updated

    def get_brief_context(self, entity_names: List[str]) -> str:
        """
        Format briefs for injection into recall context.

        Returns a markdown-formatted context block with briefs for the given entities.

        Args:
            entity_names: List of entity names to retrieve briefs for.

        Returns:
            Formatted context string (markdown).
        """
        if not entity_names:
            return ""

        lines = ["# Project Briefs\n"]

        for entity in entity_names:
            brief = self.get_brief(entity)
            if brief:
                lines.append(f"## {entity}")
                lines.append(f"**Status:** {brief.status}")
                lines.append(f"**Confidence:** {brief.confidence:.0%}")
                lines.append(f"**Entries:** {brief.entry_count}\n")

                if brief.brief:
                    lines.append(f"{brief.brief}\n")

                if brief.key_facts:
                    lines.append("**Key Facts:**")
                    for fact in brief.key_facts:
                        lines.append(f"- {fact}")
                    lines.append("")

        return "\n".join(lines)


# ─── Utility Functions ───────────────────────────────────────────────────────

def to_dict(brief: ProjectBrief) -> Dict[str, Any]:
    """
    Convert ProjectBrief to a JSON-serializable dict.

    Args:
        brief: ProjectBrief instance.

    Returns:
        Dict with all fields.
    """
    return asdict(brief)


def from_dict(d: Dict[str, Any]) -> ProjectBrief:
    """
    Reconstruct ProjectBrief from a dict (e.g., from JSON).

    Args:
        d: Dict with keys matching ProjectBrief fields.

    Returns:
        ProjectBrief instance.

    Raises:
        KeyError or ValueError if required fields are missing.
    """
    return ProjectBrief(
        entity_name=str(d["entity_name"]),
        brief=str(d.get("brief", "")),
        key_facts=list(d.get("key_facts", [])),
        status=str(d.get("status", "active")),
        last_updated=str(d.get("last_updated", "")),
        entry_count=int(d.get("entry_count", 0)),
        confidence=float(d.get("confidence", 0.0)),
    )
