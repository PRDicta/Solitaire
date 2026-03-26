"""
The Librarian — Temporal Reasoning Module

State-over-time tracking for entities and relationships.
Detects status changes, version bumps, decisions, completions, and other
temporal signals from content. Records structured timeline events in SQLite.

Design principles:
    - Lightweight heuristics: no LLM required for signal detection
    - Pure SQLite: single file, no external dependencies
    - Incremental: events recorded during ingestion
    - Portable: follows the same patterns as knowledge_graph.py
"""

import json
import sqlite3
import uuid
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field


# ─── Data Types ──────────────────────────────────────────────────────────────

@dataclass
class TimelineEvent:
    """A single temporal event in an entity's history."""
    id: str
    entity_name: str
    timestamp: str              # ISO datetime when the event occurred
    event_type: str             # "created", "updated", "status_change", "decision", "relationship_change"
    description: str            # What happened
    source_entry_id: Optional[str] = None  # Rolodex entry that produced this
    old_value: Optional[str] = None        # Previous state (for changes)
    new_value: Optional[str] = None        # New state (for changes)
    metadata: Dict = field(default_factory=dict)


@dataclass
class EntityTimeline:
    """A complete timeline for an entity."""
    entity_name: str
    events: List[TimelineEvent]
    current_state: Optional[str] = None      # Most recent known state
    first_seen: Optional[str] = None         # ISO timestamp
    last_seen: Optional[str] = None          # ISO timestamp


# ─── Schema ──────────────────────────────────────────────────────────────────

TEMPORAL_REASONING_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_timeline (
    id TEXT PRIMARY KEY,
    entity_name TEXT NOT NULL,
    entity_name_lower TEXT NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    source_entry_id TEXT,
    old_value TEXT,
    new_value TEXT,
    event_time DATETIME NOT NULL,
    created_at DATETIME NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_timeline_entity ON entity_timeline(entity_name_lower);
CREATE INDEX IF NOT EXISTS idx_timeline_time ON entity_timeline(event_time DESC);
CREATE INDEX IF NOT EXISTS idx_timeline_type ON entity_timeline(event_type);
"""


def ensure_temporal_schema(conn: sqlite3.Connection):
    """Create the temporal reasoning tables if they don't exist."""
    conn.executescript(TEMPORAL_REASONING_SCHEMA)
    conn.commit()


# ─── Temporal Signal Detection ───────────────────────────────────────────────

# Patterns for status changes
_STATUS_KEYWORDS = {
    "completed": "completed",
    "shipped": "shipped",
    "started": "started",
    "blocked": "blocked",
    "in progress": "in_progress",
    "in-progress": "in_progress",
    "done": "done",
    "fixed": "fixed",
    "broken": "broken",
    "finished": "finished",
    "wrapped up": "finished",
}

# Patterns for version bumps
_VERSION_PATTERN = re.compile(r'\b(?:v|version)\s*(\d+(?:\.\d+)*)\b', re.IGNORECASE)
_VERSION_UPGRADE_PATTERN = re.compile(
    r'(?:upgraded|updated)\s+(?:from|to)\s+(?:v|version)?\s*(\d+(?:\.\d+)*)',
    re.IGNORECASE
)

# Patterns for decisions
_DECISION_KEYWORDS = {"decided", "chose", "picked", "selected", "switched to", "went with"}

# Patterns for completions
_COMPLETION_KEYWORDS = {"finished", "completed", "done with", "wrapped up", "accomplished"}

# Patterns for temporal markers
_TEMPORAL_MARKERS = {
    "today", "yesterday", "this morning", "this afternoon", "this evening",
    "tonight", "last week", "last month", "last year", "next week", "next month"
}


# Stopword filter: common words that get capitalized at sentence starts
# or in conversational text but are never real entities.
_ENTITY_STOPWORDS = {
    # Single words that appear capitalized at sentence starts
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "i", "we", "he", "she", "they", "you", "me", "my", "our", "your",
    "all", "some", "any", "each", "every", "both", "few", "many", "much",
    "now", "then", "here", "there", "when", "where", "what", "which", "who",
    "how", "why", "well", "just", "also", "too", "very", "quite", "still",
    "not", "no", "yes", "ok", "done", "let", "got", "get", "set", "run",
    "but", "and", "or", "so", "if", "as", "at", "by", "for", "in", "of",
    "on", "to", "up", "off", "out", "with", "from", "into", "over",
    "after", "before", "since", "until", "while", "during", "about",
    # Common sentence starters that aren't entities
    "however", "therefore", "meanwhile", "furthermore", "additionally",
    "finally", "first", "second", "third", "next", "last",
    # Contractions and fragments
    "s", "t", "re", "ve", "ll", "d", "m",
    # Generic nouns used as fallback
    "task", "thing", "stuff", "item", "part", "way", "note",
    # AI-generated sentence starters
    "looking", "working", "building", "running", "getting", "going",
    "starting", "finishing", "checking", "adding", "updating", "fixing",
    "partially", "currently", "recently", "previously", "already",
    # Adverbs/modifiers that appear capitalized
    "yet", "never", "always", "often", "only", "even",
    "once", "again", "maybe", "perhaps", "probably", "certainly",
    "really", "actually", "basically", "essentially", "literally",
}

# Minimum entity name length (characters)
_MIN_ENTITY_LENGTH = 3


def _is_valid_entity(name: str) -> bool:
    """Check if an extracted entity name passes quality filters."""
    if not name or len(name) < _MIN_ENTITY_LENGTH:
        return False
    # Reject ALL-CAPS single words (COMPLETED, STARTED, PIPELINE, etc.)
    # These are status keywords or formatting artifacts, not entities.
    words = name.split()
    if len(words) == 1 and name.isupper():
        return False
    # Check each word against stopwords
    words_lower = [w.lower() for w in words]
    if all(w in _ENTITY_STOPWORDS for w in words_lower):
        return False
    # Single-word entities must be at least 4 chars (avoids "Now", "The", etc.)
    if len(words) == 1 and len(name) < 4:
        return False
    # Reject contraction fragments and possessive artifacts
    if name in ("s", "t", "re", "ve", "ll", "d", "m"):
        return False
    return True


def _extract_entity_name_from_context(content: str, kg_entities: Optional[set] = None) -> Optional[str]:
    """
    Heuristic: extract a likely entity name from content.

    Strategy (in priority order):
    1. If kg_entities provided, check if any known entity appears in the content.
    2. Fall back to capitalized phrase extraction with stopword filtering.
    """
    # Strategy 1: Cross-reference with knowledge graph entities
    if kg_entities:
        content_lower = content.lower()
        # Check longer entity names first (more specific = better match)
        for entity in sorted(kg_entities, key=len, reverse=True):
            if entity.lower() in content_lower:
                return entity

    # Strategy 2: Capitalized phrase extraction with filtering
    lines = content.split('\n')
    for line in lines[:5]:  # Check first few lines
        # Look for capitalized phrases (2-4 words)
        matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b', line)
        if matches:
            # Prefer longer matches, filter through quality check
            for match in sorted(matches, key=len, reverse=True):
                if len(match.split()) <= 4 and _is_valid_entity(match):
                    return match
    return None


def _detect_status_changes(content: str, kg_entities: Optional[set] = None) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
    """
    Detect status change signals in content.
    Returns list of (entity_name, old_status, new_status, description).
    """
    changes = []
    content_lower = content.lower()

    for keyword, status in _STATUS_KEYWORDS.items():
        if keyword in content_lower:
            # Simple heuristic: if we see a status keyword, extract surrounding context
            pattern = re.compile(
                rf'(\b\w+(?:\s+\w+){{0,2}})\s+(?:is\s+)?{re.escape(keyword)}',
                re.IGNORECASE
            )
            matches = pattern.finditer(content)
            for match in matches:
                entity = match.group(1).strip()
                changes.append((entity, None, status, f"Status changed to {status}"))

    return changes


def _detect_version_bumps(content: str, kg_entities: Optional[set] = None) -> List[Tuple[str, str, str, str]]:
    """
    Detect version change signals.
    Returns list of (entity_name, old_version, new_version, description).
    """
    changes = []

    # Detect explicit upgrade/update patterns
    for match in _VERSION_UPGRADE_PATTERN.finditer(content):
        version = match.group(1)
        context = content[max(0, match.start() - 50):match.end()]
        entity = _extract_entity_name_from_context(context, kg_entities)
        if entity:
            changes.append((entity, None, f"v{version}", f"Updated to version {version}"))

    # Detect version mentions
    for match in _VERSION_PATTERN.finditer(content):
        version = match.group(1)
        context = content[max(0, match.start() - 50):match.end()]
        entity = _extract_entity_name_from_context(context, kg_entities)
        if entity:
            changes.append((entity, None, f"v{version}", f"Version {version}"))

    return changes


def _detect_decisions(content: str, kg_entities: Optional[set] = None) -> List[Tuple[str, str, str]]:
    """
    Detect decision signals.
    Returns list of (entity_name, decision_description, rationale).
    """
    decisions = []
    content_lower = content.lower()

    for keyword in _DECISION_KEYWORDS:
        pattern = re.compile(
            rf'(\b\w+(?:\s+\w+){{0,2}})\s+(?:{re.escape(keyword)})\s+(?:to\s+)?(.{{20,100}}?)(?:\.|,|$)',
            re.IGNORECASE
        )
        for match in pattern.finditer(content):
            entity = match.group(1).strip()
            decision = match.group(2).strip()
            decisions.append((entity, decision, f"Decided: {decision}"))

    return decisions


def _detect_completions(content: str, kg_entities: Optional[set] = None) -> List[Tuple[str, str, str]]:
    """
    Detect completion signals.
    Returns list of (entity_name, task_description, completion_note).
    """
    completions = []

    for keyword in _COMPLETION_KEYWORDS:
        pattern = re.compile(
            rf'(?:{re.escape(keyword)})\s+(?:with\s+)?(.{{20,100}}?)(?:\.|,|$)',
            re.IGNORECASE
        )
        for match in pattern.finditer(content):
            task = match.group(1).strip()
            context = content[max(0, match.start() - 50):match.end()]
            entity = _extract_entity_name_from_context(context, kg_entities) or "Task"
            completions.append((entity, task, f"Completed: {task}"))

    return completions


def _detect_temporal_markers(content: str) -> List[str]:
    """
    Detect temporal markers (today, yesterday, last week, etc.).
    Returns list of found markers.
    """
    markers = []
    content_lower = content.lower()

    for marker in _TEMPORAL_MARKERS:
        if marker in content_lower:
            markers.append(marker)

    return markers


# ─── Temporal Reasoner ───────────────────────────────────────────────────────

class TemporalReasoner:
    """
    SQLite-native temporal reasoning engine.

    Tracks state-over-time for entities, detects temporal signals during
    ingestion, and provides timeline queries.

    Usage:
        tr = TemporalReasoner(conn)
        tr.ensure_schema()

        # During ingestion
        events = tr.process_entry_temporal(content, entry_id="abc123")

        # During recall
        timeline = tr.get_timeline("The Librarian")
        recent = tr.get_changes_since("2026-02-28T00:00:00")
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._schema_ensured = False
        self._kg_entities: Optional[set] = None

    def load_kg_entities(self) -> set:
        """Load known entity names from the knowledge graph for cross-referencing.

        Caches the result for the lifetime of this reasoner instance.
        Filters out KG entities that fail the same quality checks as extracted entities.
        Returns a set of entity name strings.
        """
        if self._kg_entities is not None:
            return self._kg_entities
        try:
            rows = self.conn.execute(
                "SELECT DISTINCT entity FROM knowledge_graph WHERE entity IS NOT NULL"
            ).fetchall()
            self._kg_entities = {
                (r["entity"] if isinstance(r, sqlite3.Row) else r[0])
                for r in rows
                if (r["entity"] if isinstance(r, sqlite3.Row) else r[0]) and
                   _is_valid_entity(r["entity"] if isinstance(r, sqlite3.Row) else r[0])
            }
        except Exception:
            self._kg_entities = set()
        return self._kg_entities

    def ensure_schema(self):
        """Create tables if needed."""
        if not self._schema_ensured:
            ensure_temporal_schema(self.conn)
            self._schema_ensured = True

    def record_event(
        self,
        entity_name: str,
        event_type: str,
        description: str,
        source_entry_id: Optional[str] = None,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        event_time: Optional[str] = None,
    ) -> TimelineEvent:
        """
        Record a temporal event for an entity.

        Args:
            entity_name: The entity this event is about
            event_type: One of "created", "updated", "status_change", "decision", "relationship_change"
            description: Human-readable description of what happened
            source_entry_id: The rolodex entry that produced this event
            old_value: Previous value (for state changes)
            new_value: New value (for state changes)
            event_time: When the event occurred (defaults to now)

        Returns:
            The recorded TimelineEvent
        """
        self.ensure_schema()

        event_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()
        event_time = event_time or now
        entity_name_lower = entity_name.lower().strip()

        # Build metadata
        metadata = {}
        if old_value or new_value:
            metadata["state_transition"] = {
                "old": old_value,
                "new": new_value,
            }

        self.conn.execute(
            """INSERT INTO entity_timeline
               (id, entity_name, entity_name_lower, event_type, description,
                source_entry_id, old_value, new_value, event_time, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_id, entity_name, entity_name_lower, event_type, description,
             source_entry_id, old_value, new_value, event_time, now,
             json.dumps(metadata))
        )
        self.conn.commit()

        return TimelineEvent(
            id=event_id,
            entity_name=entity_name,
            timestamp=event_time,
            event_type=event_type,
            description=description,
            source_entry_id=source_entry_id,
            old_value=old_value,
            new_value=new_value,
            metadata=metadata,
        )

    def get_timeline(self, entity_name: str, limit: int = 50) -> EntityTimeline:
        """
        Get the complete timeline for an entity.

        Args:
            entity_name: The entity to get history for
            limit: Maximum number of events to return

        Returns:
            EntityTimeline with all events sorted chronologically
        """
        self.ensure_schema()
        entity_name_lower = entity_name.lower().strip()

        rows = self.conn.execute(
            """SELECT id, entity_name, event_type, description, source_entry_id,
                      old_value, new_value, event_time, metadata
               FROM entity_timeline
               WHERE entity_name_lower = ?
               ORDER BY event_time ASC
               LIMIT ?""",
            (entity_name_lower, limit)
        ).fetchall()

        events = []
        for row in rows:
            events.append(TimelineEvent(
                id=row[0],
                entity_name=row[1],
                event_type=row[2],
                description=row[3],
                source_entry_id=row[4],
                old_value=row[5],
                new_value=row[6],
                timestamp=row[7],
                metadata=json.loads(row[8]) if row[8] else {},
            ))

        # Compute aggregate properties
        current_state = None
        first_seen = None
        last_seen = None

        if events:
            # Most recent state: look at the last "status_change" event
            status_events = [e for e in events if e.event_type == "status_change"]
            if status_events:
                current_state = status_events[-1].new_value

            first_seen = events[0].timestamp
            last_seen = events[-1].timestamp

        return EntityTimeline(
            entity_name=entity_name,
            events=events,
            current_state=current_state,
            first_seen=first_seen,
            last_seen=last_seen,
        )

    def get_current_state(self, entity_name: str) -> Optional[str]:
        """Get the most recent known state of an entity."""
        timeline = self.get_timeline(entity_name, limit=100)

        # Look for the most recent status_change or "updated" event
        for event in reversed(timeline.events):
            if event.event_type in ("status_change", "updated"):
                return event.new_value or event.description

        return timeline.current_state

    def get_changes_since(
        self,
        since_iso: str,
        entity_name: Optional[str] = None,
    ) -> List[TimelineEvent]:
        """
        Get all timeline events since a given timestamp.

        Args:
            since_iso: ISO datetime string to filter from
            entity_name: Optional entity to filter to (if None, return all)

        Returns:
            List of TimelineEvent objects in chronological order
        """
        self.ensure_schema()

        if entity_name:
            entity_name_lower = entity_name.lower().strip()
            rows = self.conn.execute(
                """SELECT id, entity_name, event_type, description, source_entry_id,
                          old_value, new_value, event_time, metadata
                   FROM entity_timeline
                   WHERE entity_name_lower = ? AND event_time > ?
                   ORDER BY event_time DESC""",
                (entity_name_lower, since_iso)
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT id, entity_name, event_type, description, source_entry_id,
                          old_value, new_value, event_time, metadata
                   FROM entity_timeline
                   WHERE event_time > ?
                   ORDER BY event_time DESC""",
                (since_iso,)
            ).fetchall()

        events = []
        for row in rows:
            events.append(TimelineEvent(
                id=row[0],
                entity_name=row[1],
                event_type=row[2],
                description=row[3],
                source_entry_id=row[4],
                old_value=row[5],
                new_value=row[6],
                timestamp=row[7],
                metadata=json.loads(row[8]) if row[8] else {},
            ))

        return events

    def process_entry_temporal(
        self,
        content: str,
        entry_id: str,
        created_at: Optional[str] = None,
        event_time: Optional[str] = None,
    ) -> List[TimelineEvent]:
        """
        Process temporal signals from ingested content.

        Heuristically detects:
        - Status changes (completed, shipped, started, blocked, done, fixed)
        - Version bumps (v1.2, updated to version 2, etc.)
        - Decisions (decided to, chose, picked, switched to)
        - Completions (finished, wrapped up, accomplished)
        - Temporal markers (today, yesterday, last week, etc.)

        Called during ingestion. No LLM required.

        Args:
            content: The text to process
            entry_id: The rolodex entry ID (for provenance)
            created_at: When the entry was created
            event_time: When the content describes (defaults to created_at or now)

        Returns:
            List of TimelineEvent objects created
        """
        self.ensure_schema()

        if not event_time:
            event_time = created_at or datetime.now(timezone.utc).isoformat()

        # Load KG entities for cross-referencing (cached after first call)
        kg_entities = self.load_kg_entities()

        events_created = []

        # Detect status changes
        for entity_name, old_status, new_status, description in _detect_status_changes(content, kg_entities):
            event = self.record_event(
                entity_name=entity_name,
                event_type="status_change",
                description=description,
                source_entry_id=entry_id,
                old_value=old_status,
                new_value=new_status,
                event_time=event_time,
            )
            events_created.append(event)

        # Detect version bumps
        for entity_name, old_ver, new_ver, description in _detect_version_bumps(content, kg_entities):
            event = self.record_event(
                entity_name=entity_name,
                event_type="updated",
                description=description,
                source_entry_id=entry_id,
                old_value=old_ver,
                new_value=new_ver,
                event_time=event_time,
            )
            events_created.append(event)

        # Detect decisions
        for entity_name, decision_desc, description in _detect_decisions(content, kg_entities):
            event = self.record_event(
                entity_name=entity_name,
                event_type="decision",
                description=description,
                source_entry_id=entry_id,
                new_value=decision_desc,
                event_time=event_time,
            )
            events_created.append(event)

        # Detect completions (with KG entity awareness)
        for entity_name, task_desc, description in _detect_completions(content, kg_entities=kg_entities):
            event = self.record_event(
                entity_name=entity_name,
                event_type="status_change",
                description=description,
                source_entry_id=entry_id,
                old_value=None,
                new_value="completed",
                event_time=event_time,
            )
            events_created.append(event)

        return events_created

    def get_temporal_context(
        self,
        entity_names: List[str],
        since_hours: float = 48.0,
    ) -> str:
        """
        Format recent timeline events for injection into recall context.

        Returns a human-readable summary of recent temporal events.
        Useful for providing state context to the LLM.

        Args:
            entity_names: List of entities to include
            since_hours: How far back to look (defaults to 48 hours)

        Returns:
            Formatted string for context injection
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=since_hours)).isoformat()

        result = []
        for entity_name in entity_names:
            timeline = self.get_timeline(entity_name)
            recent_events = [e for e in timeline.events if e.timestamp > cutoff]

            if recent_events:
                result.append(f"\n# {entity_name}")
                for event in recent_events:
                    result.append(f"  - [{event.event_type}] {event.description}")

                if timeline.current_state:
                    result.append(f"  Current state: {timeline.current_state}")

        if not result:
            return ""

        return "## Recent Entity Timeline\n" + "".join(result)

    # ─── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return timeline statistics."""
        self.ensure_schema()

        total_events = self.conn.execute(
            "SELECT COUNT(*) FROM entity_timeline"
        ).fetchone()[0]

        # Count by event type
        type_counts = self.conn.execute(
            "SELECT event_type, COUNT(*) FROM entity_timeline GROUP BY event_type"
        ).fetchall()

        # Top entities
        top_entities = self.conn.execute(
            """SELECT entity_name, COUNT(*) as event_count
               FROM entity_timeline
               GROUP BY entity_name
               ORDER BY event_count DESC
               LIMIT 10"""
        ).fetchall()

        return {
            "total_events": total_events,
            "event_types": {
                row[0]: row[1] for row in type_counts
            },
            "top_entities": [
                {"name": row[0], "event_count": row[1]}
                for row in top_entities
            ],
        }
