"""
Reader for another Solitaire instance's rolodex.db.

Opens a foreign rolodex database in read-only mode and yields entries
as IngestCandidates. This is entry-level import only: raw content comes
across, but identity graph nodes, persona configs, and session state
do not. Those are treated as belonging to the source instance.

Design decisions:
- Read-only SQLite connection (file: URI with mode=ro).
- Skips entries that are already marked as external imports (no import chains).
- Maps rolodex categories to IngestContentType.
- Uses the entry's original created_at as the candidate timestamp.
- Dedup key uses the source entry's ID + content hash to prevent
  re-import across instances.
"""

import os
import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Iterator, Dict, Any
from datetime import datetime, timezone

from .reader_base import ReaderBase
from ..core.types import (
    IngestCandidate,
    IngestContentType,
    EnrichmentHint,
)


# Map rolodex categories to IngestContentType
_CATEGORY_MAP = {
    "user_knowledge": IngestContentType.PREFERENCE,
    "preference": IngestContentType.PREFERENCE,
    "behavioral": IngestContentType.PREFERENCE,
    "correction": IngestContentType.PREFERENCE,
    "decision": IngestContentType.FACT,
    "fact": IngestContentType.FACT,
    "factual_knowledge": IngestContentType.FACT,
    "project_knowledge": IngestContentType.FACT,
    "reference": IngestContentType.DOCUMENT,
    "definition": IngestContentType.DOCUMENT,
    "implementation": IngestContentType.DOCUMENT,
    "instruction": IngestContentType.PREFERENCE,
    "note": IngestContentType.CONVERSATION,
    "example": IngestContentType.DOCUMENT,
    "warning": IngestContentType.FACT,
    "friction": IngestContentType.CONVERSATION,
    "breakthrough": IngestContentType.CONVERSATION,
    "pivot": IngestContentType.CONVERSATION,
    "disposition_drift": IngestContentType.OTHER,
}

# Categories that map to high-confidence entries (already structured)
_HIGH_CONFIDENCE_CATEGORIES = {
    "user_knowledge", "preference", "behavioral", "correction",
    "decision", "instruction",
}


class SolitaireReader(ReaderBase):
    """Reads entries from another Solitaire instance's rolodex.db.

    Config:
        path: str - Path to the foreign rolodex.db file.
        skip_external: bool - Skip entries already marked as external imports (default True).
        categories: list[str] - Only import these categories (optional, imports all if omitted).
    """

    @property
    def source_id(self) -> str:
        return "solitaire-instance"

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        path = config.get("path", "")
        if not path:
            return {"valid": False, "error": "No path provided"}
        if not os.path.isfile(path):
            return {"valid": False, "error": f"File not found: {path}"}

        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                # Verify it's a Solitaire database
                tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}

                if "rolodex_entries" not in tables:
                    return {"valid": False, "error": "Not a Solitaire database (missing rolodex_entries table)"}

                count = conn.execute("SELECT COUNT(*) FROM rolodex_entries").fetchone()[0]
                if count == 0:
                    return {"valid": False, "error": "Database is empty"}

                return {"valid": True, "entry_count": count}
            finally:
                conn.close()
        except sqlite3.Error as e:
            return {"valid": False, "error": f"SQLite error: {e}"}

    def read(self, config: Dict[str, Any]) -> Iterator[IngestCandidate]:
        """Yield IngestCandidates from a foreign Solitaire rolodex.

        Opens the database read-only and streams entries. Never modifies
        the source database.
        """
        path = config["path"]
        skip_external = config.get("skip_external", True)
        category_filter = set(config.get("categories", []))

        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            yield IngestCandidate(
                source_ref=f"file://{os.path.abspath(path)}",
                raw_content="",
                content_type=IngestContentType.OTHER,
                enrichment_hint=EnrichmentHint.SKIP,
                confidence=0.0,
                source_id=self.source_id,
                metadata={"reader_error": f"Failed to open database: {e}"},
                dedup_key=f"solitaire:{path}:open-error",
            )
            return

        try:
            query = "SELECT id, content, category, tags, metadata, created_at FROM rolodex_entries"
            cursor = conn.execute(query)

            for row in cursor:
                entry_id = row["id"]
                content = row["content"]
                category = row["category"]
                tags_raw = row["tags"]
                meta_raw = row["metadata"]
                created_at_raw = row["created_at"]

                if not content or not content.strip():
                    continue

                # Parse metadata
                try:
                    metadata = json.loads(meta_raw) if meta_raw else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                # Skip external imports to prevent import chains
                if skip_external and metadata.get("import_source_id"):
                    continue

                # Category filter
                if category_filter and category not in category_filter:
                    continue

                # Parse tags
                try:
                    tags = json.loads(tags_raw) if tags_raw else []
                except (json.JSONDecodeError, TypeError):
                    tags = []

                # Map category to content type
                content_type = _CATEGORY_MAP.get(category, IngestContentType.OTHER)

                # Confidence based on category structure
                confidence = 0.8 if category in _HIGH_CONFIDENCE_CATEGORIES else 0.6

                # Enrichment: entries from another Solitaire instance are
                # pre-enriched, but we want entity extraction for our own KG
                enrichment_hint = EnrichmentHint.PARTIAL

                # Parse timestamp
                timestamp = None
                if created_at_raw:
                    try:
                        ts = datetime.fromisoformat(created_at_raw.replace('Z', '+00:00'))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        timestamp = ts
                    except (ValueError, TypeError):
                        pass

                # Dedup key: source entry ID + content hash
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
                dedup_key = f"solitaire:{entry_id}:{content_hash}"

                # Build tags for our system
                import_tags = list(tags) + [
                    "source:solitaire-instance",
                    f"original-category:{category}",
                    f"source-db:{os.path.basename(path)}",
                ]

                yield IngestCandidate(
                    source_ref=f"solitaire://{entry_id}",
                    raw_content=content,
                    content_type=content_type,
                    enrichment_hint=enrichment_hint,
                    confidence=confidence,
                    source_id=self.source_id,
                    timestamp=timestamp,
                    metadata={
                        "source_entry_id": entry_id,
                        "source_category": category,
                        "source_db": path,
                        "original_metadata": metadata,
                    },
                    tags=import_tags,
                    dedup_key=dedup_key,
                )

        except sqlite3.Error as e:
            yield IngestCandidate(
                source_ref=f"file://{os.path.abspath(path)}",
                raw_content="",
                content_type=IngestContentType.OTHER,
                enrichment_hint=EnrichmentHint.SKIP,
                confidence=0.0,
                source_id=self.source_id,
                metadata={"reader_error": f"Read error: {e}"},
                dedup_key=f"solitaire:{path}:read-error",
            )
        finally:
            conn.close()
