"""
Reader for Librarian JSONL export files (entries.jsonl format).

Format: one JSON object per line. Each line represents a RolodexEntry
with fields: id, content, content_type, category, tags, created_at,
metadata, provenance, etc.

This reader is for importing entries from another Solitaire/Librarian
instance. It maps the source entry's category and content_type to
IngestContentType, preserving as much metadata as possible.

The JSONL format has two flavors:
1. Canonical store (entries.jsonl): includes _seq, _type, _op, _ts headers
2. Plain export: just the entry fields, one per line

This reader handles both.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from datetime import datetime

from .reader_base import ReaderBase
from ..core.types import (
    IngestCandidate,
    IngestContentType,
    EnrichmentHint,
)


# Map Librarian categories to IngestContentType
_CATEGORY_MAP: Dict[str, IngestContentType] = {
    "user_knowledge": IngestContentType.PREFERENCE,
    "preference": IngestContentType.PREFERENCE,
    "fact": IngestContentType.FACT,
    "factual_knowledge": IngestContentType.FACT,
    "project_knowledge": IngestContentType.FACT,
    "definition": IngestContentType.FACT,
    "reference": IngestContentType.DOCUMENT,
    "implementation": IngestContentType.DOCUMENT,
    "instruction": IngestContentType.DOCUMENT,
    "decision": IngestContentType.FACT,
    "note": IngestContentType.CONVERSATION,
    "behavioral": IngestContentType.PREFERENCE,
    "correction": IngestContentType.CONVERSATION,
    "friction": IngestContentType.CONVERSATION,
    "breakthrough": IngestContentType.CONVERSATION,
    "pivot": IngestContentType.CONVERSATION,
    "warning": IngestContentType.FACT,
    "example": IngestContentType.DOCUMENT,
    "disposition_drift": IngestContentType.OTHER,
}


def _parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse a timestamp string from JSONL. Handles multiple formats."""
    if not ts_str:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


class JSONLReader(ReaderBase):
    """Reads Solitaire/Librarian JSONL entry export files.

    Config:
        path: str - Path to the .jsonl file.
        skip_archived: bool - Skip entries with non-null archived_at (default True).
        skip_superseded: bool - Skip entries with non-null superseded_by (default True).
    """

    @property
    def source_id(self) -> str:
        return "jsonl"

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        path = config.get("path", "")
        if not path:
            return {"valid": False, "error": "No path provided"}
        if not os.path.isfile(path):
            return {"valid": False, "error": f"File not found: {path}"}
        if not path.endswith(".jsonl"):
            return {"valid": False, "error": "File must have .jsonl extension"}

        # Quick validation: try reading first line
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    obj = json.loads(first_line)
                    if "content" not in obj:
                        return {"valid": False, "error": "First entry missing 'content' field"}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return {"valid": False, "error": f"Invalid JSONL: {e}"}

        return {"valid": True}

    def read(self, config: Dict[str, Any]) -> Iterator[IngestCandidate]:
        """Yield IngestCandidates from a JSONL file.

        Handles both canonical store format (with _seq/_type/_op headers)
        and plain export format.
        """
        path = config["path"]
        skip_archived = config.get("skip_archived", True)
        skip_superseded = config.get("skip_superseded", True)

        line_num = 0
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line_num += 1
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    yield IngestCandidate(
                        source_ref=f"file://{os.path.abspath(path)}#L{line_num}",
                        raw_content="",
                        content_type=IngestContentType.OTHER,
                        enrichment_hint=EnrichmentHint.SKIP,
                        confidence=0.0,
                        source_id=self.source_id,
                        metadata={"reader_error": f"JSON parse error at line {line_num}: {e}"},
                        dedup_key=f"jsonl:{path}:L{line_num}:error",
                    )
                    continue

                # Skip non-entry records in canonical store format
                record_type = obj.get("_type", "entry")
                if record_type != "entry":
                    continue

                # Skip delete operations
                if obj.get("_op") == "delete":
                    continue

                content = obj.get("content", "").strip()
                if not content:
                    continue

                # Skip archived entries
                if skip_archived and obj.get("archived_at"):
                    continue

                # Skip superseded entries
                if skip_superseded and obj.get("superseded_by"):
                    continue

                # Map category to content type
                category = obj.get("category", "note")
                content_type = _CATEGORY_MAP.get(category, IngestContentType.OTHER)

                # Entries from another Librarian are already enriched.
                # Use PARTIAL hint: they have structure but may lack KG
                # integration in the new instance.
                enrichment_hint = EnrichmentHint.PARTIAL

                # Higher confidence for structured entries
                confidence = 0.9 if obj.get("tags") else 0.7

                # Parse timestamp
                timestamp = _parse_timestamp(obj.get("created_at"))

                # Build tags: preserve original tags + add provenance
                tags = list(obj.get("tags", []))
                if category:
                    tags.append(f"original-category:{category}")

                # Source ref: original entry ID + file location
                entry_id = obj.get("id", "")
                source_ref = f"file://{os.path.abspath(path)}#L{line_num}"
                if entry_id:
                    source_ref = f"librarian-entry://{entry_id}"

                # Dedup key: original ID is best (globally unique UUID)
                # Fall back to content hash if no ID
                if entry_id:
                    dedup_key = f"jsonl:{entry_id}"
                else:
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
                    dedup_key = f"jsonl:{path}:{content_hash}"

                # Preserve rich metadata
                metadata = {
                    "original_id": entry_id,
                    "original_category": category,
                    "original_content_type": obj.get("content_type", ""),
                    "original_provenance": obj.get("provenance", ""),
                    "original_source_type": obj.get("source_type", ""),
                    "original_conversation_id": obj.get("conversation_id", ""),
                    "original_tier": obj.get("tier", ""),
                    "original_access_count": obj.get("access_count", 0),
                    "line_number": line_num,
                }

                # Carry over the original metadata bag if present
                orig_meta = obj.get("metadata")
                if isinstance(orig_meta, dict):
                    metadata["original_metadata"] = orig_meta

                yield IngestCandidate(
                    source_ref=source_ref,
                    raw_content=content,
                    content_type=content_type,
                    enrichment_hint=enrichment_hint,
                    confidence=confidence,
                    source_id=self.source_id,
                    timestamp=timestamp,
                    metadata=metadata,
                    tags=tags,
                    dedup_key=dedup_key,
                )
