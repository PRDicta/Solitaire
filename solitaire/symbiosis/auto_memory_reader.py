"""
Reader for Cowork's .auto-memory format.

Format: a directory containing markdown files with YAML frontmatter.
Each file represents one memory entry. MEMORY.md is an index file (skipped).

File structure:
    ---
    name: <memory name>
    description: <one-line description>
    type: user | feedback | project | reference
    ---

    <memory content body>

The reader maps Cowork's type field to IngestContentType:
    user       -> PREFERENCE  (user role, goals, preferences)
    feedback   -> PREFERENCE  (behavioral guidance)
    project    -> FACT        (project state, decisions, context)
    reference  -> DOCUMENT    (pointers to external resources)
"""

import os
import hashlib
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from datetime import datetime, timezone

from .reader_base import ReaderBase
from ..core.types import (
    IngestCandidate,
    IngestContentType,
    EnrichmentHint,
)


# Cowork type -> IngestContentType mapping
_TYPE_MAP: Dict[str, IngestContentType] = {
    "user": IngestContentType.PREFERENCE,
    "feedback": IngestContentType.PREFERENCE,
    "project": IngestContentType.FACT,
    "reference": IngestContentType.DOCUMENT,
}


def _parse_frontmatter(text: str) -> tuple:
    """Parse YAML frontmatter from markdown text.

    Returns (frontmatter_dict, body_text). If no frontmatter is found,
    returns (empty dict, full text).
    """
    stripped = text.strip()
    if not stripped.startswith("---"):
        return {}, text

    # Find closing ---
    end_idx = stripped.find("---", 3)
    if end_idx == -1:
        return {}, text

    frontmatter_raw = stripped[3:end_idx].strip()
    body = stripped[end_idx + 3:].strip()

    # Parse YAML manually (no pyyaml dependency required)
    frontmatter = {}
    for line in frontmatter_raw.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            # Strip quotes
            if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            frontmatter[key] = value

    return frontmatter, body


def _compute_dedup_key(file_path: str, content: str) -> str:
    """Stable dedup key: source path + content hash.

    This means re-running the import on the same .auto-memory directory
    won't create duplicates, but editing a file will re-import it.
    """
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"auto-memory:{file_path}:{content_hash}"


class AutoMemoryReader(ReaderBase):
    """Reads Cowork .auto-memory directories.

    Config:
        path: str - Path to the .auto-memory directory.
    """

    @property
    def source_id(self) -> str:
        return "auto-memory"

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        path = config.get("path", "")
        if not path:
            return {"valid": False, "error": "No path provided"}
        if not os.path.isdir(path):
            return {"valid": False, "error": f"Directory not found: {path}"}

        # Check for at least one .md file
        md_files = [f for f in os.listdir(path) if f.endswith(".md") and f != "MEMORY.md"]
        if not md_files:
            return {"valid": False, "error": "No memory files found (only MEMORY.md index)"}

        return {"valid": True, "file_count": len(md_files)}

    def read(self, config: Dict[str, Any]) -> Iterator[IngestCandidate]:
        """Yield IngestCandidates from .auto-memory markdown files.

        Skips MEMORY.md (index file). Processes all other .md files.
        Never raises on individual file errors; logs to metadata instead.
        """
        path = config["path"]
        dir_path = Path(path)

        for file_path in sorted(dir_path.glob("*.md")):
            # Skip the index file
            if file_path.name == "MEMORY.md":
                continue

            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception as e:
                # Yield an error candidate so the orchestrator can track it
                yield IngestCandidate(
                    source_ref=file_path.as_uri() if hasattr(file_path, 'as_uri') else f"file://{file_path}",
                    raw_content="",
                    content_type=IngestContentType.OTHER,
                    enrichment_hint=EnrichmentHint.SKIP,
                    confidence=0.0,
                    source_id=self.source_id,
                    metadata={"reader_error": str(e), "file": str(file_path)},
                    dedup_key=f"auto-memory:{file_path}:error",
                )
                continue

            if not text.strip():
                continue

            frontmatter, body = _parse_frontmatter(text)

            if not body.strip():
                # Frontmatter-only file with no content. Skip.
                continue

            # Map Cowork type to IngestContentType
            cowork_type = frontmatter.get("type", "").lower()
            content_type = _TYPE_MAP.get(cowork_type, IngestContentType.OTHER)

            # Confidence: structured frontmatter means higher confidence
            confidence = 0.8 if frontmatter else 0.5

            # Enrichment hint: Cowork memories are human-written prose,
            # but they lack entity extraction, KG integration, temporal
            # context. Full enrichment is appropriate.
            enrichment_hint = EnrichmentHint.FULL

            # Build tags from frontmatter
            tags = []
            if cowork_type:
                tags.append(f"cowork-type:{cowork_type}")
            name = frontmatter.get("name", "")
            if name:
                tags.append(f"memory-name:{name}")

            # Use file modification time as timestamp if available
            try:
                mtime = os.path.getmtime(file_path)
                timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc)
            except Exception:
                timestamp = None

            # Build source_ref as URI
            source_ref = f"file://{file_path.resolve()}"

            # Compose the content: include the name/description as context
            # so the enrichment pipeline has the full picture
            content_parts = []
            if name:
                content_parts.append(name)
            description = frontmatter.get("description", "")
            if description:
                content_parts.append(description)
            content_parts.append(body)
            full_content = "\n\n".join(content_parts)

            yield IngestCandidate(
                source_ref=source_ref,
                raw_content=full_content,
                content_type=content_type,
                enrichment_hint=enrichment_hint,
                confidence=confidence,
                source_id=self.source_id,
                timestamp=timestamp,
                metadata={
                    "file": str(file_path),
                    "frontmatter": frontmatter,
                    "cowork_type": cowork_type,
                    "original_name": name,
                    "original_description": description,
                },
                tags=tags,
                dedup_key=_compute_dedup_key(str(file_path), full_content),
            )
