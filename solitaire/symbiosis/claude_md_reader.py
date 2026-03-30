"""
Reader for CLAUDE.md and INSTRUCTIONS.md files.

These files contain user-defined instructions, preferences, project context,
and behavioral guidance. They're high-signal, low-volume: typically one file
per project, dense with preferences and standing instructions.

The reader splits the file into sections (by markdown headers) and yields
each section as a separate IngestCandidate. This gives the enrichment pipeline
finer-grained entries to work with rather than one monolithic blob.

Content mapping:
  - Sections with preference/style/format keywords -> PREFERENCE
  - Sections with project/architecture/setup keywords -> FACT
  - Everything else -> DOCUMENT
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple
from datetime import datetime, timezone

from .reader_base import ReaderBase
from ..core.types import (
    IngestCandidate,
    IngestContentType,
    EnrichmentHint,
)


# Keywords that signal content type
_PREFERENCE_KEYWORDS = {
    "prefer", "style", "format", "tone", "writing", "always", "never",
    "don't", "do not", "avoid", "rule", "standard", "convention",
    "mandatory", "required", "behavioral", "persona",
}

_FACT_KEYWORDS = {
    "project", "architecture", "setup", "structure", "stack", "deploy",
    "database", "api", "endpoint", "schema", "config", "environment",
    "build", "test", "ci", "pipeline",
}


def _classify_section(header: str, body: str) -> IngestContentType:
    """Classify a section based on header and body keywords."""
    text = (header + " " + body[:500]).lower()

    pref_hits = sum(1 for kw in _PREFERENCE_KEYWORDS if kw in text)
    fact_hits = sum(1 for kw in _FACT_KEYWORDS if kw in text)

    if pref_hits >= 2 or pref_hits > fact_hits:
        return IngestContentType.PREFERENCE
    elif fact_hits >= 2:
        return IngestContentType.FACT
    return IngestContentType.DOCUMENT


def _split_sections(text: str) -> List[Tuple[str, str]]:
    """Split markdown text into (header, body) pairs by top-level headers.

    Lines starting with # or ## are treated as section boundaries.
    Content before the first header gets an empty-string header.
    """
    sections = []
    current_header = ""
    current_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r'^#{1,3}\s+', stripped):
            # Save previous section
            if current_lines or current_header:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_header, body))
            current_header = stripped.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last section
    if current_lines or current_header:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_header, body))

    return sections


class ClaudeMdReader(ReaderBase):
    """Reads CLAUDE.md and INSTRUCTIONS.md files.

    Config:
        path: str - Path to the .md file.
    """

    @property
    def source_id(self) -> str:
        return "claude-md"

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        path = config.get("path", "")
        if not path:
            return {"valid": False, "error": "No path provided"}
        if not os.path.isfile(path):
            return {"valid": False, "error": f"File not found: {path}"}

        name = os.path.basename(path)
        if name not in ("CLAUDE.md", "INSTRUCTIONS.md"):
            return {"valid": False, "error": f"Expected CLAUDE.md or INSTRUCTIONS.md, got {name}"}

        try:
            size = os.path.getsize(path)
            if size == 0:
                return {"valid": False, "error": "File is empty"}
        except OSError as e:
            return {"valid": False, "error": str(e)}

        return {"valid": True, "file_size": size}

    def read(self, config: Dict[str, Any]) -> Iterator[IngestCandidate]:
        """Yield IngestCandidates from a CLAUDE.md or INSTRUCTIONS.md file.

        Each markdown section becomes a separate candidate.
        """
        path = config["path"]
        file_path = Path(path)

        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            yield IngestCandidate(
                source_ref=f"file://{file_path.resolve()}",
                raw_content="",
                content_type=IngestContentType.OTHER,
                enrichment_hint=EnrichmentHint.SKIP,
                confidence=0.0,
                source_id=self.source_id,
                metadata={"reader_error": str(e)},
                dedup_key=f"claude-md:{path}:error",
            )
            return

        if not text.strip():
            return

        # Get file timestamp
        try:
            mtime = os.path.getmtime(path)
            timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc)
        except Exception:
            timestamp = None

        sections = _split_sections(text)

        # If the file doesn't split into sections, yield it as one entry
        if not sections:
            sections = [("", text.strip())]

        filename = file_path.name

        for idx, (header, body) in enumerate(sections):
            content_type = _classify_section(header, body)

            # Confidence: CLAUDE.md files are human-written, high-signal
            confidence = 0.85

            # Build content with header context
            if header:
                full_content = f"{filename}: {header}\n\n{body}"
            else:
                full_content = f"{filename}\n\n{body}"

            content_hash = hashlib.sha256(full_content.encode("utf-8")).hexdigest()[:16]
            dedup_key = f"claude-md:{path}:{idx}:{content_hash}"

            tags = [
                f"source-file:{filename}",
                "source:claude-md",
            ]
            if header:
                tags.append(f"section:{header[:50]}")

            yield IngestCandidate(
                source_ref=f"file://{file_path.resolve()}#section-{idx}",
                raw_content=full_content,
                content_type=content_type,
                enrichment_hint=EnrichmentHint.FULL,
                confidence=confidence,
                source_id=self.source_id,
                timestamp=timestamp,
                metadata={
                    "file": str(file_path),
                    "section_index": idx,
                    "section_header": header,
                    "filename": filename,
                },
                tags=tags,
                dedup_key=dedup_key,
            )
