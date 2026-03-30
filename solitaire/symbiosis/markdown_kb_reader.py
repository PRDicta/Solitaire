"""
Reader for generic markdown knowledge bases.

Handles directories containing plain .md files without any particular
frontmatter convention. Each file becomes one IngestCandidate. Large
files (> 4KB) are split into sections by headers, similar to the
ClaudeMdReader approach.

This is the catch-all reader for users who keep notes, documentation,
or knowledge in markdown files that don't follow any specific format.
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


# Files to skip (common non-content markdown files)
_SKIP_FILES = {
    "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE.md",
    "CODE_OF_CONDUCT.md", "SECURITY.md", "MEMORY.md",
}

# Size threshold for splitting into sections
_SPLIT_THRESHOLD = 4096  # 4 KB


def _split_by_headers(text: str) -> List[Tuple[str, str]]:
    """Split markdown into (header, body) pairs by headers."""
    sections = []
    current_header = ""
    current_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r'^#{1,3}\s+', stripped):
            if current_lines or current_header:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_header, body))
            current_header = stripped.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or current_header:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_header, body))

    return sections


class MarkdownKBReader(ReaderBase):
    """Reads directories of plain markdown files.

    Config:
        path: str - Path to the directory containing .md files.
        skip_files: list[str] - Additional filenames to skip (optional).
        min_files: int - Minimum .md file count to consider valid (default 1).
    """

    @property
    def source_id(self) -> str:
        return "markdown-kb"

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        path = config.get("path", "")
        if not path:
            return {"valid": False, "error": "No path provided"}
        if not os.path.isdir(path):
            return {"valid": False, "error": f"Directory not found: {path}"}

        min_files = config.get("min_files", 1)
        skip = _SKIP_FILES | set(config.get("skip_files", []))

        md_files = [
            f for f in os.listdir(path)
            if f.endswith(".md") and f not in skip
        ]

        if len(md_files) < min_files:
            return {
                "valid": False,
                "error": f"Found {len(md_files)} .md files, need at least {min_files}",
            }

        return {"valid": True, "file_count": len(md_files)}

    def read(self, config: Dict[str, Any]) -> Iterator[IngestCandidate]:
        """Yield IngestCandidates from a directory of markdown files.

        Small files yield one candidate each. Large files are split
        by headers into multiple candidates.
        """
        path = config["path"]
        dir_path = Path(path)
        skip = _SKIP_FILES | set(config.get("skip_files", []))

        md_files = sorted([
            f for f in os.listdir(path)
            if f.endswith(".md") and f not in skip
        ])

        for filename in md_files:
            file_path = dir_path / filename

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
                    metadata={"reader_error": str(e), "file": str(file_path)},
                    dedup_key=f"markdown-kb:{file_path}:error",
                )
                continue

            if not text.strip():
                continue

            # Get file timestamp
            try:
                mtime = os.path.getmtime(file_path)
                timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc)
            except Exception:
                timestamp = None

            # For large files, split into sections
            if len(text.encode("utf-8")) > _SPLIT_THRESHOLD:
                sections = _split_by_headers(text)
                if len(sections) > 1:
                    yield from self._yield_sections(
                        file_path, filename, sections, timestamp
                    )
                    continue

            # Small file or no headers: yield as single entry
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            yield IngestCandidate(
                source_ref=f"file://{file_path.resolve()}",
                raw_content=f"{filename}\n\n{text.strip()}",
                content_type=IngestContentType.DOCUMENT,
                enrichment_hint=EnrichmentHint.FULL,
                confidence=0.5,  # Unstructured, needs full enrichment
                source_id=self.source_id,
                timestamp=timestamp,
                metadata={
                    "file": str(file_path),
                    "filename": filename,
                },
                tags=[f"file:{filename}", "source:markdown-kb"],
                dedup_key=f"markdown-kb:{file_path}:{content_hash}",
            )

    def _yield_sections(
        self,
        file_path: Path,
        filename: str,
        sections: List[Tuple[str, str]],
        timestamp,
    ) -> Iterator[IngestCandidate]:
        """Yield one candidate per section of a large markdown file."""
        for idx, (header, body) in enumerate(sections):
            if header:
                full_content = f"{filename}: {header}\n\n{body}"
            else:
                full_content = f"{filename}\n\n{body}"

            content_hash = hashlib.sha256(full_content.encode("utf-8")).hexdigest()[:16]

            tags = [f"file:{filename}", "source:markdown-kb"]
            if header:
                tags.append(f"section:{header[:50]}")

            yield IngestCandidate(
                source_ref=f"file://{file_path.resolve()}#section-{idx}",
                raw_content=full_content,
                content_type=IngestContentType.DOCUMENT,
                enrichment_hint=EnrichmentHint.FULL,
                confidence=0.5,
                source_id=self.source_id,
                timestamp=timestamp,
                metadata={
                    "file": str(file_path),
                    "filename": filename,
                    "section_index": idx,
                    "section_header": header,
                    "total_sections": len(sections),
                },
                tags=tags,
                dedup_key=f"markdown-kb:{file_path}:{idx}:{content_hash}",
            )
