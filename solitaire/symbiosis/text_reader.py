"""
Reader for plain text and markdown files.

Handles individual text files or directories of text files.
Each file becomes one IngestCandidate. For large files, the reader
chunks by paragraph breaks to keep individual entries at a
reasonable size for the enrichment pipeline.

Supported extensions: .txt, .md, .markdown, .text, .rst
"""

import os
import hashlib
from pathlib import Path
from typing import Iterator, Dict, Any, List, Set
from datetime import datetime, timezone

from .reader_base import ReaderBase
from ..core.types import (
    IngestCandidate,
    IngestContentType,
    EnrichmentHint,
)


_TEXT_EXTENSIONS: Set[str] = {".txt", ".md", ".markdown", ".text", ".rst"}

# Rough token estimate: 1 token per 4 chars
_MAX_CHUNK_CHARS = 8000  # ~2000 tokens per chunk


def _chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> List[str]:
    """Split text into chunks at paragraph boundaries.

    Tries to keep chunks under max_chars. If a single paragraph exceeds
    the limit, it gets its own chunk (not split mid-paragraph).
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        current.append(para)
        current_len += para_len + 2  # +2 for \n\n

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text]


class TextReader(ReaderBase):
    """Reads plain text and markdown files.

    Config:
        path: str - Path to a file or directory.
        recursive: bool - Recurse into subdirectories (default False).
        extensions: list - File extensions to include (default: txt, md, etc).
        max_chunk_chars: int - Split files larger than this (default 8000).
    """

    @property
    def source_id(self) -> str:
        return "text"

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        path = config.get("path", "")
        if not path:
            return {"valid": False, "error": "No path provided"}
        if not os.path.exists(path):
            return {"valid": False, "error": f"Path not found: {path}"}

        extensions = set(config.get("extensions", _TEXT_EXTENSIONS))

        if os.path.isfile(path):
            ext = Path(path).suffix.lower()
            if ext not in extensions:
                return {"valid": False, "error": f"Unsupported extension: {ext}"}
            return {"valid": True, "file_count": 1}

        # Directory: count matching files
        count = 0
        recursive = config.get("recursive", False)
        if recursive:
            for root, _, files in os.walk(path):
                count += sum(1 for f in files if Path(f).suffix.lower() in extensions)
        else:
            count = sum(
                1 for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and Path(f).suffix.lower() in extensions
            )

        if count == 0:
            return {"valid": False, "error": "No matching text files found"}
        return {"valid": True, "file_count": count}

    def read(self, config: Dict[str, Any]) -> Iterator[IngestCandidate]:
        """Yield IngestCandidates from text files."""
        path = config["path"]
        recursive = config.get("recursive", False)
        extensions = set(config.get("extensions", _TEXT_EXTENSIONS))
        max_chars = config.get("max_chunk_chars", _MAX_CHUNK_CHARS)

        if os.path.isfile(path):
            yield from self._read_file(path, max_chars)
        else:
            if recursive:
                for root, _, files in os.walk(path):
                    for fname in sorted(files):
                        if Path(fname).suffix.lower() in extensions:
                            yield from self._read_file(os.path.join(root, fname), max_chars)
            else:
                for fname in sorted(os.listdir(path)):
                    fpath = os.path.join(path, fname)
                    if os.path.isfile(fpath) and Path(fname).suffix.lower() in extensions:
                        yield from self._read_file(fpath, max_chars)

    def _read_file(self, file_path: str, max_chars: int) -> Iterator[IngestCandidate]:
        """Read a single text file, chunking if necessary."""
        try:
            text = Path(file_path).read_text(encoding="utf-8")
        except Exception as e:
            yield IngestCandidate(
                source_ref=f"file://{os.path.abspath(file_path)}",
                raw_content="",
                content_type=IngestContentType.OTHER,
                enrichment_hint=EnrichmentHint.SKIP,
                confidence=0.0,
                source_id=self.source_id,
                metadata={"reader_error": str(e), "file": file_path},
                dedup_key=f"text:{file_path}:error",
            )
            return

        text = text.strip()
        if not text:
            return

        # Determine content type from extension
        ext = Path(file_path).suffix.lower()
        content_type = IngestContentType.DOCUMENT if ext in (".md", ".markdown", ".rst") else IngestContentType.OTHER

        # Get file timestamp
        try:
            mtime = os.path.getmtime(file_path)
            timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc)
        except Exception:
            timestamp = None

        # Chunk if necessary
        chunks = _chunk_text(text, max_chars)

        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            # Header for context
            fname = os.path.basename(file_path)
            if len(chunks) > 1:
                full_content = f"Source: {fname} (part {chunk_idx + 1}/{len(chunks)})\n\n{chunk}"
            else:
                full_content = f"Source: {fname}\n\n{chunk}"

            content_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]
            dedup_key = f"text:{file_path}:{chunk_idx}:{content_hash}"

            yield IngestCandidate(
                source_ref=f"file://{os.path.abspath(file_path)}",
                raw_content=full_content,
                content_type=content_type,
                enrichment_hint=EnrichmentHint.FULL,
                confidence=0.5,
                source_id=self.source_id,
                timestamp=timestamp,
                metadata={
                    "file": file_path,
                    "filename": fname,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "extension": ext,
                },
                tags=[f"filename:{fname}", f"ext:{ext}"],
                dedup_key=dedup_key,
            )
