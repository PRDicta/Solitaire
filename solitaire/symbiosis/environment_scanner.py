"""
Environment scanner for Smart Capture onboarding.

Probes the local filesystem for known memory system signatures.
Returns a ranked list of detected sources with size estimates.

Design:
- Reads only filesystem metadata (existence, size, mtime) during detection.
  No file contents are read until the user consents to ingestion.
- Probes run in priority order, fast filesystem checks only.
- Each probe returns a DetectedSource or None.
"""

import os
import json
import time
import logging
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DetectedSource:
    """A memory source found during environment scanning."""
    source_id: str              # Reader type (e.g., "auto-memory", "claude-md")
    display_name: str           # Human-readable ("Claude Code memory", "ChatGPT export")
    path: str                   # Filesystem path
    entry_count_estimate: int   # Rough count of ingestible entries
    size_bytes: int             # Total size on disk
    age_range: Optional[Tuple[datetime, datetime]] = None  # (oldest, newest)
    confidence: float = 0.0     # 0.0-1.0, how sure we are
    reader_available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "source_id": self.source_id,
            "display_name": self.display_name,
            "path": self.path,
            "entry_count_estimate": self.entry_count_estimate,
            "size_bytes": self.size_bytes,
            "confidence": self.confidence,
            "reader_available": self.reader_available,
        }
        if self.age_range:
            d["oldest"] = self.age_range[0].isoformat()
            d["newest"] = self.age_range[1].isoformat()
            delta = self.age_range[1] - self.age_range[0]
            d["age_days"] = delta.days
        return d

    @property
    def size_description(self) -> str:
        """Human-readable size string."""
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        elif self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.0f} KB"
        else:
            return f"{self.size_bytes / (1024 * 1024):.1f} MB"

    @property
    def age_description(self) -> str:
        """Human-readable age string (e.g., '3 months', 'over a year')."""
        if not self.age_range:
            return "unknown timespan"
        delta = datetime.now(timezone.utc) - self.age_range[0]
        days = delta.days
        if days < 7:
            return "a few days"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''}"
        elif days < 365:
            months = days // 30
            return f"{months} month{'s' if months != 1 else ''}"
        else:
            years = days // 365
            return f"over {years} year{'s' if years != 1 else ''}"


@dataclass
class ScanResult:
    """Aggregated result from a full environment scan."""
    sources: List[DetectedSource] = field(default_factory=list)
    total_size_bytes: int = 0
    total_entry_estimate: int = 0
    scan_duration_ms: float = 0.0
    llm_detected: Optional[str] = None
    memory_system_detected: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sources": [s.to_dict() for s in self.sources],
            "total_size_bytes": self.total_size_bytes,
            "total_entry_estimate": self.total_entry_estimate,
            "scan_duration_ms": round(self.scan_duration_ms, 1),
            "llm_detected": self.llm_detected,
            "memory_system_detected": self.memory_system_detected,
            "source_count": len(self.sources),
        }

    @property
    def has_sources(self) -> bool:
        return len(self.sources) > 0

    @property
    def combined_age_description(self) -> str:
        """Oldest age across all detected sources."""
        oldest = None
        for s in self.sources:
            if s.age_range and (oldest is None or s.age_range[0] < oldest):
                oldest = s.age_range[0]
        if not oldest:
            return "some"
        delta = datetime.now(timezone.utc) - oldest
        days = delta.days
        if days < 7:
            return "a few days"
        elif days < 30:
            weeks = days // 7
            return f"about {weeks} week{'s' if weeks != 1 else ''}"
        elif days < 365:
            months = days // 30
            return f"about {months} month{'s' if months != 1 else ''}"
        else:
            years = days // 365
            return f"over {years} year{'s' if years != 1 else ''}"

    @property
    def total_size_description(self) -> str:
        if self.total_size_bytes < 1024:
            return f"{self.total_size_bytes} B"
        elif self.total_size_bytes < 1024 * 1024:
            return f"{self.total_size_bytes / 1024:.0f} KB"
        else:
            return f"{self.total_size_bytes / (1024 * 1024):.1f} MB"


# ─── Default scan paths ────────────────────────────────────────────────────

def _default_scan_paths(workspace: Optional[str] = None) -> List[Path]:
    """Build the default list of paths to scan.

    Args:
        workspace: The Solitaire workspace directory. If None, uses cwd.
    """
    cwd = Path(workspace) if workspace else Path.cwd()
    home = Path.home()

    paths = [
        cwd,
        home / ".claude",
        home / ".claude" / "projects",
        cwd / ".auto-memory",
        cwd / "memory",
    ]
    return [p for p in paths if p.exists()]


# ─── Individual probes ──────────────────────────────────────────────────────

def _dir_size_and_mtimes(dirpath: Path, extensions: Optional[set] = None) -> Tuple[int, List[float]]:
    """Get total file size and modification times for a directory.

    If extensions is provided, only count files with those suffixes.
    """
    total = 0
    mtimes = []
    try:
        for entry in os.scandir(dirpath):
            if not entry.is_file():
                continue
            if extensions and not any(entry.name.endswith(ext) for ext in extensions):
                continue
            try:
                stat = entry.stat()
                total += stat.st_size
                mtimes.append(stat.st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return total, mtimes


def _mtime_range(mtimes: List[float]) -> Optional[Tuple[datetime, datetime]]:
    """Convert a list of mtimes to (oldest, newest) datetime tuple."""
    if not mtimes:
        return None
    return (
        datetime.fromtimestamp(min(mtimes), tz=timezone.utc),
        datetime.fromtimestamp(max(mtimes), tz=timezone.utc),
    )


def _probe_claude_code(scan_paths: List[Path]) -> Optional[DetectedSource]:
    """Detect Claude Code's .claude directory with project memory."""
    for base in scan_paths:
        claude_dir = base / ".claude" if base.name != ".claude" else base
        if not claude_dir.is_dir():
            continue

        # Check for project memory directories
        projects_dir = claude_dir / "projects"
        if not projects_dir.is_dir():
            continue

        # Count memory files across project directories
        total_files = 0
        total_size = 0
        all_mtimes = []

        try:
            for project_entry in os.scandir(projects_dir):
                if not project_entry.is_dir():
                    continue
                memory_dir = Path(project_entry.path) / "memory"
                if not memory_dir.is_dir():
                    continue
                size, mtimes = _dir_size_and_mtimes(memory_dir, {".md"})
                # Subtract 1 for MEMORY.md index file if present
                md_files = [f for f in os.listdir(memory_dir)
                            if f.endswith(".md") and f != "MEMORY.md"]
                total_files += len(md_files)
                total_size += size
                all_mtimes.extend(mtimes)
        except OSError:
            continue

        if total_files > 0:
            return DetectedSource(
                source_id="auto-memory",
                display_name="Claude Code memory",
                path=str(projects_dir),
                entry_count_estimate=total_files,
                size_bytes=total_size,
                age_range=_mtime_range(all_mtimes),
                confidence=0.9,
                reader_available=True,
            )

    return None


def _probe_auto_memory(scan_paths: List[Path]) -> Optional[DetectedSource]:
    """Detect .auto-memory or memory/ directories with markdown files."""
    for base in scan_paths:
        for dirname in [".auto-memory", "memory"]:
            mem_dir = base / dirname if base.name != dirname else base
            if not mem_dir.is_dir():
                continue

            md_files = [f for f in os.listdir(mem_dir)
                        if f.endswith(".md") and f != "MEMORY.md"]
            if not md_files:
                continue

            size, mtimes = _dir_size_and_mtimes(mem_dir, {".md"})

            # Check for YAML frontmatter in first file as confidence signal
            confidence = 0.7
            try:
                first_file = mem_dir / md_files[0]
                with open(first_file, "r", encoding="utf-8") as f:
                    header = f.read(50)
                if header.strip().startswith("---"):
                    confidence = 0.9
            except (OSError, UnicodeDecodeError):
                pass

            return DetectedSource(
                source_id="auto-memory",
                display_name="Auto-memory files",
                path=str(mem_dir),
                entry_count_estimate=len(md_files),
                size_bytes=size,
                age_range=_mtime_range(mtimes),
                confidence=confidence,
                reader_available=True,
            )

    return None


def _probe_claude_md(scan_paths: List[Path]) -> Optional[DetectedSource]:
    """Detect standalone CLAUDE.md or INSTRUCTIONS.md files."""
    for base in scan_paths:
        for name in ["CLAUDE.md", "INSTRUCTIONS.md"]:
            filepath = base / name
            if not filepath.is_file():
                continue

            try:
                stat = filepath.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                return DetectedSource(
                    source_id="claude-md",
                    display_name=f"{name} instructions",
                    path=str(filepath),
                    entry_count_estimate=1,
                    size_bytes=stat.st_size,
                    age_range=(mtime, mtime),
                    confidence=0.95,
                    reader_available=True,
                )
            except OSError:
                continue

    return None


def _probe_solitaire_instance(scan_paths: List[Path], own_db: Optional[str] = None) -> Optional[DetectedSource]:
    """Detect another Solitaire instance's rolodex.db."""
    for base in scan_paths:
        db_path = base / "rolodex.db"
        if not db_path.is_file():
            continue

        # Skip our own database
        if own_db and os.path.abspath(str(db_path)) == os.path.abspath(own_db):
            continue

        try:
            stat = db_path.stat()
            # Quick schema check: open read-only and look for rolodex_entries
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                count = conn.execute("SELECT COUNT(*) FROM rolodex_entries").fetchone()[0]
                # Get date range
                row = conn.execute(
                    "SELECT MIN(created_at), MAX(created_at) FROM rolodex_entries"
                ).fetchone()
                age_range = None
                if row and row[0] and row[1]:
                    try:
                        oldest = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
                        newest = datetime.fromisoformat(row[1].replace('Z', '+00:00'))
                        if oldest.tzinfo is None:
                            oldest = oldest.replace(tzinfo=timezone.utc)
                        if newest.tzinfo is None:
                            newest = newest.replace(tzinfo=timezone.utc)
                        age_range = (oldest, newest)
                    except (ValueError, TypeError):
                        pass
            finally:
                conn.close()

            return DetectedSource(
                source_id="solitaire-instance",
                display_name="Solitaire instance",
                path=str(db_path),
                entry_count_estimate=count,
                size_bytes=stat.st_size,
                age_range=age_range,
                confidence=0.95,
                reader_available=True,
            )
        except (sqlite3.Error, OSError):
            continue

    return None


def _probe_chatgpt_export(scan_paths: List[Path]) -> Optional[DetectedSource]:
    """Detect ChatGPT conversation export files."""
    for base in scan_paths:
        json_path = base / "conversations.json"
        if not json_path.is_file():
            continue

        try:
            stat = json_path.stat()
            # Quick validation: check first bytes for JSON array
            with open(json_path, "r", encoding="utf-8") as f:
                header = f.read(200)

            if not header.strip().startswith("["):
                continue

            # Try to count conversations without loading entire file
            # For large files, estimate from file size
            if stat.st_size > 10 * 1024 * 1024:  # > 10 MB
                # Rough estimate: ~10KB per conversation average
                count_estimate = stat.st_size // (10 * 1024)
            else:
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0 and "mapping" in data[0]:
                        count_estimate = len(data)
                    else:
                        continue
                except (json.JSONDecodeError, MemoryError):
                    count_estimate = stat.st_size // (10 * 1024)

            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            return DetectedSource(
                source_id="chatgpt-export",
                display_name="ChatGPT export",
                path=str(json_path),
                entry_count_estimate=count_estimate,
                size_bytes=stat.st_size,
                age_range=(mtime, mtime),  # Can't know convo dates without reading
                confidence=0.85,
                reader_available=True,
            )
        except (OSError, UnicodeDecodeError):
            continue

    return None


def _probe_markdown_kb(scan_paths: List[Path]) -> Optional[DetectedSource]:
    """Detect generic markdown knowledge bases (10+ .md files in a directory)."""
    for base in scan_paths:
        if not base.is_dir():
            continue

        md_files = []
        try:
            for entry in os.scandir(base):
                if entry.is_file() and entry.name.endswith(".md"):
                    md_files.append(entry)
        except OSError:
            continue

        if len(md_files) < 10:
            continue

        total_size = 0
        mtimes = []
        for entry in md_files:
            try:
                stat = entry.stat()
                total_size += stat.st_size
                mtimes.append(stat.st_mtime)
            except OSError:
                continue

        return DetectedSource(
            source_id="markdown-kb",
            display_name=f"Markdown files in {base.name}",
            path=str(base),
            entry_count_estimate=len(md_files),
            size_bytes=total_size,
            age_range=_mtime_range(mtimes),
            confidence=0.6,
            reader_available=True,
        )

    return None


def _probe_llm_config(scan_paths: List[Path]) -> Optional[str]:
    """Detect LLM configuration (model detection only, no ingestion).

    Returns a string identifying the detected LLM, or None.
    Does NOT read file contents for security. Only checks file existence.
    """
    for base in scan_paths:
        # Claude Code indicators
        claude_dir = base / ".claude" if base.name != ".claude" else base
        if claude_dir.is_dir():
            settings = claude_dir / "settings.json"
            if settings.is_file():
                return "claude-code"

        # Common LLM config files (existence check only)
        if (base / ".openai").is_dir() or (base / ".openai_api_key").is_file():
            return "openai"

    return None


# ─── Main scanner ───────────────────────────────────────────────────────────

def scan_environment(
    workspace: Optional[str] = None,
    extra_paths: Optional[List[str]] = None,
    own_db: Optional[str] = None,
) -> ScanResult:
    """Run a full environment scan for existing memory sources.

    Args:
        workspace: Solitaire workspace directory. Defaults to cwd.
        extra_paths: Additional paths to scan beyond defaults.
        own_db: Path to this Solitaire instance's rolodex.db (excluded from detection).

    Returns:
        ScanResult with all detected sources.
    """
    start = time.time()
    result = ScanResult()

    scan_paths = _default_scan_paths(workspace)
    if extra_paths:
        for p in extra_paths:
            path = Path(p)
            if path.exists() and path not in scan_paths:
                scan_paths.append(path)

    # Run probes in priority order
    probes = [
        _probe_claude_code,
        lambda paths: _probe_auto_memory(paths),
        _probe_claude_md,
        lambda paths: _probe_solitaire_instance(paths, own_db),
        _probe_chatgpt_export,
        _probe_markdown_kb,
    ]

    seen_paths = set()
    for probe in probes:
        try:
            source = probe(scan_paths)
            if source and source.path not in seen_paths:
                result.sources.append(source)
                seen_paths.add(source.path)
        except Exception as e:
            logger.debug(f"Probe failed: {e}")

    # LLM detection (informational only)
    result.llm_detected = _probe_llm_config(scan_paths)

    # Detect memory system type from sources
    if any(s.source_id == "auto-memory" for s in result.sources):
        result.memory_system_detected = "auto-memory"
    elif any(s.source_id == "solitaire-instance" for s in result.sources):
        result.memory_system_detected = "solitaire"
    elif any(s.source_id == "chatgpt-export" for s in result.sources):
        result.memory_system_detected = "chatgpt"

    # Aggregate totals
    result.total_size_bytes = sum(s.size_bytes for s in result.sources)
    result.total_entry_estimate = sum(s.entry_count_estimate for s in result.sources)
    result.scan_duration_ms = (time.time() - start) * 1000

    return result
