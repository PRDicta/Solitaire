"""
The Librarian — Resident Knowledge Loader (Tier 1)
Loads persona-scoped knowledge files at boot time within a token budget.
Files in the resident/ directory are always in context from the first message.
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from solitaire.core.types import estimate_tokens


# Files with these extensions are loaded
LOADABLE_EXTENSIONS = {".md", ".yaml", ".yml", ".txt"}

# Filename prefix pattern for priority ordering (e.g., 01_, 02_)
PRIORITY_RE = re.compile(r"^(\d+)[_\-]")


@dataclass
class ResidentKnowledgeEntry:
    """A single resident knowledge file loaded into context."""
    source_file: str      # Filename relative to resident/
    content: str           # Raw file content
    token_count: int       # Estimated token count
    priority: int          # Sort order (from filename prefix, lower = first)
    format: str            # "markdown" | "yaml" | "text"


@dataclass
class ResidentLoadResult:
    """Result of loading resident knowledge for a persona."""
    entries: List[ResidentKnowledgeEntry] = field(default_factory=list)
    tokens_used: int = 0
    budget_total: int = 4000
    files_skipped: int = 0          # Files that didn't fit in budget
    skipped_files: List[str] = field(default_factory=list)
    load_time_estimate_ms: float = 0.0
    is_heavy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries_loaded": len(self.entries),
            "tokens_used": self.tokens_used,
            "budget_total": self.budget_total,
            "budget_remaining": self.budget_total - self.tokens_used,
            "files_skipped": self.files_skipped,
            "load_time_estimate_ms": round(self.load_time_estimate_ms, 1),
            "is_heavy": self.is_heavy,
        }


def _get_priority(filename: str) -> int:
    """Extract numeric priority from filename prefix. Lower = loaded first."""
    m = PRIORITY_RE.match(filename)
    return int(m.group(1)) if m else 999


def _get_format(filename: str) -> str:
    """Determine content format from file extension."""
    ext = Path(filename).suffix.lower()
    if ext in (".yaml", ".yml"):
        return "yaml"
    if ext == ".md":
        return "markdown"
    return "text"


def load_resident_knowledge(
    resident_path: str,
    budget_tokens: int = 4000,
) -> ResidentLoadResult:
    """Load all knowledge files from a persona's resident/ directory.

    Files are sorted by filename prefix (01_foo.md before 02_bar.md).
    Loading stops when the next file would exceed the token budget.

    Args:
        resident_path: Absolute path to the resident/ directory
        budget_tokens: Maximum tokens to load

    Returns:
        ResidentLoadResult with entries and metadata
    """
    result = ResidentLoadResult(budget_total=budget_tokens)

    if not resident_path or not os.path.isdir(resident_path):
        return result

    # Collect loadable files
    candidates = []
    for fname in os.listdir(resident_path):
        fpath = os.path.join(resident_path, fname)
        if not os.path.isfile(fpath):
            continue
        ext = Path(fname).suffix.lower()
        if ext not in LOADABLE_EXTENSIONS:
            continue
        candidates.append((fname, fpath))

    # Sort by priority prefix
    candidates.sort(key=lambda x: (_get_priority(x[0]), x[0]))

    # Load within budget
    tokens_used = 0
    for fname, fpath in candidates:
        try:
            content = Path(fpath).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        token_count = estimate_tokens(content)

        # Budget check: skip if adding this file would exceed budget
        if tokens_used + token_count > budget_tokens:
            result.files_skipped += 1
            result.skipped_files.append(fname)
            continue

        entry = ResidentKnowledgeEntry(
            source_file=fname,
            content=content,
            token_count=token_count,
            priority=_get_priority(fname),
            format=_get_format(fname),
        )
        result.entries.append(entry)
        tokens_used += token_count

    result.tokens_used = tokens_used

    # Estimate load time (heuristic: local reads are fast, but budget matters for context)
    result.load_time_estimate_ms = len(result.entries) * 5 + tokens_used * 0.01
    result.is_heavy = (
        tokens_used > budget_tokens * 0.80
        or result.load_time_estimate_ms > 500
    )

    return result


def render_resident_block(result: ResidentLoadResult) -> str:
    """Render loaded resident knowledge as a context block for boot injection.

    Returns empty string if no entries were loaded.
    """
    if not result.entries:
        return ""

    parts = ["═══ RESIDENT KNOWLEDGE ═══\n"]
    for entry in result.entries:
        parts.append(f"── {entry.source_file} ({entry.token_count} tokens) ──")
        parts.append(entry.content.strip())
        parts.append("")  # blank line between entries

    parts.append("═══ END RESIDENT KNOWLEDGE ═══")
    return "\n".join(parts)
