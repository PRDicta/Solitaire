"""
Fabric Layer Enrichment C: Front-Loaded Indexed Knowledge

Selects the single most relevant indexed knowledge pack at boot
and loads it into the preamble alongside resident knowledge.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from solitaire.core.types import estimate_tokens

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


# ─── Data Structures ─────────────────────────────────────────────────────────

class IndexedPackInfo:
    """Metadata about an indexed knowledge pack."""
    def __init__(
        self,
        name: str,
        path: str,
        domain: str = "",
        description: str = "",
        keywords: List[str] = None,
        total_tokens: int = 0,
    ):
        self.name = name
        self.path = path
        self.domain = domain
        self.description = description
        self.keywords = keywords or []
        self.total_tokens = total_tokens


# ─── Pack Discovery ──────────────────────────────────────────────────────────

def discover_indexed_packs(indexed_path: str) -> List[IndexedPackInfo]:
    """
    Discover all indexed knowledge packs in the persona's indexed/ directory.

    Each pack is a subdirectory with a manifest.yaml and content files.
    """
    if not indexed_path or not os.path.isdir(indexed_path):
        return []

    packs = []
    for entry in os.listdir(indexed_path):
        pack_dir = os.path.join(indexed_path, entry)
        if not os.path.isdir(pack_dir):
            continue

        manifest_path = os.path.join(pack_dir, "manifest.yaml")
        name = entry
        domain = ""
        description = ""
        keywords = []

        if os.path.isfile(manifest_path) and yaml:
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = yaml.safe_load(f) or {}
                name = manifest.get("name", entry)
                domain = manifest.get("domain", "")
                description = manifest.get("description", "")
                # Extract keywords from manifest fields
                keywords = _extract_keywords_from_manifest(manifest)
            except Exception:
                pass

        # Estimate total tokens
        total_tokens = _estimate_pack_tokens(pack_dir)

        packs.append(IndexedPackInfo(
            name=name,
            path=pack_dir,
            domain=domain,
            description=description,
            keywords=keywords,
            total_tokens=total_tokens,
        ))

    return packs


# ─── Pack Selection ──────────────────────────────────────────────────────────

def select_best_pack(
    packs: List[IndexedPackInfo],
    intent_text: str = "",
    briefing_text: str = "",
    budget_tokens: int = 3000,
) -> Optional[IndexedPackInfo]:
    """
    Select the single most relevant indexed pack based on available signals.

    Priority:
    1. Intent signal (highest confidence)
    2. Briefing text (work stream alignment)
    3. Largest pack (fallback — most general knowledge)

    Returns None if no packs exist or none fit the budget.
    """
    if not packs:
        return None

    # Filter to packs that fit the budget
    eligible = [p for p in packs if p.total_tokens <= budget_tokens]
    if not eligible:
        # Try loading partial content from the smallest pack
        eligible = sorted(packs, key=lambda p: p.total_tokens)[:1]

    if not eligible:
        return None

    # Score each pack
    scored: List[Tuple[float, IndexedPackInfo]] = []
    for pack in eligible:
        score = _score_pack(pack, intent_text, briefing_text)
        scored.append((score, pack))

    scored.sort(key=lambda x: -x[0])

    # Return best match, but only if it has a non-trivial score
    # (or if there's no signal at all, return the first one as fallback)
    best_score, best_pack = scored[0]
    if best_score > 0 or (not intent_text and not briefing_text):
        return best_pack

    return None


def _score_pack(
    pack: IndexedPackInfo,
    intent_text: str,
    briefing_text: str,
) -> float:
    """Score a pack against available signals."""
    score = 0.0
    pack_words = set(pack.keywords)
    pack_words.update(_tokenize(pack.name))
    pack_words.update(_tokenize(pack.domain))
    pack_words.update(_tokenize(pack.description))

    # Score against intent (weight: 2.0)
    if intent_text:
        intent_words = set(_tokenize(intent_text))
        overlap = pack_words & intent_words
        if overlap:
            score += len(overlap) * 2.0

    # Score against briefing (weight: 1.0)
    if briefing_text:
        briefing_words = set(_tokenize(briefing_text))
        overlap = pack_words & briefing_words
        if overlap:
            score += len(overlap) * 1.0

    return score


# ─── Pack Loading ────────────────────────────────────────────────────────────

def load_pack_content(
    pack: IndexedPackInfo,
    budget_tokens: int = 3000,
) -> str:
    """
    Load the content files from a pack within the token budget.

    Skips manifest.yaml. Loads files in sorted order (priority prefixed).
    """
    if not os.path.isdir(pack.path):
        return ""

    loadable_ext = {".md", ".txt", ".yaml", ".yml", ".json"}
    candidates = []
    for fname in os.listdir(pack.path):
        if fname == "manifest.yaml":
            continue
        fpath = os.path.join(pack.path, fname)
        if not os.path.isfile(fpath):
            continue
        ext = Path(fname).suffix.lower()
        if ext not in loadable_ext:
            continue
        candidates.append((fname, fpath))

    candidates.sort(key=lambda x: x[0])

    parts = [f"── ACTIVE DOMAIN: {pack.name} ──"]
    tokens_used = estimate_tokens(parts[0])

    for fname, fpath in candidates:
        try:
            content = Path(fpath).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        file_tokens = estimate_tokens(content)
        if tokens_used + file_tokens > budget_tokens:
            # Try to fit a truncated version
            remaining = budget_tokens - tokens_used - 10
            if remaining > 200:
                char_limit = remaining * 4
                content = content[:char_limit] + "\n[truncated]"
                file_tokens = estimate_tokens(content)
            else:
                break

        parts.append(content.strip())
        parts.append("")
        tokens_used += file_tokens

    if len(parts) <= 1:
        return ""

    return "\n".join(parts)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_keywords_from_manifest(manifest: dict) -> List[str]:
    """Extract searchable keywords from all manifest text fields."""
    keywords = []
    for field in ["name", "domain", "description"]:
        val = manifest.get(field, "")
        if val:
            keywords.extend(_tokenize(val))

    # Also check for explicit keywords/tags fields
    for field in ["keywords", "tags", "topics"]:
        val = manifest.get(field, [])
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    keywords.extend(_tokenize(item))

    return list(set(keywords))


def _tokenize(text: str) -> List[str]:
    """Simple word tokenization for keyword matching."""
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    # Remove very common words
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "are",
        "was", "were", "been", "have", "has", "had", "will", "can",
        "not", "but", "all", "any", "each", "every", "more", "most",
        "other", "some", "such", "than", "too", "very", "also",
    }
    return [w for w in words if w not in stopwords]


def _estimate_pack_tokens(pack_dir: str) -> int:
    """Estimate total tokens in a pack directory."""
    total_chars = 0
    for fname in os.listdir(pack_dir):
        if fname == "manifest.yaml":
            continue
        fpath = os.path.join(pack_dir, fname)
        if os.path.isfile(fpath):
            try:
                total_chars += os.path.getsize(fpath)
            except OSError:
                pass
    return total_chars // 4  # Rough estimate
