"""
Priority ranker for Smart Capture first-chunk selection.

Determines which entries from a detected source should be ingested first
when the corpus is too large for immediate ingestion. Two ranking paths:

1. LLM classification (primary): One API call reads a batch of candidates
   and returns tier assignments. More accurate for unstructured content.

2. Heuristic scoring (fallback): Fast, no API dependency. Used when LLM
   is unavailable (no API key, rate limited, user opts out).

Priority tiers:
  Tier 1: User identity, preferences, corrections, behavioral guidance
  Tier 2: Strategic decisions, active project context, key relationships
  Tier 3: Conversation history, reference material, transient context

Within each tier, entries sort by recency (newest first).
"""

import logging
from typing import List, Optional, Callable, Iterator
from datetime import datetime, timezone
from dataclasses import dataclass

from ..core.types import IngestCandidate, IngestContentType

logger = logging.getLogger(__name__)


# ─── Size classification ────────────────────────────────────────────────────

class IngestionStrategy:
    IMMEDIATE = "immediate"    # < 2 MB or < 500 entries
    CHUNKED = "chunked"        # 2 MB - 50 MB
    LARGE = "large"            # > 50 MB


@dataclass
class IngestionPlan:
    """Plan for how to ingest a detected source."""
    strategy: str                   # IngestionStrategy value
    first_chunk_entries: int        # How many entries in the first chunk
    first_chunk_size_mb: float      # Estimated size of first chunk
    background_remaining: int       # Entries left for background sync
    estimated_time_seconds: float   # Rough time estimate for first chunk
    total_entries: int = 0
    total_size_bytes: int = 0

    def to_dict(self):
        return {
            "strategy": self.strategy,
            "first_chunk_entries": self.first_chunk_entries,
            "first_chunk_size_mb": round(self.first_chunk_size_mb, 2),
            "background_remaining": self.background_remaining,
            "estimated_time_seconds": round(self.estimated_time_seconds, 1),
            "total_entries": self.total_entries,
            "total_size_bytes": self.total_size_bytes,
        }


# Budget defaults
DEFAULT_FIRST_CHUNK_MB = 10
DEFAULT_FIRST_CHUNK_ENTRIES = 2000

# Size thresholds
IMMEDIATE_SIZE_BYTES = 2 * 1024 * 1024   # 2 MB
IMMEDIATE_ENTRY_COUNT = 500
LARGE_SIZE_BYTES = 50 * 1024 * 1024      # 50 MB


def classify_corpus(
    entry_count: int,
    size_bytes: int,
    first_chunk_mb: float = DEFAULT_FIRST_CHUNK_MB,
    first_chunk_entries: int = DEFAULT_FIRST_CHUNK_ENTRIES,
) -> IngestionPlan:
    """Classify a corpus and produce an ingestion plan.

    Args:
        entry_count: Estimated number of ingestible entries.
        size_bytes: Total size on disk.
        first_chunk_mb: Budget for first chunk in MB.
        first_chunk_entries: Budget for first chunk in entry count.
    """
    if size_bytes <= IMMEDIATE_SIZE_BYTES or entry_count <= IMMEDIATE_ENTRY_COUNT:
        return IngestionPlan(
            strategy=IngestionStrategy.IMMEDIATE,
            first_chunk_entries=entry_count,
            first_chunk_size_mb=size_bytes / (1024 * 1024),
            background_remaining=0,
            estimated_time_seconds=max(1, entry_count * 0.05),
            total_entries=entry_count,
            total_size_bytes=size_bytes,
        )

    # Calculate first chunk size
    chunk_entries = min(entry_count, first_chunk_entries)
    avg_entry_size = size_bytes / max(entry_count, 1)
    chunk_bytes = chunk_entries * avg_entry_size
    chunk_mb = chunk_bytes / (1024 * 1024)

    # Clamp to MB budget
    if chunk_mb > first_chunk_mb:
        chunk_entries = int(first_chunk_mb * 1024 * 1024 / max(avg_entry_size, 1))
        chunk_mb = first_chunk_mb

    remaining = max(0, entry_count - chunk_entries)

    strategy = IngestionStrategy.LARGE if size_bytes > LARGE_SIZE_BYTES else IngestionStrategy.CHUNKED

    return IngestionPlan(
        strategy=strategy,
        first_chunk_entries=chunk_entries,
        first_chunk_size_mb=chunk_mb,
        background_remaining=remaining,
        estimated_time_seconds=max(1, chunk_entries * 0.05),
        total_entries=entry_count,
        total_size_bytes=size_bytes,
    )


# ─── Heuristic scoring (offline fallback) ───────────────────────────────────

def heuristic_priority_score(candidate: IngestCandidate) -> float:
    """Score a candidate for first-chunk selection using heuristics.

    This is the fallback scorer when LLM classification is unavailable.
    Higher score = higher priority for first-chunk inclusion.
    """
    score = 0.0

    # Content type signals
    if candidate.content_type == IngestContentType.PREFERENCE:
        score += 3.0
    elif candidate.content_type == IngestContentType.FACT:
        score += 2.0
    elif candidate.content_type == IngestContentType.DOCUMENT:
        score += 1.0
    # CONVERSATION and OTHER get 0.0 base

    # Tag signals (reader-specific enrichment)
    tags = candidate.tags or []
    tag_str = " ".join(tags).lower()

    if "user" in tag_str:
        score += 2.0
    if "feedback" in tag_str or "correction" in tag_str:
        score += 2.5
    if "preference" in tag_str:
        score += 2.0
    if "decision" in tag_str or "strategic" in tag_str:
        score += 1.5
    if "behavioral" in tag_str or "instruction" in tag_str:
        score += 2.0

    # Confidence boost (well-structured entries are more likely to be useful)
    score += candidate.confidence * 0.5

    # Recency: strong signal within tier, not just a tiebreaker
    if candidate.timestamp:
        age_days = (datetime.now(timezone.utc) - candidate.timestamp).days
        recency_bonus = max(0.0, 1.0 - (age_days / 365))
        score += recency_bonus * 1.2

    return score


def rank_candidates_heuristic(
    candidates: List[IngestCandidate],
    budget: int = DEFAULT_FIRST_CHUNK_ENTRIES,
) -> List[IngestCandidate]:
    """Rank and trim candidates to budget using heuristic scoring.

    Returns the top candidates sorted by priority score (descending).
    """
    scored = [(heuristic_priority_score(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:budget]]


# ─── LLM classification (primary path) ─────────────────────────────────────

# Tier assignments from LLM classification
TIER_SCORES = {
    1: 10.0,  # Identity, preferences, corrections, behavioral
    2: 5.0,   # Strategic decisions, active projects, relationships
    3: 1.0,   # Conversation history, reference, transient
}


def build_classification_prompt(candidates: List[IngestCandidate], batch_size: int = 50) -> str:
    """Build a prompt for LLM-based tier classification.

    Sends a batch of candidate summaries and asks the LLM to assign
    each to a priority tier (1, 2, or 3).
    """
    entries = []
    for i, c in enumerate(candidates[:batch_size]):
        # Truncate content for the prompt (first 200 chars)
        preview = c.raw_content[:200].replace("\n", " ").strip()
        source = c.source_id
        ctype = c.content_type.value if c.content_type else "unknown"
        entries.append(f"[{i}] source={source} type={ctype}: {preview}")

    entry_block = "\n".join(entries)

    return f"""Classify each entry into a priority tier for a memory system's first-chunk ingestion.

Tier 1 (highest priority): User identity (name, role, company), stated preferences, corrections, behavioral guidance, standing instructions.
Tier 2: Strategic decisions, active project context, key relationships, deadlines.
Tier 3 (lowest priority): Conversation history, reference material, debugging sessions, transient technical context.

Respond with ONLY a JSON array of objects, one per entry:
[{{"index": 0, "tier": 1}}, {{"index": 1, "tier": 3}}, ...]

Entries:
{entry_block}"""


def parse_classification_response(response: str, count: int) -> List[int]:
    """Parse LLM classification response into tier assignments.

    Returns a list of tier numbers (1, 2, or 3) aligned to input indices.
    Falls back to tier 3 for any entry that can't be parsed.
    """
    import json

    tiers = [3] * count  # Default everything to tier 3

    # Find JSON array in response
    text = response.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        logger.warning("Could not find JSON array in classification response")
        return tiers

    try:
        items = json.loads(text[start:end + 1])
        for item in items:
            idx = item.get("index", -1)
            tier = item.get("tier", 3)
            if 0 <= idx < count and tier in (1, 2, 3):
                tiers[idx] = tier
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning(f"Failed to parse classification response: {e}")

    return tiers


def rank_candidates_llm(
    candidates: List[IngestCandidate],
    tier_assignments: List[int],
    budget: int = DEFAULT_FIRST_CHUNK_ENTRIES,
) -> List[IngestCandidate]:
    """Rank candidates using LLM tier assignments + recency within tier.

    Args:
        candidates: The full candidate list.
        tier_assignments: Tier (1, 2, 3) for each candidate, aligned by index.
        budget: Maximum entries to return.
    """
    now = datetime.now(timezone.utc)

    def sort_key(pair):
        tier, candidate = pair
        tier_score = TIER_SCORES.get(tier, 1.0)
        recency = 0.0
        if candidate.timestamp:
            age_days = (now - candidate.timestamp).days
            recency = max(0.0, 1.0 - (age_days / 365))
        return -(tier_score + recency * 1.2)

    paired = list(zip(tier_assignments, candidates))
    paired.sort(key=sort_key)

    return [c for _, c in paired[:budget]]
