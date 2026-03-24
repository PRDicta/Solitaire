"""
Fabric Layer Enrichment B: Intent-Aware Context Loading

Accepts an intent signal (user's first message, browser tabs, etc.)
and pre-loads relevant context from the knowledge base before the
first response.
"""

import sqlite3
from typing import List, Optional

from solitaire.core.types import estimate_tokens


def build_intent_context_block(
    conn: sqlite3.Connection,
    intent_text: str,
    recall_fn,
    budget_tokens: int = 2000,
) -> str:
    """
    Build a context block from intent-driven recall.

    Args:
        conn: SQLite connection (unused directly, but available for entity lookups)
        intent_text: The intent signal (user message, tab titles, etc.)
        recall_fn: A callable that runs recall search. Expected signature:
                   recall_fn(query: str, limit: int) -> List[dict]
                   Each dict should have at least 'content' and optionally
                   'source_type', 'created_at', 'tags'.
        budget_tokens: Maximum tokens for the block.

    Returns:
        Assembled context block string, or empty string if nothing relevant found.
    """
    if not intent_text or not intent_text.strip():
        return ""

    # Clean and truncate intent for query
    query = intent_text.strip()[:500]

    # Detect continuation signals: thin phrases that mean "pick up where we left off"
    # These need expansion to pull pending/open items rather than literal matching
    continuation_phrases = {
        "next phase", "next step", "pick up", "continue", "where we left off",
        "carry on", "what's next", "let's go", "back to it", "resume",
        "what were we doing", "pending", "open items",
    }
    is_continuation = any(
        cp in query.lower() for cp in continuation_phrases
    )

    # Run recall — for continuations, also search for pending/open threads
    try:
        results = recall_fn(query, limit=12)
        if is_continuation:
            continuation_results = recall_fn(
                "pending next session remaining deferred open thread",
                limit=8,
            )
            # Merge, deduplicating by content prefix
            seen_prefixes = {r.get("content", "")[:80] for r in results}
            for cr in continuation_results:
                prefix = cr.get("content", "")[:80]
                if prefix not in seen_prefixes:
                    results.append(cr)
                    seen_prefixes.add(prefix)
    except Exception:
        return ""

    if not results:
        return ""

    # Build block within budget
    parts = [f"═══ RELEVANT CONTEXT (intent: {_truncate(query, 80)}) ═══", ""]

    tokens_used = estimate_tokens(parts[0]) + 20  # header + footer budget
    entries_included = 0

    for entry in results:
        content = entry.get("content", "")
        if not content:
            continue

        source = entry.get("source_type", "")
        source_tag = f" [{source}]" if source else ""
        entry_text = f"{content.strip()}{source_tag}"

        entry_tokens = estimate_tokens(entry_text)
        if tokens_used + entry_tokens > budget_tokens:
            # Try truncating this entry to fit remaining budget
            remaining = budget_tokens - tokens_used - 10
            if remaining > 100:
                char_limit = remaining * 4
                entry_text = entry_text[:char_limit] + "..."
                entry_tokens = estimate_tokens(entry_text)
            else:
                break

        parts.append(f"[{entries_included + 1}] {entry_text}")
        parts.append("")
        tokens_used += entry_tokens + 2
        entries_included += 1

    if entries_included == 0:
        return ""

    parts.append("═══ END RELEVANT CONTEXT ═══")
    return "\n".join(parts)


def parse_intent_from_args(
    intent_arg: Optional[str] = None,
    tab_titles: Optional[List[str]] = None,
) -> str:
    """
    Combine intent signals into a single query string.

    Priority: explicit intent > tab titles > empty.
    """
    signals = []

    if intent_arg and intent_arg.strip():
        signals.append(intent_arg.strip())

    if tab_titles:
        # Filter out generic titles
        generic = {"new tab", "google", "about:blank", "untitled"}
        meaningful = [t for t in tab_titles if t.lower().strip() not in generic]
        if meaningful:
            signals.append("Browser context: " + "; ".join(meaningful[:5]))

    return " | ".join(signals) if signals else ""


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
