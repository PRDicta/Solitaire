"""
The Librarian — Entry Reclassifier

Scans the rolodex for miscategorized entries and proposes reclassifications
based on multi-signal heuristics. Three tiers of confidence:

Tier 1 (near-certain): Tag evidence directly contradicts category.
  - Entry tagged user_knowledge but category is note.

Tier 2 (high confidence): User-authored short entries with directive language.
  - Preferences: "I prefer", "don't want", "from now on"
  - Corrections: "that's wrong", "actually,", "incorrect"
  - Decisions: "I decided", "let's go with", "the plan is"

Tier 3 (medium confidence): Assistant-authored entries attributed to the owner.
  - Contains owner-directive verb (wants, confirmed, corrected, etc.)
  - Attributed via tags (attributed:user, attributed:named)

Also handles source_type boosts: entries tagged attributed:user/named
get source_type upgraded to user_knowledge for 3x retrieval boost.

Dry-run by default. Only commits changes when --commit is passed.
"""

import sqlite3
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ─── Data Types ──────────────────────────────────────────────────────────────

@dataclass
class Reclassification:
    """A proposed change to an entry's category or source_type."""
    entry_id: str
    content_preview: str
    created_at: str
    current_category: str
    current_source_type: str
    proposed_category: Optional[str] = None      # None = no category change
    proposed_source_type: Optional[str] = None    # None = no source_type change
    tier: int = 0                                  # 1, 2, or 3
    confidence: float = 0.0                        # 0.0-1.0
    reason: str = ""
    signals: List[str] = field(default_factory=list)


@dataclass
class ReclassifyResult:
    """Result of a reclassification scan."""
    total_scanned: int = 0
    proposals: List[Reclassification] = field(default_factory=list)
    committed: int = 0
    skipped: int = 0

    @property
    def by_tier(self) -> Dict[int, List[Reclassification]]:
        tiers: Dict[int, List[Reclassification]] = {}
        for p in self.proposals:
            tiers.setdefault(p.tier, []).append(p)
        return tiers

    @property
    def summary(self) -> str:
        lines = [f"Scanned {self.total_scanned} entries."]
        lines.append(f"Found {len(self.proposals)} reclassification candidates.")
        for tier_num in sorted(self.by_tier.keys()):
            items = self.by_tier[tier_num]
            lines.append(f"  Tier {tier_num}: {len(items)} entries")
            # Group by proposed category
            cats: Dict[str, int] = {}
            boosts = 0
            for item in items:
                if item.proposed_category:
                    cats[item.proposed_category] = cats.get(item.proposed_category, 0) + 1
                if item.proposed_source_type:
                    boosts += 1
            for cat, count in sorted(cats.items()):
                lines.append(f"    -> {cat}: {count}")
            if boosts:
                lines.append(f"    -> source_type boost: {boosts}")
        if self.committed:
            lines.append(f"Committed: {self.committed}")
        if self.skipped:
            lines.append(f"Skipped: {self.skipped}")
        return "\n".join(lines)


# ─── Tier 1: Tag-Category Mismatch ──────────────────────────────────────────

def _scan_tier1(conn: sqlite3.Connection) -> List[Reclassification]:
    """Entries tagged user_knowledge but categorized as note."""
    rows = conn.execute("""
        SELECT id, content, created_at, category, source_type, tags
        FROM rolodex_entries
        WHERE superseded_by IS NULL
          AND category = 'note'
          AND tags LIKE '%user_knowledge%'
    """).fetchall()

    results = []
    for row in rows:
        entry_id, content, created_at, category, source_type, tags = row
        results.append(Reclassification(
            entry_id=entry_id,
            content_preview=(content or "")[:200],
            created_at=created_at or "",
            current_category=category or "note",
            current_source_type=source_type or "conversation",
            proposed_category="user_knowledge",
            proposed_source_type="user_knowledge" if source_type != "user_knowledge" else None,
            tier=1,
            confidence=0.95,
            reason="Tagged user_knowledge but categorized as note",
            signals=["tag:user_knowledge", "category:note"],
        ))
    return results


# ─── Tier 2: User-Authored Directive Language ────────────────────────────────

_PREFERENCE_PATTERNS = [
    re.compile(r"\bI prefer\b", re.IGNORECASE),
    re.compile(r"\bI don'?t want\b", re.IGNORECASE),
    re.compile(r"\bdo not want\b", re.IGNORECASE),
    re.compile(r"\bdoes not want\b", re.IGNORECASE),
    re.compile(r"\bI want you to\b", re.IGNORECASE),
    re.compile(r"\bfrom now on\b", re.IGNORECASE),
    re.compile(r"\bgoing forward\b", re.IGNORECASE),
    re.compile(r"\bstop doing\b", re.IGNORECASE),
    re.compile(r"\bnever use\b", re.IGNORECASE),
    re.compile(r"\balways use\b", re.IGNORECASE),
    re.compile(r"\bI'd prefer\b", re.IGNORECASE),
    re.compile(r"\bI'd rather\b", re.IGNORECASE),
    re.compile(r"\bplease don'?t\b", re.IGNORECASE),
    re.compile(r"\bI'd like you to\b", re.IGNORECASE),
]

_CORRECTION_PATTERNS = [
    re.compile(r"\bthat'?s wrong\b", re.IGNORECASE),
    re.compile(r"\bthat'?s not\b", re.IGNORECASE),
    re.compile(r"\bthat is wrong\b", re.IGNORECASE),
    re.compile(r"\bnot correct\b", re.IGNORECASE),
    re.compile(r"\bactually,", re.IGNORECASE),
    re.compile(r"\byou misunder", re.IGNORECASE),
    re.compile(r"\bincorrect\b", re.IGNORECASE),
    re.compile(r"\bmy name is\b", re.IGNORECASE),
    re.compile(r"\blast name\b.*\bis\b", re.IGNORECASE),
    re.compile(r"\bno,\s", re.IGNORECASE),
    re.compile(r"\bnot what I\b", re.IGNORECASE),
]

_DECISION_PATTERNS = [
    re.compile(r"\bI decided\b", re.IGNORECASE),
    re.compile(r"\bI'?ve decided\b", re.IGNORECASE),
    re.compile(r"\blet'?s go with\b", re.IGNORECASE),
    re.compile(r"\bwe'?ll go with\b", re.IGNORECASE),
    re.compile(r"\blet'?s do\b", re.IGNORECASE),
    re.compile(r"\bthe plan is\b", re.IGNORECASE),
    re.compile(r"\bfinal decision\b", re.IGNORECASE),
    re.compile(r"\bI want to go with\b", re.IGNORECASE),
    re.compile(r"\bI chose\b", re.IGNORECASE),
    re.compile(r"\bwe decided\b", re.IGNORECASE),
]


def _match_patterns(content: str, patterns: list) -> List[str]:
    """Return list of pattern labels that matched."""
    matched = []
    for p in patterns:
        if p.search(content):
            matched.append(p.pattern)
    return matched


def _scan_tier2(conn: sqlite3.Connection) -> List[Reclassification]:
    """User-authored, short entries with directive language."""
    rows = conn.execute("""
        SELECT id, content, created_at, category, source_type, tags
        FROM rolodex_entries
        WHERE superseded_by IS NULL
          AND category = 'note'
          AND tags LIKE '%role:user%'
          AND length(content) < 500
          AND tags NOT LIKE '%user_knowledge%'
    """).fetchall()

    results = []
    for row in rows:
        entry_id, content, created_at, category, source_type, tags = row
        if not content:
            continue

        # Check each category of patterns
        pref_matches = _match_patterns(content, _PREFERENCE_PATTERNS)
        corr_matches = _match_patterns(content, _CORRECTION_PATTERNS)
        deci_matches = _match_patterns(content, _DECISION_PATTERNS)

        # Skip continuation summaries (these are system-generated, not real user speech)
        if "The summary below covers" in content or "This session is being continued" in content:
            continue

        if pref_matches:
            results.append(Reclassification(
                entry_id=entry_id,
                content_preview=content[:200],
                created_at=created_at or "",
                current_category=category or "note",
                current_source_type=source_type or "conversation",
                proposed_category="preference",
                proposed_source_type="user_knowledge",
                tier=2,
                confidence=0.85,
                reason="User-authored entry with preference language",
                signals=["role:user", f"patterns:{len(pref_matches)}"] + pref_matches[:2],
            ))
        elif corr_matches:
            results.append(Reclassification(
                entry_id=entry_id,
                content_preview=content[:200],
                created_at=created_at or "",
                current_category=category or "note",
                current_source_type=source_type or "conversation",
                proposed_category="correction",
                proposed_source_type="user_knowledge",
                tier=2,
                confidence=0.80,
                reason="User-authored entry with correction language",
                signals=["role:user", f"patterns:{len(corr_matches)}"] + corr_matches[:2],
            ))
        elif deci_matches:
            results.append(Reclassification(
                entry_id=entry_id,
                content_preview=content[:200],
                created_at=created_at or "",
                current_category=category or "note",
                current_source_type=source_type or "conversation",
                proposed_category="decision",
                proposed_source_type="user_knowledge",
                tier=2,
                confidence=0.80,
                reason="User-authored entry with decision language",
                signals=["role:user", f"patterns:{len(deci_matches)}"] + deci_matches[:2],
            ))

    return results


# ─── Tier 3: Attributed Assistant Entries ────────────────────────────────────

# Tier 3a: Durable intent patterns — these carry strategic decisions, preferences,
# architectural direction, or standing instructions that persist beyond the session.
_DURABLE_INTENT_PATTERNS = [
    # User + strong directive verbs (decisions, preferences, instructions).
    # Owner-name patterns removed: they were compiled with literal "{owner}"
    # (never substituted) and never matched. Owner-name matching requires
    # runtime compilation with the actual owner name; deferred to a future pass.
    re.compile(r"\bUser\b.*\bprefers?\b", re.IGNORECASE),
    re.compile(r"\bUser\b.*\bconfirmed\b", re.IGNORECASE),
    re.compile(r"\bUser\b.*\bcorrected\b", re.IGNORECASE),
    re.compile(r"\bUser\b.*\bdecided\b", re.IGNORECASE),
    re.compile(r"\bUser\b.*\bapproved\b", re.IGNORECASE),
    re.compile(r"\bUser\b.*\binstructed\b", re.IGNORECASE),
    re.compile(r"\bUser\b.*\brejected\b", re.IGNORECASE),
]

# Tier 3a (lower confidence): "wants to" patterns need extra filtering because
# "User wants clarification" is ephemeral but "User wants to build X" is durable.
_WANTS_PATTERNS = [
    re.compile(r"\bUser\b.*\bwants?\s+to\b", re.IGNORECASE),
    re.compile(r"\bUser\b.*\bwants?\b", re.IGNORECASE),
]

# Ephemeral signals — these indicate session-scoped questions or requests,
# not durable preferences or decisions. Entries matching these are demoted
# to boost-only (source_type upgrade but no category change).
_EPHEMERAL_PATTERNS = [
    re.compile(r"\bwants?\s+(?:clarification|information|to\s+know|to\s+see|to\s+check|to\s+verify|to\s+understand)\b", re.IGNORECASE),
    re.compile(r"\basked\s+(?:about|whether|if|how|what|where|when|why|for\s+options)\b", re.IGNORECASE),
    re.compile(r"\basked\s+(?:two|three|four|a)\s+\w+\s+questions?\b", re.IGNORECASE),
    re.compile(r"\bwants?\s+to\s+know\b", re.IGNORECASE),
    re.compile(r"\bsaid\s", re.IGNORECASE),  # "{owner} said" is usually quoting, not directing
    re.compile(r"\basked\s+for\s+(?:a|the)\b", re.IGNORECASE),  # "asked for a recommendation"
]

# Durable "wants to" signals — these indicate strategic intent that persists.
_DURABLE_WANTS_SIGNALS = [
    re.compile(r"\bwants?\s+to\s+(?:build|implement|create|design|ship|adopt|evolve|bring|use|keep|add|integrate|scope|overhaul)\b", re.IGNORECASE),
    re.compile(r"\bwants?\s+(?:100%|full|every|all)\b", re.IGNORECASE),
    re.compile(r"\bwants?\s+\w+\s+(?:kept|triggered|resilience|enabled|disabled)\b", re.IGNORECASE),
    re.compile(r"\bwants?\s+to\s+(?:evolve|refine|improve|extend|scale)\b", re.IGNORECASE),
]


def _is_ephemeral(content: str) -> bool:
    """Check if content looks like a session-scoped question rather than durable intent."""
    return any(p.search(content) for p in _EPHEMERAL_PATTERNS)


def _is_durable_want(content: str) -> bool:
    """Check if a 'wants' pattern carries durable strategic intent."""
    return any(p.search(content) for p in _DURABLE_WANTS_SIGNALS)


def _scan_tier3(conn: sqlite3.Connection) -> List[Reclassification]:
    """Attributed assistant entries with owner-directive patterns.

    Split into three sub-tiers:
    3a: Durable intent (confirmed, decided, approved, etc.) -> category + source_type
    3b: "Wants to" with durable signal -> category + source_type (slightly lower confidence)
    3c: Remaining attributed entries -> source_type boost only
    """
    rows = conn.execute("""
        SELECT id, content, created_at, category, source_type, tags
        FROM rolodex_entries
        WHERE superseded_by IS NULL
          AND category = 'note'
          AND (tags LIKE '%attributed:user%' OR tags LIKE '%attributed:named%')
          AND tags NOT LIKE '%user_knowledge%'
    """).fetchall()

    results = []
    categorized_ids = set()

    for row in rows:
        entry_id, content, created_at, category, source_type, tags = row
        if not content:
            continue

        # Skip system noise
        if content.startswith("{") or "fuse_detected" in content or len(content) < 30:
            continue

        # Skip continuation summaries
        if "The summary below covers" in content or "This session is being continued" in content:
            continue

        # 3a: Strong durable directive verbs (confirmed, decided, approved, etc.)
        durable_matches = _match_patterns(content, _DURABLE_INTENT_PATTERNS)
        if durable_matches and not _is_ephemeral(content):
            categorized_ids.add(entry_id)
            results.append(Reclassification(
                entry_id=entry_id,
                content_preview=content[:200],
                created_at=created_at or "",
                current_category=category or "note",
                current_source_type=source_type or "conversation",
                proposed_category="user_knowledge",
                proposed_source_type="user_knowledge",
                tier=3,
                confidence=0.75,
                reason="Attributed entry with durable owner directive language",
                signals=["attributed", "durable"] + durable_matches[:2],
            ))
            continue

        # 3b: "Wants to" patterns — only if they carry durable strategic signal
        wants_matches = _match_patterns(content, _WANTS_PATTERNS)
        if wants_matches:
            if _is_ephemeral(content):
                # Ephemeral "wants" — demote to boost-only (handled in 3c below)
                pass
            elif _is_durable_want(content):
                categorized_ids.add(entry_id)
                results.append(Reclassification(
                    entry_id=entry_id,
                    content_preview=content[:200],
                    created_at=created_at or "",
                    current_category=category or "note",
                    current_source_type=source_type or "conversation",
                    proposed_category="user_knowledge",
                    proposed_source_type="user_knowledge",
                    tier=3,
                    confidence=0.70,
                    reason="Attributed entry with durable 'wants to' intent",
                    signals=["attributed", "durable_want"] + wants_matches[:2],
                ))
                continue

    # 3c: All remaining attributed entries -> source_type boost only (no category change)
    for row in rows:
        entry_id, content, created_at, category, source_type, tags = row
        if entry_id in categorized_ids:
            continue
        if not content or content.startswith("{") or len(content) < 30:
            continue
        if "The summary below covers" in content or "This session is being continued" in content:
            continue
        if source_type == "user_knowledge":
            continue  # Already boosted

        results.append(Reclassification(
            entry_id=entry_id,
            content_preview=content[:200],
            created_at=created_at or "",
            current_category=category or "note",
            current_source_type=source_type or "conversation",
            proposed_category=None,  # No category change
            proposed_source_type="user_knowledge",
            tier=3,
            confidence=0.60,
            reason="Attributed entry -- source_type boost for retrieval weight",
            signals=["attributed", "boost_only"],
        ))

    return results


# ─── Main Scanner ────────────────────────────────────────────────────────────

def scan_reclassifications(conn: sqlite3.Connection) -> ReclassifyResult:
    """
    Scan the rolodex for miscategorized entries.
    Returns proposals without modifying anything.
    """
    total = conn.execute(
        "SELECT COUNT(*) FROM rolodex_entries WHERE superseded_by IS NULL AND category = 'note'"
    ).fetchone()[0]

    t1 = _scan_tier1(conn)
    t2 = _scan_tier2(conn)
    t3 = _scan_tier3(conn)

    result = ReclassifyResult(
        total_scanned=total,
        proposals=t1 + t2 + t3,
    )
    return result


def commit_reclassifications(
    conn: sqlite3.Connection,
    proposals: List[Reclassification],
    min_tier: int = 1,
    max_tier: int = 3,
    min_confidence: float = 0.0,
    rolodex=None,
) -> Tuple[int, int]:
    """
    Apply approved reclassifications to the database.

    Args:
        conn: Database connection (used as fallback if rolodex is None)
        proposals: List of Reclassification objects to apply
        min_tier: Minimum tier to commit (1-3)
        max_tier: Maximum tier to commit (1-3)
        min_confidence: Minimum confidence threshold
        rolodex: Optional Rolodex instance. When provided, reclassifications
            go through Rolodex.reclassify_entry() which maintains all write
            invariants (FTS, JSONL, cache). Without it, only the raw SQLite
            row is updated (legacy behavior).

    Returns:
        (committed_count, skipped_count)
    """
    committed = 0
    skipped = 0

    for p in proposals:
        if p.tier < min_tier or p.tier > max_tier:
            skipped += 1
            continue
        if p.confidence < min_confidence:
            skipped += 1
            continue

        if not p.proposed_category and not p.proposed_source_type:
            skipped += 1
            continue

        try:
            if rolodex is not None:
                # Preferred path: maintains FTS, JSONL, and cache invariants
                ok = rolodex.reclassify_entry(
                    entry_id=p.entry_id,
                    category=p.proposed_category,
                    source_type=p.proposed_source_type,
                )
                if ok:
                    committed += 1
                else:
                    skipped += 1
            else:
                # Legacy fallback: raw SQL (FTS and JSONL not updated)
                updates = []
                params: list = []
                if p.proposed_category:
                    updates.append("category = ?")
                    params.append(p.proposed_category)
                if p.proposed_source_type:
                    updates.append("source_type = ?")
                    params.append(p.proposed_source_type)
                params.append(p.entry_id)
                sql = f"UPDATE rolodex_entries SET {', '.join(updates)} WHERE id = ?"
                conn.execute(sql, params)
                committed += 1
        except Exception:
            skipped += 1

    if committed > 0 and rolodex is None:
        conn.commit()

    return committed, skipped
