"""
The Librarian — User Facts Store & Contradiction Detection

Provides:
1. Provenance tagging at ingestion (user-stated vs assistant-inferred)
2. Explicit user facts store (biographical, high-authority, checked before general recall)
3. Contradiction detection tuned for objective facts

User facts are structured extractions from user_knowledge entries:
  - subject: what the fact is about ("the user", "Acme Corp", "wife")
  - predicate: the relationship/attribute ("lives in", "founded", "name is")
  - value: the asserted value ("California", "2024", "Sarah")
  - provenance: who stated it ("user-stated" | "assistant-inferred")

Contradiction detection runs at ingestion time. When a new fact conflicts
with an existing fact on the same subject+predicate, the system:
  - Flags the conflict in the ingestion output
  - Marks the older fact as potentially stale (not deleted)
  - Lets the user or assistant resolve it explicitly

Design principles:
  - No LLM calls. All extraction is heuristic/pattern-based.
  - Additive only. Never blocks ingestion on failure.
  - User-stated facts always outrank assistant-inferred.
  - Objective facts (locations, dates, names, counts) get contradiction checks.
    Subjective content (opinions, preferences, assessments) does not.
"""

import json
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class UserFact:
    """A structured fact about the user or their world."""
    id: str = ""
    subject: str = ""          # "the user", "Acme Corp", "wife", "dog"
    predicate: str = ""        # "lives in", "age", "name is", "founded"
    value: str = ""            # "California", "38", "Sarah", "2024"
    provenance: str = "unknown"  # user-stated | assistant-inferred | system
    source_entry_id: str = ""  # FK to rolodex_entries.id
    fact_type: str = "biographical"  # biographical | relational | temporal | quantitative
    confidence: float = 1.0    # 0.0-1.0
    created_at: str = ""       # ISO timestamp
    superseded_by: Optional[str] = None  # FK to another UserFact.id
    superseded_at: Optional[str] = None


@dataclass
class Contradiction:
    """A detected conflict between two facts."""
    existing_fact: UserFact
    new_fact: UserFact
    conflict_type: str = ""    # value_change | temporal_conflict | count_mismatch
    description: str = ""


# ─── Provenance Classification ───────────────────────────────────────────────

def classify_provenance(role: str, as_user_knowledge: bool = False,
                        content: str = "") -> str:
    """
    Determine provenance tag for an ingested entry.

    Rules:
      - role="user" → "user-stated" (the user said it)
      - role="user" + --user-knowledge → "user-stated" (explicitly flagged)
      - role="assistant" + --user-knowledge → "assistant-inferred"
        (assistant extracted this as a fact, user didn't say it verbatim)
      - role="assistant" → "assistant-inferred"
      - role="system" → "system"

    The key distinction: did the user say this, or did the assistant
    conclude it? Even if an assistant entry is marked user_knowledge,
    it was the assistant's interpretation.
    """
    if role == "system":
        return "system"
    if role == "user":
        return "user-stated"
    if role == "assistant":
        return "assistant-inferred"
    return "unknown"


# ─── Fact Extraction (Heuristic) ─────────────────────────────────────────────

# Patterns for extracting structured facts from text.
# Each pattern produces (subject, predicate, value, fact_type).

_FACT_PATTERNS = [
    # Location: "I live in X", "I'm based in X", "I moved to X"
    (r"\b(?:i|{owner})\s+(?:live|lives|am based|is based|reside|resides)\s+in\s+([^.,;]+?)(?:\s+and\s|\.|,|;|$)",
     "{owner}", "lives in", "biographical"),  # Note: {owner} is replaced at runtime with owner_name variable
    (r"\b(?:i|{owner})\s+(?:moved|relocated)\s+to\s+([^.,;]+?)(?:\s+and\s|\.|,|;|$)",
     "{owner}", "lives in", "biographical"),  # Note: {owner} is replaced at runtime with owner_name variable

    # Age: "I'm X years old", "I am X"
    (r"\b(?:i'm|i am|{owner} is)\s+(\d{1,3})\s+(?:years?\s+old|yo)\b",
     "{owner}", "age", "quantitative"),  # Note: {owner} is replaced at runtime with owner_name variable

    # Name patterns: "my X is named Y", "my X's name is Y"
    (r"\bmy\s+(wife|husband|partner|daughter|son|dog|cat|child)\s*(?:'s)?\s*(?:name\s+is|is\s+named|is\s+called)\s+(\w+)",
     None, "name is", "relational"),  # subject derived from capture group

    # Founding/creation: "I founded X in Y", "X was founded in Y"
    (r"\b(?:i|{owner})\s+(?:founded|started|created|launched)\s+(.+?)\s+in\s+(\d{4})\b",
     "{owner}", "founded", "temporal"),  # Note: {owner} is replaced at runtime with owner_name variable

    # Employment: "I work at X", "I work for X"
    (r"\b(?:i|{owner})\s+(?:work|works)\s+(?:at|for)\s+([^.,;]+?)(?:\s+and\s|\.|,|;|$)",
     "{owner}", "works at", "biographical"),  # Note: {owner} is replaced at runtime with owner_name variable

    # Role: "I'm a X", "I am a X", "my role is X"
    (r"\b(?:i'm|i am)\s+(?:a|an|the)\s+([^.,;]+?)(?:\s+and\s|\.|,|;|$)",
     "{owner}", "role is", "biographical"),  # Note: {owner} is replaced at runtime with owner_name variable

    # Preferences: "I prefer X", "I like X", "I use X"
    (r"\b(?:i|{owner})\s+(?:prefer|prefers|like|likes|use|uses)\s+([^.,;]+?)(?:\s+and\s|\.|,|\s+over\s|\s+for\s|;|$)",
     "{owner}", "prefers", "biographical"),  # Note: {owner} is replaced at runtime with owner_name variable

    # Timezone: "I'm in X timezone", "my timezone is X"
    (r"\b(?:my\s+)?timezone\s+is\s+(.+?)(?:\.|,|$)",
     "{owner}", "timezone is", "biographical"),  # Note: {owner} is replaced at runtime with owner_name variable
    (r"\b(?:i'm|i am)\s+in\s+(?:the\s+)?(.+?)\s+timezone\b",
     "{owner}", "timezone is", "biographical"),  # Note: {owner} is replaced at runtime with owner_name variable

    # Email: "my email is X"
    (r"\bmy\s+email\s+(?:is|address\s+is)\s+(\S+@\S+)",
     "{owner}", "email is", "biographical"),  # Note: {owner} is replaced at runtime with owner_name variable

    # Company details: "organization is X", "organization does X"
    (r"\b(?:organization|mycompany)\s+(?:is|does|provides|offers|specializes)\s+(.+?)(?:\.|,|$)",
     "mycompany", "is", "biographical"),
]

# Objective fact predicates that should trigger contradiction detection.
# Subjective predicates (prefers, likes, thinks) are excluded.
_OBJECTIVE_PREDICATES = {
    "lives in", "age", "name is", "founded", "works at", "role is",
    "timezone is", "email is", "is",
}


def extract_facts(content: str, role: str,
                  as_user_knowledge: bool = False) -> List[UserFact]:
    """
    Extract structured facts from content using heuristic patterns.

    Only extracts from content that looks like factual statements.
    Returns empty list if nothing matches (which is fine — most messages
    don't contain extractable biographical facts).
    """
    import uuid

    provenance = classify_provenance(role, as_user_knowledge, content)
    content_lower = content.lower().strip()
    facts = []
    seen = set()  # Dedup on (subject, predicate)

    for pattern, default_subject, predicate, fact_type in _FACT_PATTERNS:
        for match in re.finditer(pattern, content_lower, re.IGNORECASE):
            groups = match.groups()

            if default_subject is None and predicate == "name is":
                # Special: "my wife is named Sarah" → subject="wife", value="Sarah"
                if len(groups) >= 2:
                    subject = groups[0].strip().title()
                    value = groups[1].strip().title()
                else:
                    continue
            elif predicate == "founded" and len(groups) >= 2:
                # "I founded Acme in 2024" → subject="Acme", value="2024"
                subject = groups[0].strip().title()
                value = groups[1].strip()
                predicate = "founded in"
            else:
                subject = default_subject or "{owner}"
                value = groups[0].strip() if groups else ""

            if not value:
                continue

            # Clean up value
            value = re.sub(r'\s+', ' ', value).strip()
            # Cap value length (facts should be concise)
            if len(value) > 200:
                continue

            dedup_key = (subject.lower(), predicate.lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            facts.append(UserFact(
                id=str(uuid.uuid4())[:8],
                subject=subject,
                predicate=predicate,
                value=value,
                provenance=provenance,
                fact_type=fact_type,
                confidence=1.0 if provenance == "user-stated" else 0.7,
                created_at=datetime.utcnow().isoformat(),
            ))

    return facts


# ─── User Facts Store (SQLite) ───────────────────────────────────────────────

class UserFactsStore:
    """
    Persistent store for structured user facts.

    Table: user_facts
    Queried before general recall to ground responses in known facts.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ensure_table()

    def _ensure_table(self):
        """Create user_facts table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value TEXT NOT NULL,
                provenance TEXT NOT NULL DEFAULT 'unknown',
                source_entry_id TEXT,
                fact_type TEXT DEFAULT 'biographical',
                confidence REAL DEFAULT 1.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                superseded_by TEXT,
                superseded_at DATETIME,
                FOREIGN KEY (source_entry_id) REFERENCES rolodex_entries(id)
            )
        """)
        # Index for contradiction queries (subject + predicate lookups)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_facts_subject_predicate
            ON user_facts (subject, predicate)
        """)
        # Index for active (non-superseded) facts
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_facts_active
            ON user_facts (superseded_by) WHERE superseded_by IS NULL
        """)
        self.conn.commit()

    def store_fact(self, fact: UserFact) -> str:
        """Store a user fact. Returns the fact ID."""
        self.conn.execute("""
            INSERT OR REPLACE INTO user_facts
            (id, subject, predicate, value, provenance, source_entry_id,
             fact_type, confidence, created_at, superseded_by, superseded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fact.id, fact.subject, fact.predicate, fact.value,
            fact.provenance, fact.source_entry_id, fact.fact_type,
            fact.confidence, fact.created_at, fact.superseded_by,
            fact.superseded_at,
        ))
        self.conn.commit()
        return fact.id

    def store_facts(self, facts: List[UserFact]) -> List[str]:
        """Store multiple facts. Returns list of fact IDs."""
        ids = []
        for fact in facts:
            ids.append(self.store_fact(fact))
        return ids

    def get_active_facts(self, subject: Optional[str] = None,
                         predicate: Optional[str] = None) -> List[UserFact]:
        """
        Get all active (non-superseded) facts, optionally filtered.

        This is the primary query for grounding responses in known facts.
        """
        sql = "SELECT * FROM user_facts WHERE superseded_by IS NULL"
        params = []

        if subject:
            sql += " AND LOWER(subject) = LOWER(?)"
            params.append(subject)
        if predicate:
            sql += " AND LOWER(predicate) = LOWER(?)"
            params.append(predicate)

        sql += " ORDER BY confidence DESC, created_at DESC"

        cursor = self.conn.execute(sql, params)
        return [self._row_to_fact(row) for row in cursor.fetchall()]

    def get_all_active_facts(self) -> List[UserFact]:
        """Get all active facts for boot-time loading."""
        return self.get_active_facts()

    def query_facts(self, query: str) -> List[UserFact]:
        """
        Search facts by keyword match against subject, predicate, and value.
        Returns active facts only.
        """
        query_lower = query.lower()
        words = query_lower.split()

        sql = """
            SELECT * FROM user_facts
            WHERE superseded_by IS NULL
            AND (
                LOWER(subject) LIKE ?
                OR LOWER(predicate) LIKE ?
                OR LOWER(value) LIKE ?
            )
            ORDER BY confidence DESC, created_at DESC
            LIMIT 20
        """
        pattern = f"%{query_lower}%"
        results = []
        cursor = self.conn.execute(sql, (pattern, pattern, pattern))
        results = [self._row_to_fact(row) for row in cursor.fetchall()]

        # Also try individual words if full query didn't match well
        if len(results) < 3 and len(words) > 1:
            seen_ids = {f.id for f in results}
            for word in words:
                if len(word) < 3:
                    continue
                pattern = f"%{word}%"
                cursor = self.conn.execute(sql, (pattern, pattern, pattern))
                for row in cursor.fetchall():
                    fact = self._row_to_fact(row)
                    if fact.id not in seen_ids:
                        results.append(fact)
                        seen_ids.add(fact.id)

        return results[:20]

    def supersede_fact(self, old_fact_id: str, new_fact_id: str):
        """Mark an old fact as superseded by a new one."""
        self.conn.execute("""
            UPDATE user_facts
            SET superseded_by = ?, superseded_at = ?
            WHERE id = ?
        """, (new_fact_id, datetime.utcnow().isoformat(), old_fact_id))
        self.conn.commit()

    def count_active(self) -> int:
        """Count active (non-superseded) facts."""
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM user_facts WHERE superseded_by IS NULL"
        )
        return cursor.fetchone()[0]

    def _row_to_fact(self, row) -> UserFact:
        """Convert a DB row to a UserFact."""
        if isinstance(row, sqlite3.Row):
            return UserFact(
                id=row["id"],
                subject=row["subject"],
                predicate=row["predicate"],
                value=row["value"],
                provenance=row["provenance"],
                source_entry_id=row["source_entry_id"] or "",
                fact_type=row["fact_type"] or "biographical",
                confidence=row["confidence"] or 1.0,
                created_at=row["created_at"] or "",
                superseded_by=row["superseded_by"],
                superseded_at=row["superseded_at"],
            )
        # Tuple fallback
        return UserFact(
            id=row[0], subject=row[1], predicate=row[2], value=row[3],
            provenance=row[4], source_entry_id=row[5] or "",
            fact_type=row[6] or "biographical", confidence=row[7] or 1.0,
            created_at=row[8] or "", superseded_by=row[9],
            superseded_at=row[10],
        )


# ─── Contradiction Detection ─────────────────────────────────────────────────

def detect_contradictions(store: UserFactsStore,
                          new_facts: List[UserFact]) -> List[Contradiction]:
    """
    Check new facts against existing facts for contradictions.

    Only checks objective predicates (locations, dates, names, counts).
    Skips subjective predicates (preferences, opinions).

    Contradiction types:
      - value_change: same subject+predicate, different value
        ("lives in California" vs "lives in Oregon")
      - count_mismatch: numeric values that differ
        ("age 38" vs "age 39")
      - temporal_conflict: date/year values that conflict
        ("founded in 2024" vs "founded in 2023")
    """
    contradictions = []

    for new_fact in new_facts:
        # Only check objective predicates
        if new_fact.predicate.lower() not in _OBJECTIVE_PREDICATES:
            continue

        # Find existing active facts with same subject + predicate
        existing = store.get_active_facts(
            subject=new_fact.subject,
            predicate=new_fact.predicate,
        )

        for old_fact in existing:
            # Skip if values are the same (case-insensitive)
            if old_fact.value.lower().strip() == new_fact.value.lower().strip():
                continue

            # Determine conflict type
            conflict_type = "value_change"
            old_numeric = _extract_number(old_fact.value)
            new_numeric = _extract_number(new_fact.value)

            if old_numeric is not None and new_numeric is not None:
                if _looks_like_year(old_fact.value) or _looks_like_year(new_fact.value):
                    conflict_type = "temporal_conflict"
                else:
                    conflict_type = "count_mismatch"

            description = (
                f"{new_fact.subject}'s {new_fact.predicate}: "
                f"was '{old_fact.value}' ({old_fact.provenance}, "
                f"{old_fact.created_at[:10] if old_fact.created_at else '?'}), "
                f"now '{new_fact.value}' ({new_fact.provenance})"
            )

            contradictions.append(Contradiction(
                existing_fact=old_fact,
                new_fact=new_fact,
                conflict_type=conflict_type,
                description=description,
            ))

    return contradictions


def resolve_contradictions(store: UserFactsStore,
                           contradictions: List[Contradiction],
                           auto_resolve: bool = True) -> List[Dict[str, Any]]:
    """
    Resolve detected contradictions.

    Resolution rules (when auto_resolve=True):
      1. User-stated always beats assistant-inferred.
      2. Newer user-stated beats older user-stated.
      3. If both are assistant-inferred, newer wins but flag for review.

    Returns list of resolution dicts for the ingestion output.
    """
    resolutions = []

    for c in contradictions:
        old = c.existing_fact
        new = c.new_fact

        resolution = {
            "conflict": c.description,
            "type": c.conflict_type,
            "old_fact_id": old.id,
            "new_fact_id": new.id,
        }

        if not auto_resolve:
            resolution["action"] = "flagged_for_review"
            resolutions.append(resolution)
            continue

        # Rule 1: user-stated beats assistant-inferred
        if new.provenance == "user-stated" and old.provenance == "assistant-inferred":
            store.supersede_fact(old.id, new.id)
            resolution["action"] = "superseded"
            resolution["reason"] = "user-stated overrides assistant-inferred"

        # Rule 2: newer user-stated beats older user-stated
        elif new.provenance == "user-stated" and old.provenance == "user-stated":
            store.supersede_fact(old.id, new.id)
            resolution["action"] = "superseded"
            resolution["reason"] = "newer user statement supersedes older"

        # Rule 3: both assistant-inferred — newer wins, flag for review
        elif new.provenance == "assistant-inferred" and old.provenance == "assistant-inferred":
            store.supersede_fact(old.id, new.id)
            resolution["action"] = "superseded_with_review"
            resolution["reason"] = "both inferred; newer wins but verify with user"

        # Rule 4: assistant-inferred trying to override user-stated — reject
        elif new.provenance == "assistant-inferred" and old.provenance == "user-stated":
            resolution["action"] = "rejected"
            resolution["reason"] = "assistant-inferred cannot override user-stated fact"

        else:
            store.supersede_fact(old.id, new.id)
            resolution["action"] = "superseded"
            resolution["reason"] = "newer fact replaces older"

        resolutions.append(resolution)

    return resolutions


# ─── Context Block Builder ───────────────────────────────────────────────────

def build_facts_block(facts: List[UserFact]) -> str:
    """
    Build a compact context block from user facts for inclusion in boot/recall.

    Format is concise and scannable:
      ═══ KNOWN FACTS ═══
      {owner}: lives in California | works at Acme | timezone is Pacific
      Acme: is a content agency | founded in 2024
      Wife: name is Sarah
      ═══ END KNOWN FACTS ═══
    """
    if not facts:
        return ""

    # Group by subject
    by_subject: Dict[str, List[UserFact]] = {}
    for f in facts:
        by_subject.setdefault(f.subject, []).append(f)

    lines = ["═══ KNOWN FACTS ═══"]
    for subject, subject_facts in by_subject.items():
        # Sort by confidence desc, then created_at desc
        subject_facts.sort(key=lambda f: (-f.confidence, f.created_at or ""), reverse=False)
        pairs = [f"{f.predicate} {f.value}" for f in subject_facts]
        lines.append(f"  {subject}: {' | '.join(pairs)}")
    lines.append("═══ END KNOWN FACTS ═══")

    return "\n".join(lines)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_number(value: str) -> Optional[float]:
    """Extract a number from a value string, if present."""
    match = re.search(r'(\d+\.?\d*)', value)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _looks_like_year(value: str) -> bool:
    """Check if a value looks like a year (1900-2100)."""
    match = re.search(r'\b(19|20)\d{2}\b', value)
    return match is not None


# ─── Pipeline Integration ────────────────────────────────────────────────────

def process_at_ingestion(
    conn: sqlite3.Connection,
    content: str,
    role: str,
    entry_ids: List[str],
    as_user_knowledge: bool = False,
) -> Dict[str, Any]:
    """
    Run the full user facts pipeline at ingestion time.

    Steps:
      1. Classify provenance
      2. Extract facts from content
      3. Store facts with source_entry_id links
      4. Detect contradictions against existing facts
      5. Auto-resolve contradictions
      6. Return stats for ingestion output

    This is the single entry point called from cmd_ingest.
    """
    stats: Dict[str, Any] = {}

    # Step 1: Provenance
    provenance = classify_provenance(role, as_user_knowledge, content)
    stats["provenance"] = provenance

    # Step 2: Extract facts
    facts = extract_facts(content, role, as_user_knowledge)
    if not facts:
        return stats

    # Step 3: Link to source entries and store
    store = UserFactsStore(conn)
    for i, fact in enumerate(facts):
        if i < len(entry_ids):
            fact.source_entry_id = entry_ids[i]

    # Step 4: Detect contradictions BEFORE storing new facts
    contradictions = detect_contradictions(store, facts)

    # Step 5: Resolve contradictions
    resolutions = []
    if contradictions:
        resolutions = resolve_contradictions(store, contradictions)
        stats["contradictions"] = [
            {
                "conflict": r["conflict"],
                "type": r["type"],
                "action": r["action"],
                "reason": r.get("reason", ""),
            }
            for r in resolutions
        ]

    # Step 6: Store facts (skip rejected ones)
    rejected_ids = {
        r["new_fact_id"] for r in resolutions
        if r.get("action") == "rejected"
    }
    stored = []
    for fact in facts:
        if fact.id not in rejected_ids:
            store.store_fact(fact)
            stored.append(fact)

    stats["facts_extracted"] = len(facts)
    stats["facts_stored"] = len(stored)
    stats["facts_rejected"] = len(rejected_ids)

    return stats
