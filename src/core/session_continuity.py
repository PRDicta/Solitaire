"""
The Librarian — Session Continuity Engine

Solves the "cold boot" problem: when a session starts, the LLM needs to know
what happened in the previous session(s). Since 99% of sessions are never
formally closed (user just closes the window), this module builds continuity
context directly from the rolodex entries.

Key insight: the conversations table is unreliable (185 of 191 session IDs
in rolodex_entries have no matching conversations row). So we work directly
from rolodex_entries, grouping by conversation_id and using entry timestamps
to determine session boundaries.

Pipeline (runs at boot, before context block is built):
1. Find recent sessions by scanning distinct conversation_ids in rolodex_entries
2. For each, gather entries and synthesize a heuristic summary
3. Optionally close the session in the conversations table (if row exists)
4. Build a continuity block for the most recent N sessions

The continuity block is what the LLM sees. It's the "here's where we left off"
signal that makes session transitions feel seamless.

Design principles:
- No LLM calls. Summaries are synthesized from entry content heuristically.
- Works even when conversations table is out of sync with rolodex_entries.
- Idempotent. Running twice produces the same result.
- Fast. Must complete in <2 seconds to not delay boot.
- Never blocks boot on failure.
"""

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class SessionSummary:
    """A synthesized summary of a session."""
    session_id: str
    summary: str
    topics: List[str]
    entry_count: int
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_minutes: float = 0.0
    key_entries: List[str] = field(default_factory=list)


@dataclass
class ContinuityBlock:
    """The context block inserted at boot for session continuity."""
    text: str
    sessions_closed: int
    sessions_summarized: int
    most_recent_summary: Optional[SessionSummary] = None


# ─── Core Engine ─────────────────────────────────────────────────────────────

class SessionContinuityEngine:
    """
    Runs at boot to build continuity context from recent sessions.
    Works directly from rolodex_entries, bypassing the conversations table.
    """

    def __init__(self, conn: sqlite3.Connection, current_session_id: str):
        self.conn = conn
        self.current_session_id = current_session_id

    def run(self, max_sessions: int = 3) -> ContinuityBlock:
        """
        Main entry point. Called at boot before context block is built.

        Returns:
            ContinuityBlock with text for the context window.
        """
        # Step 1: Find recent sessions from rolodex_entries
        recent_sessions = self._find_recent_sessions(limit=max_sessions + 2)

        # Step 2: Synthesize summaries for each
        summaries = []
        sessions_closed = 0
        for session_id, first_at, last_at, entry_count in recent_sessions:
            if session_id == self.current_session_id:
                continue
            if len(summaries) >= max_sessions:
                break

            summary = self._synthesize_session(session_id, first_at, last_at)
            if summary:
                summaries.append(summary)
                # Try to close in conversations table if row exists and is active
                closed = self._try_close_session(session_id, summary)
                if closed:
                    sessions_closed += 1

        # Step 3: Build continuity block
        text = self._build_continuity_text(summaries)

        return ContinuityBlock(
            text=text,
            sessions_closed=sessions_closed,
            sessions_summarized=len(summaries),
            most_recent_summary=summaries[0] if summaries else None,
        )

    def _find_recent_sessions(self, limit: int = 5) -> List[Tuple[str, str, str, int]]:
        """
        Find recent sessions by scanning rolodex_entries directly.

        Returns list of (session_id, first_entry_at, last_entry_at, entry_count),
        ordered by most recent activity.
        """
        rows = self.conn.execute("""
            SELECT conversation_id,
                   MIN(created_at) as first_at,
                   MAX(created_at) as last_at,
                   COUNT(*) as cnt
            FROM rolodex_entries
            WHERE superseded_by IS NULL
              AND conversation_id != ''
              AND conversation_id IS NOT NULL
            GROUP BY conversation_id
            HAVING cnt >= 2
            ORDER BY last_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [(row[0], row[1], row[2], row[3]) for row in rows]

    def _synthesize_session(self, session_id: str, first_at: str,
                            last_at: str) -> Optional[SessionSummary]:
        """
        Build a summary of a session from its ingested entries.
        No LLM calls — purely heuristic extraction.
        """
        entries = self.conn.execute("""
            SELECT id, content, category, tags, created_at, source_type
            FROM rolodex_entries
            WHERE conversation_id = ?
              AND superseded_by IS NULL
            ORDER BY created_at ASC
        """, (session_id,)).fetchall()

        if not entries:
            return None

        entry_data = []
        for e in entries:
            tags = []
            try:
                tags = json.loads(e[3]) if e[3] else []
            except (json.JSONDecodeError, TypeError):
                pass
            entry_data.append({
                "id": e[0],
                "content": e[1] or "",
                "category": e[2] or "note",
                "tags": tags,
                "created_at": e[4],
                "source_type": e[5] or "conversation",
            })

        duration = _time_diff_minutes(first_at, last_at)
        topics = _extract_topics(entry_data)
        key_entries = _pick_key_entries(entry_data, max_entries=5)
        summary_text = _compose_summary(entry_data, topics, key_entries, duration)

        return SessionSummary(
            session_id=session_id,
            summary=summary_text,
            topics=topics[:10],
            entry_count=len(entries),
            started_at=first_at,
            ended_at=last_at,
            duration_minutes=duration,
            key_entries=[e["content"][:200] for e in key_entries],
        )

    def _try_close_session(self, session_id: str,
                           summary: SessionSummary) -> bool:
        """
        Try to close a session in the conversations table.
        Returns True if a row was updated, False otherwise.
        No-op if the row doesn't exist or is already ended.
        """
        try:
            cursor = self.conn.execute("""
                UPDATE conversations
                SET status = 'ended',
                    ended_at = COALESCE(?, ended_at),
                    summary = COALESCE(summary, ?)
                WHERE id = ?
                  AND status = 'active'
                  AND (summary IS NULL OR summary = '')
            """, (summary.ended_at or datetime.utcnow().isoformat(),
                  summary.summary, session_id))
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception:
            return False

    def _build_continuity_text(self, summaries: List[SessionSummary]) -> str:
        """
        Build the continuity block text that gets injected into context.

        This is the critical output. The LLM reads this and knows what
        happened in recent sessions without needing to be told.
        """
        if not summaries:
            return ""

        lines = ["═══ SESSION CONTINUITY ═══", ""]

        for i, s in enumerate(summaries[:3]):
            label = "Most recent" if i == 0 else "Prior" if i == 1 else "Earlier"
            time_str = _format_session_time(s.ended_at or s.started_at)

            lines.append(f"[{label} — {time_str}]")
            lines.append(s.summary)

            meta_parts = []
            if s.entry_count > 0:
                meta_parts.append(f"{s.entry_count} entries")
            if s.duration_minutes > 1:
                dur = s.duration_minutes
                if dur > 60:
                    meta_parts.append(f"~{dur / 60:.1f}h")
                else:
                    meta_parts.append(f"~{int(dur)} min")
            if meta_parts:
                lines.append(f"({', '.join(meta_parts)})")
            lines.append("")

        lines.append("═══ END SESSION CONTINUITY ═══")
        return "\n".join(lines)


# ─── Heuristic Summary Synthesis ─────────────────────────────────────────────

def _extract_topics(entries: List[Dict]) -> List[str]:
    """Extract topic keywords from entry tags, deduplicated."""
    noise = {
        "prose", "code", "conversational", "structured", "mixed",
        "source:reingestion", "role:user", "role:assistant",
    }
    date_pattern = re.compile(
        r"^\d{4}(-\d{2})?$|^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$"
        r"|^(january|february|march|april|may|june|july|august|september|october|november|december)$"
        r"|^\d{1,2}(am|pm)$|^(morning|afternoon|evening|night)$"
        r"|^\d{4}-\d{2}-\d{2}$"
    )

    topic_counts: Dict[str, int] = {}
    for e in entries:
        for tag in e.get("tags", []):
            tag_lower = tag.lower().strip()
            if tag_lower in noise or date_pattern.match(tag_lower):
                continue
            if len(tag_lower) < 3:
                continue
            # Skip year-only tags
            if re.match(r"^\d{4}$", tag_lower):
                continue
            topic_counts[tag_lower] = topic_counts.get(tag_lower, 0) + 1

    sorted_topics = sorted(topic_counts.items(), key=lambda x: -x[1])
    return [t[0] for t in sorted_topics[:15]]


_CATEGORY_PRIORITY = {
    "user_knowledge": 10,
    "correction": 9,
    "decision": 8,
    "breakthrough": 7,
    "pivot": 6,
    "preference": 5,
    "implementation": 4,
    "fact": 3,
    "warning": 3,
    "note": 2,
    "reference": 1,
    "example": 1,
    "definition": 1,
}


_NOISE_PATTERNS = [
    re.compile(r"^(?:It works\.\s+)?Base directory", re.IGNORECASE),
    re.compile(r"^#+ .*(?:SKILL|Guide|Overview|Processing|Quick Start|Installation)", re.IGNORECASE),
    re.compile(r"^from \w+ import |^import \w+"),
    re.compile(r"^\{\"(signal|status|housekeeping|error)\""),
    re.compile(r"^```|^---$|^==="),
    re.compile(r"^\[Chief Librarian\]"),
    re.compile(r"^This (?:guide|skill|tool) (?:covers|provides|helps)", re.IGNORECASE),
    re.compile(r"^No response requested"),
    re.compile(r"^Processing\.\.\.|^Loading\.\.\.|^Booting"),
]


def _is_noise(content: str) -> bool:
    """Check if content is system boilerplate or noise."""
    content = content.strip()
    if len(content) < 10:
        return True
    # Check the first meaningful line
    first_line = content.split('\n')[0].strip()
    if any(p.search(first_line) for p in _NOISE_PATTERNS):
        return True
    return False


def _clean_content(content: str) -> str:
    """Strip noise prefixes from content before summarizing."""
    lines = content.strip().split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(p.search(stripped) for p in _NOISE_PATTERNS):
            continue
        cleaned.append(stripped)
    return ' '.join(cleaned) if cleaned else content.strip()


def _pick_key_entries(entries: List[Dict], max_entries: int = 5) -> List[Dict]:
    """Pick the most significant entries from a session."""
    scored = []
    for e in entries:
        content = e.get("content", "").strip()

        # Skip noise entries entirely
        if _is_noise(content):
            continue

        cat = e.get("category", "note")
        priority = _CATEGORY_PRIORITY.get(cat, 1)

        if e.get("source_type") == "user_knowledge":
            priority += 3

        content_len = len(content)
        if content_len > 200:
            priority += 1
        if content_len > 500:
            priority += 1

        scored.append((priority, e))

    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:max_entries]]


def _compose_summary(entries: List[Dict], topics: List[str],
                     key_entries: List[Dict], duration_minutes: float) -> str:
    """
    Compose a concise summary from entry data.
    Lead with the most significant content, keep it scannable.
    """
    if not entries:
        return "Empty session."

    fragments = []
    total_chars = 0
    max_chars = 600

    for e in key_entries:
        content = e.get("content", "").strip()
        if not content:
            continue

        # Clean noise prefixes before extracting sentence
        content = _clean_content(content)
        if not content or _is_noise(content):
            continue

        first = _first_sentence(content)

        # Skip if extracted sentence is itself noise
        if _is_noise(first):
            continue
        if len(first) > 150:
            first = first[:147] + "..."

        if total_chars + len(first) > max_chars:
            break

        fragments.append(first)
        total_chars += len(first)

    summary = " ".join(fragments) if fragments else _first_sentence(entries[-1]["content"])

    if topics and total_chars < max_chars - 50:
        topic_str = ", ".join(topics[:5])
        summary += f" Topics: {topic_str}."

    return summary.strip()


def _first_sentence(text: str) -> str:
    """Extract the first sentence from text."""
    text = text.strip()
    match = re.search(r'[.!?](?:\s|$)', text)
    if match and match.end() < 200:
        return text[:match.end()].strip()
    if len(text) > 150:
        cut = text[:150].rfind(" ")
        if cut > 80:
            return text[:cut] + "..."
        return text[:150] + "..."
    return text


def _time_diff_minutes(start: str, end: str) -> float:
    """Calculate time difference in minutes between two ISO timestamps."""
    try:
        t1 = datetime.fromisoformat(start.replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return max(0, (t2 - t1).total_seconds() / 60)
    except (ValueError, TypeError):
        return 0.0


def _format_session_time(timestamp: Optional[str]) -> str:
    """Format a timestamp for the continuity block header."""
    if not timestamp:
        return "unknown time"
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        # Convert UTC to PT (rough: -8 hours, good enough for display)
        pt = dt - timedelta(hours=8)
        # Cross-platform: %-d and %-I are POSIX-only, %#d and %#I are Windows-only
        day = pt.day
        hour12 = pt.hour % 12 or 12
        minute = pt.strftime("%M")
        ampm = pt.strftime("%p")
        month = pt.strftime("%b")
        return f"{month} {day}, {hour12}:{minute} {ampm} PT"
    except (ValueError, TypeError, OSError):
        return timestamp[:16]


# ─── Integration Entry Point ─────────────────────────────────────────────────

def build_continuity_at_boot(
    conn: sqlite3.Connection,
    current_session_id: str,
    max_sessions: int = 3,
) -> Dict[str, Any]:
    """
    Single entry point called from boot code.

    Returns dict with:
    - continuity_block: str (the text to inject into context)
    - sessions_closed: int
    - sessions_summarized: int
    - stats: dict with details

    Never raises — returns empty block on any failure.
    """
    try:
        engine = SessionContinuityEngine(conn, current_session_id)
        result = engine.run(max_sessions=max_sessions)
        return {
            "continuity_block": result.text,
            "sessions_closed": result.sessions_closed,
            "sessions_summarized": result.sessions_summarized,
            "most_recent": {
                "session_id": result.most_recent_summary.session_id[:8],
                "summary": result.most_recent_summary.summary[:300],
                "entry_count": result.most_recent_summary.entry_count,
                "ended_at": result.most_recent_summary.ended_at,
            } if result.most_recent_summary else None,
        }
    except Exception as e:
        return {
            "continuity_block": "",
            "sessions_closed": 0,
            "sessions_summarized": 0,
            "error": str(e),
        }
