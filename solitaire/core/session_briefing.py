"""
Fabric Layer Enrichment A: Session Briefing Block

Replaces the fragmented continuity_block + recent_context + last_session
with a single synthesized situational briefing covering the last 72 hours.

Organizes by active work streams, not by session. Heuristic synthesis,
no LLM calls.
"""

import json
import sqlite3
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from solitaire.core.types import estimate_tokens


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class WorkStream:
    """An active work stream aggregated across sessions."""
    label: str
    topic_id: Optional[str] = None
    cluster_id: Optional[str] = None
    last_active: Optional[str] = None
    last_active_relative: str = ""
    entry_count: int = 0
    session_ids: List[str] = field(default_factory=list)
    key_details: List[str] = field(default_factory=list)
    is_user_named: bool = False


@dataclass
class SessionDigest:
    """Condensed session info for the briefing."""
    session_id: str
    started_at: str
    ended_at: Optional[str]
    summary: str
    entry_count: int
    duration_minutes: float
    topics: List[str] = field(default_factory=list)
    high_signal_entries: List[str] = field(default_factory=list)


@dataclass
class BriefingResult:
    """Output of the briefing builder."""
    block: str
    sessions_in_window: int
    work_streams_found: int
    open_threads: int
    tokens_used: int
    # Carry forward for JSON output (replaces continuity_stats)
    sessions_closed: int = 0
    sessions_summarized: int = 0


# ─── Core Builder ────────────────────────────────────────────────────────────

def build_briefing_block(
    conn: sqlite3.Connection,
    current_session_id: str,
    window_hours: int = 72,
    budget_tokens: int = 1500,
    project_clusterer=None,
) -> BriefingResult:
    """
    Build a situational briefing from recent sessions.

    Args:
        conn: SQLite connection to the rolodex database
        current_session_id: Current session ID (excluded from briefing)
        window_hours: How far back to look (default 72h)
        budget_tokens: Max token budget for the block
        project_clusterer: Optional ProjectClusterer instance for work streams

    Returns:
        BriefingResult with the assembled block and metadata.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()

    # Step 1: Gather recent sessions within the window
    digests = _gather_session_digests(conn, current_session_id, cutoff)

    if not digests:
        # No recent sessions at all — nothing to brief on
        return BriefingResult(
            block="",
            sessions_in_window=0,
            work_streams_found=0,
            open_threads=0,
            tokens_used=0,
        )

    # Step 2: Close unclosed sessions (preserve continuity engine's side effect)
    sessions_closed = _close_unclosed_sessions(conn, digests, current_session_id)

    # Step 3: Identify active work streams
    work_streams = _identify_work_streams(conn, digests, project_clusterer)

    # Step 4: Detect open threads
    open_threads = _detect_open_threads(conn, digests)

    # Step 4b: Load pending contradictions
    contradiction_alerts = _load_contradiction_alerts(conn)

    # Step 5: Assemble the briefing
    block = _assemble_briefing(
        digests, work_streams, open_threads, budget_tokens,
        contradiction_alerts=contradiction_alerts,
    )

    return BriefingResult(
        block=block,
        sessions_in_window=len(digests),
        work_streams_found=len(work_streams),
        open_threads=len(open_threads),
        tokens_used=estimate_tokens(block),
        sessions_closed=sessions_closed,
        sessions_summarized=len(digests),
    )


# ─── Session Gathering ───────────────────────────────────────────────────────

def _gather_session_digests(
    conn: sqlite3.Connection,
    current_session_id: str,
    cutoff: str,
) -> List[SessionDigest]:
    """Gather condensed session digests within the time window."""
    # Find sessions with recent activity
    rows = conn.execute("""
        SELECT conversation_id,
               MIN(created_at) as first_at,
               MAX(created_at) as last_at,
               COUNT(*) as cnt
        FROM rolodex_entries
        WHERE superseded_by IS NULL
          AND conversation_id != ''
          AND conversation_id IS NOT NULL
          AND conversation_id != ?
          AND created_at >= ?
        GROUP BY conversation_id
        HAVING cnt >= 2
        ORDER BY last_at DESC
        LIMIT 10
    """, (current_session_id, cutoff)).fetchall()

    digests = []
    for row in rows:
        session_id, first_at, last_at, entry_count = row
        digest = _build_session_digest(conn, session_id, first_at, last_at, entry_count)
        if digest:
            digests.append(digest)

    return digests


def _build_session_digest(
    conn: sqlite3.Connection,
    session_id: str,
    first_at: str,
    last_at: str,
    entry_count: int,
) -> Optional[SessionDigest]:
    """Build a condensed digest for a single session."""
    # Check for existing summary in conversations table
    conv_row = conn.execute(
        "SELECT summary FROM conversations WHERE id = ?", (session_id,)
    ).fetchone()
    existing_summary = (conv_row[0] or "") if conv_row else ""

    # Get high-signal entries (corrections, implementations, user_knowledge)
    high_signal = conn.execute("""
        SELECT content, source_type, category
        FROM rolodex_entries
        WHERE conversation_id = ?
          AND superseded_by IS NULL
          AND (source_type IN ('correction', 'implementation', 'user_knowledge')
               OR category IN ('correction', 'implementation', 'user_knowledge'))
        ORDER BY created_at DESC
        LIMIT 5
    """, (session_id,)).fetchall()

    # Extract topics from tags
    tag_rows = conn.execute("""
        SELECT tags FROM rolodex_entries
        WHERE conversation_id = ?
          AND superseded_by IS NULL
          AND tags IS NOT NULL AND tags != '[]'
        ORDER BY created_at DESC
        LIMIT 20
    """, (session_id,)).fetchall()

    topics = _extract_topics_from_tags(tag_rows)

    # If no existing summary, synthesize one from entries
    if not existing_summary:
        existing_summary = _quick_synthesize(conn, session_id, topics)

    duration = _time_diff_minutes(first_at, last_at)

    return SessionDigest(
        session_id=session_id,
        started_at=first_at,
        ended_at=last_at,
        summary=existing_summary,
        entry_count=entry_count,
        duration_minutes=duration,
        topics=topics[:8],
        high_signal_entries=[row[0][:200] for row in high_signal],
    )


# ─── Work Stream Identification ──────────────────────────────────────────────

def _identify_work_streams(
    conn: sqlite3.Connection,
    digests: List[SessionDigest],
    project_clusterer=None,
) -> List[WorkStream]:
    """Identify active work streams from session digests and project clusters."""
    streams = []

    # Try project clusterer first (best signal)
    if project_clusterer:
        try:
            suggestions = project_clusterer.suggest_focus(limit=5)
            for s in suggestions:
                last_active = s.get("last_active", "")
                streams.append(WorkStream(
                    label=s.get("project_label") or "Unnamed",
                    topic_id=s.get("topic_id"),
                    cluster_id=s.get("cluster_id"),
                    last_active=last_active,
                    last_active_relative=_relative_time(last_active),
                    entry_count=s.get("entry_count", 0),
                    is_user_named=s.get("is_user_named", False),
                ))
        except Exception:
            pass

    # Only use clusterer streams if they have real labels
    named_streams = [s for s in streams if s.label and s.label != "Unnamed"]
    if named_streams:
        _enrich_streams_from_digests(named_streams, digests)
        return named_streams[:5]

    # Fallback: group by topic overlap across sessions (better labels from tags)
    return _streams_from_topic_overlap(digests)[:5]


def _enrich_streams_from_digests(
    streams: List[WorkStream],
    digests: List[SessionDigest],
) -> None:
    """Add session-level detail to work streams by matching topics."""
    for stream in streams:
        if not stream.label:
            continue
        label_words = set(stream.label.lower().split())
        for digest in digests:
            digest_words = set(w.lower() for t in digest.topics for w in t.split())
            if label_words & digest_words:
                stream.session_ids.append(digest.session_id[:8])
                # Pull the most relevant high-signal entry
                for hs in digest.high_signal_entries:
                    if any(w in hs.lower() for w in label_words):
                        stream.key_details.append(hs[:150])
                        break


def _streams_from_topic_overlap(digests: List[SessionDigest]) -> List[WorkStream]:
    """Fallback: identify work streams from topic co-occurrence across sessions."""
    topic_sessions: Dict[str, List[str]] = {}
    topic_last_seen: Dict[str, str] = {}
    topic_counts: Dict[str, int] = {}

    for digest in digests:
        for topic in digest.topics:
            t = topic.lower()
            if t not in topic_sessions:
                topic_sessions[t] = []
                topic_counts[t] = 0
            topic_sessions[t].append(digest.session_id[:8])
            topic_counts[t] += digest.entry_count
            if not topic_last_seen.get(t) or digest.ended_at > topic_last_seen.get(t, ""):
                topic_last_seen[t] = digest.ended_at or digest.started_at

    # Topics that appear in 2+ sessions are likely work streams
    multi_session_topics = [
        (t, len(sessions), topic_counts[t])
        for t, sessions in topic_sessions.items()
        if len(sessions) >= 2
    ]
    multi_session_topics.sort(key=lambda x: (-x[1], -x[2]))

    streams = []
    for topic, session_count, entry_count in multi_session_topics[:5]:
        streams.append(WorkStream(
            label=topic.title(),
            last_active=topic_last_seen.get(topic),
            last_active_relative=_relative_time(topic_last_seen.get(topic, "")),
            entry_count=entry_count,
            session_ids=topic_sessions[topic],
        ))

    return streams


# ─── Open Thread Detection ───────────────────────────────────────────────────

def _detect_open_threads(
    conn: sqlite3.Connection,
    digests: List[SessionDigest],
) -> List[str]:
    """Detect open threads from recent session entries.

    Looks for:
    - Entries with 'pending', 'todo', 'next', 'deferred' in content
    - Implementation entries that mention 'remaining' or 'phase'
    - Session summaries that mention future work
    """
    threads = []
    session_ids = [d.session_id for d in digests[:5]]

    if not session_ids:
        return threads

    placeholders = ",".join(["?"] * len(session_ids))
    rows = conn.execute(f"""
        SELECT content, source_type
        FROM rolodex_entries
        WHERE conversation_id IN ({placeholders})
          AND superseded_by IS NULL
          AND (
            content LIKE '%remaining%'
            OR content LIKE '%pending%'
            OR content LIKE '%next step%'
            OR content LIKE '%next session%'
            OR content LIKE '%deferred%'
            OR content LIKE '%todo%'
            OR content LIKE '%phase%remaining%'
          )
        ORDER BY created_at DESC
        LIMIT 10
    """, session_ids).fetchall()

    seen = set()
    for row in rows:
        content = row[0] or ""
        # Extract the sentence containing the trigger word
        thread = _extract_thread_sentence(content)
        if thread and thread not in seen:
            seen.add(thread)
            threads.append(thread)

    # Also check session summaries for future-looking language
    for digest in digests[:3]:
        if digest.summary:
            for pattern in [r"(?:next|remaining|deferred|todo)[^.]*\.", r"phase \d[^.]*\."]:
                matches = re.findall(pattern, digest.summary, re.IGNORECASE)
                for m in matches:
                    m = m.strip()
                    if m and m not in seen and len(m) > 20:
                        seen.add(m)
                        threads.append(m)

    return threads[:5]


def _extract_thread_sentence(content: str) -> Optional[str]:
    """Extract the most relevant sentence from content for open thread detection."""
    sentences = re.split(r'[.!?\n]', content)
    # Prioritize strong signals (explicit future work) over weak ones (incidental "remaining")
    strong_triggers = {"pending work", "next session", "next step", "deferred to",
                       "todo:", "remaining work", "queued for", "open thread"}
    weak_triggers = {"remaining", "pending", "phase"}

    best = None
    best_is_strong = False

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20 or len(sentence) > 200:
            continue
        lower = sentence.lower()
        is_strong = any(tw in lower for tw in strong_triggers)
        is_weak = any(tw in lower for tw in weak_triggers)
        if is_strong and not best_is_strong:
            best = sentence
            best_is_strong = True
        elif is_weak and not best:
            best = sentence

    return best


# ─── Contradiction Alert Loading ────────────────────────────────────────────

def _load_contradiction_alerts(conn: sqlite3.Connection) -> List[str]:
    """Load pending contradictions for briefing display."""
    try:
        rows = conn.execute(
            """
            SELECT pc.description
            FROM pending_contradictions pc
            WHERE pc.resolved_at IS NULL
            ORDER BY pc.detected_at DESC
            LIMIT 3
            """
        ).fetchall()
        return [row[0] for row in rows]
    except Exception:
        return []


# ─── Briefing Assembly ───────────────────────────────────────────────────────

def _assemble_briefing(
    digests: List[SessionDigest],
    work_streams: List[WorkStream],
    open_threads: List[str],
    budget_tokens: int,
    contradiction_alerts: Optional[List[str]] = None,
) -> str:
    """Assemble the final briefing block within token budget."""
    parts = ["═══ SITUATIONAL BRIEFING ═══", ""]

    # Section 1: Active work streams
    if work_streams:
        parts.append("Active work streams:")
        for ws in work_streams[:4]:
            time_str = ws.last_active_relative or _relative_time(ws.last_active or "")
            parts.append(f"- {ws.label} (last active: {time_str}, {ws.entry_count} entries)")
            if ws.key_details:
                parts.append(f"  {ws.key_details[0]}")
        parts.append("")

    # Section 2: Recent session summaries
    parts.append("Recent sessions:")
    for digest in digests[:4]:
        time_str = _relative_time(digest.ended_at or digest.started_at)
        dur = digest.duration_minutes
        dur_str = f"~{dur / 60:.1f}h" if dur > 60 else f"~{int(dur)}m"
        summary = digest.summary[:250] if digest.summary else "(no summary)"
        parts.append(f"- [{time_str}, {dur_str}] {summary}")
    parts.append("")

    # Section 3: Open threads
    if open_threads:
        parts.append("Open threads:")
        for thread in open_threads[:3]:
            parts.append(f"- {thread[:150]}")
        parts.append("")

    # Section 4: Contradiction alerts
    if contradiction_alerts:
        parts.append("Pending contradictions (needs resolution):")
        for alert in contradiction_alerts[:3]:
            parts.append(f"- {alert[:200]}")
        parts.append("")

    parts.append("═══ END BRIEFING ═══")

    block = "\n".join(parts)

    # Trim if over budget
    tokens = estimate_tokens(block)
    if tokens > budget_tokens:
        # Progressively trim: contradiction alerts first, then open threads, sessions, streams
        if contradiction_alerts and len(contradiction_alerts) > 1:
            return _assemble_briefing(digests, work_streams, open_threads, budget_tokens, contradiction_alerts[:1])
        if open_threads:
            return _assemble_briefing(digests, work_streams, open_threads[:1], budget_tokens, [])
        if len(digests) > 2:
            return _assemble_briefing(digests[:2], work_streams, [], budget_tokens, [])
        if len(work_streams) > 2:
            return _assemble_briefing(digests, work_streams[:2], [], budget_tokens, [])
        # Last resort: hard truncate
        target_chars = budget_tokens * 4
        block = block[:target_chars] + "\n\n═══ END BRIEFING ═══"

    return block


# ─── Close Unclosed Sessions (side effect preserved from continuity engine) ──

def _close_unclosed_sessions(
    conn: sqlite3.Connection,
    digests: List[SessionDigest],
    current_session_id: str,
) -> int:
    """Close any sessions that are still marked active in conversations table."""
    closed = 0
    for digest in digests:
        if digest.session_id == current_session_id:
            continue
        try:
            cursor = conn.execute("""
                UPDATE conversations
                SET status = 'ended',
                    ended_at = COALESCE(?, ended_at),
                    summary = COALESCE(summary, ?)
                WHERE id = ?
                  AND status = 'active'
                  AND (summary IS NULL OR summary = '')
            """, (
                digest.ended_at or datetime.now(timezone.utc).isoformat(),
                digest.summary,
                digest.session_id,
            ))
            conn.commit()
            if cursor.rowcount > 0:
                closed += 1
        except Exception:
            pass
    return closed


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_topics_from_tags(tag_rows: list) -> List[str]:
    """Extract meaningful topics from tag JSON arrays."""
    noise = {
        "prose", "code", "conversational", "structured", "mixed",
        "source:reingestion", "role:user", "role:assistant",
    }
    date_pattern = re.compile(
        r"^\d{4}(-\d{2})?$|^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$"
        r"|^\d{4}-\d{2}-\d{2}$|^(january|february|march|april|may|june|july|august"
        r"|september|october|november|december)$"
        r"|^\d{4}$|^(morning|afternoon|evening|night)$|^\d+[apm]+$"
    )
    topic_counts: Dict[str, int] = {}
    for row in tag_rows:
        try:
            tags = json.loads(row[0]) if row[0] else []
        except (json.JSONDecodeError, TypeError):
            continue
        for tag in tags:
            if not tag or not isinstance(tag, str):
                continue
            t = tag.lower().strip()
            if t in noise or date_pattern.match(t) or len(t) < 3:
                continue
            # Skip attributed: prefixed tags
            if t.startswith("attributed:"):
                continue
            topic_counts[t] = topic_counts.get(t, 0) + 1

    # Return most frequent topics
    sorted_topics = sorted(topic_counts.items(), key=lambda x: -x[1])
    return [t for t, _ in sorted_topics]


def _quick_synthesize(
    conn: sqlite3.Connection,
    session_id: str,
    topics: List[str],
) -> str:
    """Quick heuristic summary when no conversation summary exists."""
    # Get first and last non-trivial entries
    entries = conn.execute("""
        SELECT content, source_type
        FROM rolodex_entries
        WHERE conversation_id = ?
          AND superseded_by IS NULL
          AND length(content) > 50
        ORDER BY created_at ASC
        LIMIT 3
    """, (session_id,)).fetchall()

    if not entries:
        if topics:
            return f"Session covering: {', '.join(topics[:5])}"
        return "Session with minimal captured content."

    # Use the first substantive entry as the summary seed
    first_content = entries[0][0][:200]
    topic_str = f" Topics: {', '.join(topics[:4])}." if topics else ""

    return f"{first_content}{topic_str}"


def _time_diff_minutes(start: str, end: str) -> float:
    """Calculate time difference in minutes between two ISO timestamps."""
    try:
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return (e - s).total_seconds() / 60
    except (ValueError, TypeError):
        return 0


def _relative_time(timestamp: str) -> str:
    """Convert ISO timestamp to relative time string."""
    if not timestamp:
        return "unknown"
    try:
        t = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        # Handle timezone-aware vs naive
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        delta = now - t
        if delta.days == 0:
            hours = delta.seconds // 3600
            if hours == 0:
                minutes = delta.seconds // 60
                return f"{minutes}m ago" if minutes > 0 else "just now"
            return f"{hours}h ago"
        elif delta.days == 1:
            return "yesterday"
        else:
            return f"{delta.days}d ago"
    except (ValueError, TypeError):
        return "unknown"
