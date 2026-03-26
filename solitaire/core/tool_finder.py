"""
Solitaire — Proactive Tool Finding (Phase 19: Self-Learning Layer 2c)

When pattern detection (2b) identifies recurring gap signals, this module
searches for relevant tools, MCP servers, or skills that could fill the gap.
Proposals are surfaced during boot or in-session. Nothing installs silently.

The module is registry-agnostic: actual search is delegated to pluggable
SearchProvider callables. This lets different deployment environments
(agentskills.io, MCP registry, web search, platform marketplaces) plug in
without changing core logic.

Flow:
    1. Gap signals from retrieval_patterns.py trigger search
    2. search_for_gap() queries registered providers
    3. Results become ToolProposal records (persisted in tool_proposals table)
    4. get_pending_proposals() surfaces them at boot or in-session
    5. User approves/dismisses via confirm_proposal() / dismiss_proposal()
    6. Usage tracking via record_tool_usage() feeds back into the loop
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from .retrieval_patterns import detect_gap_signals


# ─── Constants ──────────────────────────────────────────────────────────────

PROPOSAL_STALENESS_DAYS = 30    # Dismiss proposals older than this
MAX_PROPOSALS_PER_GAP = 3       # Don't flood: top 3 results per gap signal
USAGE_REVIEW_DAYS = 14          # Installed tools unused for this long get flagged
MIN_GAP_OCCURRENCES = 3         # Only search for gaps with 3+ occurrences


# ─── Schema ─────────────────────────────────────────────────────────────────

TOOL_PROPOSALS_SCHEMA = """
-- Tool proposals: tracks discovered, proposed, installed, and used tools
CREATE TABLE IF NOT EXISTS tool_proposals (
    id TEXT PRIMARY KEY,
    gap_pattern TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_description TEXT NOT NULL,
    tool_source TEXT NOT NULL,
    source_url TEXT DEFAULT '',
    permissions TEXT DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'proposed',
    proposed_at DATETIME NOT NULL,
    decided_at DATETIME,
    installed_at DATETIME,
    last_used DATETIME,
    use_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_tool_proposals_status
    ON tool_proposals(status);
CREATE INDEX IF NOT EXISTS idx_tool_proposals_gap
    ON tool_proposals(gap_pattern);
CREATE INDEX IF NOT EXISTS idx_tool_proposals_proposed
    ON tool_proposals(proposed_at DESC);
"""


def ensure_tool_proposals_schema(conn: sqlite3.Connection):
    """Create the tool_proposals table if it doesn't exist."""
    conn.executescript(TOOL_PROPOSALS_SCHEMA)
    conn.commit()


# ─── Search Provider Interface ──────────────────────────────────────────────

# A SearchProvider is a callable with signature:
#   (query: str, limit: int) -> List[Dict[str, Any]]
#
# Each result dict must contain:
#   - name: str          (tool/skill/MCP server name)
#   - description: str   (what it does)
#   - source: str        (registry name, e.g. "agentskills.io", "mcp-registry")
#   - url: str           (where to get it)
#   - permissions: list   (what access it needs, e.g. ["network", "filesystem"])
#
# Optional fields:
#   - relevance: float   (0-1 relevance score, for ranking)
#   - install_command: str (e.g. "npx @modelcontextprotocol/install ...")
#   - metadata: dict     (anything else)

SearchProvider = Callable[[str, int], List[Dict[str, Any]]]


# ─── Core Functions ─────────────────────────────────────────────────────────

def search_for_gap(
    conn: sqlite3.Connection,
    gap_pattern: str,
    providers: List[SearchProvider],
    limit: int = MAX_PROPOSALS_PER_GAP,
) -> List[Dict[str, Any]]:
    """
    Search registered providers for tools matching a gap pattern.

    Args:
        conn: Database connection (for deduplication against existing proposals).
        gap_pattern: The gap signal's query pattern (space-separated key terms).
        providers: List of search provider callables.
        limit: Max results per provider.

    Returns:
        List of tool result dicts from providers, deduplicated against
        existing proposals for this gap.
    """
    ensure_tool_proposals_schema(conn)

    # Check what's already proposed for this gap (avoid re-proposing)
    existing = conn.execute(
        """SELECT tool_name, tool_source FROM tool_proposals
           WHERE gap_pattern = ? AND status IN ('proposed', 'approved', 'installed')""",
        (gap_pattern,),
    ).fetchall()
    existing_keys = {(r["tool_name"], r["tool_source"]) for r in existing}

    all_results = []
    for provider in providers:
        try:
            results = provider(gap_pattern, limit)
            for r in results:
                key = (r.get("name", ""), r.get("source", ""))
                if key not in existing_keys:
                    all_results.append(r)
                    existing_keys.add(key)
        except Exception:
            continue  # Provider failure is non-fatal

    # Sort by relevance if available, take top N
    all_results.sort(key=lambda x: x.get("relevance", 0.5), reverse=True)
    return all_results[:limit]


def generate_proposals(
    conn: sqlite3.Connection,
    providers: List[SearchProvider],
    gap_window_days: int = 14,
    min_occurrences: int = MIN_GAP_OCCURRENCES,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Run the full gap-to-search pipeline.

    1. Detect gap signals from retrieval patterns
    2. For each qualifying gap, search providers for relevant tools
    3. Store new proposals in tool_proposals table
    4. Return list of newly created proposals

    Args:
        conn: Database connection.
        providers: Search provider callables.
        gap_window_days: Window for gap signal detection.
        min_occurrences: Minimum gap occurrences before triggering search.
        now: Override current time (for testing).

    Returns:
        List of newly created proposal dicts.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    ensure_tool_proposals_schema(conn)

    # Get gap signals
    gaps = detect_gap_signals(
        conn, threshold=min_occurrences, window_days=gap_window_days, now=now,
    )

    if not gaps:
        return []

    new_proposals = []
    for gap in gaps:
        pattern = gap["query_pattern"]

        # Search providers for this gap
        results = search_for_gap(conn, pattern, providers)

        for tool in results:
            proposal_id = str(uuid.uuid4())
            proposal = {
                "id": proposal_id,
                "gap_pattern": pattern,
                "tool_name": tool.get("name", "unknown"),
                "tool_description": tool.get("description", ""),
                "tool_source": tool.get("source", "unknown"),
                "source_url": tool.get("url", ""),
                "permissions": json.dumps(tool.get("permissions", [])),
                "status": "proposed",
                "proposed_at": now.isoformat(),
                "metadata": json.dumps({
                    "gap_occurrences": gap["occurrences"],
                    "sample_queries": gap.get("sample_queries", []),
                    "relevance": tool.get("relevance", 0.5),
                    "install_command": tool.get("install_command", ""),
                }),
            }

            conn.execute(
                """INSERT INTO tool_proposals
                   (id, gap_pattern, tool_name, tool_description, tool_source,
                    source_url, permissions, status, proposed_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    proposal["id"], proposal["gap_pattern"],
                    proposal["tool_name"], proposal["tool_description"],
                    proposal["tool_source"], proposal["source_url"],
                    proposal["permissions"], proposal["status"],
                    proposal["proposed_at"], proposal["metadata"],
                ),
            )
            new_proposals.append(proposal)

    conn.commit()
    return new_proposals


def get_pending_proposals(
    conn: sqlite3.Connection,
    include_stale: bool = False,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Get all proposals awaiting user decision.

    Called during boot to surface tool suggestions. Returns proposals
    in order of gap severity (most-asked-about gaps first).

    Args:
        conn: Database connection.
        include_stale: If True, include proposals older than PROPOSAL_STALENESS_DAYS.
        now: Override current time (for testing).

    Returns:
        List of proposal dicts with: id, gap_pattern, tool_name,
        tool_description, tool_source, source_url, permissions, proposed_at, metadata.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    ensure_tool_proposals_schema(conn)

    if include_stale:
        rows = conn.execute(
            """SELECT * FROM tool_proposals
               WHERE status = 'proposed'
               ORDER BY proposed_at DESC""",
        ).fetchall()
    else:
        cutoff = (now - timedelta(days=PROPOSAL_STALENESS_DAYS)).isoformat()
        rows = conn.execute(
            """SELECT * FROM tool_proposals
               WHERE status = 'proposed' AND proposed_at > ?
               ORDER BY proposed_at DESC""",
            (cutoff,),
        ).fetchall()

    proposals = []
    for r in rows:
        proposals.append({
            "id": r["id"],
            "gap_pattern": r["gap_pattern"],
            "tool_name": r["tool_name"],
            "tool_description": r["tool_description"],
            "tool_source": r["tool_source"],
            "source_url": r["source_url"],
            "permissions": json.loads(r["permissions"]) if r["permissions"] else [],
            "proposed_at": r["proposed_at"],
            "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
        })

    return proposals


def confirm_proposal(
    conn: sqlite3.Connection,
    proposal_id: str,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    User approves a tool proposal. Marks it as approved (ready for installation).

    Args:
        conn: Database connection.
        proposal_id: The proposal to approve.
        now: Override current time.

    Returns:
        Dict with status and proposal details, including install_command if available.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    ensure_tool_proposals_schema(conn)

    row = conn.execute(
        "SELECT * FROM tool_proposals WHERE id = ?", (proposal_id,),
    ).fetchone()

    if not row:
        return {"status": "error", "message": "Proposal not found"}

    if row["status"] != "proposed":
        return {"status": "error", "message": f"Proposal already {row['status']}"}

    conn.execute(
        """UPDATE tool_proposals SET status = 'approved', decided_at = ?
           WHERE id = ?""",
        (now.isoformat(), proposal_id),
    )
    conn.commit()

    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    return {
        "status": "approved",
        "proposal_id": proposal_id,
        "tool_name": row["tool_name"],
        "tool_source": row["tool_source"],
        "source_url": row["source_url"],
        "permissions": json.loads(row["permissions"]) if row["permissions"] else [],
        "install_command": metadata.get("install_command", ""),
    }


def dismiss_proposal(
    conn: sqlite3.Connection,
    proposal_id: str,
    reason: str = "",
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    User dismisses a tool proposal.

    Args:
        conn: Database connection.
        proposal_id: The proposal to dismiss.
        reason: Optional reason for dismissal.
        now: Override current time.

    Returns:
        Dict with status.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    ensure_tool_proposals_schema(conn)

    row = conn.execute(
        "SELECT status FROM tool_proposals WHERE id = ?", (proposal_id,),
    ).fetchone()

    if not row:
        return {"status": "error", "message": "Proposal not found"}

    metadata_update = {}
    if reason:
        metadata_update["dismiss_reason"] = reason

    # Update metadata with dismiss reason (json_patch not available in all SQLite builds)
    existing_meta = conn.execute(
        "SELECT metadata FROM tool_proposals WHERE id = ?", (proposal_id,),
    ).fetchone()
    meta = json.loads(existing_meta["metadata"]) if existing_meta and existing_meta["metadata"] else {}
    meta.update(metadata_update)

    conn.execute(
        """UPDATE tool_proposals SET status = 'dismissed', decided_at = ?,
           metadata = ? WHERE id = ?""",
        (now.isoformat(), json.dumps(meta), proposal_id),
    )
    conn.commit()

    return {"status": "dismissed", "proposal_id": proposal_id}


def mark_installed(
    conn: sqlite3.Connection,
    proposal_id: str,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Mark a tool as successfully installed.

    Called after the host agent completes installation. Moves status
    from 'approved' to 'installed'.

    Args:
        conn: Database connection.
        proposal_id: The proposal to mark installed.
        now: Override current time.

    Returns:
        Dict with status.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    ensure_tool_proposals_schema(conn)

    row = conn.execute(
        "SELECT status FROM tool_proposals WHERE id = ?", (proposal_id,),
    ).fetchone()

    if not row:
        return {"status": "error", "message": "Proposal not found"}

    conn.execute(
        """UPDATE tool_proposals SET status = 'installed', installed_at = ?
           WHERE id = ?""",
        (now.isoformat(), proposal_id),
    )
    conn.commit()

    return {"status": "installed", "proposal_id": proposal_id}


def record_tool_usage(
    conn: sqlite3.Connection,
    tool_name: str,
    tool_source: str,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Record that an installed tool was used.

    Increments use_count and updates last_used for matching proposals.
    Matches by tool_name + tool_source rather than proposal_id so it works
    even if the tool was installed through a different path.

    Args:
        conn: Database connection.
        tool_name: Name of the tool that was used.
        tool_source: Source registry of the tool.
        now: Override current time.

    Returns:
        Dict with status and updated use_count.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    ensure_tool_proposals_schema(conn)

    row = conn.execute(
        """SELECT id, use_count FROM tool_proposals
           WHERE tool_name = ? AND tool_source = ? AND status = 'installed'
           ORDER BY installed_at DESC LIMIT 1""",
        (tool_name, tool_source),
    ).fetchone()

    if not row:
        return {"status": "not_tracked", "message": "No installed proposal found for this tool"}

    new_count = (row["use_count"] or 0) + 1
    conn.execute(
        """UPDATE tool_proposals SET use_count = ?, last_used = ?
           WHERE id = ?""",
        (new_count, now.isoformat(), row["id"]),
    )
    conn.commit()

    return {"status": "ok", "proposal_id": row["id"], "use_count": new_count}


def get_unused_tools(
    conn: sqlite3.Connection,
    review_days: int = USAGE_REVIEW_DAYS,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Find installed tools that haven't been used recently.

    These are candidates for removal review. A tool is flagged if:
    - It's installed and has never been used, OR
    - It's installed and last_used is older than review_days ago

    Args:
        conn: Database connection.
        review_days: Days without use before flagging.
        now: Override current time.

    Returns:
        List of dicts: [{tool_name, tool_source, installed_at, last_used,
                        use_count, days_unused}]
    """
    if now is None:
        now = datetime.now(timezone.utc)

    ensure_tool_proposals_schema(conn)

    cutoff = (now - timedelta(days=review_days)).isoformat()

    rows = conn.execute(
        """SELECT * FROM tool_proposals
           WHERE status = 'installed'
           AND (last_used IS NULL OR last_used < ?)
           ORDER BY installed_at ASC""",
        (cutoff,),
    ).fetchall()

    unused = []
    for r in rows:
        if r["last_used"]:
            try:
                lu_dt = datetime.fromisoformat(
                    r["last_used"].replace('Z', '+00:00')
                ).replace(tzinfo=None)
                days_unused = (now - lu_dt).total_seconds() / 86400.0
            except (ValueError, AttributeError):
                days_unused = review_days + 1
        else:
            # Never used: days since installation
            try:
                inst_dt = datetime.fromisoformat(
                    r["installed_at"].replace('Z', '+00:00')
                ).replace(tzinfo=None)
                days_unused = (now - inst_dt).total_seconds() / 86400.0
            except (ValueError, AttributeError):
                days_unused = review_days + 1

        unused.append({
            "tool_name": r["tool_name"],
            "tool_source": r["tool_source"],
            "installed_at": r["installed_at"],
            "last_used": r["last_used"],
            "use_count": r["use_count"] or 0,
            "days_unused": round(days_unused, 1),
            "proposal_id": r["id"],
        })

    return unused


def get_tool_report(
    conn: sqlite3.Connection,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Full tool finding report. Suitable for inclusion in the patterns report.

    Returns:
        Dict with: pending_proposals, installed_tools, unused_tools,
        dismissed_count, generated_at.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    ensure_tool_proposals_schema(conn)

    pending = get_pending_proposals(conn, now=now)
    unused = get_unused_tools(conn, now=now)

    installed_rows = conn.execute(
        """SELECT tool_name, tool_source, installed_at, last_used, use_count
           FROM tool_proposals WHERE status = 'installed'
           ORDER BY use_count DESC""",
    ).fetchall()

    installed = [{
        "tool_name": r["tool_name"],
        "tool_source": r["tool_source"],
        "installed_at": r["installed_at"],
        "last_used": r["last_used"],
        "use_count": r["use_count"] or 0,
    } for r in installed_rows]

    dismissed_count = conn.execute(
        "SELECT COUNT(*) as c FROM tool_proposals WHERE status = 'dismissed'",
    ).fetchone()["c"]

    return {
        "pending_proposals": pending,
        "installed_tools": installed,
        "unused_tools": unused,
        "dismissed_count": dismissed_count,
        "generated_at": now.isoformat(),
    }


# ─── Boot Integration ──────────────────────────────────────────────────────

def format_boot_proposals(proposals: List[Dict[str, Any]]) -> str:
    """
    Format pending proposals as a text block for boot context injection.

    This is what the host agent sees during boot. It includes enough
    information for the agent to present the proposals to the user and
    handle approval/dismissal.

    Args:
        proposals: List from get_pending_proposals().

    Returns:
        Formatted text block, or empty string if no proposals.
    """
    if not proposals:
        return ""

    lines = ["⚡ TOOL SUGGESTIONS (based on recurring knowledge gaps):"]
    for i, p in enumerate(proposals, 1):
        perms = p.get("permissions", [])
        perm_str = f" | Permissions: {', '.join(perms)}" if perms else ""
        meta = p.get("metadata", {})
        gap_count = meta.get("gap_occurrences", "?")

        lines.append(
            f"  {i}. {p['tool_name']} ({p['tool_source']})"
        )
        lines.append(
            f"     {p['tool_description']}"
        )
        lines.append(
            f"     Gap: \"{p['gap_pattern']}\" ({gap_count} queries){perm_str}"
        )
        lines.append(
            f"     → Approve or dismiss? (proposal_id: {p['id']})"
        )

    lines.append("")
    lines.append("To approve: engine.confirm_proposal(proposal_id)")
    lines.append("To dismiss: engine.dismiss_proposal(proposal_id)")

    return "\n".join(lines)
