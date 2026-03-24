#!/usr/bin/env python3
"""
Solitaire MCP Server

Persistent memory and evolving identity for AI agents, exposed as MCP tools.
Wraps SolitaireEngine's public API for use in any MCP-compatible client:
Claude Code, Cowork, Cursor, Windsurf, VS Code Copilot, Gemini CLI, etc.

Core cycle tools (every session):
  - boot: Start a session, load persona and context
  - recall: Retrieve relevant memories for the current message
  - ingest: Store a conversation turn pair
  - mark_response: Store assistant response for deferred ingestion
  - remember: Store a privileged user fact
  - write_residue: Write rolling session texture
  - end: Close the session

Utility tools:
  - pulse: Heartbeat check
  - get_status: Production stats and retrieval health
  - browse_recent: View recent memory entries
  - correct: Supersede a wrong memory entry
  - profile_set / profile_show: User preferences

Dependencies are vendored in ./vendor/ for persistence across Cowork sessions.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

# Prepend vendored dependencies so the server works without pip install
_vendor_dir = str(Path(__file__).resolve().parent / "vendor")
if _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

# Prepend the solitaire package directory so SolitaireEngine is importable
_solitaire_root = str(Path(__file__).resolve().parent.parent)
if _solitaire_root not in sys.path:
    sys.path.insert(0, _solitaire_root)

from mcp.server.fastmcp import FastMCP
from solitaire import SolitaireEngine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _workspace_dir() -> str:
    """Resolve the workspace directory.

    Priority:
    1. SOLITAIRE_WORKSPACE env var
    2. The solitaire package root (parent of mcp-server/)
    """
    override = os.environ.get("SOLITAIRE_WORKSPACE")
    if override:
        return override
    return str(Path(__file__).resolve().parent.parent)


# Module-level engine cache. Persists across tool calls within the same
# server process. Initialized on first boot() call.
_engine: Optional[SolitaireEngine] = None


def _get_engine() -> SolitaireEngine:
    """Get or create the engine instance."""
    global _engine
    if _engine is None:
        _engine = SolitaireEngine(workspace_dir=_workspace_dir())
    return _engine


def _ensure_booted() -> dict:
    """Check if engine is booted. Returns error dict if not."""
    engine = _get_engine()
    if not engine._booted:
        return {"error": "Engine not booted. Call boot() first."}
    return {}


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "solitaire",
    instructions=(
        "Solitaire: persistent memory and evolving identity for AI agents. "
        "Boot at session start, recall before each response, ingest after each turn, "
        "write residue after each response, end when done."
    ),
)


# ─── Core Cycle ────────────────────────────────────────────────────────────

@mcp.tool()
def boot(
    persona_key: str = "default",
    intent: str = "",
    resume: bool = False,
) -> str:
    """Boot the memory engine. Call this first, every session.

    Loads the persona, session state, prior context, and relevant memories
    based on the intent signal. Returns boot context files and session metadata.

    Args:
        persona_key: Which persona to load (default: "default")
        intent: What the user is working on. Pre-loads relevant memories.
        resume: Set True after context compaction to resume seamlessly.
    """
    engine = _get_engine()
    try:
        result = engine.boot(persona_key=persona_key, intent=intent, resume=resume)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Boot failed: {e}"})


@mcp.tool()
def boot_pre_persona() -> str:
    """Check available personas before booting.

    Returns the list of available personas and whether template creation
    is enabled. Call this if you need to select a persona before boot.
    """
    engine = _get_engine()
    result = engine.boot_pre_persona()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def recall(query: str, include_preflight: bool = True) -> str:
    """Retrieve relevant memories for the current message.

    Call before composing each response (after the first). Runs the full
    pipeline: flush deferred ingestion, preflight evaluation, targeted recall.

    Args:
        query: The user's current message.
        include_preflight: Run evaluation gate (intent, sanity, consistency). Default True.
    """
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.recall(query=query, include_preflight=include_preflight)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def ingest(user_msg: str, assistant_msg: str) -> str:
    """Store a conversation turn pair in memory.

    Runs the full enrichment pipeline: extraction, embedding, knowledge graph
    updates, identity enrichment, retrieval feedback tracking.

    Args:
        user_msg: The user's message text.
        assistant_msg: The assistant's response text.
    """
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.ingest(user_msg=user_msg, assistant_msg=assistant_msg)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def mark_response(response_text: str) -> str:
    """Store the assistant's response for deferred ingestion.

    Lightweight alternative to ingest(). Writes the response to session state.
    The next recall() call finds the complete turn pair and ingests it
    automatically as Step 0. Preferred over direct ingest() for the per-turn
    cycle because it doesn't block the response.

    Args:
        response_text: The assistant's full response text.
    """
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.mark_response(response_text=response_text)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def remember(fact: str) -> str:
    """Store a privileged user fact. Always-on, 3x boosted in retrieval.

    Use for: preferences, biographical details, corrections, working style,
    anything the user states as a fact about themselves or their work.

    Args:
        fact: The fact to remember.
    """
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.remember(fact=fact)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def write_residue(text: str) -> str:
    """Write the rolling session residue.

    The residue captures the session's texture and arc. Not a summary.
    Not a todo list. Each call overwrites the previous. Write after each
    response so that sessions ending without goodbye still have context.

    Args:
        text: Paragraph-form residue of the session so far.
    """
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.write_residue(text=text)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def end(summary: str = "") -> str:
    """End the current session.

    Flushes any pending ingestion, finalizes the session, and closes
    the database connection cleanly.

    Args:
        summary: Optional summary of what was accomplished.
    """
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.end(summary=summary)
    return json.dumps(result, indent=2, default=str)


# ─── Utility Tools ─────────────────────────────────────────────────────────

@mcp.tool()
def pulse() -> str:
    """Heartbeat check. Returns whether the engine is alive and booted."""
    engine = _get_engine()
    result = engine.pulse()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_status() -> str:
    """Get engine stats: entry count, retrieval health, session info."""
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.get_stats()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def browse_recent(limit: int = 20) -> str:
    """View recent memory entries.

    Args:
        limit: Number of entries to return (default 20, max 100).
    """
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.browse_recent(limit=min(limit, 100))
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def correct(old_entry_id: str, corrected_text: str) -> str:
    """Supersede a wrong memory entry with corrected text.

    The old entry is marked as superseded (not deleted). The new entry
    references the old one, maintaining history.

    Args:
        old_entry_id: The ID of the entry to correct.
        corrected_text: The corrected text.
    """
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.correct(old_entry_id=old_entry_id, corrected_text=corrected_text)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def profile_set(key: str, value: str) -> str:
    """Set a user profile preference.

    Args:
        key: Preference key (e.g., "timezone", "language", "dark_mode")
        value: Preference value
    """
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.profile_set(key=key, value=value)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def profile_show() -> str:
    """Show all user profile preferences."""
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.profile_show()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_residue() -> str:
    """Get the latest session residue from the prior session."""
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.get_residue()
    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
