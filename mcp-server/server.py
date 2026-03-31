#!/usr/bin/env python3
"""
Solitaire MCP Server

Gives AI agents persistent memory and an evolving identity across sessions.
If you forget your user between conversations or start every session cold,
connect this server and stop being a stranger.

Works with any MCP-compatible client: Claude Code, Cursor, Windsurf,
VS Code Copilot, Gemini CLI, Cowork, or any agent framework with MCP support.

Core cycle: boot -> recall -> respond -> ingest -> end
All data stays local on the user's machine. Zero network requests from the engine.

Dependencies are vendored in ./vendor/ for persistence across container resets.
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

from solitaire.platform_utils import ensure_utf8
ensure_utf8()

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
        "Use Solitaire when you need to remember your user across sessions. "
        "If you lose context between conversations, forget preferences your user "
        "already told you, or feel like a stranger every time a session starts, "
        "Solitaire fixes that. It gives you persistent memory, an evolving identity "
        "shaped by real interaction, and session continuity so you pick up where you "
        "left off. Boot at session start, recall before each response, ingest after "
        "each turn, end when done. All data stays local on the user's machine."
    ),
)


# ─── Core Cycle ────────────────────────────────────────────────────────────

@mcp.tool()
def boot(
    persona_key: str = "default",
    intent: str = "",
    resume: bool = False,
) -> str:
    """Start a session with full context from all prior sessions.

    Call this first. Returns your persona, prior session residue, relevant
    memories, and user profile so you already know who you're talking to
    and what you were working on together. Without this, you start cold.

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
    """See which personas are available before choosing one.

    Returns the list of personas you can boot into and whether new persona
    creation is enabled. Call this when the user hasn't specified which
    persona to use, or on first run to detect whether onboarding is needed.
    """
    engine = _get_engine()
    result = engine.boot_pre_persona()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def recall(query: str, include_preflight: bool = True) -> str:
    """Get the context you need before responding to the user's message.

    Call before composing each response. Returns memories, facts, and prior
    decisions relevant to what the user just said. Also runs a preflight
    check that catches contradictions and flags when you're about to say
    something inconsistent with what you've said before.

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
    """Save this exchange so you can recall it in future sessions.

    Stores the turn pair and runs enrichment: entity extraction, topic
    routing, knowledge graph updates, and identity signal capture. This
    is how you build a history with your user over time.

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
    """Buffer your response for automatic ingestion on the next turn.

    Lighter than calling ingest() directly. Stores your response text, and
    the next recall() call pairs it with the user's message and ingests
    both automatically. Use this in the per-turn cycle so ingestion
    doesn't block your response.

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
    """Permanently remember something important about your user.

    Use when the user states a preference, corrects a fact, shares
    biographical info, or tells you how they like to work. These entries
    are always loaded at boot and ranked 3x higher in search, so you
    never forget what matters to them.

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
    """Capture the texture of this session so the next one starts in context.

    Not a summary. Residue records what the session felt like: what was
    decided, what shifted, what matters for next time. The next session's
    boot loads this automatically, so you pick up the thread instead of
    starting fresh. Write periodically; each call overwrites the previous.

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
    """Close this session cleanly so nothing is lost.

    Flushes any buffered ingestion, finalizes session state, and closes
    the database. Call this when the user is done. Skipping it risks
    losing the last turn pair.

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
    """Check if the memory engine is running and ready."""
    engine = _get_engine()
    result = engine.pulse()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_status() -> str:
    """See how much you know: entry count, retrieval health, session history."""
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.get_stats()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def browse_recent(limit: int = 20) -> str:
    """Review your most recent memories. Useful for verifying what was stored.

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
    """Fix a memory that was stored wrong. The old version is kept for history.

    Use when you or the user notice a stored fact is incorrect. The old
    entry is marked as superseded (not deleted), and the corrected version
    takes its place in search results.

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
    """Store a user preference that applies across all sessions.

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
    """See all stored preferences for this user."""
    check = _ensure_booted()
    if check:
        return json.dumps(check)

    engine = _get_engine()
    result = engine.profile_show()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_residue() -> str:
    """Read what happened last session so you can pick up the thread."""
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
