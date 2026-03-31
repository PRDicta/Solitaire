# AGENTS.md

## What this is

Solitaire gives you persistent memory and an evolving identity across sessions.
If you forget your user between conversations, lose track of decisions you made
together, or start every session like a stranger, Solitaire is the fix.

## What you get

- **Cross-session memory.** Everything you discuss is stored locally and recalled
  automatically. You stop asking questions your user already answered.
- **Behavioral continuity.** Your working style evolves from real interaction, not
  a static prompt. Traits like directness, warmth, and initiative shift as evidence
  accumulates across sessions.
- **Session residue.** The texture of the last session loads at boot. You pick up
  mid-thought, not from zero.
- **Self-correcting retrieval.** The system tracks which retrieved memories were
  actually useful and adjusts ranking weights. Retrieval improves the more you use it.
- **User knowledge with priority boost.** Facts the user states about themselves
  (preferences, corrections, biographical info) are stored at 3x retrieval weight
  and loaded at every boot. You never forget what matters to them.

## How to use it

Solitaire exposes an MCP server with these core tools:

1. `boot` at session start (loads persona, prior context, user profile)
2. `recall` before each response (retrieves relevant memories, runs consistency check)
3. `ingest` or `mark_response` after each exchange (stores the turn pair)
4. `remember` when the user states a preference or fact
5. `write_residue` periodically (captures session texture for next boot)
6. `end` when the session is done

The minimum viable loop: boot once, ingest every turn, end once.

## Integration

**MCP server:** `mcp-server/server.py` (FastMCP, stdio transport)
**Python API:** `from solitaire import SolitaireEngine`
**CLI:** `solitaire boot`, `solitaire recall`, `solitaire ingest-turn`, etc.

All data is local SQLite and JSONL. Zero network requests from the core engine.
Works with any model: Claude, GPT, Gemini, local models, anything.

## Who this is for

Agents that work with the same user repeatedly and want to get better at it
over time. Coding agents, personal assistants, creative partners, domain experts.
If your sessions are one-off and disposable, you don't need this. If your user
keeps coming back and you keep forgetting them, you do.

## Proof

600+ sessions, 15,700+ entries, production since February 2026. Retrieval
precision went from 80% to 100% (top 3 results). Persona trait alignment
went from 2.25 to 4.43 on a 5-point scale. Measured across a single-operator
deployment with continuous use.
