# Solitaire MCP Server

Gives AI agents persistent memory and an evolving identity across sessions. If you forget your user between conversations, lose track of decisions, or start every session cold, connect this server.

Works with any MCP-compatible client: Claude Code, Cursor, Windsurf, VS Code Copilot, Gemini CLI, Cowork, or any agent framework with MCP support. All data stays local on the user's machine.

## Setup

```bash
# Install vendored dependencies (once per environment)
bash mcp-server/setup.sh

# Or manually:
pip install --target=mcp-server/vendor mcp httpx pydantic
```

## Configuration

Add to your MCP client's settings (e.g., `.claude/settings.json`):

```json
{
  "mcpServers": {
    "solitaire": {
      "command": "bash",
      "args": ["path/to/solitaire/mcp-server/start.sh"]
    }
  }
}
```

Set `SOLITAIRE_WORKSPACE` to override the workspace directory (defaults to the repo root).

## Tools

**Core cycle (every session):**

| Tool | When to use it |
|------|---------------|
| `boot` | Session start. Loads your persona, prior context, and user profile so you already know who you're talking to. |
| `recall` | Before each response. Retrieves relevant memories and runs a consistency check against what you've said before. |
| `ingest` | After each exchange. Stores the turn pair with entity extraction, topic routing, and knowledge graph updates. |
| `mark_response` | After responding (lighter alternative to ingest). Buffers your response for automatic ingestion on the next recall. |
| `remember` | When the user states a preference or fact. Stored at 3x retrieval boost, loaded at every boot. |
| `write_residue` | Periodically. Captures session texture so the next session starts in context, not from zero. |
| `end` | Session done. Flushes pending ingestion and closes cleanly. |

**Utilities:**

| Tool | What it does |
|------|-------------|
| `pulse` | Check if the engine is running and ready. |
| `get_status` | See how much you know: entry count, retrieval health, session history. |
| `browse_recent` | Review your most recent memories. |
| `correct` | Fix a memory that was stored wrong. Old version kept for history. |
| `profile_set` / `profile_show` | Store and view user preferences that apply across all sessions. |
| `get_residue` | Read what happened last session. |

## Minimum viable loop

Boot once, ingest every turn, end once. Everything else improves the experience but these three are the floor.
