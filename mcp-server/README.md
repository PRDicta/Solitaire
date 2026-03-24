# Solitaire MCP Server

Exposes Solitaire's memory engine as MCP (Model Context Protocol) tools for use in any compatible AI client: Claude Code, Cowork, Cursor, Windsurf, VS Code Copilot, Gemini CLI, etc.

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
- `boot` — Start a session, load persona and context
- `recall` — Retrieve relevant memories for the current message
- `ingest` — Store a conversation turn pair
- `mark_response` — Store assistant response for deferred ingestion
- `remember` — Store a privileged user fact
- `write_residue` — Write rolling session texture
- `end` — Close the session

**Utilities:**
- `pulse` — Heartbeat check
- `get_status` — Engine stats and retrieval health
- `browse_recent` — View recent memory entries
- `correct` — Supersede a wrong memory entry
- `profile_set` / `profile_show` — User preferences
- `get_residue` — Read the latest session residue
