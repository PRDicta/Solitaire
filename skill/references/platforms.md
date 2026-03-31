# Platform Compatibility

Solitaire works as a subprocess (CLI) or Python import on any platform that can run Python 3.10+. The integration surface is the same everywhere: call commands, parse JSON, inject context.

## Tested Platforms

| Platform | Integration | Status | Notes |
|----------|------------|--------|-------|
| **Claude Code / Cowork** | INSTRUCTIONS.md + bash subprocess | Production | Original development platform. Full lifecycle coverage. |
| **Hermes** | SKILL.md (native format) | Compatible | agentskills.io origin platform. SKILL.md is the native skill format. |
| **OpenClaw** | SKILL.md (native format) | Compatible | Reads SKILL.md natively. |
| **Gemini CLI** | SKILL.md compatible | Compatible | Supports skill installation from SKILL.md. |
| **Cursor / VS Code** | Extension wrapper | Planned | Requires custom extension to call CLI. Not in v1 scope. |
| **Dify** | Plugin (5 tools) | Prototype complete | Wraps SolitaireEngine as Dify tool nodes: boot, ingest, recall, remember, end. |

## Integration Patterns

### Pattern 1: Subprocess (recommended for most platforms)

The host agent calls `solitaire` as a bash subprocess. Works on any platform that can execute shell commands.

```
Host Agent → bash: solitaire boot --persona default → JSON → inject into prompt
Host Agent → bash: solitaire auto-recall "message"  → JSON → inject into prompt
Host Agent → bash: solitaire ingest-turn "u" "a"    → JSON → (discard or log)
```

### Pattern 2: Python Import (for Python-native platforms)

Direct API access without subprocess overhead. Same semantics, typed return values.

```python
from solitaire import SolitaireEngine
engine = SolitaireEngine(workspace_dir="./data")
engine.boot(persona_key="default")
context = engine.recall(query="...")
engine.ingest(user_msg="...", assistant_msg="...")
```

### Pattern 3: INSTRUCTIONS.md / System Prompt (for prompt-driven agents)

The agent's instruction file tells it when to call Solitaire. The agent itself decides when to invoke the commands based on the instructions. This is how Claude Code and Cowork integrate: the INSTRUCTIONS.md file contains lifecycle rules, and the model follows them.

## Host Agent Integration Contract

Five things the host agent must provide for Solitaire to function:

**1. Persistent workspace directory.** A path on disk that survives between sessions.
Solitaire creates and maintains: `rolodex.db` (SQLite knowledge graph),
`personas/` (persona configurations and accumulated identity), `.solitaire_session`
(session state), and boot context files. Set via `SOLITAIRE_WORKSPACE` env var or
pass to `SolitaireEngine(workspace_dir=...)`.

**2. Python 3.10+ runtime.** Solitaire is a Python package. No external services,
no containers, no Docker. One `pip install` and it runs.

**3. Command execution.** Either subprocess (`solitaire <command>`) or Python import
(`from solitaire import SolitaireEngine`). The subprocess path is universal. The
import path eliminates process overhead for high-frequency calls.

**4. System prompt injection.** Boot and recall return context blocks (plain text
strings). The host agent must inject these into the model's system prompt or
conversation prefix. This is what gives the model access to past context. Without
injection, Solitaire stores data but the model never sees it.

**5. Per-turn lifecycle hook.** The host agent must call `ingest-turn` after each
user/assistant exchange. This is the memory pipeline. Optionally call `auto-recall`
before composing responses and `residue write` after ingesting. The minimum viable
loop is: boot once, ingest every turn, end once.

## Platform-Specific Notes

### Claude Code
Two integration modes, complementary:

**Instructions-driven (minimum viable):** INSTRUCTIONS.md tells the model when to
call Solitaire commands. The model follows lifecycle rules autonomously. Works but
depends on the model remembering to ingest every turn.

**Hook-driven (recommended):** A `Stop` hook auto-ingests every exchange via the
Claude Code transcript. Drop `skill/hooks/claude-code-auto-ingest.py` into your
`.claude/hooks/` directory and add to `.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "python .claude/hooks/claude-code-auto-ingest.py",
        "timeout": 45
      }]
    }]
  }
}
```

The hook reads the session transcript, extracts the last user+assistant exchange,
and calls `solitaire ingest-turn` automatically. Deduplication prevents double
ingestion if the model also calls ingest-turn manually. This makes ingestion
infrastructure rather than instruction compliance.

Set `SOLITAIRE_WORKSPACE` env var to point at your workspace directory, or the
hook defaults to cwd. Set `SOLITAIRE_CMD` if `solitaire` is not on PATH.

**Claim scanner (recommended):** A second `Stop` hook scans assistant responses for
unverified state assertions about remote or unobserved systems. Drop
`skill/hooks/claude-code-claim-scanner.py` into `.claude/hooks/` and add it
alongside the auto-ingest hook:

```json
{
  "hooks": {
    "Stop": [{
      "matcher": "",
      "hooks": [
        {
          "type": "command",
          "command": "python .claude/hooks/claude-code-auto-ingest.py",
          "timeout": 45
        },
        {
          "type": "command",
          "command": "python .claude/hooks/claude-code-claim-scanner.py",
          "timeout": 30
        }
      ]
    }]
  }
}
```

The claim scanner detects when the model makes definitive claims about systems it
cannot directly observe (remote machines, unverified installations, state inferred
from screenshots). When claims are detected, it writes a marker file that the
preflight evaluation gate picks up on the next turn, injecting a "Stop. Think.
Check. Be Sure." block before the model composes its response. This makes
verification structural rather than behavioral.

Disable with `SOLITAIRE_CLAIM_SCANNER=0` environment variable.

### Cowork
The original development platform. Integration is via `INSTRUCTIONS.md` (maps to
`CLAUDE.md` in the Cowork workspace). Cowork's platform harness handles the
per-turn lifecycle natively: the runtime calls ingest-turn and residue write as
part of its turn processing. The model does not need to remember. Production-tested
across 300+ sessions.

### Hermes
SKILL.md is the native skill format (agentskills.io origin). Drop the `solitaire-skill/`
directory into Hermes' skill discovery path. Hermes reads the YAML frontmatter for
metadata and progressive disclosure for instructions. The `install` and `verify` fields
in frontmatter point to the install/verify scripts.

### OpenClaw
SKILL.md native support. Same as Hermes: drop the directory, OpenClaw discovers it.

### Gemini CLI
SKILL.md compatible. Gemini CLI reads skill manifests in the agentskills.io format.
Install via its skill management interface.

### Cursor / VS Code
Not natively compatible with SKILL.md. Requires a custom extension wrapper that
calls the Solitaire CLI and injects context into the editor's AI assistant prompt.
Not in v1 scope. The subprocess pattern works if the extension can execute shell
commands and modify the system prompt.

### Dify
Plugin shipped (Session 10). Five tools: boot, ingest, recall, remember, end.
Provider handles credential validation (workspace directory, optional persona key,
optional Anthropic API key). Module-level engine cache persists across tool calls
within the same plugin process. Marketplace-ready with README and PRIVACY docs.
