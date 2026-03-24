---
name: solitaire
description: >
  Persistent memory and evolving identity for AI agents. Gives your agent a
  knowledge graph, session continuity, persona management, and adaptive retrieval
  that survives across sessions. Boot at session start, ingest every turn. Your
  agent remembers everything.
version: 0.1.0
author: Dicta Technologies
license: AGPL-3.0-or-later
homepage: https://github.com/PRDicta/Solitaire
requires:
  runtime: python>=3.10
  packages:
    - pyyaml>=6.0
    - click>=8.0
  optional:
    embeddings: [sentence-transformers>=2.2.0, torch>=2.0.0]
    llm: [anthropic>=0.40.0]
    compression: [tiktoken>=0.5.0]
tags:
  - memory
  - persistence
  - identity
  - agents
  - knowledge-graph
  - retrieval
  - persona
  - session-continuity
platforms:
  - claude-code
  - cowork
  - hermes
  - openclaw
  - gemini-cli
triggers:
  - solitaire
  - persistent memory
  - remember
  - recall
  - session continuity
  - knowledge graph
  - persona
  - identity
install: scripts/install.sh
verify: scripts/verify.sh
---

# Solitaire

Persistent memory and evolving identity for AI agents. Solitaire stores every
conversation turn in a local SQLite knowledge graph, retrieves relevant context
on demand, and maintains an evolving persona that grows with use. Model-agnostic.
Runs locally. Your data never leaves your machine.

---

<!-- LEVEL 1: Instructions -->

## How It Works

Solitaire operates as a subprocess or Python import. The host agent calls it at
four points in the conversation lifecycle:

**Boot** initializes the engine, loads the persona, and returns pre-built context
(recent memories, identity graph, session residue). The host agent injects this
into the model's system prompt.

**Auto-recall** runs before composing each response. It classifies the user's
intent, checks consistency with prior statements, and retrieves relevant memories.

**Ingest-turn** runs after each exchange. Stores both messages, extracts entities
for the knowledge graph, updates temporal reasoning, enriches the identity graph,
and evaluates which recalled entries the model actually used.

**End** finalizes the session: adjusts retrieval weights, updates project clusters,
writes the final residue, and closes cleanly.

## Quick Start

Add these calls to your agent's instruction surface (system prompt, INSTRUCTIONS.md,
or equivalent):

```bash
# Session start (first message):
solitaire boot --persona default --intent "what the user said"

# Before composing each response (turn 2+):
solitaire auto-recall "the user's current message"

# After each exchange:
solitaire ingest-turn "user message" "assistant response"
solitaire residue write "rolling summary of session texture"

# Session end:
solitaire end "what was accomplished"
```

All commands output JSON to stdout. Housekeeping goes to stderr. Parse the
`boot_files.context` path from boot, read that file, and inject its contents
into your model's system prompt.

## What the Host Agent Provides

1. **Workspace directory** -- persistent storage on disk. Solitaire creates
   `rolodex.db`, `personas/`, and session state files here. Set via
   `SOLITAIRE_WORKSPACE` env var or current working directory.

2. **Python subprocess execution** -- call `solitaire <command>` and parse JSON
   from stdout. Or import `SolitaireEngine` directly for zero-overhead integration.

3. **System prompt injection** -- include Solitaire's boot context block in the
   model's prompt. This is what makes the model aware of past sessions.

4. **Per-turn hook** -- call `ingest-turn` after each user/assistant exchange.
   Without this, nothing persists.

## Environment

| Variable | Purpose | Default |
|----------|---------|---------|
| `SOLITAIRE_WORKSPACE` | Root directory for all persistent data | `cwd` |
| `ANTHROPIC_API_KEY` | Enables enhanced mode (LLM summarization) | None (verbatim mode) |

**Verbatim mode** (no API key): heuristic extraction. All features work. Keyword
search if embeddings aren't installed.

**Enhanced mode** (with key): LLM-powered summarization, richer entity extraction,
better topic routing.

## Persona System

Each instance supports multiple personas. A persona is a named configuration:
disposition traits (observance, assertiveness, conviction, warmth, humor,
initiative, empathy), a north star, and domain scope. New users create one at
first run through a guided onboarding flow. The persona accumulates context
and develops an evolving identity graph over time.

## Self-Learning

Solitaire tracks which memories the model actually uses in its responses. Entries
recalled but consistently ignored lose weight. Entries recalled and used gain
weight. The system detects hot topics, dead zones (stale context), and gap signals
(recurring queries with no good results). Gaps trigger proactive tool-finding
through pluggable search providers.

---

<!-- LEVEL 2: Full Configuration -->

## Installation

```bash
pip install solitaire
```

From source:

```bash
git clone https://github.com/PRDicta/Solitaire.git
cd Solitaire && pip install -e .
```

Optional extras:

```bash
pip install solitaire[embeddings]    # Semantic search via sentence-transformers
pip install solitaire[llm]           # Anthropic API for enhanced ingestion
pip install solitaire[compression]   # tiktoken for token counting
pip install solitaire[all]           # Everything
```

## INSTRUCTIONS.md Template

Copy this into your agent's instruction file. Adjust paths if needed:

```markdown
# Solitaire Memory System

## Boot (every session, before first response)
Run: solitaire boot --persona default --intent "<user's first message>"
Parse JSON. Read the file at boot_files.context. Inject into system context.

## Per-Turn (every exchange after the first)
Before responding: solitaire auto-recall "<user's current message>"
After responding: solitaire ingest-turn "<user msg>" "<assistant msg>"
After ingesting: solitaire residue write "<session texture paragraph>"

## Session End
When user signals done: solitaire end "<summary>"
```

## Python API

For direct integration without subprocess overhead:

```python
from solitaire import SolitaireEngine

engine = SolitaireEngine(workspace_dir="/path/to/data")
result = engine.boot(persona_key="default", intent="working on financials")

# Per-turn cycle
context = engine.recall(query="user's message")
engine.ingest(user_msg="...", assistant_msg="...")
engine.write_residue(text="session arc so far")

engine.remember(fact="User prefers dark mode")
engine.end(summary="Completed Q1 review")
```

All methods return plain dicts or strings. No model-specific formatting.

## Output Format

All CLI commands: JSON to stdout, diagnostics to stderr.

Exit codes: `0` success, `1` internal error, `2` user error (bad args).

## Full Command Reference

See [references/commands.md](references/commands.md) for the complete command
reference with all subcommands, flags, options, and output schemas.
