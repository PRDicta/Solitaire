<p align="center">
  <img src="assets/banner.jpg" alt="Solitaire for Agents" width="100%">
</p>

<p align="center">
  <a href="https://github.com/PRDicta/Solitaire-for-Agents/actions/workflows/ci.yml"><img src="https://github.com/PRDicta/Solitaire-for-Agents/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/sessions-400%2B-orange.svg" alt="400+ sessions">
  <img src="https://img.shields.io/badge/entries-13%2C000%2B-orange.svg" alt="13,000+ entries">
</p>

# Solitaire for Agents

Open-source identity and memory layer for AI agents.

**Memory is solved. Identity isn't.**

Most memory tools help agents retrieve facts from prior conversations. Solitaire goes further: it helps agents build a stable understanding of the user over time, so behavior changes across sessions instead of just recall quality. Session 50 should feel fundamentally different from session 5, because something real has accumulated between them.

## Why Solitaire exists

AI models are still too stateless in practice. Even with "memory," most agents feel like smart strangers. They may remember your name or a project detail from last week, but the collaboration itself doesn't deepen. The model recalls facts without becoming a better partner.

Solitaire gives agents persistent memory, persistent identity, and persistent behavioral context, so they can carry forward not just what matters to you, but how to work with you.

## Proof

| Metric | Value |
|--------|-------|
| Real sessions of longitudinal use | 400+ |
| Accumulated memory entries | 13,000+ |
| Precision@3 improvement | 80% → 100% |
| Persona trait alignment | 2.25/5 → 4.43/5 |
| Dify integration | Shipped |
| Claude Code / Cowork usage | Production |
| Supporting research papers | 2 |

## What makes it different

**Identity, not just retrieval.** Most memory tools store facts and retrieve them later. Solitaire also builds a behavioral identity layer: persona compilation, disposition modeling, voice and profile shaping, and session residues that help the agent maintain continuity across time.

**Self-correcting retrieval.** Solitaire tracks which retrieved memories actually prove useful in responses and adjusts weighting over time. Entries that help get surfaced more. Entries that don't fade back. The retrieval system calibrates itself without being told what "right" means.

**Local-first by default.** The memory engine runs locally. Storage stays in SQLite and JSONL inside your workspace. No cloud dependency for the core system. Your data never leaves your machine.

**Model-agnostic.** Claude, Gemini, a custom LLM, or another host: Solitaire returns context in plain structures and lets the host decide how to inject it. It doesn't care what model you use. It cares about the relationship between the model and the person using it.

## Getting started

### Cowork (recommended, zero-config)

The fastest way to use Solitaire. No terminal, no code.

1. [Download the latest release](https://github.com/PRDicta/Solitaire-for-Agents/releases/latest) (.zip file)
2. Unzip the folder anywhere on your computer
3. Open Cowork and select the unzipped folder as your workspace

Solitaire boots automatically on your first message, walks you through creating a partner, and starts learning from every conversation.

### From source (developers)

Requirements: Python 3.10+, git.

```bash
git clone https://github.com/PRDicta/Solitaire-for-Agents.git
cd Solitaire-for-Agents
pip install -e .
```

Optional dependencies:

```bash
pip install -e ".[embeddings]"   # sentence-transformers + torch
pip install -e ".[llm]"          # anthropic SDK
pip install -e ".[all]"          # everything
```

## Quick start

### Python API

```python
from solitaire import SolitaireEngine

engine = SolitaireEngine(workspace_dir="./my-data")

# Boot (start of session)
result = engine.boot(persona_key="default", intent="working on quarterly review")

# Recall (before composing a response)
context = engine.recall(query="Q1 revenue figures")

# Ingest (after each exchange)
engine.ingest(user_msg="What was our Q1 revenue?", assistant_msg="Based on...")

# Remember (privileged facts, always loaded at boot)
engine.remember(fact="Client X prefers email over Slack")

# End (close session cleanly)
engine.end(summary="Reviewed Q1 numbers and client preferences")
```

### CLI

```bash
solitaire boot --persona default --intent "working on quarterly review"
solitaire auto-recall "Q1 revenue figures"
solitaire ingest-turn "What was our Q1 revenue?" "Based on..."
solitaire remember "Client X prefers email over Slack"
solitaire end "Reviewed Q1 numbers"
```

All commands output JSON to stdout and diagnostics to stderr.

## Use cases

Solitaire is for developers who want agents to feel continuous across time.

- **Coding agents** that remember project history and user preferences
- **Personal assistants** that maintain working context across sessions
- **AI products** that need durable personalization instead of shallow recall
- **Enterprise teams** who want local-first memory and identity infrastructure under their control

## Platform compatibility

| Platform | Integration | Status |
|----------|------------|--------|
| Claude Code / Cowork | CLAUDE.md + bash subprocess | Production (400+ sessions) |
| Hermes | SKILL.md (native format) | Compatible |
| OpenClaw | SKILL.md (native format) | Compatible |
| Gemini CLI | SKILL.md compatible | Compatible |
| Dify | Plugin (5 tools, marketplace-ready) | Shipped |

## External memory import (symbiosis adapter)

Solitaire can ingest memories from external sources, so context from other tools and conversations carries forward into the knowledge graph. The symbiosis adapter handles source connection, deduplication, and optional live sync.

### Supported sources

| Source | Format | What it imports |
|--------|--------|-----------------|
| Cowork auto-memory | `.auto-memory/` directory (markdown + YAML frontmatter) | User preferences, feedback, project notes, references |
| Librarian JSONL | `.jsonl` export files | Structured rolodex entries with full metadata |
| ChatGPT export | `conversations.json` from ChatGPT data export | Conversation text, chunked for extraction |
| Plain text | `.txt`, `.md`, `.rst` files or directories | Documents, notes, any unstructured text |

### Sync tiers

- **One-shot** (manual): Import once on demand.
- **Periodic**: Auto-sync at a configurable interval (default: hourly).
- **Live-watch**: File-system polling detects new or changed files and syncs automatically.

Re-running an import on the same source is safe. Deduplication keys prevent double-ingestion.

### Quick start

```python
from solitaire.symbiosis import ReaderRegistry, SyncEngine

registry = ReaderRegistry()
registry.auto_discover()

engine = SyncEngine(rolodex=rolodex, conn=conn, registry=registry)
engine.connect(source_id="chatgpt-export", name="my-chatgpt", config={"path": "conversations.json"})
result = engine.sync("my-chatgpt")
print(f"Imported {result.imported} entries, skipped {result.skipped_duplicate} duplicates")
```

### Integration contract

Five things the host agent must provide:

1. **Persistent workspace directory.** A path on disk that survives between sessions. Set via `SOLITAIRE_WORKSPACE` env var or pass to `SolitaireEngine(workspace_dir=...)`.
2. **Python 3.10+ runtime.** One install and it runs.
3. **Command execution.** Subprocess (`solitaire <command>`) or Python import.
4. **System prompt injection.** Boot and recall return context blocks. The host agent injects these into the model's prompt.
5. **Per-turn lifecycle hook.** Call `ingest-turn` after each exchange. Optionally call `auto-recall` before responses and `residue write` after ingesting.

The minimum viable loop: boot once, ingest every turn, end once.

## Architecture

```
Host Agent (Claude, Gemini, custom LLM, etc.)
    │
    ├── CLI (solitaire command, 53 commands)
    ├── Python API (SolitaireEngine class)
    ├── Skill manifest (SKILL.md for Hermes/OpenClaw/Gemini)
    └── Dify plugin (5 tool nodes)
          │
    SolitaireEngine (model-agnostic public API)
          │
    ┌─────┼─────────────────────────┐
    │     │                         │
  Core Domain              Retrieval Layer
  - Persona & identity     - Vector + keyword search
  - Session management     - Query expansion
  - Self-learning stack    - Reranking
  - Intent routing         - Evaluation gate
  - Pattern detection      - Context builder
    │                               │
  Indexing Layer            Storage Layer
  - Ingestion queue        - SQLite knowledge graph
  - Entity extraction      - Identity graph
  - Topic routing          - JSONL turn archives
  - Embedding generation   - Session state
```

## What's stored locally

After your first session, the workspace contains:

| Path | Contents |
|------|----------|
| `rolodex.db` | SQLite knowledge graph (memories, entities, relationships) |
| `personas/` | Persona configurations (traits, north star, domain scope) |
| `sessions/` | Session state and metadata |
| `*.jsonl` | Verbatim turn pair archives |

All of these are gitignored by default. Your memory data never enters version control.

## Privacy

Solitaire's memory engine makes zero network requests. All storage is local SQLite and JSONL files in your workspace directory. The AI model sees your context during conversations (that's the point), but the database itself never leaves your machine.

The optional `[llm]` and `[embeddings]` dependencies make network calls to their respective APIs only when explicitly configured. The core memory system works without them.

## Research context

Solitaire is accompanied by two papers:

- *Artificial Relational Intelligence: Why the AGI Threshold Is the Wrong Target*
- *From Memory to Partnership: How Persistent External Context Transforms Human-AI Interaction*

The papers formalize ideas that emerged from real use. They should be read as strong early evidence and conceptual framing, not as a claim that the work is complete.

## License

AGPL-3.0. Commercial license available from Dicta Technologies Inc.

See [LICENSE](LICENSE) for full terms. Contact licensing@usedicta.com for commercial inquiries.
