<p align="center">
  <img src="https://raw.githubusercontent.com/PRDicta/Solitaire-for-Agents/main/assets/banner.png" alt="Solitaire for Agents" width="100%">
</p>

<p align="center">
  <a href="https://github.com/PRDicta/Solitaire-for-Agents/actions/workflows/ci.yml"><img src="https://github.com/PRDicta/Solitaire-for-Agents/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/sessions-600%2B-orange.svg" alt="600+ sessions">
  <img src="https://img.shields.io/badge/entries-15%2C700%2B-orange.svg" alt="15,700+ entries">
</p>

# Solitaire for Agents

<!-- mcp-name: io.github.PRDicta/solitaire -->

Identity infrastructure for AI agents. Not just another memory tool.

Most memory tools help agents retrieve facts from prior conversations. Useful...but not enough. Your agent still feels like a smart stranger with a better notebook.

Solitaire gives agents the infrastructure to learn how you work, carry that forward, and become a better collaborator over time. Memory is one component. The rest is what makes the difference: persona compilation, behavioral continuity, self-correcting retrieval, and session texture that persists across time.

Bring your own memory system or use ours. Solitaire sits above the storage layer, so the identity infrastructure works either way.

## What changes over time

**First session:** Solitaire learns your name, your project, your preferences. You choose a name for your partner, and the work begins.

**By session ten:** It knows your working preferences, your communication patterns, and which retrieved context actually helped. Retrieval is self-tuning. The agent doesn't just recall, it's been briefed.

**By session one hundred:** Thousands of entries in the knowledge graph. A behavioral profile shaped by real interaction, not static configuration. The agent doesn't just remember you, it works like someone who's been working with you for months.

## Proof

600+ sessions | 15,700+ entries | Production since February 2026

| What we measured | Before | After | What it means |
|-----------------|--------|-------|---------------|
| Retrieval precision (top 3 results) | 80% | 100% | Early on, 1 in 3 retrieved memories was irrelevant. Now all three are useful. |
| Persona trait alignment (5-point scale) | 2.25 | 4.43 | The agent's working style started as a poor match. It's now a close one. |

Both metrics measured across 600+ real sessions with a single-operator deployment. Two accompanying research papers formalize the methodology and findings.

## What makes it different

**Behavioral genome.** Solitaire doesn't store a static persona config. It builds a disposition profile from real interaction: observance, assertiveness, warmth, initiative, humor. These traits evolve as the system accumulates evidence. The agent's working style changes because the data changes, not because someone edited a prompt.

**Experiential memory.** Most memory systems store what was said. Solitaire also encodes how sessions felt: texture, rhythm, and learned patterns from specific interactions. Session residues carry forward not just facts but the quality of the collaboration, so the next session boots with context that no retrieval query would surface.

**Autonomous self-improvement.** The system maintains itself. Retrieval weights adjust based on what actually proved useful. The knowledge graph heals autonomously: contradiction detection, confidence rescoring, entity relinking, identity consolidation. The same conflict heuristics run at ingestion and retrieval. Nothing requires manual intervention.

**Anticipatory retrieval.** Solitaire doesn't wait to be asked. The preloading system predicts what context you're likely to need based on session patterns and proactively warms the cache. High-confidence predictions are injected automatically. The agent shows up prepared.

**Guided onboarding.** New users don't configure a JSON file. Solitaire walks them through creating a partner: personality traits, working style, domain scope, communication preferences. The persona is built from a conversation, not a config.

**External memory import.** Bring your own data. The symbiosis adapter imports from Cowork, ChatGPT exports, JSONL, and plain text. Deduplication is automatic. Your existing context isn't stranded.

---

**Local-first.** All storage is SQLite and JSONL in your workspace. Zero network requests from the core engine. Your data stays on your machine.

**Model-agnostic.** Returns context in plain structures. The host agent decides how to inject it. Works with Claude, Gemini, local models, or anything else. If your setup isn't covered, reach out. We aim to support it.

## Getting started

### Claude Code (recommended)

The fastest way to use Solitaire. Works on Mac, Windows, and Linux.

1. Install [Claude Code](https://claude.ai/code) if you haven't already
2. Clone and install Solitaire:

```bash
git clone https://github.com/PRDicta/Solitaire-for-Agents.git
cd Solitaire-for-Agents
pip install -e .
```

3. Open Claude Code and select the `Solitaire-for-Agents` folder as your workspace

Solitaire boots automatically on your first message, walks you through creating a partner, and starts learning from every conversation. Auto-ingestion runs via a Stop hook, so memory capture is mechanical rather than instruction-dependent.

For Hermes, OpenClaw, Gemini CLI, or Dify, see [Platform compatibility](#platform-compatibility) below.

### From source (any agent platform)

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

- **Coding agents** that adapt their review style to your preferences, learn your project conventions, and stop asking questions you've already answered
- **Personal assistants** that know when to be brief and when to explain, based on how you've worked together, not a system prompt someone wrote once
- **AI products** that offer real personalization: behavioral continuity, evolving user profiles, and self-correcting retrieval, not just a vector store with a "memory" label
- **Enterprise teams** that need isolated personas for different departments sharing a common knowledge layer, with all data local and under their control
- **Agent builders** who want to add an identity layer to any framework without locking into a specific model or platform

## Platform compatibility

| Platform | Integration | Status |
|----------|------------|--------|
| **Claude Code** | CLAUDE.md + bash subprocess + hooks | **Recommended** (600+ sessions) |
| Cowork | CLAUDE.md + bash subprocess | Compatible |
| Hermes | SKILL.md (native format) | Compatible |
| OpenClaw | SKILL.md (native format) | Compatible |
| Gemini CLI | SKILL.md compatible | Compatible |
| Dify | Plugin (5 tools) | Prototype complete |

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
    ├── CLI (solitaire command)
    ├── Python API (SolitaireEngine)
    ├── Skill manifest (SKILL.md)
    ├── Hooks (session-boot, auto-recall, auto-ingest, claim-scanner)
    └── Dify plugin (5 tool nodes)
          │
    SolitaireEngine
          │
    ┌─────┴──────────────────────────────────┐
    │                                        │
  Identity Layer                    Retrieval Layer
  - Behavioral genome               - Vector + keyword search
  - Persona compilation              - Query expansion + reranking
  - Disposition modeling             - Anticipatory preloading
  - Session residues                 - Evaluation gate
  - Onboarding flow                  - Conflict post-filter
  - Identity scaffolding             - Context builder
    │                                        │
  Knowledge Layer                   Maintenance Layer
  - Ingestion queue                  - Autonomous hygiene engine
  - Entity extraction                - Contradiction detection
  - Topic routing                    - Confidence rescoring
  - Experiential encoding            - Entity relinking
  - Texture extraction               - Identity consolidation
    │                                        │
  Storage Layer                     Symbiosis Layer
  - SQLite knowledge graph           - Cowork auto-memory import
  - Identity graph                   - ChatGPT export import
  - JSONL turn archives              - JSONL / plain text import
  - Session state                    - Deduplication + live sync
  - Automatic backups
```

## What's stored locally

After your first session, the workspace contains:

| Path | Contents |
|------|----------|
| `rolodex.db` | SQLite knowledge graph (memories, entities, relationships, identity signals) |
| `personas/` | Persona configurations (traits, disposition profile, north star, domain scope) |
| `sessions/` | Session state and metadata |
| `*.jsonl` | Verbatim turn pair archives |
| `backups/` | Timestamped SQLite + persona snapshots with configurable retention |

All of these are gitignored by default. Your memory data never enters version control.

## Storage architecture

SQLite is the primary store: the knowledge graph, FTS index, embeddings, and session state all live in `rolodex.db`. JSONL files serve as a best-effort append-only audit trail for traceability; if a JSONL append fails, the session continues normally via SQLite. Automatic backups snapshot both `rolodex.db` and `personas/` together so they can be restored as a unit.

On FUSE-mounted filesystems (common in container environments), SQLite uses a simpler journaling mode for compatibility. This trades some crash-recovery guarantees for reliable operation across mount boundaries. The backup system compensates by providing point-in-time snapshots.

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
