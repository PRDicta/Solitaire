# Solitaire for Agents

Your agent forgets everything between sessions. Every conversation starts from zero: no memory of past decisions, no sense of the user's preferences, no continuity. The industry's answer has been to bolt on a vector database and call it "memory," but retrieving similar text isn't remembering. It's search.

Solitaire gives your agent a persistent cognitive layer: identity that evolves, a knowledge graph that grows, and retrieval that learns from its own performance. Your agent doesn't just recall what the user said. It knows what they care about, what they've been working on, and how to pick up exactly where it left off. Session 50 feels fundamentally different from session 5 because something real has accumulated between them.

This isn't a memory plugin. It's a cognitive architecture for AI agents.

## What your agent gets

**An evolving identity, not just a system prompt.** Solitaire maintains a persona layer with disposition traits, domain expertise, a north star, and a working style that shapes every response. This identity isn't static. It deepens as the relationship with the user develops: understanding of their projects, preferences, and communication patterns compounds over time. What the agent produces after months of use is qualitatively different from what any system prompt can achieve.

**Self-correcting retrieval.** Most retrieval systems treat all stored content equally. Solitaire tracks which recalled entries the agent actually uses in its responses and which it ignores. Entries that prove useful gain weight. Entries that consistently go unused lose significance. The retrieval system calibrates itself, surfacing the right context without being told what "right" means.

**Pattern detection and gap awareness.** The engine monitors hot topics (what the user keeps coming back to), dead zones (what they've moved on from), and gap signals (recurring queries where Solitaire can't provide good context). Gap signals trigger a proactive tool-finding pipeline that searches for relevant skills, MCP servers, or plugins and proposes them for approval. Your agent identifies its own blind spots and works to fill them.

**Seamless session continuity.** Each session ends with a residue: a narrative encoding of the session's arc, key decisions, and emotional register. On boot, the agent receives a tiered context package (persona, residue, briefing, identity graph, known facts) that reconstructs the working relationship in full. No re-explaining. No lost context. No "remind me what we were doing."

**Local-first, model-agnostic.** All data stays on the user's machine. Solitaire never makes network requests from its core engine. The API returns plain dicts and strings with no model-specific formatting. Claude, GPT, Gemini, Llama, a custom LLM: the host agent decides how to inject context into its own prompt. Solitaire doesn't care what model powers your agent. It cares about the relationship between the agent and the person using it.

## Install

```bash
pip install solitaire
```

Or install from source:

```bash
git clone https://github.com/PRDicta/Solitaire-for-Agents.git
cd Solitaire-for-Agents
pip install -e .
```

Optional dependencies for enhanced features:

```bash
pip install solitaire[embeddings]   # sentence-transformers + torch
pip install solitaire[llm]          # anthropic SDK
pip install solitaire[all]          # everything
```

**Requirements:** Python 3.10+. No external services, no containers, no Docker.

## Quick start

### Python API

```python
from solitaire import SolitaireEngine

engine = SolitaireEngine(workspace_dir="./my-agent-data")

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

### Cowork (zero-config)

Drop this folder into Cowork as your workspace. Solitaire boots automatically on your first message, walks you through creating a partner, and starts learning from every conversation. No code required.

## Platform compatibility

Solitaire runs on any platform that can execute Python 3.10+ and persist a directory between sessions.

| Platform | Integration | Status |
|----------|------------|--------|
| Claude Code / Cowork | CLAUDE.md + bash subprocess | Production (300+ sessions) |
| Hermes | SKILL.md (native format) | Compatible |
| OpenClaw | SKILL.md (native format) | Compatible |
| Gemini CLI | SKILL.md compatible | Compatible |
| Dify | Plugin (5 tools, marketplace-ready) | Shipped |

### Integrating Solitaire into your agent

Five things the host agent must provide:

1. **Persistent workspace directory.** A path on disk that survives between sessions. Set via `SOLITAIRE_WORKSPACE` env var or pass to `SolitaireEngine(workspace_dir=...)`.
2. **Python 3.10+ runtime.** One `pip install` and it runs.
3. **Command execution.** Subprocess (`solitaire <command>`) or Python import.
4. **System prompt injection.** Boot and recall return context blocks. The host agent injects these into the model's prompt. Without injection, Solitaire stores data but the model never sees it.
5. **Per-turn lifecycle hook.** Call `ingest-turn` after each exchange. Optionally call `auto-recall` before responses and `residue write` after ingesting.

The minimum viable loop: boot once, ingest every turn, end once.

## Architecture

```
Your Agent (Claude, GPT, Gemini, Llama, custom LLM, etc.)
    │
    ├── CLI (solitaire command, 53 commands)
    ├── Python API (SolitaireEngine class)
    ├── Skill manifest (SKILL.md for Hermes/OpenClaw/Gemini)
    └── Dify plugin (5 tool nodes)
          │
    SolitaireEngine (model-agnostic cognitive layer)
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

All of these are gitignored by default. Your agent's memory data never enters version control.

## Privacy

Solitaire's core engine makes zero network requests. All storage is local SQLite and JSONL files in the workspace directory. The AI model sees context during conversations (that's the point), but the database never leaves the user's machine.

The optional `[llm]` and `[embeddings]` dependencies make network calls to their respective APIs (Anthropic, HuggingFace) only when explicitly invoked. The core system works without them.

## License

AGPL-3.0. Commercial license available from Dicta Technologies Inc.

See [LICENSE](LICENSE) for full terms. Contact licensing@usedicta.com for commercial inquiries.
