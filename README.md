# Solitaire

AI models are stateless. Every session starts from zero. The industry's answer so far has been to bolt on a vector database and call it "memory," but retrieving similar text isn't remembering. It's search. There's no understanding of what matters, no sense of how a relationship has evolved, no ability to learn from what worked and what didn't.

Solitaire is the layer that changes that. It gives any AI model a persistent self: an identity that evolves, a knowledge graph that grows, and a retrieval system that learns from its own performance. The model doesn't just recall what you said. It knows what you care about, what you've been working on, and how to pick up exactly where you left off. Session 50 feels fundamentally different from session 5 because something real has accumulated between them.

This isn't a memory plugin. It's cognitive infrastructure for AI agents.

## What Solitaire actually does

**Gives the model an identity, not just a prompt.** Solitaire maintains a persona layer with disposition traits, domain expertise, a north star, and a working style that shapes every response. This identity isn't static. It deepens as the relationship develops: the agent's understanding of you, your projects, your preferences, and your communication patterns compounds over time. What you get back after months of use is qualitatively different from what any system prompt can produce.

**Learns what matters by watching what works.** Most retrieval systems treat all stored content equally. Solitaire doesn't. It tracks which recalled entries the model actually uses in its responses and which it ignores. Entries that prove useful gain weight. Entries that consistently go unused lose significance. The retrieval system calibrates itself over time, surfacing the right context without being told what "right" means.

**Detects patterns you haven't articulated.** The engine monitors hot topics (what you keep coming back to), dead zones (what you've moved on from), and gap signals (what you need but Solitaire can't yet answer well). Gap signals trigger a proactive tool-finding pipeline that searches for relevant skills, MCP servers, or plugins and proposes them for your approval. The system identifies its own blind spots and works to fill them.

**Makes session boundaries invisible.** Each session ends with a residue: a narrative encoding of the session's arc, key decisions, and emotional register. On boot, the agent receives a tiered context package (persona, residue, briefing, identity graph, known facts) that reconstructs the working relationship in full. No re-explaining. No lost context. No "remind me what we were doing."

**Runs locally, works with any model.** All data stays on your machine. Solitaire never makes network requests from its memory engine. The core API returns plain dicts and strings with no model-specific formatting. Claude, Gemini, a custom LLM: the host decides how to inject context into its own prompt. Solitaire doesn't care what model you use. It cares about the relationship between the model and the person using it.

## Install

```bash
pip install solitaire
```

Or install from source:

```bash
git clone https://github.com/PRDicta/Solitaire.git
cd Solitaire
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

### Integration contract

Five things the host agent must provide:

1. **Persistent workspace directory.** A path on disk that survives between sessions. Set via `SOLITAIRE_WORKSPACE` env var or pass to `SolitaireEngine(workspace_dir=...)`.
2. **Python 3.10+ runtime.** One `pip install` and it runs.
3. **Command execution.** Subprocess (`solitaire <command>`) or Python import.
4. **System prompt injection.** Boot and recall return context blocks. The host agent injects these into the model's prompt. Without injection, Solitaire stores data but the model never sees it.
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

The optional `[llm]` and `[embeddings]` dependencies make network calls to their respective APIs (Anthropic, HuggingFace) only when explicitly invoked. The core memory system works without them.

## License

AGPL-3.0. Commercial license available from Dicta Technologies Inc.

See [LICENSE](LICENSE) for full terms. Contact licensing@usedicta.com for commercial inquiries.
