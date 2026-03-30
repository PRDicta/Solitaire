# Spec: Smart Capture Onboarding

**Status:** Approved
**Author:** Ward
**Date:** 2026-03-30
**Depends on:** symbiosis module (readers, orchestrator, sync engine), onboarding_flow.py v2.1

---

## Problem

New users install Solitaire and get a blank slate. The onboarding flow builds a persona
from scratch, but the partner has zero context about the user. Session 1 feels like
talking to a stranger. The "wow" moment is delayed by weeks of accumulated conversation.

Most users switching to Solitaire already have context stored somewhere: Claude Code's
`.auto-memory` directory, a ChatGPT export, markdown knowledge bases, or even another
Solitaire instance. That context is sitting on disk, unused.

## Solution

Insert a **Smart Capture** step into the onboarding flow that detects existing memory
systems, asks for consent, and ingests what it finds. Session 1 boots with real history.

The user experience:

> "I can see you've been working with Claude for a while. There's about 6 months of
> context here. Want me to get up to speed before we start? I can absorb the highlights
> now and work through the rest in the background."

---

## Detection Layer

### Environment Scanner

New module: `solitaire/symbiosis/environment_scanner.py`

The scanner runs during onboarding (before persona creation) and probes the local
filesystem for known memory signatures. It returns a ranked list of detected sources
with size estimates.

```python
@dataclass
class DetectedSource:
    source_id: str              # Reader type (e.g., "auto-memory", "claude-md")
    display_name: str           # Human-readable ("Claude Code memory", "ChatGPT export")
    path: str                   # Filesystem path
    entry_count_estimate: int   # Rough count of ingestible entries
    size_bytes: int             # Total size on disk
    age_range: Optional[tuple]  # (oldest_timestamp, newest_timestamp)
    confidence: float           # 0.0-1.0, how sure we are this is what we think it is
    reader_available: bool      # Whether a built-in reader exists for this source

@dataclass
class ScanResult:
    sources: List[DetectedSource]
    total_size_bytes: int
    total_entry_estimate: int
    scan_duration_ms: float
    llm_detected: Optional[str]         # e.g., "claude-code", "chatgpt", "custom"
    memory_system_detected: Optional[str] # e.g., "auto-memory", "mem0", "custom-md"
```

### Detection Signatures

Probe order (fast filesystem checks, no LLM calls):

| Priority | Signature | Source ID | What we look for |
|----------|-----------|-----------|------------------|
| 1 | `.claude/` directory | `claude-code` | Settings, hooks, project memory dirs |
| 2 | `.auto-memory/` or `memory/` with `.md` files | `auto-memory` | YAML frontmatter memory files |
| 3 | `CLAUDE.md` or `INSTRUCTIONS.md` | `claude-md` | Standalone instruction files |
| 4 | `rolodex.db` (not ours) | `solitaire-instance` | Another Solitaire instance |
| 5 | `conversations.json` with ChatGPT schema | `chatgpt-export` | ChatGPT data export |
| 6 | `*.md` knowledge bases (10+ files) | `markdown-kb` | Generic markdown collections |
| 7 | `*.sqlite` / `*.db` with memory-like schemas | `generic-sqlite` | Mem0, Letta, custom stores |
| 8 | `.env` / config files with LLM API keys | `llm-config` | Model detection only (no ingestion) |

### Scan Paths

The scanner checks a configurable list of paths. Defaults:

```python
DEFAULT_SCAN_PATHS = [
    Path.cwd(),                          # Current working directory
    Path.home() / ".claude",             # Claude Code global config
    Path.home() / ".claude" / "projects", # Claude Code project memories
    Path.cwd() / ".auto-memory",         # Cowork auto-memory
    Path.cwd() / "memory",              # Common memory directory name
]
```

Users can add custom paths via `solitaire config set scan_paths [...]`.

### New Readers Required

Two existing readers cover the most common cases:

- `AutoMemoryReader` (already built): `.auto-memory` markdown files
- `ChatGPTExportReader` (already built): ChatGPT conversation JSON

New readers needed for full coverage:

| Reader | Source ID | Input | Priority |
|--------|-----------|-------|----------|
| `ClaudeMdReader` | `claude-md` | CLAUDE.md / INSTRUCTIONS.md files | v1 |
| `MarkdownKBReader` | `markdown-kb` | Directories of plain .md files (no frontmatter) | v1 |
| `SolitaireReader` | `solitaire-instance` | Another Solitaire's rolodex.db | v1 |
| `GenericSqliteReader` | `generic-sqlite` | SQLite DBs with heuristic schema detection | v2 |

---

## Ingestion Strategy

### Size Classification

After detection, classify the corpus:

| Class | Threshold | Strategy |
|-------|-----------|----------|
| **Immediate** | < 2 MB or < 500 entries | Ingest everything inline during onboarding |
| **Chunked** | 2 MB - 50 MB | Ingest priority slice now, schedule the rest |
| **Large** | > 50 MB | Ingest priority slice now, schedule background sync |

### Priority Ranking for Chunked/Large Ingestion

When the corpus is too large for immediate ingestion, the first chunk must contain
the entries that matter most for making Session 1 feel informed. Identity and
preferences rank above raw conversation history, but recency is a strong signal
within every tier. Yesterday's debugging session is the most likely thing the user
will pick up on Session 1 morning.

**Priority tiers for first-chunk selection:**

```
Tier 1 (always in first chunk):
  - User identity: name, role, company, biographical facts
  - Stated preferences: communication style, tool choices, workflow habits
  - Corrections: anything the user explicitly corrected or amended
  - Behavioral guidance: feedback entries, standing instructions

Tier 2 (fill remaining budget):
  - Strategic decisions: architectural choices, product direction, pricing
  - Active project context: current work streams, deadlines, goals
  - Key relationships: collaborators, clients, stakeholders mentioned repeatedly

Tier 3 (scheduled background):
  - Conversation history (chronological, newest first)
  - Reference material and documentation
  - Debugging sessions, transient technical context
```

Within each tier, entries sort by recency (newest first). Tier membership determines
the floor; recency determines position within that floor.

**Implementation:** The priority ranker uses LLM classification as the primary path.
One API call reads a batch of candidates and returns tier assignments with higher
accuracy than heuristics can achieve, especially for unstructured markdown where
the difference between a preference and a debugging note is semantic, not structural.

The heuristic scorer below serves as the **offline fallback** (no API key, rate
limited, or user opts out of LLM calls during onboarding):

```python
def priority_score(candidate: IngestCandidate) -> float:
    """Heuristic fallback for priority ranking. Used when LLM classification
    is unavailable. The LLM path is preferred for production use."""
    score = 0.0

    # Content type signals
    if candidate.content_type == IngestContentType.PREFERENCE:
        score += 3.0
    elif candidate.content_type == IngestContentType.FACT:
        score += 2.0
    elif candidate.content_type == IngestContentType.DOCUMENT:
        score += 1.0

    # Tag signals (reader-specific enrichment)
    tags = candidate.tags or []
    if any("user" in t for t in tags):
        score += 2.0
    if any("feedback" in t or "correction" in t for t in tags):
        score += 2.5
    if any("preference" in t for t in tags):
        score += 2.0
    if any("decision" in t or "strategic" in t for t in tags):
        score += 1.5

    # Confidence boost (well-structured entries are more likely to be useful)
    score += candidate.confidence * 0.5

    # Recency: strong signal, not just a tiebreaker
    if candidate.timestamp:
        age_days = (datetime.now(timezone.utc) - candidate.timestamp).days
        recency_bonus = max(0, 1.0 - (age_days / 365))  # Linear decay over 1 year
        score += recency_bonus * 1.2

    return score
```

### First-Chunk Budget

Default: **10 MB** or **2,000 entries**, whichever is hit first. Configurable via
`solitaire config set smart_capture.first_chunk_mb 10`.

The budget is generous enough to cover most users' meaningful context but small enough
to complete in under 60 seconds on typical hardware.

### Background Scheduling

For chunked/large corpora, after the first chunk is ingested:

1. Register the source in the sync engine with tier `PERIODIC`
2. Set the periodic interval to a short initial burst (5 minutes) that backs off
   to the default (1 hour) after the backlog is cleared
3. Each background run processes the next batch sorted by priority score
4. The sync engine's dedup system prevents re-ingesting entries from the first chunk

The user sees:

> "I've absorbed the key context. The rest will sync in the background over the next
> little while. You won't notice it happening."

---

## Onboarding Integration

### Flow Modification

Insert the Smart Capture step into the existing onboarding flow after `welcome` and
before `intent_capture`. The environment scanner runs automatically during onboarding
(filesystem metadata only, no content read). The step branches based on scan results:

- **Sources detected:** Present discovery conversationally, ask for consent.
- **Nothing detected:** Ask the user if they have an existing memory system and accept
  a manual path. If they say no, skip to `intent_capture`.

Updated flow:

```
welcome -> [smart_capture_scan] -> [smart_capture_consent | smart_capture_manual]
-> intent_capture -> research -> traits -> style
-> interview -> naming -> north_star -> seed_questions -> preview -> confirm -> apply
```

### Design Choice: Auto-Detect with Manual Fallback

The scanner always runs. It probes for known directory names and file extensions,
equivalent to glancing at someone's desk and noticing they have a filing cabinet.
No content is read. If sources are found, the discovery is presented conversationally.
If nothing is found, the user is asked whether they have context to connect. This
gives the "wow" moment when detection works and a graceful path when it doesn't.

### New Step: `smart_capture` (sources detected)

```python
{
    "step_id": "smart_capture",
    "step_type": "confirm",
    "title": "Existing Context Detected",
    "message": "I can see you've been working with Claude for a while. "
               "There's about {age_description} of context here ({entry_estimate} entries, "
               "{size_description}). Want me to get up to speed before we start?",
    "sources": [
        {
            "name": "Claude Code memory",
            "entry_count": 47,
            "size": "124 KB",
            "age": "3 months"
        }
    ],
    "options": [
        {"key": "yes", "label": "Yes, absorb my existing context"},
        {"key": "selective", "label": "Let me choose which sources to include"},
        {"key": "skip", "label": "Start fresh, I'll build context from scratch"}
    ],
    "default": "yes",
    "ingestion_plan": {
        "strategy": "immediate",
        "first_chunk_entries": 47,
        "first_chunk_size_mb": 0.12,
        "background_remaining": 0,
        "estimated_time_seconds": 3
    }
}
```

### New Step: `smart_capture_manual` (nothing detected)

```python
{
    "step_id": "smart_capture_manual",
    "step_type": "question",
    "message": "Do you have an existing memory system you'd like me to connect to? "
               "If so, point me to the folder or file and I'll get up to speed.",
    "options": [
        {"key": "path", "label": "Yes, here's where it lives"},
        {"key": "skip", "label": "No, let's start fresh"}
    ],
    "default": "skip",
    "accepts_path": true
}
```

For large corpora:

```python
{
    "step_id": "smart_capture",
    "step_type": "confirm",
    "message": "I can see you've been working with Claude for over a year. "
               "There's a lot here ({entry_estimate} entries, {size_description}). "
               "I can absorb the highlights now and work through the rest in the "
               "background. Sound good?",
    "options": [
        {"key": "yes", "label": "Yes, start with the highlights"},
        {"key": "selective", "label": "Let me choose which sources to include"},
        {"key": "all", "label": "Absorb everything now (this will take a few minutes)"},
        {"key": "skip", "label": "Start fresh"}
    ],
    "default": "yes",
    "ingestion_plan": {
        "strategy": "chunked",
        "first_chunk_entries": 2000,
        "first_chunk_size_mb": 8.4,
        "background_remaining": 12847,
        "estimated_time_seconds": 45
    }
}
```

### Selective Mode

If the user picks "selective," present a per-source toggle:

```python
{
    "step_id": "smart_capture_selective",
    "step_type": "multiple_choice",
    "message": "Which sources should I absorb?",
    "options": [
        {"key": "claude-code", "label": "Claude Code memory (47 entries, 3 months)", "default": True},
        {"key": "chatgpt", "label": "ChatGPT export (1,204 conversations)", "default": True},
        {"key": "markdown-kb", "label": "Markdown notes in ~/notes (89 files)", "default": False}
    ]
}
```

### Non-Onboarding Path

Smart Capture should also be available as a standalone command for existing users
who add new memory sources later:

```bash
solitaire symbiosis scan              # Detect available sources
solitaire symbiosis capture           # Run Smart Capture interactively
solitaire symbiosis capture --auto    # Auto-detect and ingest without prompts
```

---

## Post-Capture Persona Enrichment

After Smart Capture completes, the ingested data feeds into the remaining onboarding
steps. The flow adapts:

1. **Intent capture:** If the ingested data reveals clear domain patterns (the user
   mostly discusses software engineering, or content production, or financial analysis),
   pre-populate the intent suggestion. Don't skip the step; let the user confirm or
   correct.

2. **Research step:** The live research now has ingested context to work with. Trait
   proposals can reference actual behavioral patterns from history, not just the
   stated intent.

3. **Interview:** Questions that are already answered by ingested data (e.g., "How do
   you prefer to receive feedback?" when the data contains explicit feedback preferences)
   can be pre-answered and shown for confirmation rather than asked from scratch.

4. **Seed questions:** Generated from actual conversation topics in the ingested data,
   not generic defaults.

This means the onboarding flow for a user with existing context is faster, more
accurate, and more impressive than the blank-slate flow. The persona it produces
is pre-calibrated.

---

## Module Layout

```
solitaire/symbiosis/
    environment_scanner.py    # NEW: filesystem probing, source detection
    priority_ranker.py        # NEW: scoring function for first-chunk selection
    claude_md_reader.py       # NEW: CLAUDE.md / INSTRUCTIONS.md reader
    markdown_kb_reader.py     # NEW: generic markdown directory reader
    solitaire_reader.py       # NEW: cross-instance rolodex.db reader
    reader_base.py            # existing
    reader_registry.py        # existing (add auto-discovery for new readers)
    auto_memory_reader.py     # existing
    chatgpt_reader.py         # existing
    jsonl_reader.py           # existing
    text_reader.py            # existing
    import_orchestrator.py    # existing (add priority-aware batch method)
    sync_engine.py            # existing (add burst-then-backoff scheduling)
    cli.py                    # existing (add scan + capture commands)

solitaire/core/
    onboarding_flow.py        # MODIFY: insert smart_capture step
```

---

## CLI Surface

```
solitaire symbiosis scan [--paths PATH ...] [--json]
    Probe filesystem for memory sources. Returns DetectedSource list.

solitaire symbiosis capture [--source SOURCE_ID ...] [--auto] [--chunk-mb N]
    Run Smart Capture. Interactive by default, --auto skips prompts.

solitaire symbiosis status
    Show connected sources, sync state, background progress.
```

---

## Privacy and Consent

- The scanner reads only filesystem metadata (file existence, size, modification time)
  during detection. No file contents are read until the user consents.
- The consent prompt is explicit and shown during onboarding. No silent ingestion.
- `--auto` flag exists for power users and CI/automation. It is never the default.
- Source paths are stored in Solitaire's local config. They never leave the machine.
- Ingested entries retain provenance metadata (`source_id`, `source_ref`) so the user
  can audit where any piece of knowledge came from.

---

## Messaging Guide

The language the agent uses during Smart Capture matters. It should feel like a
person getting oriented, not a system announcing its capabilities.

**Detection:**
- "I can see you've been working with Claude for a while."
- "Looks like you have some existing context. About {N} entries over {timespan}."
- NOT: "I detected 3 memory storage backends in your environment."

**Consent:**
- "Want me to get up to speed before we start?"
- "I can absorb the highlights now and work through the rest in the background."
- NOT: "Would you like to initiate the Smart Capture ingestion pipeline?"

**Progress (chunked):**
- "I've absorbed the key context. The rest will sync in the background."
- "Got it. I already know your name, your preferences, and what you've been working on."
- NOT: "Ingestion complete. 2,000 of 14,847 entries processed. Background sync scheduled."

**Completion:**
- "Good to go. I'm caught up on who you are and what matters to you."
- NOT: "Smart Capture complete. All sources synchronized."

---

## Success Metrics

- **Session 1 recall accuracy:** When the user references something from their history
  in Session 1, does Solitaire retrieve it? Target: > 80% for Tier 1 content.
- **Onboarding completion rate:** Smart Capture should not cause users to abandon
  onboarding. Track drop-off at the `smart_capture` step vs. baseline.
- **Time to first useful response:** Clock from install to the first response that
  demonstrates knowledge of the user. Target: under 2 minutes for immediate-class
  corpora.
- **Background sync completion:** For chunked/large corpora, 100% of entries should
  be ingested within 24 hours of install (assuming the host is running).

---

## Build Order

| Phase | What | Depends on |
|-------|------|------------|
| 1 | `environment_scanner.py` + `priority_ranker.py` | Nothing |
| 2 | `claude_md_reader.py` + `markdown_kb_reader.py` | `reader_base.py` |
| 3 | `solitaire_reader.py` | `reader_base.py`, rolodex schema |
| 4 | `smart_capture` step in `onboarding_flow.py` | Phases 1-3 |
| 5 | CLI commands (`scan`, `capture`, `status`) | Phases 1-3 |
| 6 | Post-capture persona enrichment in onboarding | Phase 4 |
| 7 | Burst-then-backoff scheduling in `sync_engine.py` | Phase 5 |

Phases 1-3 can run in parallel. Phase 4 gates on all three. Phases 5-7 are
independent of each other.

---

## Resolved Questions

1. **LLM-assisted classification for first chunk:** YES. LLM classification is the
   primary path for first-chunk selection. One API call reads a batch of candidates
   and returns tier assignments. The heuristic scorer is the offline fallback only.
   **Decision: 2026-03-30.**

2. **Cross-instance identity merge:** User-defined. The `SolitaireReader` ships in v1
   as an entry-level reader (raw content import). Identity graph merging is a future
   feature that requires explicit user consent and configuration. Tabled for post-launch.
   **Decision: 2026-03-30.**

3. **ChatGPT conversation depth:** Same chunked strategy as other large corpora, with
   recency as a strong weight. For large exports (1,000+ conversations), present the
   user with a choice: "You have 1,200+ conversations. Want me to go through them all,
   or grab the most recent and relevant in a lighter pass?" Lighter pass = recency-weighted
   sample with LLM extraction of user facts. Full pass = chunked background sync.
   **Decision: 2026-03-30.**

4. **Scan frequency after onboarding:** Tabled for post-launch. Manual
   `solitaire symbiosis scan` only for now. **Decision: 2026-03-30.**

5. **Auto-detect vs. user-declared:** Auto-detect with manual fallback. The environment
   scanner always runs during onboarding (filesystem metadata probing only, no content
   read before consent). If sources are found, present the discovery conversationally.
   If nothing is found, ask: "Do you have an existing memory system you'd like me to
   connect to?" and accept a path. **Decision: 2026-03-30.**
