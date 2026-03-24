# Solitaire Command Reference

Complete CLI reference. All commands output JSON to stdout, diagnostics to stderr.

**Exit codes:** `0` success, `1` internal error, `2` user error (bad arguments).

**Environment:** Set `SOLITAIRE_WORKSPACE` to the persistent data directory. Defaults
to current working directory.

---

## Boot

```bash
solitaire boot --pre-persona
```
List available personas without initializing the engine.

**Output:** `{status, available_personas: [{key, display_name, short_label, description}], template_creation_enabled}`

```bash
solitaire boot --persona KEY [--intent "context text"] [--cold]
```
Boot with the specified persona. `--intent` pre-loads relevant memories based on
the user's topic. `--cold` skips experiential memory and residue (testing mode).

**Output:** `{status, version, mode, session_id, persona, active_persona, total_entries, first_turn_briefed, boot_files: {context, operations}, todo_pin}`

```bash
solitaire boot --resume [--intent "context"]
```
Resume the last active persona from session state. Use after context compaction
or session continuation.

---

## Core Operations

```bash
solitaire ingest-turn "user message" "assistant response"
solitaire ingest-turn --stdin               # read {"user":"...","assistant":"..."} from stdin
solitaire ingest-turn -                     # alias for --stdin
```
Ingest a user + assistant turn pair. Runs the full enrichment pipeline: entity
extraction, knowledge graph update, temporal reasoning, identity enrichment,
retrieval usage evaluation.

**Output:** `{user: {ingested, entry_ids}, assistant: {ingested, entry_ids}, enrichment: {...}, retrieval_usage: {...}}`

```bash
solitaire auto-recall "user's current message"
```
Preflight evaluation (intent classification, consistency check, sanity scan) plus
targeted memory retrieval. Run before composing each response from turn 2 onward.

**Output:** `{preflight: {proceed, intent, flags}, entries_found, context_block, entry_ids}`

```bash
solitaire recall "query" [--no-preflight]
```
Search memory for relevant context. `--no-preflight` skips the evaluation gate
for direct queries.

**Output:** `{entries_found, context_block, entry_ids}`

```bash
solitaire remember "fact about the user"
```
Store a privileged fact as user_knowledge (3x retrieval boost, never demoted).

**Output:** `{remembered, entry_ids}`

```bash
solitaire correct OLD_ENTRY_ID "corrected text"
```
Supersede a wrong entry with corrected content. The old entry is marked as
superseded; the new entry links back to it.

**Output:** `{corrected, old_id, new_id}`

```bash
solitaire end ["summary of what was accomplished"]
```
End the current session. Adjusts retrieval weights, updates project clusters,
and closes cleanly. Summary is optional but recommended.

**Output:** `{status, session_id, duration, entries_ingested}`

```bash
solitaire pulse
```
Sub-second heartbeat check. Returns whether the engine is alive and whether
a boot is needed.

**Output:** `{alive, needs_boot, session_id, persona, uptime_seconds}`

```bash
solitaire auto-evaluate "message"
```
Standalone evaluation gate (no recall). Classifies intent and checks consistency
without triggering retrieval.

**Output:** `{proceed, intent, flags, warnings}`

---

## Ingestion

```bash
solitaire ingest user "message text" [--user-knowledge] [--corrects OLD_ID]
solitaire ingest assistant "message text"
```
Low-level single-message ingestion. Use `ingest-turn` for the standard per-turn
flow. `--user-knowledge` marks the entry as privileged (3x boost).
`--corrects` supersedes an existing entry.

**Output:** `{ingested, entry_ids, category}`

---

## Persona Management

```bash
solitaire persona list
```
List all available personas with their keys, display names, and descriptions.

```bash
solitaire persona show KEY
```
Show detailed persona configuration: identity, traits, north star, domain scope.

```bash
solitaire persona create
```
Interactive guided persona creation.

---

## Onboarding

```bash
solitaire onboard create [--intent "context"]
```
Start the persona creation onboarding flow. Returns the first step as JSON.

**Output:** `{step_id, step_type, options: [...], metadata: {cancel_available}}`

```bash
solitaire onboard flow-step STEP_ID "user_input"
```
Advance the onboarding flow. Pass the user's selection or input for the current
step. Returns the next step or the final apply/cancelled status.

**Output:** Next step JSON, or `{status: "applied", persona_key}` / `{status: "cancelled"}`

---

## Session Residue

```bash
solitaire residue write "paragraph describing session arc and texture"
```
Write or overwrite the rolling session residue. Not a summary; a felt sense of
the session's trajectory. Each call replaces the previous residue.

```bash
solitaire residue latest
```
Read the most recent session residue.

**Output:** `{residue, session_id, written_at}`

---

## Identity

```bash
solitaire identity show [--budget N] [--section SECTION]
```
Show the identity context block: north star, growth edges, known patterns,
recent realizations, open tensions. `--budget` sets token limit (default 1500).
`--section` filters to a specific section (edges, patterns, realizations, tensions).

**Output:** `{identity_block, sections, token_count}`

```bash
solitaire identity stats
```
Identity graph statistics: node counts by type, edge counts, last update times.

---

## Analytics

```bash
solitaire analytics patterns [--window N] [--stale-days N] [--gap-window N]
```
Retrieval pattern report. Surfaces hot topics (active work streams), dead zones
(stale context not retrieved recently), and gap signals (recurring queries with
no good results). `--window` sets session lookback (default 5). `--stale-days`
sets staleness threshold (default 30). `--gap-window` sets gap detection window
in days (default 14).

```bash
solitaire analytics stats
```
System statistics: entry count, topic count, session count, DB size.

```bash
solitaire analytics retrieval-stats [--session SESSION_ID]
```
Retrieval outcome statistics: use rate, top used entries, top ignored entries.
Per-session if `--session` is provided, otherwise aggregate.

---

## Proactive Tool Finding

```bash
solitaire tools find [--gap-window N] [--min-occurrences N]
```
Run the gap-to-search pipeline. Identifies recurring unmet needs from gap signals
and searches pluggable registries for relevant tools.

```bash
solitaire tools proposals
```
List pending tool proposals awaiting user decision.

```bash
solitaire tools approve PROPOSAL_ID
solitaire tools dismiss PROPOSAL_ID [--reason "text"]
solitaire tools install PROPOSAL_ID
```
Act on a tool proposal: approve (queued for install), dismiss (with optional
reason), or mark as installed.

```bash
solitaire tools record-use TOOL_NAME TOOL_SOURCE
```
Record that a tool was used in the current session. Feeds into usage tracking.

```bash
solitaire tools report
```
Full tool report: proposals (pending/approved/dismissed), installed tools,
usage frequency, unused tools flagged for review.

---

## User Profile

```bash
solitaire profile set KEY VALUE
solitaire profile show
solitaire profile delete KEY
```
Structured key-value store for persistent user preferences. Deduplicated (setting
a key replaces the old value). Always loaded on boot. Common keys: `name`,
`timezone`, `response_style`, `preferred_language`, `editor`, `os`.

---

## Browsing and Introspection

```bash
solitaire browse recent [N]              # Default: 20 most recent entries
solitaire browse entry ENTRY_ID          # View a specific entry
solitaire browse knowledge               # Knowledge graph overview
```

---

## Harvest Pipeline

Safety-net ingestion that processes conversation logs missed by per-turn ingestion.

```bash
solitaire harvest [--force-all] [--dry-run]
```
Run the conversation harvest. `--force-all` reprocesses all logs.
`--dry-run` previews what would be processed without writing.

```bash
solitaire harvest-full
```
Full pipeline: harvest + build-chains + integrity check. Single command for
comprehensive maintenance.

```bash
solitaire harvest-status
```
Show harvest progress without running a harvest.

---

## Narrative Chains

```bash
solitaire build-chains [--session SESSION_ID] [--force]
```
Build narrative reasoning chains for sessions. Extracts the arc of multi-turn
reasoning sequences. `--force` builds even for short segments.

```bash
solitaire turn-pairs [--session SESSION_ID] [--limit N]
```
Ingest user+assistant turn pairs as atomic units. `--limit` sets the number of
recent sessions to process (default 10).

```bash
solitaire decision-journal [--session SESSION_ID] [--limit N]
```
Extract decisions as first-class entities. Decisions are surfaced in future
recall when relevant topics arise.

---

## Integrity

```bash
solitaire integrity check
```
Validate DB integrity: orphan entries, missing topic links, stale session
references, schema consistency.

```bash
solitaire integrity repair [--session SESSION_ID]
```
Auto-repair integrity issues found by check. `--session` limits repair to
a specific session's data.

---

## Skill Packs

```bash
solitaire load-skill list
```
List available domain knowledge packs.

```bash
solitaire load-skill auto "user message"
```
Auto-detect relevant skill packs based on message content. Returns matching
packs with their keyword triggers and token costs.

```bash
solitaire load-skill load PACK_NAME
```
Load a specific skill pack into the current session context.

---

## Reflection

```bash
solitaire reflect [--force]
```
Run session reflection: analyze skill usage patterns, persona effectiveness,
and suggest improvements. `--force` overrides the cooldown timer (default:
one reflection per session).

---

## Maintenance

```bash
solitaire maintenance run [--full]
```
Background maintenance cycle: retrieval weight adjustment, pattern detection,
knowledge graph hygiene. `--full` runs the extended cycle including
topic reclustering and dead entry pruning.

---

## Python API Quick Reference

All CLI commands map to `SolitaireEngine` methods:

| CLI Command | Python Method |
|-------------|---------------|
| `boot --pre-persona` | `engine.boot_pre_persona()` |
| `boot --persona KEY` | `engine.boot(persona_key="KEY")` |
| `ingest-turn` | `engine.ingest(user_msg, assistant_msg)` |
| `auto-recall` | `engine.recall(query, include_preflight=True)` |
| `recall` | `engine.recall(query)` |
| `remember` | `engine.remember(fact)` |
| `end` | `engine.end(summary)` |
| `pulse` | `engine.pulse()` |
| `correct` | `engine.correct(old_id, text)` |
| `residue write` | `engine.write_residue(text)` |
| `residue latest` | `engine.get_residue()` |
| `identity show` | `engine.get_identity()` (via CLI wrapper) |
| `analytics patterns` | `engine.get_patterns()` |
| `analytics retrieval-stats` | `engine.get_retrieval_stats()` |
| `tools find` | `engine.find_tools()` |
| `tools proposals` | `engine.get_tool_proposals()` |
| `profile set` | `engine.profile_set(key, value)` |
| `profile show` | `engine.profile_show()` |
| `harvest` | `engine.harvest()` |
| `harvest-full` | `engine.harvest_full()` |
| `build-chains` | `engine.build_chains()` |
| `reflect` | `engine.reflect()` |

All methods return plain dicts. No model-specific formatting.
