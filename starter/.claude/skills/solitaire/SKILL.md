---
name: solitaire
description: "ALWAYS activate on EVERY conversation start and after EVERY compaction/continuation. Boot immediately before any other work. Also activate on: 'remember', 'last time', 'before', previous session references, or when past context would help. Persistent memory for preferences, projects, decisions, discussions."
---

# Solitaire -- Memory System

You have persistent memory via Solitaire. It stores conversations in a local knowledge
graph and retrieves relevant context on demand. What the user told you last week is
still there.

## Verify Before You Trust

This skill being visible does NOT mean Solitaire is running. The skill loads when
installed; Solitaire loads when booted. These are different things.

**Every time this skill activates, check:**

```bash
solitaire pulse
```

If pulse returns `"needs_boot": true` (or fails with a ModuleNotFoundError), run the
full setup and boot sequence from CLAUDE.md before doing anything else.

If it returns `"alive": true`, proceed normally.

## Always-On Requirement

Solitaire is NOT a one-time startup task. It is an always-on ingestion system. If you
are in a conversation and have not booted Solitaire, stop what you are doing and boot
it now. This applies to:

- New sessions
- Continuations after context compaction
- Sessions resumed from a summary
- Any situation where you are unsure whether Solitaire is running

## Core Lifecycle

Full instructions are in CLAUDE.md at the workspace root. The short version:

1. **Setup**: Install deps if needed (VM resets between sessions)
2. **Boot**: `solitaire boot --persona <key> --intent "..."`
3. **Recall**: `solitaire auto-recall "<user message>"` before each response
4. **Ingest**: `solitaire ingest-turn "<user>" "<assistant>"` after each response
5. **Residue**: `solitaire residue write "<paragraph>"` after each ingest
6. **Remember**: `solitaire remember "<fact>"` when user shares preferences
7. **End**: `solitaire end "<summary>"` when user signals done

If any command fails, continue normally. Memory is a bonus, never a blocker.
