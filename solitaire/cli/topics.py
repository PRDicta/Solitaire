"""
Topics commands — topic management and backfill.

Commands:
    solitaire topics list
    solitaire topics stats
    solitaire topics backfill
"""
import json
import uuid
from datetime import datetime, timezone

import click

from ._engine import get_engine, output_json, output_error


@click.group()
@click.pass_context
def topics(ctx):
    """Topic management operations."""
    pass


@topics.command("list")
@click.pass_context
def topics_list(ctx):
    """List all topics with entry counts."""
    engine = get_engine(ctx)
    conn = engine._lib.rolodex.conn

    rows = conn.execute(
        "SELECT label, entry_count, description FROM topics ORDER BY entry_count DESC LIMIT 50"
    ).fetchall()

    if not rows:
        output_json({"status": "empty", "message": "No topics yet. Run 'topics backfill' to populate."})
        return

    results = []
    for row in rows:
        results.append({
            "label": row["label"] if hasattr(row, "keys") else row[0],
            "entry_count": row["entry_count"] if hasattr(row, "keys") else row[1],
            "description": (row["description"] if hasattr(row, "keys") else row[2]) or "",
        })
    output_json({"status": "ok", "topics": results, "total": len(results)})


@topics.command("stats")
@click.pass_context
def topics_stats(ctx):
    """Show topic coverage statistics."""
    engine = get_engine(ctx)
    conn = engine._lib.rolodex.conn

    total_topics = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    total_entries = conn.execute(
        "SELECT COUNT(*) FROM rolodex_entries WHERE superseded_by IS NULL"
    ).fetchone()[0]
    assigned = conn.execute(
        "SELECT COUNT(*) FROM rolodex_entries WHERE superseded_by IS NULL AND topic_id IS NOT NULL"
    ).fetchone()[0]
    unassigned = total_entries - assigned
    coverage = (assigned / total_entries * 100) if total_entries > 0 else 0

    output_json({
        "total_topics": total_topics,
        "total_entries": total_entries,
        "assigned_entries": assigned,
        "unassigned_entries": unassigned,
        "coverage_percent": round(coverage, 1),
    })


@topics.command("backfill")
@click.pass_context
def topics_backfill(ctx):
    """Batch-populate the topics table from existing entries using tag patterns.

    Runs in two passes:
      Pass 1: Create topics from predefined tag clusters
      Pass 2: Assign entries to topics based on tag overlap

    Safe to re-run: skips topics that already exist (by label).
    """
    import time as _bf_time

    engine = get_engine(ctx)
    conn = engine._lib.rolodex.conn
    _bf_start = _bf_time.time()

    # ── Topic definitions: (label, description, tag_patterns) ──
    # tag_patterns: entry must have at least one of these tags to match.
    # Order matters: first match wins (entries get one primary topic).
    _TOPIC_DEFS = [
        ("solitaire product", "Solitaire for Agents game, product development, consolidated repo",
         {"solitaire product", "solitaire", "unified product", "consolidated repo"}),
        ("librarian system", "The Librarian memory system, recall, retrieval, ingestion, rolodex",
         {"librarian", "the librarian", "librarian_cli", "rolodex", "codex"}),
        ("persona system", "Persona, disposition drift, trait drift, cognitive profile",
         {"persona", "drift", "trait drift", "disposition drift"}),
        ("identity system", "Identity graph, self-knowledge graph, experiential encodings",
         {"identity graph spec", "self-knowledge graph", "identity system", "identity", "genome"}),
        ("content operations", "Content pipeline, posts, content production",
         {"dicta", "linkedin", "post", "content", "voice", "sdr", "gamma"}),
        ("session mechanics", "Residue, session texture, boot, briefing, compaction",
         {"residue", "session texture", "boot", "session", "housekeeping"}),
        ("conversational rhythm", "Attention patterns, warmth, acknowledgment, interactional style",
         {"conversational rhythm", "attention patterns", "warmth_appreciated",
          "positive_acknowledgment", "plusvibe"}),
        ("compression engine", "YAML compression, behavioral signature scoring, commitment eval",
         {"compression engine", "yaml compression", "commitment eval harness",
          "commitment scorer", "commitment eval", "behavioral signature scorer"}),
        ("recall and retrieval", "Auto-recall, search, reranking, query expansion, FTS",
         {"recall", "retrieval", "reranker", "searcher", "fts", "auto-recall"}),
        ("infrastructure", "Git, GitHub, MCP, API, CLI tooling, Python, SQLite",
         {"git", "github", "mcp", "api", "cli", "python", "sqlite", "slack"}),
        ("business operations", "Business context, incorporation, contracts, clients, revenue",
         {"cowork", "creations"}),
        ("research", "Papers, benchmarks, LLM research, bounded context hypothesis",
         {"math", "llm", "anthropic", "reddit"}),
    ]

    created = 0
    assigned = 0
    skipped_topics = 0

    # Pass 1: Create topics (skip existing)
    topic_map = {}  # label -> topic_id
    for label, desc, _tags in _TOPIC_DEFS:
        existing = conn.execute(
            "SELECT id FROM topics WHERE label = ?", (label,)
        ).fetchone()
        if existing:
            topic_map[label] = existing["id"] if hasattr(existing, "keys") else existing[0]
            skipped_topics += 1
        else:
            topic_id = str(uuid.uuid4())
            now_ts = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """INSERT INTO topics
                   (id, label, description, created_at, last_updated, entry_count)
                   VALUES (?, ?, ?, ?, ?, 0)""",
                (topic_id, label, desc, now_ts, now_ts)
            )
            conn.execute(
                "INSERT INTO topics_fts (topic_id, label, description) VALUES (?, ?, ?)",
                (topic_id, label, desc)
            )
            topic_map[label] = topic_id
            created += 1
    conn.commit()

    # Pass 2: Assign entries. Batch for speed.
    entry_rows = conn.execute(
        """SELECT id, tags FROM rolodex_entries
           WHERE superseded_by IS NULL
             AND tags IS NOT NULL AND tags != '[]'
             AND topic_id IS NULL"""
    ).fetchall()

    now_ts = datetime.now(timezone.utc).isoformat()
    batch_updates = []
    batch_assignments = []
    topic_counts = {label: 0 for label in topic_map}

    for row in entry_rows:
        row_id = row["id"] if hasattr(row, "keys") else row[0]
        row_tags = row["tags"] if hasattr(row, "keys") else row[1]
        try:
            entry_tags = set(json.loads(row_tags)) if row_tags else set()
        except Exception:
            continue
        entry_tags_lower = {t.lower().strip() for t in entry_tags if t.strip()}

        matched_label = None
        for label, _desc, tag_patterns in _TOPIC_DEFS:
            if entry_tags_lower & tag_patterns:
                matched_label = label
                break

        if matched_label:
            tid = topic_map[matched_label]
            batch_updates.append((tid, row_id))
            batch_assignments.append((
                str(uuid.uuid4()), row_id, tid, 0.75, "backfill", now_ts
            ))
            topic_counts[matched_label] += 1
            assigned += 1

    if batch_updates:
        conn.executemany(
            "UPDATE rolodex_entries SET topic_id = ? WHERE id = ?",
            batch_updates
        )
        conn.executemany(
            """INSERT INTO topic_assignments
               (id, entry_id, topic_id, confidence, source, assigned_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            batch_assignments
        )
        for label, count in topic_counts.items():
            if count > 0:
                conn.execute(
                    "UPDATE topics SET entry_count = ?, last_updated = ? WHERE id = ?",
                    (count, now_ts, topic_map[label])
                )
        conn.commit()

    elapsed = _bf_time.time() - _bf_start
    unassigned = conn.execute(
        "SELECT COUNT(*) FROM rolodex_entries WHERE superseded_by IS NULL AND topic_id IS NULL"
    ).fetchone()[0]

    output_json({
        "status": "ok",
        "topics_created": created,
        "topics_skipped": skipped_topics,
        "entries_assigned": assigned,
        "entries_unassigned": unassigned,
        "topic_breakdown": {k: v for k, v in topic_counts.items() if v > 0},
        "elapsed_seconds": round(elapsed, 2),
    })
