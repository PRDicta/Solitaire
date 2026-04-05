"""
Batch review commands — LLM-upgraded heuristic output.

Works without a booted session. Connects directly to the persona's
rolodex.db since review is a maintenance operation, not a live session.

Commands:
    solitaire review run [--category CATEGORY] [--limit N]
    solitaire review apply --decisions JSON
    solitaire review status
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

import click

from ._engine import output_json, output_error


def _get_review_conn(ctx) -> sqlite3.Connection:
    """Get a database connection for review operations.

    Finds the active persona's rolodex.db without requiring a booted
    engine. Falls back to workspace-level rolodex.db.
    """
    workspace = ctx.obj.get("workspace", os.getcwd())

    # Try to find persona from session state
    session_file = Path(workspace) / ".solitaire_session"
    persona_key = None
    if session_file.exists():
        try:
            with open(session_file) as f:
                data = json.load(f)
                persona_key = data.get("persona_key")
        except Exception:
            pass

    # Look for rolodex.db in persona dir or workspace
    search_paths = []
    if persona_key:
        search_paths.append(Path(workspace) / "personas" / persona_key / "rolodex.db")
    search_paths.append(Path(workspace) / "sessions" / "rolodex.db")
    search_paths.append(Path(workspace) / "rolodex.db")

    # Also check all persona dirs
    persona_dir = Path(workspace) / "personas"
    if persona_dir.is_dir():
        for p in persona_dir.iterdir():
            if p.is_dir():
                search_paths.append(p / "rolodex.db")

    for db_path in search_paths:
        if db_path.exists():
            return sqlite3.connect(str(db_path))

    raise FileNotFoundError(
        f"No rolodex.db found. Searched: {[str(p) for p in search_paths[:3]]}"
    )


@click.group(name="review")
@click.pass_context
def review(ctx):
    """Batch review: model-judged upgrade of heuristic output."""
    pass


@review.command("run")
@click.option("--category", default="auto",
              help="Review category (auto, commitment_signals, identity_candidates, "
                   "disposition_drift, growth_edge_evolution, lifecycle_validation)")
@click.option("--limit", default=20, help="Maximum items to gather")
@click.pass_context
def review_run(ctx, category, limit):
    """Phase 1: gather items for review. Outputs structured JSON."""
    try:
        conn = _get_review_conn(ctx)
    except FileNotFoundError as e:
        output_error(str(e))
        return

    try:
        from ..core.batch_review import run_review_gather
        result = run_review_gather(
            conn=conn,
            category=category,
            limit=limit,
        )
        output_json(result or {"status": "empty", "items": [], "item_count": 0})
    except Exception as e:
        output_error(str(e))
    finally:
        conn.close()


@review.command("apply")
@click.option("--decisions", required=True, help="JSON array of review decisions")
@click.option("--dry-run", is_flag=True, help="Preview changes without writing")
@click.pass_context
def review_apply(ctx, decisions, dry_run):
    """Phase 2: apply model decisions. Writes corrections back."""
    try:
        conn = _get_review_conn(ctx)
    except FileNotFoundError as e:
        output_error(str(e))
        return

    try:
        decisions_list = json.loads(decisions)
    except (json.JSONDecodeError, TypeError) as e:
        output_error(f"Invalid JSON: {e}")
        return

    try:
        from ..core.batch_review import run_review_apply
        result = run_review_apply(
            conn=conn,
            decisions=decisions_list,
            dry_run=dry_run,
        )
        output_json(result or {"status": "no_changes"})
    except Exception as e:
        output_error(str(e))
    finally:
        conn.close()


@review.command("status")
@click.pass_context
def review_status(ctx):
    """Show review run history and override rates."""
    try:
        conn = _get_review_conn(ctx)
    except FileNotFoundError as e:
        output_error(str(e))
        return

    try:
        from ..core.batch_review import get_review_status
        result = get_review_status(conn)
        output_json(result or {"runs": [], "category_stats": {}})
    except Exception as e:
        output_error(str(e))
    finally:
        conn.close()
