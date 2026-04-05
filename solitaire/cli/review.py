"""
Batch review commands — LLM-upgraded heuristic output.

Commands:
    solitaire review run [--category CATEGORY] [--limit N]
    solitaire review apply --decisions JSON
    solitaire review status
"""
import json
import sys

import click

from ._engine import get_engine, output_json, output_error


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
    engine = get_engine(ctx)

    try:
        from ..core.batch_review import run_review_gather
        result = run_review_gather(
            conn=engine._lib.rolodex.conn,
            category=category,
            limit=limit,
        )
        output_json(result or {"status": "empty", "items": [], "item_count": 0})
    except Exception as e:
        output_error(str(e))


@review.command("apply")
@click.option("--decisions", required=True, help="JSON array of review decisions")
@click.option("--dry-run", is_flag=True, help="Preview changes without writing")
@click.pass_context
def review_apply(ctx, decisions, dry_run):
    """Phase 2: apply model decisions. Writes corrections back."""
    engine = get_engine(ctx)

    try:
        decisions_list = json.loads(decisions)
    except (json.JSONDecodeError, TypeError) as e:
        output_error(f"Invalid JSON: {e}")
        return

    try:
        from ..core.batch_review import run_review_apply
        result = run_review_apply(
            conn=engine._lib.rolodex.conn,
            decisions=decisions_list,
            dry_run=dry_run,
        )
        output_json(result or {"status": "no_changes"})
    except Exception as e:
        output_error(str(e))


@review.command("status")
@click.pass_context
def review_status(ctx):
    """Show review run history and override rates."""
    engine = get_engine(ctx)

    try:
        from ..core.batch_review import get_review_status
        result = get_review_status(engine._lib.rolodex.conn)
        output_json(result or {"runs": [], "category_stats": {}})
    except Exception as e:
        output_error(str(e))
