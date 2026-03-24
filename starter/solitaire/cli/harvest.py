"""
Harvest and integrity commands -- safety net ingestion and gap repair.

Commands:
    solitaire harvest [--force-all] [--dry-run]
    solitaire harvest-full
    solitaire harvest-status
    solitaire integrity check
    solitaire integrity repair [--session <id>]
    solitaire build-chains [--session <id>] [--force]
    solitaire turn-pairs [--session <id>] [--limit N]
    solitaire decision-journal [--session <id>] [--limit N]
    solitaire hindsight-backfill [--all] [--dry-run] [--batch-size N]
"""
import click

from ._engine import get_engine, output_json, output_error


@click.group(name="integrity")
@click.pass_context
def integrity(ctx):
    """Integrity checking and repair."""
    pass


@integrity.command("check")
@click.pass_context
def integrity_check(ctx):
    """Detect sessions with messages that weren't ingested."""
    engine = get_engine(ctx)
    result = engine.integrity_check()
    output_json(result)


@integrity.command("repair")
@click.option("--session", "session_id", default=None, help="Scope to a specific session")
@click.pass_context
def integrity_repair(ctx, session_id):
    """Re-ingest messages that have no corresponding entries."""
    engine = get_engine(ctx)
    result = engine.integrity_repair(session_id=session_id)
    output_json(result)


@click.command("hindsight-backfill")
@click.option("--all", "all_entries", is_flag=True,
              help="Process all categories (default: high-signal only)")
@click.option("--dry-run", is_flag=True,
              help="Count matches without writing to the timeline")
@click.option("--batch-size", default=500, type=int,
              help="Entries per processing batch")
@click.pass_context
def hindsight_backfill(ctx, all_entries, dry_run, batch_size):
    """Backfill temporal timeline from existing rolodex entries.

    Scans entries not yet in the entity_timeline and runs temporal
    detection (status changes, version bumps, decisions, completions).
    Reports per-category hit rates for heuristic tuning.

    By default, processes high-signal categories only:
    implementation, definition, behavioral, instruction, warning,
    disposition_drift. Use --all to process every category.
    """
    engine = get_engine(ctx)
    try:
        result = engine.hindsight_backfill(
            all_entries=all_entries,
            dry_run=dry_run,
            batch_size=batch_size,
        )
        output_json(result)
    except Exception as e:
        output_error(f"hindsight-backfill failed: {e}")
