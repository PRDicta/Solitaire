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
