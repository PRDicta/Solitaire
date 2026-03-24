"""
Browse commands -- view entries by recency, category, or ID.

Commands:
    solitaire browse recent [N]
    solitaire browse entry <id>
    solitaire browse knowledge
"""
import click

from ._engine import get_engine, output_json, output_error


@click.group()
@click.pass_context
def browse(ctx):
    """Browse rolodex entries."""
    pass


@browse.command("recent")
@click.argument("limit", default=20, type=int)
@click.pass_context
def browse_recent(ctx, limit):
    """View most recent entries."""
    engine = get_engine(ctx)
    result = engine.browse_recent(limit=limit)
    output_json(result)


@browse.command("entry")
@click.argument("entry_id")
@click.pass_context
def browse_entry(ctx, entry_id):
    """View a specific entry by ID or prefix."""
    engine = get_engine(ctx)
    result = engine.browse_entry(entry_id)
    output_json(result)


@browse.command("knowledge")
@click.pass_context
def browse_knowledge(ctx):
    """View all user_knowledge entries."""
    engine = get_engine(ctx)
    result = engine.browse_knowledge()
    output_json(result)
