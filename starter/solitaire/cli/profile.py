"""
Profile management commands.

Commands:
    solitaire profile set <key> <value>
    solitaire profile show
    solitaire profile delete <key>
"""
import click

from ._engine import get_engine, output_json, output_error


@click.group()
@click.pass_context
def profile(ctx):
    """User profile management (key-value preferences)."""
    pass


@profile.command("set")
@click.argument("key")
@click.argument("value", nargs=-1, required=True)
@click.pass_context
def profile_set(ctx, key, value):
    """Set a user profile key-value pair."""
    val_str = " ".join(value)
    engine = get_engine(ctx)
    result = engine.profile_set(key, val_str)
    output_json(result)


@profile.command("show")
@click.pass_context
def profile_show(ctx):
    """Show all user profile entries."""
    engine = get_engine(ctx)
    result = engine.profile_show()
    output_json(result)


@profile.command("delete")
@click.argument("key")
@click.pass_context
def profile_delete(ctx, key):
    """Delete a user profile entry."""
    engine = get_engine(ctx)
    result = engine.profile_delete(key)
    output_json(result)
