"""
Skill pack commands -- Tier 2 indexed knowledge operations.

Commands:
    solitaire load-skill list
    solitaire load-skill auto "message"
    solitaire load-skill load <pack-name>
"""
import click

from ._engine import get_engine, output_json, output_error


@click.group(name="load-skill")
@click.pass_context
def load_skill(ctx):
    """Tier 2 indexed knowledge pack operations."""
    pass


@load_skill.command("list")
@click.pass_context
def skill_list(ctx):
    """List available indexed packs for the active persona."""
    engine = get_engine(ctx)
    result = engine.load_skill_list()
    output_json(result)


@load_skill.command("auto")
@click.argument("message")
@click.pass_context
def skill_auto(ctx, message):
    """Auto-detect and load packs matching keywords in a message."""
    engine = get_engine(ctx)
    result = engine.load_skill_auto(message)
    output_json(result)


@load_skill.command("load")
@click.argument("pack_name")
@click.pass_context
def skill_load(ctx, pack_name):
    """Load a specific skill pack by name."""
    engine = get_engine(ctx)
    result = engine.load_skill_load(pack_name)
    output_json(result)
