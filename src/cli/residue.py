"""
Residue commands — session texture persistence.

Commands:
    solitaire residue write "text"
    solitaire residue latest
"""
import click

from ._engine import get_engine, output_json, output_error


@click.group()
@click.pass_context
def residue(ctx):
    """Session residue (rolling texture of the session arc)."""
    pass


@residue.command("write")
@click.argument("text")
@click.pass_context
def residue_write(ctx, text):
    """Write or overwrite the session residue."""
    if not text.strip():
        output_error("Residue text cannot be empty", exit_code=2)
        return
    engine = get_engine(ctx)
    result = engine.write_residue(text=text)
    output_json(result)


@residue.command("latest")
@click.pass_context
def residue_latest(ctx):
    """Get the most recent session residue."""
    engine = get_engine(ctx)
    result = engine.get_residue()
    output_json(result)
