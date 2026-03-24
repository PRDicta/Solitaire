"""
Single-message ingest and correction commands.

Commands:
    solitaire ingest user "message"
    solitaire ingest assistant "message"
    solitaire ingest user "message" --user-knowledge
    solitaire ingest user "message" --corrects <entry_id>
    solitaire correct <old_id> "corrected text"
"""
import click

from ._engine import get_engine, output_json, output_error


@click.group()
@click.pass_context
def ingest(ctx):
    """Ingest a single message (user or assistant)."""
    pass


@ingest.command("user")
@click.argument("content")
@click.option("--user-knowledge", is_flag=True, help="Mark as privileged user_knowledge")
@click.option("--corrects", "corrects_id", default=None, help="Supersede this entry ID")
@click.pass_context
def ingest_user(ctx, content, user_knowledge, corrects_id):
    """Ingest a user message."""
    engine = get_engine(ctx)
    result = engine.ingest_single(
        role="user",
        content=content,
        as_user_knowledge=user_knowledge,
        corrects_id=corrects_id,
    )
    output_json(result)


@ingest.command("assistant")
@click.argument("content")
@click.option("--corrects", "corrects_id", default=None, help="Supersede this entry ID")
@click.pass_context
def ingest_assistant(ctx, content, corrects_id):
    """Ingest an assistant message."""
    engine = get_engine(ctx)
    result = engine.ingest_single(
        role="assistant",
        content=content,
        corrects_id=corrects_id,
    )
    output_json(result)
