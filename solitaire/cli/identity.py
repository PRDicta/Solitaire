"""
Identity commands — identity graph inspection.

Commands:
    solitaire identity show
    solitaire identity stats
"""
import click

from ._engine import get_engine, output_json, output_error


@click.group()
@click.pass_context
def identity(ctx):
    """Identity graph inspection."""
    pass


@identity.command("show")
@click.option("--budget", default=1500, help="Token budget for context block")
@click.pass_context
def identity_show(ctx, budget):
    """Show the current identity context block."""
    engine = get_engine(ctx)

    try:
        from ..storage.identity_graph import IdentityGraph
        ig = IdentityGraph(engine._lib.rolodex.conn)
        block = ig.build_identity_context_block(budget_tokens=budget)
        output_json({
            "status": "ok",
            "identity_block": block or "",
        })
    except Exception as e:
        output_error(f"Failed to build identity context: {e}")


@identity.command("stats")
@click.pass_context
def identity_stats(ctx):
    """Show identity graph statistics."""
    engine = get_engine(ctx)

    try:
        from ..storage.identity_graph import IdentityGraph
        ig = IdentityGraph(engine._lib.rolodex.conn)
        stats = ig.get_stats() if hasattr(ig, 'get_stats') else {}
        output_json({
            "status": "ok",
            "stats": stats,
        })
    except Exception as e:
        output_error(f"Failed to get identity stats: {e}")
