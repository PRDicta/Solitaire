"""
Analytics commands — retrieval patterns and system stats.

Commands:
    solitaire analytics patterns
    solitaire analytics stats
    solitaire analytics retrieval-stats
"""
import click

from ._engine import get_engine, output_json, output_error


@click.group()
@click.pass_context
def analytics(ctx):
    """Retrieval analytics and system statistics."""
    pass


@analytics.command("patterns")
@click.option("--window", default=5, help="Number of recent sessions to analyze")
@click.option("--stale-days", default=30, help="Days without recall before flagging dead zone")
@click.option("--gap-window", default=14, help="Days to look back for gap signals")
@click.pass_context
def patterns(ctx, window, stale_days, gap_window):
    """Generate retrieval pattern report (hot topics, dead zones, gaps)."""
    engine = get_engine(ctx)
    result = engine.get_patterns(
        window_sessions=window,
        stale_days=stale_days,
        gap_window_days=gap_window,
    )
    output_json(result)


@analytics.command("stats")
@click.pass_context
def stats(ctx):
    """Show system statistics."""
    engine = get_engine(ctx)
    result = engine.get_stats()
    output_json(result)


@analytics.command("retrieval-stats")
@click.option("--session", "session_id", default=None, help="Scope to specific session")
@click.pass_context
def retrieval_stats(ctx, session_id):
    """Show retrieval outcome statistics (use rate, top used/ignored)."""
    engine = get_engine(ctx)
    result = engine.get_retrieval_stats(session_id=session_id)
    output_json(result)
