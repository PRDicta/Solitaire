"""
Tool finding commands — proactive tool discovery and management.

Commands:
    solitaire tools find
    solitaire tools proposals
    solitaire tools approve <id>
    solitaire tools dismiss <id>
    solitaire tools install <id>
    solitaire tools record-use <name> <source>
    solitaire tools report
"""
import click

from ._engine import get_engine, output_json, output_error


@click.group()
@click.pass_context
def tools(ctx):
    """Proactive tool finding and management."""
    pass


@tools.command("find")
@click.option("--gap-window", default=14, help="Days to look back for gap signals")
@click.option("--min-occurrences", default=3, help="Minimum gap occurrences before searching")
@click.pass_context
def tools_find(ctx, gap_window, min_occurrences):
    """Run gap-to-search pipeline: detect gaps, search for tools, create proposals."""
    engine = get_engine(ctx)
    result = engine.find_tools(
        gap_window_days=gap_window,
        min_occurrences=min_occurrences,
    )
    output_json(result)


@tools.command("proposals")
@click.pass_context
def tools_proposals(ctx):
    """List pending tool proposals awaiting user decision."""
    engine = get_engine(ctx)
    result = engine.get_tool_proposals()
    output_json({"proposals": result})


@tools.command("approve")
@click.argument("proposal_id")
@click.pass_context
def tools_approve(ctx, proposal_id):
    """Approve a tool proposal."""
    engine = get_engine(ctx)
    result = engine.approve_tool(proposal_id)
    output_json(result)


@tools.command("dismiss")
@click.argument("proposal_id")
@click.option("--reason", default="", help="Reason for dismissal")
@click.pass_context
def tools_dismiss(ctx, proposal_id, reason):
    """Dismiss a tool proposal."""
    engine = get_engine(ctx)
    result = engine.dismiss_tool(proposal_id, reason=reason)
    output_json(result)


@tools.command("install")
@click.argument("proposal_id")
@click.pass_context
def tools_install(ctx, proposal_id):
    """Mark a tool as successfully installed."""
    engine = get_engine(ctx)
    result = engine.mark_tool_installed(proposal_id)
    output_json(result)


@tools.command("record-use")
@click.argument("tool_name")
@click.argument("tool_source")
@click.pass_context
def tools_record_use(ctx, tool_name, tool_source):
    """Record that an installed tool was used."""
    engine = get_engine(ctx)
    result = engine.record_tool_use(tool_name, tool_source)
    output_json(result)


@tools.command("report")
@click.pass_context
def tools_report(ctx):
    """Full tool report: proposals, installed, unused."""
    engine = get_engine(ctx)
    result = engine.get_tool_report()
    output_json(result)
