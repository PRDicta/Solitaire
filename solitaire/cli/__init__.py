"""
Solitaire CLI -- Click-based command interface.

Thin wrappers around SolitaireEngine. All business logic lives in the engine;
the CLI handles argument parsing, JSON output formatting, and exit codes.

Exit codes:
    0 - Success
    1 - Internal error
    2 - User error (bad arguments, missing required input)
"""
import json
import os
import sys

import click

from .boot import boot
from .core import core
from .persona import persona
from .residue import residue
from .identity import identity
from .analytics import analytics
from .tools import tools
from .maintenance import maintenance
from .ingest import ingest
from .profile import profile
from .browse import browse
from .harvest import integrity, hindsight_backfill
from .onboard import onboard
from .skills import load_skill


def _json_output(data: dict, file=None) -> None:
    """Print JSON to stdout. Housekeeping goes to stderr."""
    click.echo(json.dumps(data, indent=2, default=str), file=file or sys.stdout)


def _housekeeping(msg: str) -> None:
    """Print housekeeping message to stderr as JSON."""
    click.echo(json.dumps(msg) if isinstance(msg, dict) else msg, err=True)


def _get_workspace() -> str:
    """Resolve workspace directory.

    Priority:
    1. SOLITAIRE_WORKSPACE env var
    2. Current working directory
    """
    return os.environ.get("SOLITAIRE_WORKSPACE", os.getcwd())


@click.group()
@click.version_option(package_name="solitaire")
@click.pass_context
def cli(ctx):
    """Solitaire -- Persistent memory and evolving identity for AI agents."""
    ctx.ensure_object(dict)
    ctx.obj["workspace"] = _get_workspace()


# ── Register command groups ──────────────────────────────────────────────

cli.add_command(boot)
cli.add_command(core)
cli.add_command(persona)
cli.add_command(residue)
cli.add_command(identity)
cli.add_command(analytics)
cli.add_command(tools)
cli.add_command(maintenance)
cli.add_command(ingest)
cli.add_command(profile)
cli.add_command(browse)
cli.add_command(integrity)
cli.add_command(onboard)
cli.add_command(load_skill)
cli.add_command(hindsight_backfill)


# ── Top-level aliases for frequently-used commands ───────────────────────
# These are the commands called every turn by INSTRUCTIONS.md, so they
# need to be accessible without a subgroup prefix:
#   solitaire ingest-turn "..." "..."
#   solitaire mark-response "..."
#   solitaire auto-recall "..."
#   solitaire recall "..."
#   solitaire remember "..."
#   solitaire end "..."
#   solitaire pulse
#   solitaire auto-evaluate "..."
#   solitaire correct <old_id> "text"
#   solitaire harvest
#   solitaire harvest-full
#   solitaire harvest-status
#   solitaire build-chains
#   solitaire turn-pairs
#   solitaire decision-journal
#   solitaire reflect
#   solitaire hindsight-backfill


@cli.command("ingest-turn")
@click.argument("user_msg", required=False)
@click.argument("assistant_msg", required=False)
@click.option("--stdin", "-", "from_stdin", is_flag=True,
              help='Read JSON from stdin: {"user": "...", "assistant": "..."}')
@click.pass_context
def ingest_turn_top(ctx, user_msg, assistant_msg, from_stdin):
    """Ingest a user + assistant turn pair."""
    from .core import _do_ingest_turn
    _do_ingest_turn(ctx, user_msg, assistant_msg, from_stdin)


@cli.command("mark-response")
@click.argument("response_text", required=False, default="")
@click.option("--stdin", "-", "from_stdin", is_flag=True,
              help='Read JSON from stdin: {"response": "..."}')
@click.pass_context
def mark_response_top(ctx, response_text, from_stdin):
    """Store assistant response for deferred ingestion."""
    from .core import _do_mark_response
    _do_mark_response(ctx, response_text, from_stdin)


@cli.command("auto-recall")
@click.argument("message")
@click.pass_context
def auto_recall_top(ctx, message):
    """Preflight evaluation + recall (run before composing each response)."""
    from .core import _do_recall
    _do_recall(ctx, message, include_preflight=True)


@cli.command("recall")
@click.argument("query")
@click.option("--no-preflight", is_flag=True, help="Skip evaluation gate")
@click.pass_context
def recall_top(ctx, query, no_preflight):
    """Search memory for relevant context."""
    from .core import _do_recall
    _do_recall(ctx, query, include_preflight=not no_preflight)


@cli.command("remember")
@click.argument("fact")
@click.pass_context
def remember_top(ctx, fact):
    """Store a fact as privileged user_knowledge."""
    from .core import _do_remember
    _do_remember(ctx, fact)


@cli.command("end")
@click.argument("summary", default="")
@click.pass_context
def end_top(ctx, summary):
    """End the current session."""
    from .core import _do_end
    _do_end(ctx, summary)


@cli.command("pulse")
@click.pass_context
def pulse_top(ctx):
    """Heartbeat check: is the engine alive?"""
    from ._engine import get_engine, output_json
    engine = get_engine(ctx)
    result = engine.pulse()
    output_json(result)


@cli.command("auto-evaluate")
@click.argument("message")
@click.pass_context
def auto_evaluate_top(ctx, message):
    """Standalone evaluation gate (no recall)."""
    from ._engine import get_engine, output_json
    engine = get_engine(ctx)
    result = engine.auto_evaluate(message)
    output_json(result)


@cli.command("correct")
@click.argument("old_entry_id")
@click.argument("corrected_text")
@click.pass_context
def correct_top(ctx, old_entry_id, corrected_text):
    """Supersede a wrong entry with corrected content."""
    from ._engine import get_engine, output_json
    engine = get_engine(ctx)
    result = engine.correct(old_entry_id, corrected_text)
    output_json(result)


@cli.command("harvest")
@click.option("--force-all", is_flag=True, help="Re-process all logs")
@click.option("--dry-run", is_flag=True, help="Show what would be processed")
@click.pass_context
def harvest_top(ctx, force_all, dry_run):
    """Run the conversation harvest (safety net ingestion)."""
    from ._engine import get_engine, output_json
    engine = get_engine(ctx)
    result = engine.harvest(force_all=force_all, dry_run=dry_run)
    output_json(result)


@cli.command("harvest-full")
@click.pass_context
def harvest_full_top(ctx):
    """Full pipeline: harvest + build-chains + integrity check."""
    from ._engine import get_engine, output_json
    engine = get_engine(ctx)
    result = engine.harvest_full()
    output_json(result)


@cli.command("harvest-status")
@click.pass_context
def harvest_status_top(ctx):
    """Show harvest progress without running a harvest."""
    from ._engine import get_engine, output_json
    engine = get_engine(ctx)
    result = engine.harvest_status()
    output_json(result)


@cli.command("build-chains")
@click.option("--session", "session_id", default=None, help="Target session ID")
@click.option("--force", is_flag=True, help="Build even for short segments")
@click.pass_context
def build_chains_top(ctx, session_id, force):
    """Build narrative reasoning chains for sessions."""
    from ._engine import get_engine, output_json
    engine = get_engine(ctx)
    result = engine.build_chains(session_id=session_id, force=force)
    output_json(result)


@cli.command("turn-pairs")
@click.option("--session", "session_id", default=None, help="Target session ID")
@click.option("--limit", default=10, help="Number of recent sessions")
@click.pass_context
def turn_pairs_top(ctx, session_id, limit):
    """Ingest user+assistant turn pairs as atomic units."""
    from ._engine import get_engine, output_json
    engine = get_engine(ctx)
    result = engine.turn_pairs(session_id=session_id, limit=limit)
    output_json(result)


@cli.command("decision-journal")
@click.option("--session", "session_id", default=None, help="Target session ID")
@click.option("--limit", default=10, help="Number of recent sessions")
@click.pass_context
def decision_journal_top(ctx, session_id, limit):
    """Extract decisions as first-class entities."""
    from ._engine import get_engine, output_json
    engine = get_engine(ctx)
    result = engine.decision_journal(session_id=session_id, limit=limit)
    output_json(result)


@cli.command("reflect")
@click.option("--force", is_flag=True, help="Override cooldown timer")
@click.pass_context
def reflect_top(ctx, force):
    """Run session reflection (skill usage analysis)."""
    from ._engine import get_engine, output_json
    engine = get_engine(ctx)
    result = engine.reflect(force=force)
    output_json(result)


@cli.command("patterns")
@click.option("--window", default=5, help="Number of recent sessions to analyze")
@click.pass_context
def patterns_top(ctx, window):
    """Generate retrieval pattern report."""
    from ._engine import get_engine, output_json
    engine = get_engine(ctx)
    result = engine.retrieval_patterns(window=window)
    output_json(result)