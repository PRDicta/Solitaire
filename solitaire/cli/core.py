"""
Core commands — the per-turn operations called by INSTRUCTIONS.md.

Commands:
    solitaire ingest-turn "user msg" "assistant msg"
    solitaire ingest-turn -              (read JSON from stdin)
    solitaire mark-response "assistant response"
    solitaire mark-response -            (read JSON from stdin)
    solitaire auto-recall "message"
    solitaire recall "query"
    solitaire remember "fact"
    solitaire end "summary"

These are also promoted to top-level in __init__.py so they work
without the 'core' subgroup prefix.
"""
import json
import sys

import click

from ._engine import get_engine, output_json, output_error


@click.group()
@click.pass_context
def core(ctx):
    """Core per-turn operations (ingest, recall, remember, end)."""
    pass


# ── Implementation functions (shared by subgroup and top-level aliases) ──


def _do_ingest_turn(ctx, user_msg, assistant_msg, from_stdin):
    """Shared implementation for ingest-turn."""
    if from_stdin or (user_msg == "-" and assistant_msg is None):
        # Stdin mode
        try:
            raw = sys.stdin.read()
            data = json.loads(raw)
            user_msg = data.get("user", "")
            assistant_msg = data.get("assistant", "")
        except (json.JSONDecodeError, KeyError) as e:
            output_error(
                f"Invalid JSON on stdin: {e}. Expected: "
                '{"user": "...", "assistant": "..."}',
                exit_code=2,
            )
            return

    if not user_msg or not assistant_msg:
        output_error(
            'Usage: solitaire ingest-turn "user msg" "assistant msg" '
            'OR echo \'{"user":"...","assistant":"..."}\' | solitaire ingest-turn -',
            exit_code=2,
        )
        return

    engine = get_engine(ctx)
    result = engine.ingest(user_msg=user_msg, assistant_msg=assistant_msg)
    output_json(result)


def _do_recall(ctx, query, include_preflight=True):
    """Shared implementation for recall / auto-recall."""
    engine = get_engine(ctx)
    result = engine.recall(query=query, include_preflight=include_preflight)
    output_json(result)


def _do_remember(ctx, fact, as_reference=False):
    """Shared implementation for remember."""
    engine = get_engine(ctx)
    result = engine.remember(fact=fact, as_reference=as_reference)
    output_json(result)


def _do_mark_response(ctx, response_text, from_stdin):
    """Shared implementation for mark-response."""
    if from_stdin or response_text == "-":
        try:
            raw = sys.stdin.read()
            data = json.loads(raw)
            response_text = data.get("response", "")
        except (json.JSONDecodeError, KeyError) as e:
            output_error(
                f"Invalid JSON on stdin: {e}. Expected: "
                '{"response": "..."}',
                exit_code=2,
            )
            return

    if not response_text:
        output_error(
            'Usage: solitaire mark-response "assistant response" '
            "OR echo '{\"response\":\"...\"}' | solitaire mark-response -",
            exit_code=2,
        )
        return

    engine = get_engine(ctx)
    result = engine.mark_response(response_text=response_text)
    output_json(result)


def _do_end(ctx, summary):
    """Shared implementation for end."""
    engine = get_engine(ctx)
    result = engine.end(summary=summary)
    output_json(result)


# ── Click commands within the 'core' subgroup ──


@core.command("ingest-turn")
@click.argument("user_msg", required=False)
@click.argument("assistant_msg", required=False)
@click.option("--stdin", "-", "from_stdin", is_flag=True,
              help="Read JSON from stdin")
@click.pass_context
def ingest_turn(ctx, user_msg, assistant_msg, from_stdin):
    """Ingest a user + assistant turn pair."""
    _do_ingest_turn(ctx, user_msg, assistant_msg, from_stdin)


@core.command("auto-recall")
@click.argument("message")
@click.pass_context
def auto_recall(ctx, message):
    """Preflight evaluation + recall."""
    _do_recall(ctx, message, include_preflight=True)


@core.command("recall")
@click.argument("query")
@click.option("--no-preflight", is_flag=True, help="Skip evaluation gate")
@click.pass_context
def recall(ctx, query, no_preflight):
    """Search memory for relevant context."""
    _do_recall(ctx, query, include_preflight=not no_preflight)


@core.command("remember")
@click.argument("fact")
@click.option("--reference", is_flag=True, help="Store as reference (quick link) instead of user_knowledge")
@click.pass_context
def remember(ctx, fact, reference):
    """Store a fact as privileged user_knowledge or reference."""
    _do_remember(ctx, fact, as_reference=reference)


@core.command("mark-response")
@click.argument("response_text", required=False, default="")
@click.option("--stdin", "-", "from_stdin", is_flag=True,
              help="Read JSON from stdin")
@click.pass_context
def mark_response(ctx, response_text, from_stdin):
    """Store assistant response for deferred ingestion."""
    _do_mark_response(ctx, response_text, from_stdin)


@core.command("diarize")
@click.argument("response_text", required=False, default="")
@click.argument("residue_text", required=False, default="")
@click.option("--stdin", "-", "from_stdin", is_flag=True,
              help="Read JSON from stdin")
@click.pass_context
def diarize(ctx, response_text, residue_text, from_stdin):
    """Combined closing-anchor procedure: mark response + write residue."""
    if from_stdin or response_text == "-":
        try:
            raw = sys.stdin.read()
            data = json.loads(raw)
            response_text = data.get("response", "")
            residue_text = data.get("residue", "")
        except (json.JSONDecodeError, KeyError) as e:
            output_error(
                f"Invalid JSON on stdin: {e}. Expected: "
                '{"response": "...", "residue": "..."}',
                exit_code=2,
            )
            return

    if not response_text and not residue_text:
        output_error(
            'Usage: solitaire diarize "response" "residue" '
            "OR echo '{\"response\":\"...\",\"residue\":\"...\"}' | solitaire diarize -",
            exit_code=2,
        )
        return

    engine = get_engine(ctx)
    result = engine.diarize(response_text=response_text, residue_text=residue_text)
    output_json(result)