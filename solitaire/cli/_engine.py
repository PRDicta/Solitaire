"""
Shared engine construction for CLI commands.

All CLI commands need a SolitaireEngine instance. This module handles
lazy construction, caching, and auto-resume from session state so the
engine works correctly across separate subprocess invocations.
"""
import json
import os
import sys

from ..engine import SolitaireEngine


_engine_instance = None


def get_engine(ctx, auto_resume: bool = True) -> SolitaireEngine:
    """Get or create the SolitaireEngine for this CLI invocation.

    If a .solitaire_session file exists in the workspace and auto_resume
    is True, the engine will automatically boot with --resume so that
    commands like remember, ingest-turn, and recall work in fresh
    subprocess invocations (the standard CLI usage pattern).
    """
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance

    workspace = ctx.obj.get("workspace", os.getcwd())

    # Persona dir: check for personas/ subdirectory
    persona_dir = os.path.join(workspace, "personas")
    if not os.path.isdir(persona_dir):
        persona_dir = None

    engine = SolitaireEngine(
        workspace_dir=workspace,
        persona_dir=persona_dir,
    )

    # Auto-resume from session state if not already booted.
    # This makes subprocess-per-command usage work: boot writes
    # .solitaire_session, subsequent commands auto-resume from it.
    if auto_resume and not engine._booted:
        session_file = os.path.join(workspace, ".solitaire_session")
        if os.path.exists(session_file):
            try:
                with open(session_file) as f:
                    session_data = json.load(f)
                persona_key = session_data.get("persona_key")
                if persona_key:
                    engine.boot(persona_key=persona_key, resume=True, cold=True)
            except Exception:
                pass  # Non-fatal: caller will get an unbooted engine

    _engine_instance = engine
    return _engine_instance


def output_json(data: dict) -> None:
    """Print result JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


def output_error(msg: str, exit_code: int = 1) -> None:
    """Print error JSON to stderr and exit.

    Errors go to stderr so that automation parsing stdout for success
    JSON does not see error payloads on the same stream.
    """
    print(json.dumps({"error": msg}), file=sys.stderr)
    sys.exit(exit_code)


def housekeeping(data) -> None:
    """Print housekeeping/diagnostic info to stderr."""
    if isinstance(data, dict):
        print(json.dumps(data), file=sys.stderr)
    else:
        print(str(data), file=sys.stderr)