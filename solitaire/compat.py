"""
Backward compatibility shim for librarian_cli.py.

Maps the old `python librarian_cli.py <command> <args>` invocation pattern
to SolitaireEngine method calls. Exists so that existing instruction document
references continue to work without modification.

Usage:
    # As a drop-in replacement for librarian_cli.py:
    python -m solitaire.compat boot --persona default --intent "working on X"
    python -m solitaire.compat ingest-turn "user msg" "assistant msg"
    python -m solitaire.compat recall "query"

    # Or import and use the dispatch function:
    from solitaire.compat import dispatch
    dispatch(["boot", "--persona", "default"])
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Optional


def _resolve_workspace() -> str:
    """Resolve workspace directory using the same logic as the old CLI."""
    # Check SOLITAIRE_WORKSPACE first
    ws = os.environ.get("SOLITAIRE_WORKSPACE")
    if ws:
        return ws

    # Check if we're being called from a known workspace layout
    # Old pattern: librarian_cli.py lives in <workspace>/librarian/
    script_dir = Path(__file__).resolve().parent
    # If solitaire/ is inside a workspace that has a personas/ dir
    for candidate in [script_dir.parent, script_dir.parent.parent]:
        if (candidate / "personas").is_dir():
            return str(candidate)

    # Fall back to cwd
    return os.getcwd()


def _print_json(data: dict) -> None:
    """Print JSON to stdout (matching old CLI behavior)."""
    print(json.dumps(data, indent=2, default=str))


def _print_housekeeping(data: dict) -> None:
    """Print housekeeping/diagnostic info to stderr (matching old CLI behavior)."""
    print(json.dumps(data), file=sys.stderr)


def dispatch(argv: Optional[List[str]] = None) -> None:
    """
    Dispatch a command using the old CLI argument format.

    Args:
        argv: Command arguments (without the script name).
              Defaults to sys.argv[1:].
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _print_json({
            "error": "Usage: solitaire <command> [args]",
            "commands": [
                "boot", "ingest", "ingest-turn", "recall", "auto-recall",
                "remember", "correct", "end", "pulse", "auto-evaluate",
                "profile", "browse", "persona", "residue", "identity",
                "maintain", "harvest", "harvest-full", "harvest-status",
                "integrity-check", "integrity-repair", "build-chains",
                "turn-pairs", "decision-journal", "onboard", "load-skill",
                "reflect", "patterns",
            ],
        })
        sys.exit(1)

    cmd = argv[0].lower()
    args = argv[1:]

    # Route to the solitaire click CLI
    # This gives us the cleanest path: all commands go through the same
    # click-based interface, with its argument parsing and help text.
    from .cli import cli

    # Reconstruct argv for click
    # Click expects [program_name, command, ...args]
    click_argv = [cmd] + args

    try:
        cli(click_argv, standalone_mode=False)
    except SystemExit as e:
        sys.exit(e.code if e.code else 0)
    except Exception as e:
        _print_json({"error": str(e)})
        sys.exit(1)


def main():
    """Entry point for backward compatibility."""
    # Set workspace from script location if not already set
    if "SOLITAIRE_WORKSPACE" not in os.environ:
        script_dir = Path(sys.argv[0]).resolve().parent if sys.argv else Path.cwd()
        # Old layout: librarian_cli.py is in <workspace>/librarian/
        # New layout: solitaire/ is at <workspace>/solitaire/
        for candidate in [script_dir, script_dir.parent]:
            if (candidate / "personas").is_dir():
                os.environ["SOLITAIRE_WORKSPACE"] = str(candidate)
                break

    from solitaire.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()