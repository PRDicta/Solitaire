#!/usr/bin/env python3
"""
Backward compatibility shim.

Drop this file at the old `librarian_cli.py` path to redirect all
commands to the new Solitaire CLI. No behavior change for callers;
the same command-line arguments produce the same JSON output.

Usage (same as before):
    python librarian_cli.py boot --persona chief --intent "..."
    python librarian_cli.py ingest-turn "user msg" "assistant msg"
    python librarian_cli.py recall "query"
    python librarian_cli.py end "summary"

All commands are dispatched through SolitaireEngine via the click CLI.
"""
import os
import sys

# Ensure the solitaire package is importable.
# In the old layout, this shim sits at <workspace>/librarian/librarian_cli.py
# and solitaire/ is at <workspace>/solitaire/.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_workspace = os.path.dirname(_script_dir)

# Add workspace root to path so `import solitaire` works
if _workspace not in sys.path:
    sys.path.insert(0, _workspace)

# Set workspace env var for the engine
if "SOLITAIRE_WORKSPACE" not in os.environ:
    # The workspace dir that contains personas/, rolodex.db, etc.
    # In the old layout, that's the librarian/ directory itself
    for candidate in [_script_dir, _workspace]:
        if os.path.isdir(os.path.join(candidate, "personas")):
            os.environ["SOLITAIRE_WORKSPACE"] = candidate
            break
    else:
        os.environ["SOLITAIRE_WORKSPACE"] = _script_dir

from solitaire.compat import dispatch

if __name__ == "__main__":
    dispatch()
