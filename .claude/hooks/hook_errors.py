"""Shared hook error logging.

All Solitaire hooks import this module to log failures to a known location.
Errors are written to .solitaire/hook-errors.log in the workspace directory.
The auto-recall hook reads the latest error to surface a visible cue.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = os.environ.get("SOLITAIRE_WORKSPACE", os.getcwd())
LOG_DIR = os.path.join(WORKSPACE, ".solitaire")
LOG_FILE = os.path.join(LOG_DIR, "hook-errors.log")
LATEST_FILE = os.path.join(LOG_DIR, "hook-error-latest.json")


def log_hook_error(hook_name: str, error: str) -> None:
    """Append a timestamped error to the hook error log.

    Also writes the latest error to a separate JSON file so
    auto-recall can surface it as a visible cue.
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Append to log
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {hook_name}: {error}\n")

        # Write latest for auto-recall pickup
        latest = {
            "hook": hook_name,
            "error": error[:200],
            "timestamp": ts,
        }
        with open(LATEST_FILE, "w", encoding="utf-8") as f:
            json.dump(latest, f)
    except Exception:
        pass  # Logging failures must never block


def read_and_clear_latest() -> str:
    """Read the latest hook error and clear it.

    Returns a formatted warning string, or empty string if no error.
    Called by auto-recall to inject visible cue.
    """
    try:
        if not os.path.isfile(LATEST_FILE):
            return ""
        with open(LATEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Clear after reading
        os.unlink(LATEST_FILE)
        hook = data.get("hook", "unknown")
        error = data.get("error", "unknown error")
        ts = data.get("timestamp", "")
        return f"[HOOK WARNING: {hook} failed at {ts} - {error}. See .solitaire/hook-errors.log]"
    except Exception:
        return ""
