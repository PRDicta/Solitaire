#!/usr/bin/env python3
"""UserPromptSubmit hook: auto-recall before every model response.

Fires when the user sends a message, before the model starts generating.
Runs auto-recall to retrieve relevant memory context and injects it
as additionalContext so the model sees recall results before composing.

This replaces the behavioral instruction that required the model to
manually call auto-recall before writing its persona label. Hooks have
near-perfect compliance; behavioral instructions do not.
"""

import json
import os
import subprocess
import sys

WORKSPACE = os.environ.get("SOLITAIRE_WORKSPACE", os.getcwd())


def main():
    # Read hook input from stdin
    try:
        hook_input = json.loads(sys.stdin.read())
    except Exception:
        sys.exit(0)

    # Extract user message from the prompt field
    user_message = hook_input.get("prompt", "").strip()

    # Skip trivial messages
    if not user_message or len(user_message) < 5:
        sys.exit(0)

    # Run auto-recall
    try:
        result = subprocess.run(
            ["solitaire", "auto-recall", user_message],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,  # Leave headroom within the 30s hook timeout
            cwd=WORKSPACE,
        )
        stdout = result.stdout.strip()
        if result.returncode != 0:
            from hook_errors import log_hook_error
            log_hook_error("auto-recall", f"exit {result.returncode}: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        try:
            from hook_errors import log_hook_error
            log_hook_error("auto-recall", "timed out after 25s")
        except Exception:
            pass
        sys.exit(0)
    except Exception as e:
        try:
            from hook_errors import log_hook_error
            log_hook_error("auto-recall", str(e)[:200])
        except Exception:
            pass
        sys.exit(0)

    if not stdout:
        sys.exit(0)

    # Parse the auto-recall JSON output
    recall_data = None
    for line in stdout.split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            candidate = json.loads(line)
            if "status" in candidate:
                recall_data = candidate
        except json.JSONDecodeError:
            pass

    # Fallback: try parsing multi-line JSON via brace tracking
    if not recall_data:
        lines = stdout.split("\n")
        brace_depth = 0
        json_start = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("{") and brace_depth == 0:
                json_start = i
                brace_depth = stripped.count("{") - stripped.count("}")
            elif json_start is not None:
                brace_depth += stripped.count("{") - stripped.count("}")
            if json_start is not None and brace_depth == 0:
                block = "\n".join(lines[json_start:i + 1])
                try:
                    candidate = json.loads(block)
                    if "status" in candidate:
                        recall_data = candidate
                except json.JSONDecodeError:
                    pass
                json_start = None

    if not recall_data:
        sys.exit(0)

    # Extract the context block
    context_block = recall_data.get("context_block", "").strip()
    preflight = recall_data.get("preflight", {})
    proactive_briefing = recall_data.get("proactive_briefing", {})

    # Build the injection context
    parts = []

    if context_block:
        parts.append("[AUTO-RECALL CONTEXT]")
        parts.append(context_block)

    if preflight and preflight.get("proceed") is False:
        parts.append("[PREFLIGHT GATE]")
        parts.append("proceed: false")
        if preflight.get("reason"):
            parts.append(f"reason: {preflight['reason']}")

    if proactive_briefing and proactive_briefing.get("surface"):
        parts.append("[PROACTIVE BRIEFING]")
        parts.append(json.dumps(proactive_briefing, indent=2))

    if not parts:
        # No recall context to inject — recall ran but found nothing relevant
        sys.exit(0)

    # Surface any recent hook errors as a visible cue
    try:
        from hook_errors import read_and_clear_latest
        hook_warning = read_and_clear_latest()
        if hook_warning:
            parts.insert(0, hook_warning)
            parts.insert(1, "")
    except Exception:
        pass

    context = "\n".join(parts)

    # Sanitize surrogate characters
    context = context.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    # Output hook JSON with additionalContext
    output = json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context
        }
    }, ensure_ascii=True)

    print(output)


if __name__ == "__main__":
    main()
