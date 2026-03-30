#!/usr/bin/env python3
"""Claude Code Stop hook: scan assistant responses for AI writing tells.

Drop-in hook for Claude Code's Stop event. Reads the session transcript,
extracts the most recent assistant response, and scans for surface-level
and structural writing tells (em dashes, cursed word clusters, paragraph
uniformity, etc.).

If violations are detected, writes a JSON marker file that the preflight
evaluation gate picks up on the next turn to inject a "WRITING QUALITY"
block before the model generates its next response.

Follows the same architecture as claim-scanner.py:
  Stop hook fires → scan assistant output → write marker → gate reads marker

Setup: Add to .claude/settings.json alongside auto-ingest and claim-scanner:
{
  "hooks": {
    "Stop": [{
      "matcher": "",
      "hooks": [
        {
          "type": "command",
          "command": "python .claude/hooks/writing-gate.py",
          "timeout": 30
        }
      ]
    }]
  }
}

Disable via environment variable: SOLITAIRE_WRITING_GATE=0
"""

import json
import os
import sys


def extract_last_assistant(transcript_path):
    """Extract the last assistant response from transcript.

    Same logic as claim-scanner.py — proven pattern.
    """
    if not transcript_path or not os.path.isfile(transcript_path):
        return None

    last_assistant = None
    try:
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "assistant":
                    content = event.get("message", {}).get("content", "")
                    if isinstance(content, list):
                        parts = [
                            p.get("text", "")
                            for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        content = "\n".join(parts)
                    elif not isinstance(content, str):
                        content = ""
                    if content and len(content.strip()) > 0:
                        last_assistant = content.strip()
    except Exception:
        return None

    return last_assistant


def _extract_transcript_context(transcript_path):
    """Extract user message and prior assistant turns from transcript.

    Returns a TranscriptContext with:
    - user_text: the most recent user message
    - prior_assistant_text: the second-to-last assistant response
    - prior_turns_text: all prior assistant turns concatenated
    """
    from solitaire.outbound.config import TranscriptContext

    ctx = TranscriptContext()
    if not transcript_path or not os.path.isfile(transcript_path):
        return ctx

    assistant_turns = []
    last_user = None

    try:
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = event.get("type", "")
                content = event.get("message", {}).get("content", "")
                if isinstance(content, list):
                    parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = "\n".join(parts)
                elif not isinstance(content, str):
                    content = ""
                content = content.strip()

                if msg_type == "human" and content:
                    last_user = content
                elif msg_type == "assistant" and content:
                    assistant_turns.append(content)
    except Exception:
        return ctx

    if last_user:
        ctx.user_text = last_user

    # Prior assistant text is second-to-last (last is the current response)
    if len(assistant_turns) >= 2:
        ctx.prior_assistant_text = assistant_turns[-2]
        ctx.prior_turns_text = "\n\n".join(assistant_turns[:-1])
    elif len(assistant_turns) == 1:
        # Only one assistant turn (the current one), no prior
        ctx.prior_turns_text = ""

    return ctx


def main():
    # Kill switch
    if os.environ.get("SOLITAIRE_WRITING_GATE", "1").lower() in (
        "0", "false", "no", "off",
    ):
        sys.exit(0)

    try:
        hook_input = json.loads(sys.stdin.read())
    except Exception:
        sys.exit(0)

    transcript_path = hook_input.get("transcript_path", "")
    assistant_text = extract_last_assistant(transcript_path)

    if not assistant_text:
        sys.exit(0)

    # Add the librarian root to sys.path so we can import the outbound package
    workspace = os.environ.get("SOLITAIRE_WORKSPACE", os.getcwd())
    librarian_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if librarian_root not in sys.path:
        sys.path.insert(0, librarian_root)

    try:
        from solitaire.outbound.writing_gate import scan
        from solitaire.outbound.config import (
            load_config, load_persona_traits, TranscriptContext,
        )
        from solitaire.outbound.marker import write_marker
    except ImportError as e:
        # If the outbound package isn't available, exit silently
        sys.exit(0)

    # Load persona-specific config and traits
    persona_key = os.environ.get("LIBRARIAN_PERSONA", os.environ.get("SOLITAIRE_PERSONA", "chief"))
    config = load_config(persona_key, workspace)
    persona_traits = load_persona_traits(persona_key, workspace)

    if not config.enabled:
        sys.exit(0)

    # Extract transcript context for Layers 4-5
    transcript_ctx = _extract_transcript_context(transcript_path)

    # Run the scan
    result = scan(assistant_text, config, persona_traits, transcript_ctx)

    if result.has_violations():
        marker_data = result.to_marker_dict()
        write_marker(marker_data["violations"], persona_key, workspace)

    sys.exit(0)


if __name__ == "__main__":
    main()
