#!/usr/bin/env python3
"""Claude Code Stop hook: auto-ingest the last user+assistant exchange.

Drop-in hook for Claude Code's Stop event. Reads the session transcript,
extracts the most recent turn pair, and calls `solitaire ingest-turn`.

Setup: Add to .claude/settings.json:
{
  "hooks": {
    "Stop": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "python <path-to>/claude-code-auto-ingest.py",
        "timeout": 45
      }]
    }]
  }
}

Skips ingestion when:
- No transcript available
- Last exchange was a bare ack (< 20 chars assistant response)
- Last user message was tool-only (no human text)
- Dedup: same exchange already ingested this session
"""

import json
import os
import sys
import subprocess
import hashlib
import tempfile

# --- Configuration ---
# Set SOLITAIRE_WORKSPACE or edit this path
WORKSPACE = os.environ.get("SOLITAIRE_WORKSPACE", os.getcwd())
SOLITAIRE_CMD = os.environ.get("SOLITAIRE_CMD", "solitaire")

MARKER_DIR = os.path.join(tempfile.gettempdir(), "solitaire_ingest_markers")
os.makedirs(MARKER_DIR, exist_ok=True)


def extract_last_exchange(transcript_path):
    """Extract the last user message and assistant response from transcript."""
    if not transcript_path or not os.path.isfile(transcript_path):
        return None, None

    last_user = None
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

                msg_type = event.get("type", "")

                if msg_type == "user":
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
                        last_user = content.strip()

                elif msg_type == "assistant":
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
        return None, None

    return last_user, last_assistant


def already_ingested(user_msg, assistant_msg):
    """Dedup check via temp marker file."""
    key = hashlib.md5(f"{user_msg}|{assistant_msg}".encode()).hexdigest()
    marker = os.path.join(MARKER_DIR, key)
    if os.path.exists(marker):
        return True
    try:
        with open(marker, "w") as f:
            f.write("1")
    except Exception:
        pass
    return False


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except Exception:
        sys.exit(0)

    transcript_path = hook_input.get("transcript_path", "")
    user_msg, assistant_msg = extract_last_exchange(transcript_path)

    if not user_msg or not assistant_msg:
        sys.exit(0)

    if len(assistant_msg) < 20:
        sys.exit(0)

    if already_ingested(user_msg, assistant_msg):
        sys.exit(0)

    # Call solitaire ingest-turn via stdin
    ingest_payload = json.dumps({
        "user": user_msg,
        "assistant": assistant_msg
    })

    try:
        subprocess.run(
            [SOLITAIRE_CMD, "ingest-turn", "-"],
            input=ingest_payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            cwd=WORKSPACE,
        )
    except Exception:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
