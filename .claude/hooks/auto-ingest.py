#!/usr/bin/env python3
"""Stop hook: auto-ingest the last user+assistant exchange after every response.

Reads the Claude Code transcript to extract the most recent turn pair,
then calls `solitaire ingest-turn` via stdin JSON.

Skips ingestion when:
- No transcript available
- Last exchange was a bare ack (< 20 chars assistant response)
- Last user message was a tool-only turn (no human text)
- Ingest already ran for this turn (dedup via marker file)
"""

import json
import os
import sys
import subprocess
import hashlib
import tempfile

WORKSPACE = os.environ.get("SOLITAIRE_WORKSPACE", os.getcwd())
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
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                        content = "\n".join(text_parts)
                    elif not isinstance(content, str):
                        content = ""
                    if content and len(content.strip()) > 0:
                        last_user = content.strip()

                elif msg_type == "assistant":
                    content = event.get("message", {}).get("content", "")
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                        content = "\n".join(text_parts)
                    elif not isinstance(content, str):
                        content = ""
                    if content and len(content.strip()) > 0:
                        last_assistant = content.strip()

    except Exception:
        return None, None

    return last_user, last_assistant


def already_ingested(user_msg, assistant_msg):
    """Check if this exact exchange was already ingested (dedup)."""
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

    # Call ingest-turn via stdin JSON
    ingest_payload = json.dumps({
        "user": user_msg,
        "assistant": assistant_msg
    })

    try:
        subprocess.run(
            ["solitaire", "ingest-turn", "-"],
            input=ingest_payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            cwd=WORKSPACE,
        )
    except Exception:
        pass  # Don't block the session on ingest failure

    sys.exit(0)


if __name__ == "__main__":
    main()
