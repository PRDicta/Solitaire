#!/usr/bin/env python3
"""SessionStart hook: auto-boot Solitaire and inject full context.

Runs before the model sees the first message.
Reads SOLITAIRE_PERSONA env var (set during onboarding) to determine
which persona to boot. Falls back to the first available persona.
"""

import json
import os
import subprocess
import sys

WORKSPACE = os.environ.get("SOLITAIRE_WORKSPACE", os.getcwd())
PERSONA = os.environ.get("SOLITAIRE_PERSONA", "")

# Build boot command
cmd = ["solitaire", "boot"]
if PERSONA:
    cmd += ["--persona", PERSONA]
else:
    cmd += ["--pre-persona"]

try:
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45,
        cwd=WORKSPACE,
    )
    stdout = result.stdout
    if result.returncode != 0:
        try:
            from hook_errors import log_hook_error
            log_hook_error("session-boot", f"exit {result.returncode}: {result.stderr[:200]}")
        except Exception:
            pass
except Exception as e:
    try:
        from hook_errors import log_hook_error
        log_hook_error("session-boot", str(e)[:200])
    except Exception:
        pass
    sys.exit(0)

# If no persona was set, try to extract the first available and boot it
if not PERSONA:
    try:
        pre = json.loads(stdout.strip().split("\n")[-1])
        personas = pre.get("available_personas", [])
        if personas:
            PERSONA = personas[0].get("key", "")
            if PERSONA:
                result = subprocess.run(
                    ["solitaire", "boot", "--persona", PERSONA],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=45, cwd=WORKSPACE,
                )
                stdout = result.stdout
            else:
                sys.exit(0)
        else:
            # No personas exist yet; let CLAUDE.md onboarding handle it
            sys.exit(0)
    except Exception:
        sys.exit(0)

# Parse the boot JSON (multi-line pretty-printed output)
boot_data = None
lines = stdout.strip().split("\n")
brace_depth = 0
json_start = None

for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == "{" and brace_depth == 0:
        json_start = i
        brace_depth = 1
    elif json_start is not None:
        brace_depth += stripped.count("{") - stripped.count("}")
        if brace_depth == 0:
            block = "\n".join(lines[json_start:i + 1])
            try:
                candidate = json.loads(block)
                if "boot_files" in candidate:
                    boot_data = candidate
            except json.JSONDecodeError:
                pass
            json_start = None

if not boot_data:
    sys.exit(0)

# Read tier files
boot_files = boot_data.get("boot_files", {})
partner_name = boot_data.get("active_persona", {}).get("name", PERSONA)


def read_file(path):
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


context_file = boot_files.get("context", boot_files.get("t1", ""))
ops_file = boot_files.get("operations", "")
t2_file = boot_files.get("t2", "")
auto_closed = boot_data.get("auto_closed")

context_content = read_file(context_file)
ops_content = read_file(ops_file)
t2_content = read_file(t2_file)

# Build context block
parts = ["[SOLITAIRE AUTO-BOOT COMPLETE]", ""]
parts.append(f"{partner_name} is online. All tiers loaded.")
parts.append("")

if context_content:
    parts.append("--- TIER 1 (persona, direction, residue, briefing, facts) ---")
    parts.append(context_content)
    parts.append("")

# If auto-close fired, inject a message between context and T2
if auto_closed and auto_closed.get("status") != "skipped":
    prior_sid = auto_closed.get("prior_session_id", "unknown")[:8]
    parts.append("--- PRIOR SESSION CLOSED ---")
    parts.append(f"Automatically closed prior session {prior_sid}. Residue updated from partial to final.")
    parts.append("")

if t2_content:
    parts.append("--- TIER 2 (identity, commitments, experiential, user knowledge, resident knowledge) ---")
    parts.append(t2_content)
    parts.append("")

if ops_content:
    parts.append("--- OPERATIONS (session rules, behavioral instructions) ---")
    parts.append(ops_content)
    parts.append("")

parts.append(f"[END AUTO-BOOT] Respond to the user's first message. Open with [{partner_name}] on its own line (persona label). Auto-recall and ingestion are handled by hooks; do not call them manually.")

context = "\n".join(parts)

# Sanitize surrogate characters
context = context.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

# Output hook JSON
output = json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": context
    }
}, ensure_ascii=True)

print(output)
