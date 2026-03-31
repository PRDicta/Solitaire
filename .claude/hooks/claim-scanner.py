#!/usr/bin/env python3
"""Stop hook: scan assistant responses for unverified state claims.

Reads the session transcript, extracts the most recent assistant response,
and scans for unverified assertions about remote/unobserved system state.

If claims are detected, writes a JSON marker file that the preflight
evaluation gate picks up on the next turn.

Disable via environment variable: SOLITAIRE_CLAIM_SCANNER=0
"""

import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone

WORKSPACE = os.environ.get("SOLITAIRE_WORKSPACE", os.getcwd())
MARKER_DIR = os.path.join(tempfile.gettempdir(), "solitaire_claim_markers")

# --- Patterns: what the assistant says that indicates unverified claims ---

# Definitive state claims about unobserved systems
_STATE_CLAIMS = [
    re.compile(
        r"\b(?:the|her|his|their)\s+\w+\s+"
        r"(?:isn't|is\s+not|aren't|are\s+not)\s+"
        r"(?:running|installed|working|responding|active|started|configured|connected)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:her|his|their|the)\s+\w+\s+"
        r"(?:is|are)\s+"
        r"(?:broken|corrupt|missing|outdated|misconfigured|incompatible)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+\w+\s+)?(?:never|didn't|did\s+not|hasn't|has\s+not)\s+"
        r"(?:applied|installed|updated|completed|taken\s+effect|propagated)",
        re.IGNORECASE,
    ),
]

# Recommendations for file ops on unverified targets
_FILE_OP_RECS = [
    re.compile(
        r"\b(?:you\s+should|let's|we\s+(?:should|can|need\s+to)|I'll|try\s+(?:to\s+)?)"
        r"\s*(?:delete|remove|move|rename|overwrite|replace|copy)\b"
        r".{0,60}"
        r"\b(?:on|from|at)\s+(?:her|his|their|that|the\s+other|the\s+remote)",
        re.IGNORECASE,
    ),
]

# Narrative construction markers (certainty about unobserved state)
_NARRATIVE = [
    re.compile(
        r"\b(?:what\s+happened\s+(?:is|was)|"
        r"the\s+problem\s+is\s+that|"
        r"the\s+issue\s+is\s+that|"
        r"basically\s+what'?s?\s+going\s+on)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:clearly|obviously|definitely|certainly)\s+"
        r"(?:the|her|his|their|it)\b",
        re.IGNORECASE,
    ),
]

# Verification signals (suppress false positives)
_VERIFICATION = [
    re.compile(r"\bI\s+(?:checked|verified|confirmed|ran|tested)\b", re.IGNORECASE),
    re.compile(r"\b(?:output|log|result)\s+(?:shows|says|reads|confirms)\b", re.IGNORECASE),
    re.compile(r"\bI\s+can't\s+verify\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+(?:checking|verifying)\b", re.IGNORECASE),
    re.compile(r"```[\s\S]{10,}```"),  # Code block output (likely command results)
]

# Remote system context (only flag claims that involve remote/unobserved systems)
_REMOTE_CONTEXT = [
    re.compile(
        r"\b(?:her|his|their)\s+"
        r"(?:machine|computer|laptop|desktop|server|system|setup|"
        r"installation|install|folder|directory)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:remote|ssh|rdp)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:another|different|separate|other)\s+"
        r"(?:machine|computer|server|system|environment|instance)",
        re.IGNORECASE,
    ),
]


def extract_last_assistant(transcript_path):
    """Extract the last assistant response from transcript."""
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


def has_remote_context(text):
    return any(p.search(text) for p in _REMOTE_CONTEXT)


def has_verification(text):
    return any(p.search(text) for p in _VERIFICATION)


def scan_for_claims(text):
    if not text or len(text) < 20:
        return []
    if not has_remote_context(text):
        return []
    if has_verification(text):
        return []

    claims = []
    for pattern in _STATE_CLAIMS:
        m = pattern.search(text)
        if m:
            claims.append({"category": "state_claim", "text": m.group(0)[:100]})
    for pattern in _FILE_OP_RECS:
        m = pattern.search(text)
        if m:
            claims.append({"category": "file_op_recommendation", "text": m.group(0)[:100]})
    for pattern in _NARRATIVE:
        m = pattern.search(text)
        if m:
            claims.append({"category": "narrative_construction", "text": m.group(0)[:100]})
    return claims


def write_marker(claims, workspace=None):
    ws = workspace or WORKSPACE
    try:
        os.makedirs(MARKER_DIR, exist_ok=True)
        ws_hash = hashlib.md5(ws.encode()).hexdigest()[:12]
        marker_path = os.path.join(MARKER_DIR, ws_hash)

        cats = list({c["category"] for c in claims})
        texts = [c["text"] for c in claims[:3]]
        summary = f"{len(claims)} unverified claim(s): {'; '.join(texts)}"

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "claims_detected": claims,
            "summary": summary,
            "categories": cats,
        }
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def main():
    if os.environ.get("SOLITAIRE_CLAIM_SCANNER", "1").lower() in (
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

    try:
        claims = scan_for_claims(assistant_text)
        if claims:
            write_marker(claims)
    except Exception as e:
        try:
            from hook_errors import log_hook_error
            log_hook_error("claim-scanner", str(e)[:200])
        except Exception:
            pass

    sys.exit(0)


if __name__ == "__main__":
    main()
