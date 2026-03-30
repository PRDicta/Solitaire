"""Marker file read/write for the outbound writing quality gate.

Follows the same pattern as claim-scanner.py marker files:
- Write: Stop hook writes marker after scanning assistant output
- Read: Evaluation gate reads and consumes marker on next turn
- Location: /tmp/solitaire_writing_markers/{workspace_hash}
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional


MARKER_DIR = os.path.join(tempfile.gettempdir(), "solitaire_writing_markers")


def write_marker(violations: list, persona_key: str, workspace: Optional[str] = None) -> None:
    """Write a writing quality marker for the evaluation gate to pick up.

    Args:
        violations: List of dicts with layer, category, severity, count, samples, detail, score.
        persona_key: Active persona identifier.
        workspace: Workspace directory for hashing. Defaults to CWD.
    """
    ws = workspace or os.getcwd()
    try:
        os.makedirs(MARKER_DIR, exist_ok=True)
        ws_hash = hashlib.md5(ws.encode()).hexdigest()[:12]
        marker_path = os.path.join(MARKER_DIR, ws_hash)

        # Build layer scores
        layer_scores = {
            "surface": None,
            "structural": None,
            "persona_drift": None,
            "commitment": None,
            "context": None,
        }
        # Simple aggregate: count violations per layer, normalize to 0-1
        for layer_name, layer_num in [("surface", 1), ("structural", 2)]:
            layer_violations = [v for v in violations if v.get("layer") == layer_num]
            if layer_violations:
                # More violations = higher score (worse)
                warning_count = sum(1 for v in layer_violations if v.get("severity") == "warning")
                info_count = sum(1 for v in layer_violations if v.get("severity") == "info")
                layer_scores[layer_name] = min(1.0, (warning_count * 0.3 + info_count * 0.1))

        # Summary: top 3 by severity
        sorted_v = sorted(violations, key=lambda v: {"warning": 0, "info": 1}.get(v.get("severity", "info"), 2))
        summary_parts = []
        for v in sorted_v[:3]:
            count = v.get("count", 1)
            cat = v.get("category", "unknown")
            if v.get("score") is not None:
                summary_parts.append(f"{cat} (CV={v['score']:.2f})")
            elif count > 1:
                summary_parts.append(f"{count} {cat}")
            else:
                summary_parts.append(cat)
        summary = ", ".join(summary_parts)

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scan_version": "1.0",
            "persona_key": persona_key,
            "violations": violations,
            "summary": summary,
            "layer_scores": layer_scores,
        }

        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass  # Non-fatal


def read_marker(workspace: Optional[str] = None) -> Optional[dict]:
    """Read and consume a writing quality marker, if present.

    Returns the marker data dict, or None if no marker exists.
    Deletes the marker file after reading (consume-once pattern).
    """
    ws = workspace or os.getcwd()
    try:
        if not os.path.isdir(MARKER_DIR):
            return None
        ws_hash = hashlib.md5(ws.encode()).hexdigest()[:12]
        marker_path = os.path.join(MARKER_DIR, ws_hash)
        if not os.path.isfile(marker_path):
            return None
        with open(marker_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        os.unlink(marker_path)
        return data
    except Exception:
        return None
