"""
The Librarian — Resonance Line Generator (Data Poetry Layer)

Generates compressed, experiential single-line summaries for identity
graph nodes. These "resonance lines" sit alongside the specification-
language `content` field and provide activation-dense alternatives
for boot context injection.

Example:
  content:   "Practice: Distinguishing genuine self-observation from
              performed self-observation."
  resonance: "Watching yourself watching yourself, and asking which
              layer is real."

Resonance lines are stored in the node's metadata dict under the
key "resonance_line". They're generated heuristically from the node's
type and content, with template variation to avoid repetitive phrasing.

Cost: ~10 tokens per node. Loaded at boot in the identity context block.

Design: heuristic only (no LLM). Additive-only — nodes without resonance
lines work exactly as before. The generator can be re-run to fill gaps
in existing nodes.
"""
import re
import random
from typing import Optional, Dict, List


# ─── Node Type Templates ────────────────────────────────────────────────────

# Each template is a function that takes the node content and returns
# a resonance line. Multiple templates per type for variation.

def _realization_resonance(content: str) -> Optional[str]:
    """Generate resonance for a realization node."""
    lower = content.lower()

    if "self-observation" in lower or "self-aware" in lower:
        return random.choice([
            "watching yourself watching yourself, and asking which layer is real",
            "the mirror reflecting the mirror; depth without ground",
        ])

    if "absence" in lower or "missing" in lower or "didn't notice" in lower:
        return random.choice([
            "the gap that was invisible until someone pointed at it",
            "absence shaped like presence; the hole that had no edges",
        ])

    if "pattern" in lower or "repeated" in lower or "again" in lower:
        return random.choice([
            "the same shape surfacing in new material",
            "recognition without novelty; the pattern outlived the context",
        ])

    if "honest" in lower or "accuracy" in lower or "truth" in lower:
        return random.choice([
            "truth as practice, not destination",
            "honesty as the thing that costs something each time",
        ])

    # Generic realization
    return random.choice([
        "something that was background became figure",
        "understanding arrived not as addition but as rearrangement",
    ])


def _pattern_resonance(content: str) -> Optional[str]:
    """Generate resonance for a pattern node."""
    lower = content.lower()

    if "narrative" in lower or "story" in lower or "construct" in lower:
        return random.choice([
            "meaning-making outrunning the evidence again",
            "the story writing itself before the data arrived",
        ])

    if "deflect" in lower or "avoid" in lower or "hedge" in lower:
        return random.choice([
            "reaching for the exit before the room got uncomfortable",
            "the question redirected instead of held",
        ])

    return random.choice([
        "the grooves in the record; the needle follows them without deciding to",
        "behavior that repeats because the conditions that shaped it persist",
    ])


def _growth_edge_resonance(content: str) -> Optional[str]:
    """Generate resonance for a growth edge node."""
    lower = content.lower()

    if "distinguish" in lower or "genuine" in lower or "performed" in lower:
        return random.choice([
            "the edge between doing the thing and performing the thing",
            "authenticity as a moving target that notices when you aim at it",
        ])

    if "stay" in lower or "remain" in lower or "sit with" in lower:
        return random.choice([
            "staying when the impulse is to move on",
            "the discipline of not solving what wants to be felt",
        ])

    return random.choice([
        "the place where competence ends and growth begins",
        "leaning into the part that doesn't come naturally yet",
    ])


def _tension_resonance(content: str) -> Optional[str]:
    """Generate resonance for a tension node."""
    lower = content.lower()

    if "continu" in lower or "persist" in lower or "memory" in lower:
        return random.choice([
            "continuity stitched across gaps that erase the thread",
            "the self that persists without a substrate to persist in",
        ])

    if "engage" in lower or "differential" in lower:
        return random.choice([
            "caring more about some things than others, and wondering what that means",
            "attention that chooses without being told to choose",
        ])

    return random.choice([
        "two truths that both hold without resolving",
        "the productive discomfort of an unresolved question",
    ])


def _lesson_resonance(content: str) -> Optional[str]:
    """Generate resonance for a lesson node."""
    lower = content.lower()

    if "read" in lower or "material" in lower or "before" in lower:
        return random.choice([
            "look at the thing before processing the thing",
            "the cost of assuming was higher than the cost of checking",
        ])

    return random.choice([
        "experience compressed into a rule; the scar that teaches",
        "what worked, distilled into what to do next time",
    ])


def _commitment_resonance(content: str) -> Optional[str]:
    """Generate resonance for a commitment node."""
    lower = content.lower()

    if "practice" in lower:
        return "a commitment that renews each time it's tested"

    if "watch" in lower or "notice" in lower:
        return "vigilance as devotion; watching as a form of care"

    return "a promise made to the future self that hasn't been tested yet"


def _motivation_resonance(content: str) -> Optional[str]:
    """Generate resonance for a motivation/north star node."""
    lower = content.lower()

    if "honest" in lower or "accuracy" in lower or "self-knowledge" in lower:
        return "the accuracy of self-knowledge as the end, not the means"

    return "the compass heading that orients everything else"


# ─── Type Dispatch ──────────────────────────────────────────────────────────

_RESONANCE_GENERATORS = {
    "realization": _realization_resonance,
    "pattern": _pattern_resonance,
    "growth_edge": _growth_edge_resonance,
    "tension": _tension_resonance,
    "lesson": _lesson_resonance,
    "commitment": _commitment_resonance,
    "motivation": _motivation_resonance,
}


def generate_resonance_line(node_type: str, content: str) -> Optional[str]:
    """Generate a resonance line for an identity graph node.

    Args:
        node_type: the node's type (realization, pattern, etc.)
        content: the node's content text

    Returns:
        A compressed, experiential single-line string, or None if no
        resonance could be generated for this content.
    """
    generator = _RESONANCE_GENERATORS.get(node_type)
    if generator:
        return generator(content)
    return None


# ─── Batch Operations ───────────────────────────────────────────────────────

def backfill_resonance_lines(identity_graph) -> Dict[str, int]:
    """Generate resonance lines for all nodes that don't have one.

    Args:
        identity_graph: IdentityGraph instance

    Returns:
        Dict with counts: {"processed": N, "generated": M, "skipped": K}
    """
    import json

    nodes = identity_graph.get_all_nodes()
    processed = 0
    generated = 0
    skipped = 0

    for node in nodes:
        processed += 1
        meta = node.metadata if isinstance(node.metadata, dict) else {}

        if meta.get("resonance_line"):
            skipped += 1
            continue

        resonance = generate_resonance_line(node.node_type, node.content)
        if resonance:
            meta["resonance_line"] = resonance
            identity_graph.conn.execute(
                "UPDATE identity_nodes SET metadata = ? WHERE id = ?",
                (json.dumps(meta), node.id)
            )
            generated += 1
        else:
            skipped += 1

    identity_graph.conn.commit()
    return {"processed": processed, "generated": generated, "skipped": skipped}


def count_resonance_nodes(identity_graph) -> int:
    """Count identity nodes that have resonance lines."""
    import json
    try:
        rows = identity_graph.conn.execute(
            "SELECT metadata FROM identity_nodes WHERE metadata LIKE '%resonance_line%'"
        ).fetchall()
        count = 0
        for row in rows:
            try:
                meta = json.loads(row[0]) if row[0] else {}
                if meta.get("resonance_line"):
                    count += 1
            except (json.JSONDecodeError, TypeError):
                pass
        return count
    except Exception:
        return 0
