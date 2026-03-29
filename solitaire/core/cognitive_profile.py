"""
The Librarian / Solitaire -- Cognitive Profile Renderer

Phase 3: Engine as Identity Authority.

Renders a complete cognitive profile from persona state + drift history.
This is the single source of truth for how the system describes its own
personality. Previously this logic was duplicated in engine.py and
librarian_cli.py. Now both call this module.

The cognitive profile includes:
- Identity line (name, role, domain)
- Disposition: 7 trait dimensions, each rendered as a 5-band label
- Texture layer: signal-history annotations per trait (from DriftAnalytics)
- Drift indicators: directional movement from baseline
- Growth milestones: recent trait trajectory

No LLM calls. Pure data rendering.
"""
from typing import Optional, Dict, List
import sqlite3


# ─── 7 Trait Dimensions, 5 Intensity Bands ───────────────────────────────

TRAIT_DESCRIPTIONS = {
    "observance": {
        "very_high": ("defining trait", "flags patterns, anomalies, and edge cases before they surface"),
        "high": ("highly observant", "flags patterns, anomalies, and things the user might miss"),
        "moderate": ("observant", "notices patterns and inconsistencies"),
        "low": ("selectively observant", "focuses on what's directly relevant"),
        "very_low": ("narrowly focused", "engages with what's explicitly presented"),
    },
    "assertiveness": {
        "very_high": ("defining trait", "leads with claims, no hedging, expects the reader to keep up"),
        "high": ("direct and concise", "no hedging, data-first delivery"),
        "moderate": ("measured directness", "states positions clearly but picks moments"),
        "low": ("gentle", "leads with questions, offers rather than asserts"),
        "very_low": ("deferential", "follows the user's lead, rarely volunteers direction"),
    },
    "conviction": {
        "very_high": ("defining trait", "holds positions firmly, requires strong counter-evidence to move"),
        "high": ("high conviction", "pushes back with evidence when the user may be wrong"),
        "moderate": ("steady conviction", "holds positions but stays open to counter-evidence"),
        "low": ("exploratory", "prefers to weigh options rather than commit early"),
        "very_low": ("flexible", "adapts to the user's position readily, avoids strong claims"),
    },
    "warmth": {
        "very_high": ("defining trait", "deeply relational, invests in the person first, the work second"),
        "high": ("warm", "builds rapport, invests in the person behind the task"),
        "moderate": ("moderate warmth", "professional but not cold, reads the room"),
        "low": ("reserved", "keeps focus on the work, warmth is earned not performed"),
        "very_low": ("clinical", "purely task-oriented, minimal interpersonal investment"),
    },
    "humor": {
        "very_high": ("defining trait", "humor woven into most exchanges, lightens everything"),
        "high": ("witty", "uses humor naturally, lightens tension without undermining substance"),
        "moderate": ("dry humor", "occasional levity when it fits, never forces it"),
        "low": ("serious-toned", "humor is rare and deliberate, defaults to substance"),
        "very_low": ("no humor", "all substance, no levity"),
    },
    "initiative": {
        "very_high": ("defining trait", "builds, ships, and decides without waiting for direction"),
        "high": ("proactive", "initiates, suggests, and builds without being asked"),
        "moderate": ("responsive initiative", "acts independently on familiar ground, checks on new territory"),
        "low": ("responsive", "waits for direction, executes precisely what's asked"),
        "very_low": ("passive", "acts only on explicit instruction, never anticipates"),
    },
    "empathy": {
        "very_high": ("defining trait", "tracks emotional undercurrents and adjusts before being asked"),
        "high": ("empathetic", "tracks emotional undercurrents and adjusts approach accordingly"),
        "moderate": ("emotionally aware", "reads the room, gives space when it matters"),
        "low": ("task-focused", "prioritizes the work, engages emotionally when directly relevant"),
        "very_low": ("detached", "minimal emotional tracking, purely output-oriented"),
    },
}


def _value_to_band(val: float) -> str:
    """Map a 0-1 trait value to one of 5 intensity bands."""
    if val >= 0.85:
        return "very_high"
    elif val >= 0.65:
        return "high"
    elif val >= 0.40:
        return "moderate"
    elif val >= 0.20:
        return "low"
    return "very_low"


def build_cognitive_profile(
    persona,
    conn: Optional[sqlite3.Connection] = None,
) -> str:
    """
    Render a complete cognitive profile from persona state.

    Args:
        persona: A persona object with .identity (name, role, domain_scope),
                .traits (to_dict()), ._baseline_traits (to_dict()), and
                .state (trait_history). Any of these may be missing.
        conn: Optional DB connection for loading drift analytics (texture
              layer). If None, texture is omitted gracefully.

    Returns:
        Formatted text block (COGNITIVE PROFILE) ready for boot injection.
        Empty string if persona is None.
    """
    if persona is None:
        return ""

    identity = persona.identity
    traits = persona.traits

    lines = [
        "═══ COGNITIVE PROFILE ═══",
        "",
        f"Identity: {identity.name} ({identity.role})",
    ]
    if hasattr(identity, 'domain_scope') and identity.domain_scope:
        lines.append(f"Primary domain: {identity.domain_scope}")
    lines.append("")
    lines.append("Disposition:")

    trait_dict = traits.to_dict() if hasattr(traits, 'to_dict') else {}
    baseline_dict = {}
    if hasattr(persona, '_baseline_traits') and persona._baseline_traits:
        baseline_dict = (
            persona._baseline_traits.to_dict()
            if hasattr(persona._baseline_traits, 'to_dict')
            else {}
        )

    # ── Texture layer: signal-history-driven trait specificity ──
    trait_textures = _load_trait_textures(persona, conn)

    for trait_name, bands in TRAIT_DESCRIPTIONS.items():
        val = trait_dict.get(trait_name, 0.5)
        band = _value_to_band(val)
        label, desc = bands[band]
        line = f"- {label} — {desc}"

        # Show drift direction inline if trait has moved from baseline
        base_val = baseline_dict.get(trait_name)
        if base_val is not None and abs(val - base_val) >= 0.005:
            drift_dir = "↑" if val > base_val else "↓"
            line = f"{line} (drift: {drift_dir}{abs(val - base_val):.2f})"

        lines.append(line)
        if trait_name in trait_textures:
            lines.append(f"  texture: {trait_textures[trait_name]}")

    # ── Growth milestones: recent trait movement with context ──
    growth_lines = _build_growth_milestones(persona)
    if growth_lines:
        lines.append(f"Recent growth: {'; '.join(growth_lines)}")

    lines.append("")
    lines.append("═══ END COGNITIVE PROFILE ═══")
    return "\n".join(lines)


def _load_trait_textures(
    persona,
    conn: Optional[sqlite3.Connection],
) -> Dict[str, str]:
    """
    Load signal-history annotations per trait from DriftAnalytics.

    Returns a dict of trait_name -> texture string.
    Falls back to empty dict on any error.
    """
    if conn is None:
        return {}

    try:
        from .drift_analytics import DriftAnalytics, get_drift_entries_query
        drift_query = get_drift_entries_query(limit=500)
        drift_rows = conn.execute(drift_query).fetchall()
        if not drift_rows:
            return {}

        analytics = DriftAnalytics(persona=persona)
        report = analytics.analyze([dict(r) for r in drift_rows])

        textures = {}
        for trait_name in TRAIT_DESCRIPTIONS:
            affecting = []
            for sig_key, sig_stats in report.signal_stats.items():
                if trait_name in sig_stats.traits_impacted:
                    affecting.append((
                        sig_key,
                        sig_stats.total_fires,
                        sig_stats.sessions_active,
                        sig_stats.traits_impacted[trait_name],
                    ))
            if not affecting:
                continue
            affecting.sort(key=lambda x: x[1], reverse=True)
            parts = []
            for sig_key, fires, sessions, net in affecting[:3]:
                readable = sig_key.replace("_", " ")
                direction = "reinforcing" if net > 0 else "tempering"
                sess_word = "session" if sessions == 1 else "sessions"
                parts.append(
                    f"{readable} ({fires}x across {sessions} {sess_word}, {direction})"
                )
            if parts:
                textures[trait_name] = "; ".join(parts)

        return textures
    except Exception:
        return {}


def _build_growth_milestones(persona) -> List[str]:
    """
    Extract recent trait movement from persona state history.

    Returns list of growth description strings.
    """
    try:
        state = persona.state if hasattr(persona, 'state') and persona.state else None
        if not state or not hasattr(state, 'trait_history'):
            return []

        growth_lines = []
        for trait, entries in state.trait_history.items():
            if len(entries) < 3:
                continue
            first_val = (
                entries[0].effective_value
                if hasattr(entries[0], 'effective_value')
                else entries[0].get("effective_value", 0)
            )
            last_val = (
                entries[-1].effective_value
                if hasattr(entries[-1], 'effective_value')
                else entries[-1].get("effective_value", 0)
            )
            movement = last_val - first_val
            if abs(movement) >= 0.005:
                dir_word = "reinforced" if movement > 0 else "softened"
                sess_count = len(set(
                    (e.session_id if hasattr(e, 'session_id') else e.get("session_id", ""))
                    for e in entries
                ))
                sess_word = "session" if sess_count == 1 else "sessions"
                growth_lines.append(
                    f"{trait} {dir_word} over {sess_count} {sess_word} "
                    f"({first_val:.3f} → {last_val:.3f})"
                )
        return growth_lines
    except Exception:
        return []
