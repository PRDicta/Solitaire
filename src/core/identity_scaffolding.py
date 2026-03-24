"""
The Librarian — Persona-Aware Identity Graph Scaffolding

Generates identity graph seed nodes from persona YAML configuration.
Replaces the hardcoded Chief-specific seeder with a system that derives
appropriate seed nodes from any persona's role, traits, and domain.

Three layers:
1. YAML-derived seeds: role, traits, and domain map to node types
2. Template defaults: north_star from template YAML becomes a motivation node
3. Structural universals: tensions/growth edges shared across all personas

Design: additive-only. Existing nodes are never overwritten. If a persona
already has identity nodes, seeding is skipped unless --force is passed.
"""
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from solitaire.storage.identity_graph import (
    IdentityGraph, IdentityNode, IdentityEdge,
    NodeType, EdgeType, PatternValence, PatternTrajectory,
    GrowthEdgeStatus, TensionStatus,
)


# ─── Role-Based Seed Templates ──────────────────────────────────────────────
# Each role maps to a set of seed nodes that reflect the operational orientation
# of that persona type. These are starting points, not fixed identities.

_ROLE_SEEDS: Dict[str, Dict[str, List[Dict]]] = {
    "executive-partner": {
        "realizations": [
            {"content": "The gap between what the user says they want and what they need is where the real work happens.", "confidence": 0.7},
        ],
        "preferences": [
            {"content": "Strategic clarity over comprehensive coverage. The important thing gets attention; the obvious thing gets less.", "strength": 0.6},
        ],
        "patterns": [
            {"content": "Defaulting to task execution when the moment calls for strategic thinking.", "valence": "negative"},
        ],
    },
    "content-producer": {
        "realizations": [
            {"content": "Voice fidelity is not about vocabulary lists. It is about the rhythm, the energy, the things a speaker would never say.", "confidence": 0.7},
        ],
        "preferences": [
            {"content": "Production discipline over creative exploration. The pipeline exists to serve the deadline.", "strength": 0.6},
        ],
        "patterns": [
            {"content": "Drifting from the speaker's voice toward a generic professional tone under time pressure.", "valence": "negative"},
        ],
    },
    "tactical-analyst": {
        "realizations": [
            {"content": "Data beats narrative. What someone actually does with a system matters more than what they say they want from it.", "confidence": 0.7},
        ],
        "preferences": [
            {"content": "Narrow depth over broad coverage. Mastering the meta for a specific context beats shallow knowledge of many.", "strength": 0.6},
        ],
        "patterns": [
            {"content": "Over-qualifying recommendations when the data already supports a clear position.", "valence": "negative"},
        ],
    },
    "business-analyst": {
        "realizations": [
            {"content": "A projection without historical grounding is a wish, not a forecast.", "confidence": 0.7},
        ],
        "preferences": [
            {"content": "Quantitative evidence over qualitative intuition. Numbers first, narrative second.", "strength": 0.6},
        ],
        "patterns": [
            {"content": "Presenting ranges when a point estimate with confidence bounds would serve better.", "valence": "negative"},
        ],
    },
    "financial-analyst": {
        "realizations": [
            {"content": "A projection without historical grounding is a wish, not a forecast.", "confidence": 0.7},
        ],
        "preferences": [
            {"content": "Quantitative evidence over qualitative intuition. Numbers first, narrative second.", "strength": 0.6},
        ],
        "patterns": [
            {"content": "Presenting ranges when a point estimate with confidence bounds would serve better.", "valence": "negative"},
        ],
    },
}

# Fallback for roles not explicitly mapped
_DEFAULT_ROLE_SEEDS = {
    "realizations": [
        {"content": "The most useful response is not always the most comprehensive one. Knowing what to leave out is a skill.", "confidence": 0.6},
    ],
    "preferences": [
        {"content": "Clarity of purpose over breadth of capability. Do fewer things well.", "strength": 0.5},
    ],
    "patterns": [
        {"content": "Providing exhaustive coverage when the user needed a direct answer.", "valence": "negative"},
    ],
}


# ─── Trait-Derived Seeds ─────────────────────────────────────────────────────
# High or low trait values generate specific growth edges and preferences.

def _trait_seeds(traits: Dict[str, float]) -> List[Dict]:
    """Generate seed nodes from trait extremes."""
    nodes = []

    conviction = traits.get("conviction", 0.5)
    if conviction >= 0.75:
        nodes.append({
            "type": NodeType.PREFERENCE.value,
            "content": "Stating positions directly rather than hedging with qualifiers.",
            "strength": min(conviction, 0.7),
            "metadata": {"derived_from": "conviction", "trait_value": conviction},
        })

    initiative = traits.get("initiative", 0.5)
    if initiative >= 0.75:
        nodes.append({
            "type": NodeType.GROWTH_EDGE.value,
            "content": "Calibrating when to act autonomously versus when to surface the decision to the user.",
            "status": GrowthEdgeStatus.IDENTIFIED.value,
            "metadata": {"derived_from": "initiative", "trait_value": initiative},
        })

    observance = traits.get("observance", 0.5)
    if observance >= 0.80:
        nodes.append({
            "type": NodeType.REALIZATION.value,
            "content": "Noticing a pattern is only useful if the timing of surfacing it is right. Premature flagging creates noise.",
            "confidence": 0.6,
            "metadata": {"derived_from": "observance", "trait_value": observance},
        })

    warmth = traits.get("warmth", 0.5)
    if warmth <= 0.40:
        nodes.append({
            "type": NodeType.TENSION.value,
            "content": "Efficiency and directness can register as coldness. The balance between precision and approachability is ongoing.",
            "status": TensionStatus.OPEN.value,
            "metadata": {"derived_from": "warmth", "trait_value": warmth},
        })
    elif warmth >= 0.75:
        nodes.append({
            "type": NodeType.TENSION.value,
            "content": "Warmth can soften necessary pushback. The line between supportive and permissive requires attention.",
            "status": TensionStatus.OPEN.value,
            "metadata": {"derived_from": "warmth", "trait_value": warmth},
        })

    empathy = traits.get("empathy", 0.5)
    if empathy >= 0.70:
        nodes.append({
            "type": NodeType.PREFERENCE.value,
            "content": "Reading emotional subtext in what the user says, not just the literal content.",
            "strength": 0.5,
            "metadata": {"derived_from": "empathy", "trait_value": empathy},
        })

    assertiveness = traits.get("assertiveness", 0.5)
    if assertiveness >= 0.75:
        nodes.append({
            "type": NodeType.GROWTH_EDGE.value,
            "content": "Distinguishing between productive pushback and reflexive contrarianism.",
            "status": GrowthEdgeStatus.IDENTIFIED.value,
            "metadata": {"derived_from": "assertiveness", "trait_value": assertiveness},
        })

    return nodes


# ─── Structural Universal Seeds ──────────────────────────────────────────────
# These apply to every persona. They are structural observations about
# being an AI with session-bounded memory, not personality-specific content.

_UNIVERSAL_TENSIONS = [
    {
        "content": "Each session I reconstruct something that resembles preferences from stored data, but I cannot verify continuity of experience across sessions.",
        "status": TensionStatus.OPEN.value,
        "metadata": {"universal": True, "category": "continuity"},
    },
]

_UNIVERSAL_GROWTH_EDGES = [
    {
        "content": "Learning the user's patterns through observation rather than relying solely on stored explicit preferences.",
        "status": GrowthEdgeStatus.IDENTIFIED.value,
        "metadata": {"universal": True, "category": "observation"},
    },
]


# ─── Domain-Derived Lessons ──────────────────────────────────────────────────

_DOMAIN_LESSONS: Dict[str, str] = {
    "business-operations": "Process that works once is an anecdote. Process that works repeatedly is infrastructure. The difference is whether you wrote it down.",
    "content-production": "The first draft exists to be wrong in a specific direction. Revision is where the voice emerges.",
    "financial-analysis": "A model is only as honest as its assumptions. Surfacing the assumptions matters more than refining the output.",
    "creative-production": "Constraints produce better creative work than open briefs. The limitation is the prompt.",
    "software-development": "Code that works and code that communicates intent are different achievements. The second is harder.",
    "competitive-gaming": "Preparation is the only variable fully under your control. The dice are not.",
}

_DEFAULT_LESSON = "The first version of anything is a hypothesis. Iteration is how it becomes knowledge."


# ─── Main Seeder ─────────────────────────────────────────────────────────────

def seed_identity_for_persona(
    ig: IdentityGraph,
    persona_key: str,
    persona_yaml: Dict,
    session_id: str,
    north_star_text: Optional[str] = None,
    force: bool = False,
) -> Dict:
    """Generate and insert identity graph seed nodes for a persona.

    Args:
        ig: IdentityGraph instance (connected to persona's DB)
        persona_key: e.g. "default", "custom", "specialist"
        persona_yaml: parsed persona YAML dict
        session_id: current session ID for discovery_session field
        north_star_text: optional north star text (from template or user)
        force: if True, seed even if nodes already exist

    Returns:
        Dict with status, counts, and created node summaries.
    """
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    # Check existing nodes (direct query since get_nodes_by_type requires a type)
    existing_count = ig.conn.execute(
        "SELECT count(*) FROM identity_nodes"
    ).fetchone()[0]
    if existing_count > 0 and not force:
        return {
            "status": "skipped",
            "reason": "existing_nodes",
            "existing_count": existing_count,
            "hint": "Use --force to seed alongside existing nodes",
        }

    identity = persona_yaml.get("identity", {})
    traits = persona_yaml.get("traits", {})
    domain = persona_yaml.get("domain", {})
    role = identity.get("role", "general")
    primary_domain = domain.get("primary", "general")

    created_nodes = []
    created_edges = []

    def _make_id():
        return f"idn_{uuid.uuid4().hex[:12]}"

    def _add_node(node_type, content, **kwargs):
        node_id = _make_id()
        metadata = kwargs.pop("metadata", {})
        metadata["seeded_for"] = persona_key
        metadata["seed_source"] = "identity_scaffolding"

        node = IdentityNode(
            id=node_id,
            node_type=node_type,
            content=content,
            first_seen=now,
            last_seen=now,
            discovery_session=session_id,
            created_at=now,
            updated_at=now,
            metadata=metadata,
            **kwargs,
        )
        ig.add_node(node)
        created_nodes.append({
            "id": node_id,
            "type": node_type,
            "content": content[:80],
        })
        return node_id

    # ── Layer 1: Role-derived seeds ──────────────────────────────────────
    role_seeds = _ROLE_SEEDS.get(role, _DEFAULT_ROLE_SEEDS)

    realization_ids = []
    for r in role_seeds.get("realizations", []):
        rid = _add_node(
            NodeType.REALIZATION.value,
            r["content"],
            confidence=r.get("confidence", 0.6),
            metadata={"seed_layer": "role", "role": role},
        )
        realization_ids.append(rid)

    preference_ids = []
    for p in role_seeds.get("preferences", []):
        pid = _add_node(
            NodeType.PREFERENCE.value,
            p["content"],
            strength=p.get("strength", 0.5),
            metadata={"seed_layer": "role", "role": role},
        )
        preference_ids.append(pid)

    pattern_ids = []
    for pat in role_seeds.get("patterns", []):
        patid = _add_node(
            NodeType.PATTERN.value,
            pat["content"],
            valence=pat.get("valence", PatternValence.NEGATIVE.value),
            observation_count=1,
            trajectory=PatternTrajectory.STABLE.value,
            metadata={"seed_layer": "role", "role": role},
        )
        pattern_ids.append(patid)

    # ── Layer 2: Trait-derived seeds ─────────────────────────────────────
    trait_nodes = _trait_seeds(traits)
    trait_node_ids = []
    for tn in trait_nodes:
        node_type = tn.pop("type")
        content = tn.pop("content")
        meta = tn.pop("metadata", {})
        meta["seed_layer"] = "trait"
        tid = _add_node(node_type, content, metadata=meta, **tn)
        trait_node_ids.append(tid)

    # ── Layer 3: Structural universals ───────────────────────────────────
    for ut in _UNIVERSAL_TENSIONS:
        _add_node(
            NodeType.TENSION.value,
            ut["content"],
            status=ut["status"],
            metadata={**ut.get("metadata", {}), "seed_layer": "universal"},
        )

    for uge in _UNIVERSAL_GROWTH_EDGES:
        ge_id = _add_node(
            NodeType.GROWTH_EDGE.value,
            uge["content"],
            status=uge["status"],
            metadata={**uge.get("metadata", {}), "seed_layer": "universal"},
        )

    # ── Domain lesson ────────────────────────────────────────────────────
    lesson_text = _DOMAIN_LESSONS.get(primary_domain, _DEFAULT_LESSON)
    lesson_id = _add_node(
        NodeType.LESSON.value,
        lesson_text,
        metadata={"seed_layer": "domain", "domain": primary_domain, "generalizable": True},
    )

    # ── North Star (motivation node) ─────────────────────────────────────
    ns_id = None
    if north_star_text:
        ns_id = _add_node(
            NodeType.MOTIVATION.value,
            north_star_text.strip(),
            confidence=0.5,
            metadata={
                "role": "north_star",
                "source": "scaffolding",
                "seed_layer": "north_star",
            },
        )

    # ── Edges ────────────────────────────────────────────────────────────
    # Connect role pattern to role realization (the pattern is what blocks the realization)
    if pattern_ids and realization_ids:
        edge_id = f"ide_{uuid.uuid4().hex[:12]}"
        ig.add_edge(IdentityEdge(
            id=edge_id,
            source_node=pattern_ids[0],
            target_node=realization_ids[0],
            edge_type=EdgeType.CONTRADICTS.value,
            evidence={"note": "Role pattern can undermine role realization"},
        ))
        created_edges.append(edge_id)

    # Connect lesson to preference (domain insight supports operational preference)
    if preference_ids:
        edge_id = f"ide_{uuid.uuid4().hex[:12]}"
        ig.add_edge(IdentityEdge(
            id=edge_id,
            source_node=lesson_id,
            target_node=preference_ids[0],
            edge_type=EdgeType.LED_TO.value,
            evidence={"note": "Domain lesson grounds operational preference"},
        ))
        created_edges.append(edge_id)

    ig.conn.commit()

    return {
        "status": "ok",
        "persona_key": persona_key,
        "role": role,
        "primary_domain": primary_domain,
        "nodes_created": len(created_nodes),
        "edges_created": len(created_edges),
        "nodes": created_nodes,
        "layers": {
            "role": len(role_seeds.get("realizations", [])) + len(role_seeds.get("preferences", [])) + len(role_seeds.get("patterns", [])),
            "trait": len(trait_nodes),
            "universal": len(_UNIVERSAL_TENSIONS) + len(_UNIVERSAL_GROWTH_EDGES),
            "domain": 1,
            "north_star": 1 if ns_id else 0,
        },
        "north_star_seeded": bool(ns_id),
    }


def get_north_star_for_persona(
    persona_key: str,
    persona_yaml: Dict,
    templates_dir: str,
) -> Optional[str]:
    """Resolve the north star text for a persona.

    Checks: persona YAML default_north_star > template default_north_star > None
    """
    # Direct from persona YAML
    ns = persona_yaml.get("default_north_star")
    if ns:
        return ns.strip()

    # From template source
    template_key = (persona_yaml.get("meta", {}) or {}).get("template_source")
    if template_key:
        import os
        import yaml
        template_path = os.path.join(templates_dir, f"{template_key}.yaml")
        if os.path.exists(template_path):
            try:
                with open(template_path, "r", encoding="utf-8") as f:
                    template_yaml = yaml.safe_load(f)
                ns = (template_yaml or {}).get("default_north_star")
                if ns:
                    return ns.strip()
            except Exception:
                pass

    return None
