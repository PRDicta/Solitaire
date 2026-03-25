"""
The Librarian — Onboarding Engine
Guided first-boot experience for new Librarian instances.

Detects first boot (empty or missing rolodex), presents template options
matched to user intent, and hydrates the instance with scaffolding:
persona YAML + seed knowledge + skill recommendations.

The onboarding flow is designed to be called from the CLI's boot command
when it detects a fresh instance. It returns structured JSON that the
calling agent (Claude, etc.) can use to drive an interactive conversation.
"""
import json
import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


REGISTRY_FILENAME = "REGISTRY.yaml"


def load_registry(templates_dir: str) -> Dict[str, Any]:
    """Load the template registry YAML."""
    registry_path = os.path.join(templates_dir, REGISTRY_FILENAME)
    if not os.path.exists(registry_path):
        return {}
    with open(registry_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def match_intent(
    user_input: str,
    registry: Dict[str, Any],
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Match user-expressed goals to template recommendations.

    Uses keyword overlap scoring against the intent_mappings in the registry.
    Returns top N matches with scores, sorted by relevance.

    Args:
        user_input: What the user said they want to do.
        registry: The loaded REGISTRY.yaml data.
        top_n: Maximum number of recommendations.

    Returns:
        List of dicts: [{template, pitch, score, display_name, category}]
    """
    mappings = registry.get("intent_mappings", [])
    templates = registry.get("templates", {})
    user_words = set(re.findall(r'\b\w+\b', user_input.lower()))

    scored = []
    for mapping in mappings:
        intents = mapping.get("intents", [])
        template_key = mapping.get("template", "")
        pitch = mapping.get("pitch", "")

        # Score: count how many intent keywords appear in user input
        score = 0
        for intent_phrase in intents:
            intent_words = set(re.findall(r'\b\w+\b', intent_phrase.lower()))
            overlap = len(user_words & intent_words)
            if overlap > 0:
                # Bonus for full phrase match
                if intent_phrase.lower() in user_input.lower():
                    score += overlap * 2
                else:
                    score += overlap

        if score > 0:
            template_info = templates.get(template_key, {})
            scored.append({
                "template": template_key,
                "pitch": pitch,
                "score": score,
                "display_name": template_info.get("display_name", template_key),
                "category": template_info.get("category", "general"),
                "description": template_info.get("description", ""),
                "has_scaffolding": bool(template_info.get("scaffolding")),
            })

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


def load_category_tree(templates_dir: str) -> Dict[str, Any]:
    """Load INTENT_CATEGORY_TREE.yaml from the templates directory."""
    tree_path = os.path.join(templates_dir, "INTENT_CATEGORY_TREE.yaml")
    if not os.path.exists(tree_path):
        return {}
    with open(tree_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def match_intent_v2(
    user_input: str,
    registry: Dict[str, Any],
    templates_dir: str,
    top_n: int = 3,
) -> Dict[str, Any]:
    """Hierarchical intent matching using the category tree.

    Algorithm:
    1. Score user intent against all category intent phrases.
    2. Select top category. If top two within ambiguity_threshold, flag ambiguous.
    3. Within top category, score against subcategory intents.
    4. If subcategory is ambiguous, include refinement question in result.

    Falls back to match_intent() if no category tree exists.

    Returns:
        Dict with:
            category: str - top matched category key
            category_name: str - display name
            ambiguous: bool - whether categories were close
            subcategory: Optional[str] - top subcategory if clear
            templates: List[str] - candidate template keys
            refinement: Optional[dict] - refinement question if ambiguous
            recommendations: List[dict] - template recommendations (same format as match_intent)
    """
    tree = load_category_tree(templates_dir)
    if not tree or "categories" not in tree:
        # Fallback to v1
        recs = match_intent(user_input, registry, top_n=top_n)
        return {
            "category": None,
            "category_name": None,
            "ambiguous": False,
            "subcategory": None,
            "templates": [r["template"] for r in recs],
            "refinement": None,
            "recommendations": recs,
        }

    categories = tree["categories"]
    threshold = tree.get("ambiguity_threshold", 0.80)
    user_words = set(re.findall(r'\b\w+\b', user_input.lower()))
    user_lower = user_input.lower()

    # Score categories
    cat_scores = []
    for cat_key, cat_data in categories.items():
        score = 0
        for phrase in cat_data.get("intents", []):
            phrase_lower = phrase.lower()
            # Word-boundary phrase match
            pattern = r'\b' + re.escape(phrase_lower) + r'\b'
            if re.search(pattern, user_lower):
                score += len(phrase_lower.split()) * 2
            else:
                phrase_words = set(re.findall(r'\b\w+\b', phrase_lower))
                overlap = len(user_words & phrase_words)
                if overlap > 0:
                    score += overlap
        if score > 0:
            cat_scores.append((cat_key, cat_data, score))

    if not cat_scores:
        # No category match — fallback to v1
        recs = match_intent(user_input, registry, top_n=top_n)
        return {
            "category": None,
            "category_name": None,
            "ambiguous": False,
            "subcategory": None,
            "templates": [r["template"] for r in recs],
            "refinement": None,
            "recommendations": recs,
        }

    cat_scores.sort(key=lambda x: x[2], reverse=True)
    top_cat_key, top_cat_data, top_score = cat_scores[0]

    # Check ambiguity between top two categories
    category_ambiguous = False
    if len(cat_scores) > 1:
        second_score = cat_scores[1][2]
        if second_score >= top_score * threshold:
            category_ambiguous = True

    # Score subcategories within top category
    subcats = top_cat_data.get("subcategories", {})
    subcat_scores = []
    for sub_key, sub_data in subcats.items():
        score = 0
        for phrase in sub_data.get("intents", []):
            phrase_lower = phrase.lower()
            pattern = r'\b' + re.escape(phrase_lower) + r'\b'
            if re.search(pattern, user_lower):
                score += len(phrase_lower.split()) * 2
            else:
                phrase_words = set(re.findall(r'\b\w+\b', phrase_lower))
                overlap = len(user_words & phrase_words)
                if overlap > 0:
                    score += overlap
        if score > 0:
            subcat_scores.append((sub_key, sub_data, score))

    subcat_scores.sort(key=lambda x: x[2], reverse=True)

    # Determine if subcategory is clear or ambiguous
    subcategory_ambiguous = False
    top_subcat = None
    candidate_templates = []

    if subcat_scores:
        top_sub_key, top_sub_data, top_sub_score = subcat_scores[0]
        top_subcat = top_sub_key
        candidate_templates = top_sub_data.get("templates", [])

        if len(subcat_scores) > 1:
            second_sub_score = subcat_scores[1][2]
            if second_sub_score >= top_sub_score * threshold:
                subcategory_ambiguous = True
                # Include templates from top 2 subcategories
                candidate_templates = list(candidate_templates)
                for t in subcat_scores[1][1].get("templates", []):
                    if t not in candidate_templates:
                        candidate_templates.append(t)
    else:
        # No subcategory match — include all templates from the category
        for sub_data in subcats.values():
            for t in sub_data.get("templates", []):
                if t not in candidate_templates:
                    candidate_templates.append(t)
        subcategory_ambiguous = True

    # Build recommendations from candidate templates (matching registry format)
    templates_info = registry.get("templates", {})
    recommendations = []
    for tkey in candidate_templates[:top_n]:
        tinfo = templates_info.get(tkey, {})
        if tinfo:
            recommendations.append({
                "template": tkey,
                "display_name": tinfo.get("display_name", tkey),
                "category": tinfo.get("category", "general"),
                "description": tinfo.get("description", ""),
                "has_scaffolding": bool(tinfo.get("scaffolding")),
                "score": 1,  # Scores not meaningful in v2 output
            })

    # Include refinement question if ambiguous
    refinement = None
    needs_refinement = category_ambiguous or subcategory_ambiguous
    if needs_refinement and "refinement_question" in top_cat_data:
        rq = top_cat_data["refinement_question"]
        refinement = {
            "prompt": rq.get("prompt", "Can you narrow down what you're looking for?"),
            "options": rq.get("options", []),
            "category": top_cat_key,
        }

    return {
        "category": top_cat_key,
        "category_name": top_cat_data.get("display_name", top_cat_key),
        "ambiguous": needs_refinement,
        "subcategory": top_subcat if not subcategory_ambiguous else None,
        "templates": candidate_templates[:top_n],
        "refinement": refinement,
        "recommendations": recommendations,
    }


def resolve_refinement(
    user_choice: str,
    category_key: str,
    templates_dir: str,
    registry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Resolve a refinement question answer to specific template recommendations.

    Args:
        user_choice: The subcategory key the user selected (maps_to value).
        category_key: The category that generated the refinement question.
        templates_dir: Path to persona_templates directory.
        registry: The loaded REGISTRY.yaml data.

    Returns:
        List of template recommendation dicts.
    """
    tree = load_category_tree(templates_dir)
    if not tree:
        return []

    categories = tree.get("categories", {})
    cat_data = categories.get(category_key, {})
    subcats = cat_data.get("subcategories", {})
    sub_data = subcats.get(user_choice, {})
    template_keys = sub_data.get("templates", [])

    templates_info = registry.get("templates", {})
    recommendations = []
    for tkey in template_keys:
        tinfo = templates_info.get(tkey, {})
        if tinfo:
            recommendations.append({
                "template": tkey,
                "display_name": tinfo.get("display_name", tkey),
                "category": tinfo.get("category", "general"),
                "description": tinfo.get("description", ""),
                "has_scaffolding": bool(tinfo.get("scaffolding")),
            })

    return recommendations


def get_template_scaffolding(
    template_key: str,
    registry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Get the scaffolding (skills, seed knowledge, tools) for a template."""
    templates = registry.get("templates", {})
    template = templates.get(template_key)
    if not template:
        return None
    return template.get("scaffolding")


def get_all_templates(
    registry: Dict[str, Any],
) -> List[Dict[str, str]]:
    """List all available templates for browsing."""
    templates = registry.get("templates", {})
    result = []
    for key, info in templates.items():
        result.append({
            "key": key,
            "display_name": info.get("display_name", key),
            "category": info.get("category", "general"),
            "description": info.get("description", ""),
            "file": info.get("file", ""),
        })
    return result


def build_onboarding_response(
    is_first_boot: bool,
    templates_dir: str,
    user_intent: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the onboarding payload for the CLI/agent.

    Called during boot when a fresh instance is detected.
    Returns structured JSON the agent can use to drive the conversation.

    Args:
        is_first_boot: Whether this is a brand new instance.
        templates_dir: Path to persona_templates directory.
        user_intent: Optional — if the user already stated what they want.

    Returns:
        Dict with onboarding state, recommendations, and available templates.
    """
    registry = load_registry(templates_dir)

    response = {
        "onboarding_required": is_first_boot,
        "available_templates": get_all_templates(registry),
        "build_your_own_available": True,
    }

    if user_intent:
        matches = match_intent(user_intent, registry)
        response["recommendations"] = matches
    else:
        response["recommendations"] = []

    # Provide the onboarding prompt for the agent
    response["onboarding_prompt"] = (
        "This is a fresh Librarian instance with no prior context. "
        "Ask the user what kind of tasks they're aiming for, then either: "
        "(1) recommend a matching template with pre-loaded scaffolding, or "
        "(2) offer the 'build your own' path where they customize from scratch. "
        "For templates, describe what comes pre-loaded (skills, seed knowledge) "
        "and offer to tweak the persona before finalizing. "
        "For build-your-own, ask about skills and knowledge to pre-load."
    )

    return response


def apply_scaffolding(
    template_key: str,
    templates_dir: str,
    rolodex_conn,
    session_id: str,
) -> Dict[str, Any]:
    """Apply a template's scaffolding to a fresh rolodex.

    Copies the persona YAML, ingests seed knowledge entries,
    and returns the list of recommended skills/tools.

    Args:
        template_key: Which template to apply.
        templates_dir: Path to persona_templates directory.
        rolodex_conn: SQLite connection to the rolodex.
        session_id: Current session ID for entry attribution.

    Returns:
        Dict with applied scaffolding details.
    """
    import uuid
    from datetime import datetime

    registry = load_registry(templates_dir)
    scaffolding = get_template_scaffolding(template_key, registry)
    if not scaffolding:
        return {"error": f"No scaffolding found for template: {template_key}"}

    result = {
        "template": template_key,
        "seed_knowledge_ingested": 0,
        "skills_recommended": [],
        "tools_to_build": [],
    }

    # Ingest seed knowledge as user_knowledge entries
    seed_entries = scaffolding.get("seed_knowledge", [])
    for seed in seed_entries:
        entry_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        try:
            rolodex_conn.execute(
                "INSERT INTO rolodex_entries "
                "(id, conversation_id, content, content_type, category, tags, "
                " source_range, access_count, created_at, tier, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry_id,
                    session_id,
                    seed,
                    "structured",
                    "user_knowledge",
                    json.dumps(["scaffolding", "seed", template_key]),
                    "{}",
                    0,
                    now,
                    "cold",
                    json.dumps({"source": "onboarding_scaffolding", "template": template_key}),
                ),
            )
            # FTS index
            rolodex_conn.execute(
                "INSERT INTO rolodex_fts (entry_id, content, tags, category) "
                "VALUES (?, ?, ?, ?)",
                (entry_id, seed, json.dumps(["scaffolding", "seed", template_key]), "user_knowledge"),
            )
            result["seed_knowledge_ingested"] += 1
        except Exception:
            continue

    rolodex_conn.commit()

    result["skills_recommended"] = scaffolding.get("skills", [])
    result["tools_to_build"] = scaffolding.get("tools_to_build", [])

    return result
