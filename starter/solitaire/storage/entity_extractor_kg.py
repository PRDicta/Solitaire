"""
The Librarian — Entity Extractor for Knowledge Graph

Bridge module between the retrieval-layer EntityExtractor and the
storage-layer KnowledgeGraph. Transforms raw extracted entities into
typed (name, entity_type) pairs suitable for graph insertion.

No LLM calls — entirely heuristic, runs in microseconds.
"""
import re
from typing import List, Tuple, Dict
from ..retrieval.entity_extractor import EntityExtractor, _KNOWN_PROPER_NOUNS, _FILE_PATH, _TECHNICAL_TERM


# ─── Entity Type Inference ───────────────────────────────────────────────────

# Known type mappings — authoritative overrides for common entities.
# Mirrors knowledge_graph._TYPE_HINTS but is the canonical source for
# the extraction pipeline. knowledge_graph.py uses _infer_entity_type()
# as a fallback for entities that bypass this module.
_ENTITY_TYPE_MAP: Dict[str, str] = {
    # People
    "owner": "person",
    # Organizations
    "mycompany": "org", "anthropic": "org",
    # Projects
    "librarian": "project", "example system": "project",
    "token alchemy": "project",
    "example-project": "project", "example project": "project",
    "rolodex": "project",
    # Tools & tech
    "sqlite": "tool", "fts5": "tool",
    "claude": "tool", "cowork": "tool",
    "haiku": "tool", "sonnet": "tool", "opus": "tool", "voyage": "tool",
    "neo4j": "tool", "graphiti": "tool",
    "huggingface": "tool", "github": "tool",
    "python": "tool", "fastapi": "tool", "pydantic": "tool",
    "numpy": "tool",
}

# Heuristic type signals from surrounding context
_CONTEXT_TYPE_SIGNALS = {
    "person": [
        r"\b(?:said|told|asked|suggested|mentioned|decided|prefers|wants)\b",
        r"\b(?:his|her|their)\s+\w+",
        r"'s\s+(?:idea|approach|suggestion|preference|decision|opinion)",
    ],
    "project": [
        r"\b(?:built|shipped|deployed|implemented|released|launched)\b",
        r"\b(?:repo|repository|codebase|module|package|system)\b",
        r"\b(?:v\d|version\s+\d|roadmap|milestone)\b",
    ],
    "tool": [
        r"\b(?:install|import|require|dependency|library|framework|SDK|API)\b",
        r"\b(?:plugin|extension|package|module|driver)\b",
    ],
    "org": [
        r"\b(?:company|organization|team|group|corp|inc|ltd)\b",
        r"\b(?:hired|works\s+at|employed|founded|startup)\b",
    ],
    "concept": [
        r"\b(?:pattern|principle|strategy|approach|architecture|design)\b",
        r"\b(?:idea|concept|theory|methodology|paradigm)\b",
    ],
}


def _infer_type_from_name(name: str) -> str:
    """Infer entity type from the name itself."""
    name_lower = name.lower().strip()

    # Direct lookup
    if name_lower in _ENTITY_TYPE_MAP:
        return _ENTITY_TYPE_MAP[name_lower]

    # Partial match (for compound names like "The Librarian")
    for known, etype in _ENTITY_TYPE_MAP.items():
        if known in name_lower or name_lower in known:
            return etype

    # File path heuristic
    if _FILE_PATH.search(name):
        return "file"
    if '/' in name or '\\' in name:
        return "file"

    # Technical term heuristic (snake_case, camelCase)
    if _TECHNICAL_TERM.search(name):
        return "tool"

    return ""  # Unknown — will fall through to context inference


def _infer_type_from_context(name: str, content: str) -> str:
    """Infer entity type from surrounding content."""
    # Build a window around the entity mention
    name_lower = name.lower()
    content_lower = content.lower()
    pos = content_lower.find(name_lower)
    if pos < 0:
        return "concept"  # Default fallback

    # Extract a window of ~200 chars around the mention
    start = max(0, pos - 100)
    end = min(len(content), pos + len(name) + 100)
    window = content_lower[start:end]

    # Score each type based on signal matches
    scores: Dict[str, int] = {}
    for etype, patterns in _CONTEXT_TYPE_SIGNALS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, window):
                score += 1
        if score > 0:
            scores[etype] = score

    if scores:
        return max(scores, key=scores.get)

    return "concept"  # Safe default


# ─── Multi-word Entity Detection ────────────────────────────────────────────

# Patterns for compound entity names (e.g., "The Librarian", "Token Alchemy")
_COMPOUND_ENTITY_PATTERNS = [
    # "The X" where X is capitalized
    re.compile(r"\b(The\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"),
    # Two or more capitalized words in sequence (potential project/org names)
    re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"),
]

# Compound names to exclude (common phrases, not entities)
_COMPOUND_EXCLUSIONS = {
    "the way", "the first", "the last", "the next", "the best",
    "the same", "the new", "the old", "the other", "the most",
    "the user", "the model", "the system", "the main", "the whole",
    "no one", "each other", "let me",
}


def _extract_compound_entities(content: str) -> List[str]:
    """Extract multi-word entity names from content."""
    compounds = []
    seen = set()

    for pattern in _COMPOUND_ENTITY_PATTERNS:
        for match in pattern.finditer(content):
            name = match.group(1).strip()
            name_lower = name.lower()

            # Skip exclusions
            if name_lower in _COMPOUND_EXCLUSIONS:
                continue
            # Skip very short compounds
            if len(name) < 4:
                continue
            # Skip if already seen
            if name_lower in seen:
                continue

            seen.add(name_lower)
            compounds.append(name)

    return compounds


# ─── Main Extraction Function ───────────────────────────────────────────────

_extractor = EntityExtractor()


def extract_entities_for_graph(content: str) -> Dict:
    """
    Extract entities from content for knowledge graph insertion.

    Returns:
        {
            "entities": [(name, entity_type), ...],
            "attribution": "user" | "assistant" | "",
        }

    Each entity is a (name, type) tuple where type is one of:
    person, project, tool, org, file, concept
    """
    # Use the existing extractor for raw entity detection
    extracted = _extractor.extract_from_content(content)

    entities: List[Tuple[str, str]] = []
    seen_lower: set = set()

    # 0. Pre-pass: extract alphanumeric tokens the base extractor misses
    #    (e.g., "Neo4j" gets stripped of digits by the base extractor)
    _ALPHANUM_ENTITY = re.compile(
        r"\b([A-Z][a-zA-Z]*\d+[a-zA-Z]*)\b"  # Capitalized word with embedded digits
    )
    for match in _ALPHANUM_ENTITY.finditer(content):
        token = match.group(1)
        token_lower = token.lower()
        if token_lower not in seen_lower:
            seen_lower.add(token_lower)
            etype = _infer_type_from_name(token)
            if not etype:
                etype = _infer_type_from_context(token, content)
            entities.append((token, etype))

    # 1. Process proper nouns
    for noun in extracted.proper_nouns:
        lower = noun.lower()
        if lower in seen_lower:
            continue
        # Skip if this is a digit-stripped version of an already-seen entity
        # e.g., "Neoj" when "Neo4j" is already extracted
        # Check: remove all digits from existing entities and see if they match
        stripped = re.sub(r'\d', '', lower)
        if stripped != lower:
            # This entity has no digits but there might be a digit-containing version
            pass  # It's the original, keep it
        else:
            # Check if any existing entity, with digits removed, equals this
            if any(re.sub(r'\d', '', ex) == lower and ex != lower for ex in seen_lower):
                continue
        seen_lower.add(lower)

        etype = _infer_type_from_name(noun)
        if not etype:
            etype = _infer_type_from_context(noun, content)
        entities.append((noun, etype))

    # 2. Process file paths → type "file"
    for fp in extracted.file_paths:
        fp_lower = fp.lower()
        if fp_lower in seen_lower:
            continue
        seen_lower.add(fp_lower)
        entities.append((fp, "file"))

    # 3. Process technical terms → type "tool" (code identifiers)
    for term in extracted.technical_terms:
        term_lower = term.lower()
        if term_lower in seen_lower:
            continue
        seen_lower.add(term_lower)
        entities.append((term, "tool"))

    # 4. Extract compound entities (multi-word names)
    #    These get priority — if "Token Alchemy" is found, remove the
    #    individual "Token" and "Alchemy" entries and replace with the compound.
    compounds = _extract_compound_entities(content)
    for compound in compounds:
        compound_lower = compound.lower()
        if compound_lower in seen_lower:
            continue

        etype = _infer_type_from_name(compound)
        if not etype:
            etype = _infer_type_from_context(compound, content)

        # Remove individual word entities that are subsumed by this compound
        compound_words = set(compound_lower.split())
        entities = [
            (n, t) for (n, t) in entities
            if n.lower() not in compound_words
        ]
        # Update seen set
        for w in compound_words:
            seen_lower.discard(w)
        seen_lower.add(compound_lower)
        entities.append((compound, etype))

    return {
        "entities": entities,
        "attribution": extracted.attribution,
    }
