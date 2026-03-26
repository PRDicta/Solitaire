"""
The Librarian — Query Expander

Transforms a raw user query into multiple search variants for broader recall.
Three capabilities:

1. **Synonym expansion**: Maps intent words ("struggling", "broke") to search
   terms that match how entries are actually stored ("error", "fix", "debug").

2. **Intent detection**: Identifies whether the query is experiential
   ("what was I struggling with"), factual ("how does X work"), or
   retrospective ("what did we decide about Y") and biases search accordingly.

3. **Category routing**: When experiential intent is detected, returns
   category hints so the searcher can weight entries tagged as corrections,
   friction, breakthroughs, etc.

No LLM calls — this is entirely heuristic and runs in microseconds.
"""
import os
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from .entity_extractor import EntityExtractor, ExtractedEntities


# ─── Project Aliases (loaded once at import) ────────────────────────────────
# Maps canonical names to user-facing aliases and vice versa.
# Loaded from project_aliases.yaml alongside this module.

_PROJECT_ALIASES: Dict[str, List[str]] = {}
_REVERSE_ALIASES: Dict[str, str] = {}  # alias_lower -> canonical_name

def _load_project_aliases():
    """Load project_aliases.yaml and build forward + reverse lookup."""
    global _PROJECT_ALIASES, _REVERSE_ALIASES
    yaml_path = os.path.join(os.path.dirname(__file__), "project_aliases.yaml")
    if not os.path.exists(yaml_path):
        return
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        aliases = data.get("aliases", {})
        _PROJECT_ALIASES = aliases
        for canonical, alias_list in aliases.items():
            for alias in alias_list:
                _REVERSE_ALIASES[alias.lower()] = canonical
            # Also map canonical to itself for completeness
            _REVERSE_ALIASES[canonical.lower()] = canonical
    except Exception:
        pass  # YAML parse failure is non-fatal

_load_project_aliases()


# ─── Intent Types ────────────────────────────────────────────────────────────

class QueryIntent:
    EXPERIENTIAL = "experiential"    # "what did I struggle with", "where did I get stuck"
    FACTUAL = "factual"              # "how does X work", "what is Y"
    RETROSPECTIVE = "retrospective"  # "what did we decide", "what was the plan"
    TEMPORAL = "temporal"            # "last session", "most recent session", "what did we just work on"
    EXPLORATORY = "exploratory"      # default — broad search


@dataclass
class ExpandedQuery:
    """Result of query expansion — multiple search variants + metadata."""
    original: str
    variants: List[str]              # All query variants to search
    intent: str = QueryIntent.EXPLORATORY
    category_bias: List[str] = field(default_factory=list)  # Categories to boost
    category_filter: List[str] = field(default_factory=list)  # Categories to restrict to (if strong signal)
    entities: Optional[ExtractedEntities] = None  # Phase 11: Extracted entities for re-ranking


# ─── Synonym / Intent Maps ──────────────────────────────────────────────────

# Maps user intent words → search terms that match stored entries
EXPERIENTIAL_SYNONYMS = {
    # Resolution / solution
    "solution": ["fix", "resolved", "approach", "implementation", "workaround", "answer"],
    "solved": ["fix", "solution", "resolved", "working", "answer"],
    "answer": ["solution", "fix", "resolved", "result"],
    # Frustration / struggle
    "struggling": ["error", "fix", "debug", "wrong", "failed", "broken", "issue"],
    "struggled": ["error", "fix", "debug", "wrong", "failed", "broken", "issue"],
    "stuck": ["error", "fix", "debug", "blocked", "issue", "workaround"],
    "frustrated": ["error", "fix", "wrong", "failed", "retry", "broken"],
    "confused": ["clarified", "corrected", "misunderstood", "actually"],
    "broke": ["error", "fix", "broken", "regression", "failed"],
    "failed": ["error", "fix", "failure", "wrong", "retry"],
    "wrong": ["corrected", "fix", "actually", "mistake", "wrong command"],
    "mistake": ["corrected", "fix", "actually", "wrong"],
    "problem": ["error", "fix", "issue", "debug", "workaround"],
    "issue": ["error", "fix", "bug", "debug", "problem"],
    "hard": ["difficult", "complex", "challenge", "workaround"],
    "difficult": ["complex", "challenge", "workaround", "hard"],
    # Success / breakthrough
    "breakthrough": ["solved", "fix", "working", "success", "resolved"],
    "solved": ["fix", "solution", "resolved", "working"],
    "figured out": ["solution", "resolved", "discovery", "realized"],
    "realized": ["corrected", "actually", "discovery", "understood"],
    "eureka": ["solved", "fix", "breakthrough", "working"],
    # Structure / organization
    "organizing": ["hierarchy", "structure", "consolidation", "grouping", "taxonomy", "architecture"],
    "organized": ["hierarchy", "structure", "consolidated", "grouped", "categorized"],
    "structure": ["hierarchy", "architecture", "organization", "layout", "design"],
    # Change of direction
    "pivoted": ["changed", "switched", "instead", "decided", "replaced"],
    "changed": ["switched", "replaced", "updated", "modified", "instead"],
    "switched": ["changed", "replaced", "migrated", "moved to"],
    "abandoned": ["replaced", "removed", "dropped", "instead"],
}

RETROSPECTIVE_SYNONYMS = {
    "decided": ["decision", "chose", "agreed", "plan", "went with"],
    "decision": ["decided", "chose", "agreed", "plan"],
    "chose": ["decision", "selected", "went with", "picked"],
    "plan": ["decided", "strategy", "approach", "design"],
    "agreed": ["decided", "consensus", "plan", "approach"],
}

# Domain / technical synonyms — bidirectional term bridging.
# These run on every query regardless of intent, catching cases where
# the query uses different jargon than the stored entry.
# Each key maps to terms that should also be searched.
# ─── Concept Map ─────────────────────────────────────────────────────────────
#
# Bridges high-level category queries to the vocabulary entries actually use.
# Technical synonyms map term → term. Concept maps map *intent* → *vocabulary
# domain*. When someone asks "What are Philip's formatting preferences?", the
# word "formatting" never appears in the stored entries. But entries about
# preferences use words like "instruction", "don't want", "no response
# requested", "standing rule". The concept map ensures those terms enter the
# search.
#
# Each key is a trigger: a high-level concept word the user might use.
# Each value is a list of corpus-level vocabulary that entries in that
# conceptual domain tend to contain. The expander generates an additional
# variant by combining the query's topic words with these domain terms.
#
# Design rule: these should be vocabulary patterns, not specific facts.
# "no response requested" belongs here because it's a vocabulary pattern
# for the "preferences" concept. "Philip Roy" does NOT belong here because
# it's a specific fact, not a vocabulary domain.

CONCEPT_MAP: Dict[str, List[str]] = {
    # User preferences and standing instructions
    "preference": ["instruction", "standing", "rule", "don't want", "never", "always", "requested", "drop", "stop"],
    "preferences": ["instruction", "standing", "rule", "don't want", "never", "always", "requested", "drop", "stop"],
    "formatting": ["response", "requested", "instruction", "verbal tic", "noise", "in-chat", "deliverable", "style"],
    "style": ["formatting", "voice", "tone", "preference", "instruction", "standing"],
    "rules": ["instruction", "standing", "hard rule", "never", "always", "must", "don't", "preference"],
    # Work patterns and productivity
    "work patterns": ["ADHD", "hyperfocus", "tired", "medicated", "desire", "motivation", "drag state", "focus"],
    "productivity": ["ADHD", "focus", "hyperfocus", "motivation", "routine", "workflow", "patterns"],
    "habits": ["routine", "patterns", "workflow", "preference", "tendency"],
    # Legal and licensing
    "licensing": ["license", "GPL", "AGPL", "dual-license", "commercial", "open source", "CLA", "contributor"],
    "legal": ["license", "licensing", "GPL", "AGPL", "CLA", "terms", "contributor", "intellectual property"],
    "ip": ["intellectual property", "proprietary", "private repo", "solitaire", "licensing", "patent", "copyright"],
    # Organization and structure
    "tier": ["production", "d100", "archive", "compressed", "emoji-optimized", "ultra", "folder", "hierarchy"],
    "tiers": ["production", "d100", "archive", "compressed", "emoji-optimized", "ultra", "folder", "hierarchy"],
    "organization": ["folder", "hierarchy", "structure", "tier", "category", "taxonomy", "consolidated"],
    # Roadmap and strategy
    "roadmap": ["pillars", "vision", "manifesto", "personas", "autonomy", "persistence", "partnership", "north star"],
    "strategy": ["roadmap", "plan", "direction", "vision", "positioning", "competitive", "thesis"],
    "vision": ["roadmap", "manifesto", "pillars", "north star", "thesis", "future"],
    # Competitive and market
    "competitive": ["gap", "positioning", "mem0", "zep", "letta", "cognitive layer", "market"],
    "market": ["competitive", "positioning", "distribution", "pricing", "customers", "gap"],
    # Content production
    "content": ["pipeline", "production prompt", "voice profile", "client", "routing", "workflow"],
    "pipeline": ["content", "routing", "production prompt", "voice profile", "prompt", "workflow"],
    "clients": ["neon health", "dream 100", "routing", "tier", "voice profile", "content"],
    # Project-level vocabulary bridges
    "symbiosis": ["Spec v1.2", "persona", "greeting", "resident knowledge", "indexed", "anticipatory", "clearinghouse", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6"],
    "solitaire": ["unified product", "private repo", "consolidated", "librarian", "token alchemy", "compression"],
    "commitment": ["eval harness", "behavioral signature", "shadow mode", "F1", "scorer", "detection"],
}

# Domain / technical synonyms — bidirectional term bridging.
TECHNICAL_SYNONYMS = {
    # DevOps / deployment
    "ci/cd": ["deployment pipeline", "continuous integration", "continuous deployment", "build and deploy", "github actions", "ci cd"],
    "ci cd": ["ci/cd", "deployment pipeline", "continuous integration", "continuous deployment", "build and deploy"],
    "deployment pipeline": ["ci/cd", "deploy", "shipping", "release", "github actions", "build pipeline"],
    "deployment": ["deploy", "release", "shipping", "ci/cd", "pipeline"],
    "deploy": ["deployment", "release", "shipping", "ci/cd", "pipeline"],
    "shipping code": ["deployment", "deploy", "release", "ci/cd", "pipeline"],
    "shipping": ["deployment", "deploy", "release", "ci/cd"],
    "github actions": ["ci/cd", "deployment pipeline", "workflow", "automation"],
    "docker": ["container", "containerization", "dockerfile", "image"],
    "container": ["docker", "containerization", "kubernetes", "k8s"],
    "kubernetes": ["k8s", "container orchestration", "docker", "cluster"],
    "k8s": ["kubernetes", "container orchestration", "cluster"],
    # Infrastructure
    "aws": ["amazon web services", "cloud", "ec2", "s3", "lambda", "ecs", "fargate"],
    "fargate": ["ecs", "aws", "serverless", "container"],
    "ecs": ["fargate", "aws", "container service", "docker"],
    "serverless": ["lambda", "functions", "faas", "cloud functions"],
    "lambda": ["serverless", "functions", "aws lambda"],
    # Databases
    "database": ["db", "storage", "data layer", "persistence"],
    "db": ["database", "storage", "data layer"],
    "sql": ["database", "relational", "postgres", "mysql", "sqlite"],
    "nosql": ["mongodb", "dynamodb", "document store", "non-relational"],
    "postgres": ["postgresql", "sql", "relational database"],
    "postgresql": ["postgres", "sql", "relational database"],
    "mongodb": ["mongo", "nosql", "document database"],
    "mongo": ["mongodb", "nosql", "document database"],
    "redis": ["cache", "key-value", "in-memory"],
    # Auth
    "authentication": ["auth", "login", "sign in", "identity", "jwt", "oauth"],
    "auth": ["authentication", "authorization", "login", "jwt", "oauth"],
    "jwt": ["json web token", "auth", "authentication", "token"],
    "oauth": ["authentication", "auth", "sso", "single sign-on"],
    "sso": ["single sign-on", "oauth", "authentication"],
    # Frontend
    "frontend": ["front-end", "ui", "client-side", "react", "browser"],
    "front-end": ["frontend", "ui", "client-side"],
    "backend": ["back-end", "server-side", "api", "server"],
    "back-end": ["backend", "server-side", "api"],
    "api": ["endpoint", "rest", "graphql", "backend", "interface"],
    "rest": ["restful", "api", "http", "endpoint"],
    "graphql": ["api", "query language", "schema"],
    # Testing
    "testing": ["tests", "test", "unit test", "integration test", "qa"],
    "tests": ["testing", "test suite", "unit test", "spec"],
    "unit test": ["testing", "jest", "pytest", "spec"],
    "integration test": ["testing", "e2e", "end-to-end"],
    "e2e": ["end-to-end", "integration test", "cypress", "playwright"],
    # Monitoring
    "monitoring": ["observability", "alerting", "logging", "metrics", "sentry", "datadog"],
    "observability": ["monitoring", "logging", "tracing", "metrics"],
    "logging": ["logs", "monitoring", "observability"],
    "alerting": ["alerts", "monitoring", "notifications", "pagerduty"],
    # Architecture
    "microservices": ["micro-services", "service-oriented", "distributed"],
    "monolith": ["monolithic", "single service"],
    "architecture": ["design", "structure", "system design", "patterns"],
    # Version control
    "git": ["version control", "source control", "github", "repository"],
    "github": ["git", "repository", "version control", "pull request"],
    "pull request": ["pr", "code review", "merge request"],
    "pr": ["pull request", "code review", "merge request"],
    # General dev
    "refactor": ["refactoring", "restructure", "clean up", "rewrite"],
    "refactoring": ["refactor", "restructure", "clean up"],
    "performance": ["optimization", "speed", "latency", "throughput"],
    "optimization": ["performance", "optimize", "speed", "efficiency"],
    "caching": ["cache", "redis", "memoization", "cdn"],
    "cache": ["caching", "redis", "memoization", "cdn"],
    # The Librarian — internal architecture terms
    "boot": ["boot_manifest", "manifest", "startup", "initialization", "boot system"],
    "boot system": ["boot_manifest", "manifest", "manifestmanager", "boot"],
    "manifest": ["boot_manifest", "manifestmanager", "manifestentry", "manifeststate", "boot"],
    "boot_manifest": ["manifest", "boot", "manifestmanager", "boot system"],
    "rolodex": ["rolodex_entries", "rolodex_fts", "memory", "knowledge graph", "database"],
    "persona": ["persona_registry", "personas", "disposition", "trait", "behavioral genome"],
    "personas": ["persona", "persona_registry", "disposition", "traits"],
    "identity graph": ["identity_nodes", "identity_signals", "identity_edges", "self-knowledge"],
    "reranker": ["rerank", "scoring", "ranking", "composite_score", "search ranking"],
    "query expander": ["query_expander", "synonym expansion", "intent detection", "search variants"],
    "enrichment": ["enrichment_scanner", "ingestion_queue", "background processing"],
    "compression": ["token alchemy", "yaml compression", "emoji compression", "cold warm hot"],
    "token alchemy": ["compression", "yaml compression", "emoji compression", "token reduction"],
    "voice profile": ["vp", "voice profiles", "speaker", "content production"],
    "fts": ["full-text search", "rolodex_fts", "keyword search", "fts5"],
    "full-text search": ["fts", "fts5", "keyword search", "rolodex_fts"],
}

# Phrases that signal experiential intent
EXPERIENTIAL_PHRASES = [
    r"struggl\w*", r"stuck\b", r"frustrat\w*", r"confus\w*",
    r"broke\b", r"broken\b", r"fail\w*", r"wrong\b",
    r"mistake\w*", r"problem\w*", r"issue\w*", r"hard\b",
    r"difficult\w*", r"couldn't\b", r"didn't work",
    r"went wrong", r"messed up", r"screwed up",
    r"pain\w*", r"annoying", r"headache",
]

# Phrases that signal retrospective intent
RETROSPECTIVE_PHRASES = [
    r"decid\w*", r"chose\b", r"chosen\b", r"agree\w*",
    r"plan\w*", r"approach\b", r"strategy\b",
    r"went with\b", r"settled on\b",
]

# Phrases that signal factual intent
FACTUAL_PHRASES = [
    r"how does\b", r"how do\b", r"what is\b", r"what are\b",
    r"explain\b", r"describe\b", r"definition\b",
    r"how .+ work", r"what .+ mean",
]

# Phrases that signal temporal intent (pure recency lookup, no semantic search)
# These are queries where the user wants the N most recent sessions, not content matches.
# Key distinction: "last session" = temporal, "last session about X" = retrospective.
TEMPORAL_PHRASES = [
    r"^(?:what was |what's )?(?:the |our |my )?(?:last|previous|most recent|latest) session\s*\??$",
    r"^(?:what was |what's )?(?:the |our |my )?(?:last|previous|most recent|latest) (?:few )?sessions\s*\??$",
    r"^(?:what did we (?:just )?(?:work on|do|cover|discuss))\s*\??$",
    r"^(?:what were we (?:just )?(?:working on|doing))\s*\??$",
    r"^(?:what happened )?(?:last session|previous session)\s*\??$",
    r"^(?:do you )?recall (?:the |our )?(?:last|previous|most recent) session\s*\??$",
    r"^(?:and )?(?:do you )?recall (?:the |our )?(?:last|previous|most recent) session",
    r"^(?:show|tell) me (?:the |our )?(?:last|previous|most recent) (?:few )?sessions?\s*\??$",
    # Broader temporal patterns -- queries about recent activity without naming "session"
    r"^(?:what was |what's )?(?:the )?(?:most recent|latest) (?:work|thing|stuff)\b",
    r"^(?:where did we (?:leave off|stop|end))\s*\??$",
    r"^(?:where were we|picking up where we left off)\s*\??$",
    r"^(?:what have we been (?:working on|doing|building))\s*\??$",
    r"^(?:what (?:was|were) (?:the )?last (?:thing|work|task))\b",
    r"^(?:catch me up|bring me up to speed|what did I miss)\s*\??$",
]

# Category biases per intent type
INTENT_CATEGORY_BIAS = {
    QueryIntent.EXPERIENTIAL: ["correction", "friction", "breakthrough", "pivot", "warning"],
    QueryIntent.RETROSPECTIVE: ["decision", "preference", "note"],
    QueryIntent.FACTUAL: ["definition", "implementation", "fact", "reference"],
    QueryIntent.TEMPORAL: [],  # Temporal queries bypass FTS entirely — no category bias needed
    QueryIntent.EXPLORATORY: [],  # No bias — search everything
}


# ─── Query Expander ──────────────────────────────────────────────────────────

class QueryExpander:
    """
    Expands a raw query into multiple search variants with intent metadata.
    Phase 11: Now includes entity extraction for exact-match boosting.
    Phase 12: Phrase preservation — multi-word expressions are preserved as
    FTS5 phrase queries alongside individual term variants.
    Purely heuristic — no LLM calls, runs in microseconds.
    """

    def __init__(self):
        self._entity_extractor = EntityExtractor()
        # Known multi-word phrases that should be preserved as FTS5 phrase
        # queries. These are distinctive expressions that lose meaning when
        # decomposed into individual terms. Order: longest first for greedy
        # matching.
        self._known_phrases = sorted([
            # User preferences / standing instructions
            "no response requested",
            "no diplomatic preamble",
            # Architecture
            "boot manifest",
            "manifest manager",
            "identity graph",
            "knowledge graph",
            "evaluation gate",
            "preflight evaluation",
            "query expander",
            "entity extractor",
            "token alchemy",
            "voice profile",
            "session residue",
            "experiential memory",
            "experiential encoding",
            "disposition filter",
            "behavioral genome",
            # Product / business
            "pilot checklist",
            "competitive gap",
            "personal cognitive layer",
            "dual license",
            "content pipeline",
            "production prompt",
            "dream 100",
            "neon health",
            "content ops",
            # Concepts
            "narrative identity",
            "north star",
            "model gravity",
            "compliance gravity",
        ], key=len, reverse=True)

    def expand(self, query: str) -> ExpandedQuery:
        """
        Expand a query into variants with intent detection.

        Returns an ExpandedQuery with:
        - variants: list of search strings (always includes the original)
        - intent: detected intent type
        - category_bias: categories to boost in results
        - entities: extracted entities for re-ranking
        """
        lower = query.lower().strip()

        # Step 1: Detect intent
        intent = self._detect_intent(lower)

        # Step 2: Extract entities from the query
        entities = self._entity_extractor.extract_from_query(query)

        # Step 2.5: Detect and preserve multi-word phrases (Phase 12).
        # These become FTS5 phrase queries (wrapped in double quotes)
        # that preserve phrase-level precision. Individual term variants
        # are still generated alongside, so recall broadens without
        # losing phrase precision.
        #
        # Fix: 2026-03-17. Addresses Q12 failure: "Philip formatting
        # preferences no response requested" was decomposed into
        # individual terms, missing the exact phrase match.
        detected_phrases = self._detect_phrases(lower)

        # Step 3: Generate synonym-expanded variants
        variants = self._expand_synonyms(lower, intent)

        # Step 3.5: Add phrase-preserved variants
        if detected_phrases:
            # Variant with FTS5 phrase syntax: "no response requested"
            for phrase in detected_phrases:
                fts_phrase = f'"{phrase}"'
                if fts_phrase not in variants:
                    variants.append(fts_phrase)
            # Also: remaining query terms + phrase (combined search)
            remaining = lower
            for phrase in detected_phrases:
                remaining = remaining.replace(phrase, "").strip()
            remaining = " ".join(remaining.split())  # normalize whitespace
            if remaining:
                for phrase in detected_phrases:
                    combined = f'{remaining} "{phrase}"'
                    if combined not in variants:
                        variants.append(combined)

        # Step 3.7: Concept-map expansion — bridge category-level queries
        # to corpus-level vocabulary. "formatting preferences" → also search
        # for "instruction", "requested", "verbal tic", etc.
        concept_terms = self._expand_concepts(lower)
        if concept_terms:
            topic_words = self._strip_filler(lower)
            if topic_words:
                # Variant: topic words + concept vocabulary (combined)
                concept_variant = f"{topic_words} {' '.join(concept_terms[:4])}"
                if concept_variant not in variants:
                    variants.append(concept_variant)
            # Variant: just the concept terms (broadest net)
            concept_only = " ".join(concept_terms[:5])
            if concept_only not in variants:
                variants.append(concept_only)

        # Step 3.9: Project alias expansion -- bridge user-facing names to
        # canonical internal names. "symbiosis adapter" -> also search "Spec v1.2".
        alias_variants = self._expand_project_aliases(lower)
        for av in alias_variants:
            if av not in variants:
                variants.append(av)

        # Step 4: Generate entity-focused variant
        # If entities were found, create a variant that's just the entities
        # This is what makes "Philip's analogy about books" findable
        if entities.all_entities:
            entity_variant = " ".join(entities.all_entities)
            if entity_variant not in variants:
                variants.append(entity_variant)
            # Also try entities + topic words (stripped query)
            stripped = self._strip_filler(lower)
            if stripped:
                entity_topic = " ".join(entities.all_entities) + " " + stripped
                if entity_topic not in variants:
                    variants.append(entity_topic)

        # Step 5: Generate a keyword-stripped variant (remove filler words)
        stripped = self._strip_filler(lower)
        if stripped and stripped != lower and stripped not in variants:
            variants.append(stripped)

        # Always include the original query
        if query not in variants:
            variants.insert(0, query)

        # Step 6: Get category bias based on intent
        category_bias = INTENT_CATEGORY_BIAS.get(intent, [])

        return ExpandedQuery(
            original=query,
            variants=variants[:10],  # Cap at 10 — entity + technical synonym variants need room
            intent=intent,
            category_bias=category_bias,
            entities=entities,
        )

    def _detect_phrases(self, query: str) -> List[str]:
        """Detect known multi-word phrases in the query.

        Returns phrases found, longest-first (greedy matching). Each phrase
        is returned as plain text; the caller wraps it in FTS5 quote syntax.

        Also detects user-quoted substrings: if the query contains text in
        double quotes, those are preserved as phrases regardless of whether
        they match the known phrase list.
        """
        found: List[str] = []

        # First: extract user-quoted phrases (explicit intent)
        quoted = re.findall(r'"([^"]+)"', query)
        for q in quoted:
            if q.strip() and q.strip() not in found:
                found.append(q.strip())

        # Second: match known phrases (greedy, longest first)
        remaining = query.lower()
        for phrase in self._known_phrases:
            if phrase in remaining:
                found.append(phrase)
                # Remove matched phrase to prevent sub-phrase overlap
                remaining = remaining.replace(phrase, " ")

        return found

    def _expand_project_aliases(self, query: str) -> List[str]:
        """Bridge user-facing project names to canonical internal names.

        When the query contains a known alias (e.g., "symbiosis adapter"),
        generates variants using the canonical name (e.g., "Spec v1.2")
        and vice versa. Uses longest-match-first to handle multi-word aliases.
        """
        if not _REVERSE_ALIASES:
            return []

        variants: List[str] = []
        query_lower = query.lower()

        # Sort aliases by length (longest first) for greedy matching
        sorted_aliases = sorted(_REVERSE_ALIASES.keys(), key=len, reverse=True)

        for alias in sorted_aliases:
            if alias in query_lower:
                canonical = _REVERSE_ALIASES[alias]
                # If query uses an alias, add variant with canonical name
                if alias != canonical.lower():
                    replaced = query_lower.replace(alias, canonical.lower())
                    if replaced not in variants:
                        variants.append(replaced)
                # Also add variant with just the canonical name
                if canonical.lower() not in variants and canonical.lower() != query_lower:
                    variants.append(canonical.lower())
                # Add variants with other aliases of the same canonical name
                if canonical in _PROJECT_ALIASES:
                    for other_alias in _PROJECT_ALIASES[canonical][:2]:
                        if other_alias.lower() != alias and other_alias.lower() not in variants:
                            variants.append(other_alias.lower())
                break  # One match per query to avoid explosion

        return variants[:3]  # Cap to keep variant count manageable

    def _expand_concepts(self, query: str) -> List[str]:
        """Bridge high-level category terms to corpus-level vocabulary.

        Scans the query for concept-map triggers and returns the union of
        their vocabulary domains. Uses word-boundary matching to avoid
        false positives (e.g., "tier" in "frontier" should not trigger).
        """
        terms: Set[str] = set()
        for trigger, vocabulary in CONCEPT_MAP.items():
            # Word-boundary match, case-insensitive
            if re.search(r'\b' + re.escape(trigger) + r'\b', query, re.IGNORECASE):
                terms.update(vocabulary)
        # Remove any terms that are already in the query (no point re-searching)
        query_words = set(query.lower().split())
        terms = {t for t in terms if t.lower() not in query_words}
        return sorted(terms)[:8]  # Cap to keep variants manageable

    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query from phrase patterns."""
        # Check temporal first (most specific — pure recency lookups)
        # These are whole-message patterns that match queries with no topical qualifier.
        for pattern in TEMPORAL_PHRASES:
            if re.match(pattern, query.strip(), re.IGNORECASE):
                return QueryIntent.TEMPORAL

        # Check experiential (strongest content signal)
        for pattern in EXPERIENTIAL_PHRASES:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryIntent.EXPERIENTIAL

        # Check retrospective
        for pattern in RETROSPECTIVE_PHRASES:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryIntent.RETROSPECTIVE

        # Check factual
        for pattern in FACTUAL_PHRASES:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryIntent.FACTUAL

        return QueryIntent.EXPLORATORY

    def _expand_synonyms(self, query: str, intent: str) -> List[str]:
        """Generate synonym-expanded query variants."""
        variants = [query]

        # Choose synonym map based on intent
        if intent == QueryIntent.EXPERIENTIAL:
            syn_map = EXPERIENTIAL_SYNONYMS
        elif intent == QueryIntent.RETROSPECTIVE:
            syn_map = RETROSPECTIVE_SYNONYMS
        else:
            syn_map = {**EXPERIENTIAL_SYNONYMS, **RETROSPECTIVE_SYNONYMS}

        # Find matching synonym keys in the query
        expansions: Set[str] = set()
        for trigger, replacements in syn_map.items():
            if trigger in query:
                expansions.update(replacements)

        # Build expanded variant: original topic words + synonym terms
        if expansions:
            # Extract the "topic" part of the query (remove intent words)
            topic_words = self._extract_topic_words(query, syn_map.keys())
            if topic_words:
                # Variant 1: topic + top synonyms
                top_syns = sorted(expansions)[:4]
                variants.append(f"{topic_words} {' '.join(top_syns)}")
                # Variant 2: just the synonyms (broader search)
                variants.append(" ".join(sorted(expansions)[:5]))
            else:
                # No clear topic — just use synonyms
                variants.append(" ".join(sorted(expansions)[:5]))

        # ── Technical synonym expansion (runs on every query) ──
        # Bridges domain jargon: "CI/CD" ↔ "deployment pipeline", etc.
        # Uses word-boundary matching to avoid false positives
        # ("pr" in "preferences" should NOT trigger "pull request").
        tech_expansions: Set[str] = set()
        matched_triggers: List[str] = []
        for trigger, replacements in TECHNICAL_SYNONYMS.items():
            if self._match_technical_term(trigger, query):
                tech_expansions.update(replacements)
                matched_triggers.append(trigger)

        if tech_expansions:
            # Remove matched triggers from query to isolate remaining context
            topic_words = self._strip_matched_triggers(query, matched_triggers)
            top_tech = sorted(tech_expansions, key=len)[:4]  # Prefer shorter terms
            if topic_words:
                variants.append(f"{topic_words} {' '.join(top_tech)}")
            # Also add a pure technical synonym variant
            variants.append(" ".join(sorted(tech_expansions, key=len)[:5]))

        return variants

    @staticmethod
    def _match_technical_term(trigger: str, query: str) -> bool:
        """Check if a technical trigger matches as a whole word/phrase in the query.

        Handles special characters like '/' in 'ci/cd' by escaping them
        for regex, then requiring word boundaries (or string edges).
        """
        escaped = re.escape(trigger)
        # Use word boundaries, but also allow / as a boundary character
        pattern = r'(?:^|(?<=\s)|(?<=/))' + escaped + r'(?:$|(?=\s)|(?=/))'
        return bool(re.search(pattern, query, re.IGNORECASE))

    @staticmethod
    def _strip_matched_triggers(query: str, triggers: List[str]) -> str:
        """Remove matched technical triggers from query to isolate remaining context."""
        result = query
        # Sort by length descending to strip longest phrases first
        for trigger in sorted(triggers, key=len, reverse=True):
            escaped = re.escape(trigger)
            result = re.sub(escaped, " ", result, flags=re.IGNORECASE)
        return " ".join(result.split()).strip()

    def _extract_topic_words(self, query: str, intent_words) -> str:
        """Extract the meaningful topic words, stripping intent/filler words."""
        words = query.split()
        filler = set(intent_words) | _QUERY_FILLER
        topic = [w for w in words if w.lower() not in filler]
        return " ".join(topic).strip()

    def _strip_filler(self, query: str) -> str:
        """Strip common filler words to get a tighter keyword query.

        Preserves compound terms containing '/' (like 'ci/cd') as single tokens
        so they aren't broken apart during word-level filtering.
        """
        words = query.split()
        stripped = [w for w in words if w.lower() not in _QUERY_FILLER or "/" in w]
        return " ".join(stripped).strip()


# Common filler words to remove for tighter keyword searches
_QUERY_FILLER = {
    "i", "me", "my", "we", "our", "the", "a", "an", "is", "was", "were",
    "am", "are", "been", "be", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "may", "might",
    "what", "which", "where", "when", "how", "who", "whom",
    "that", "this", "these", "those", "it", "its",
    "with", "about", "for", "from", "in", "on", "at", "to", "of", "by",
    "and", "or", "but", "not", "so", "if", "then",
    "there", "here", "just", "also", "very", "really", "quite",
}
