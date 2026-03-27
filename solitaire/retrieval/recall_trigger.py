"""
The Librarian — Recall Trigger

Proactive context surfacing layer. Analyzes a user message and generates
targeted recall queries based on detected signals:

1. Entity triggers — proper nouns, file paths, technical terms
2. Topic triggers — matches against the DB topic table
3. Back-reference triggers — "that issue", "last time", "before"
4. Project triggers — matches against known project clusters

Returns a list of RecallQuery objects, each with a query string,
priority, and signal source. The caller fires these as parallel
recalls and merges the results.

No LLM calls — entirely heuristic, runs in milliseconds.
"""
import re
import sqlite3
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
from .entity_extractor import EntityExtractor, ExtractedEntities


# ─── Data Types ──────────────────────────────────────────────────────────────

@dataclass
class RecallQuery:
    """A single recall query to fire."""
    query: str
    priority: float       # 0.0-1.0, higher = more likely relevant
    signal: str           # What triggered this query
    fresh: bool = False   # Whether to use --fresh mode (recency bias)


@dataclass
class TriggerResult:
    """Result of trigger analysis on a user message."""
    queries: List[RecallQuery] = field(default_factory=list)
    entities: Optional[ExtractedEntities] = None
    signals_detected: List[str] = field(default_factory=list)
    skip_recall: bool = False  # True if message needs no context (greetings, acks)
    temporal_only: bool = False  # True if this is a pure temporal query (bypass FTS, use session digests)
    matched_topics: List[Tuple[str, int, float]] = field(default_factory=list)    # (label, entry_count, confidence)
    matched_projects: List[Tuple[str, float]] = field(default_factory=list)       # (name, confidence)


# ─── Back-Reference Patterns ────────────────────────────────────────────────

_BACK_REFERENCE_PATTERNS = [
    # Explicit temporal references
    (r'\blast\s+(?:time|session|conversation)\b', 'temporal_back_ref', 0.9),
    (r'\bbefore\b.*\b(?:we|you|I)\b', 'temporal_back_ref', 0.7),
    (r'\bprevious(?:ly)?\b', 'temporal_back_ref', 0.7),
    (r'\bearlier\b', 'temporal_back_ref', 0.6),
    (r'\bremember\s+(?:when|that|the)\b', 'explicit_recall', 0.9),
    (r'\byou\s+(?:said|mentioned|suggested|told)\b', 'assistant_back_ref', 0.8),
    (r'\bwe\s+(?:discussed|decided|agreed|talked\s+about)\b', 'discussion_back_ref', 0.9),
    (r'\bwhat\s+(?:was|were|did)\b.*\b(?:we|you|I)\b', 'question_back_ref', 0.8),

    # Demonstrative references ("that thing", "the issue")
    (r'\bthat\s+(?:issue|bug|problem|error|thing|idea|approach|design|feature)\b', 'demonstrative_ref', 0.7),
    (r'\bthe\s+(?:issue|bug|problem|error|thing|idea|approach|design|feature)\s+(?:with|from|about|in)\b', 'demonstrative_ref', 0.7),

    # Continuation signals
    (r'\bpick\s+(?:up|back\s+up)\b', 'continuation', 0.8),
    (r'\bcontinue\s+(?:with|on|from|where)\b', 'continuation', 0.8),
    (r'\bwhere\s+(?:we|I)\s+left\s+off\b', 'continuation', 0.9),
    (r"\blet'?s?\s+(?:keep|resume|get\s+back\s+to)\b", 'continuation', 0.8),
]

# ─── Skip Patterns (messages that need no recall) ───────────────────────────

_SKIP_PATTERNS = [
    r'^(?:hi|hey|hello|morning|afternoon|evening|yo)[\s!.]*$',
    r'^(?:ok|okay|sure|thanks|thank\s+you|got\s+it|sounds\s+good|great|nice|cool|yep|yes|no|nah)[\s!.]*$',
    r'^(?:let\'?s?\s+(?:go|do\s+it|proceed|start|rock|roll))[\s!.]*$',
]

# ─── Topic Cache ─────────────────────────────────────────────────────────────

class _TopicCache:
    """Lightweight cache of DB topics for fast matching."""

    def __init__(self):
        self._labels: Dict[str, int] = {}     # label → entry_count
        self._keywords: Dict[str, str] = {}   # keyword → topic_label
        self._loaded = False

    def load(self, conn: sqlite3.Connection):
        """Load topics from DB into memory."""
        if self._loaded:
            return
        try:
            rows = conn.execute(
                "SELECT label, entry_count FROM topics WHERE entry_count >= 1 ORDER BY entry_count DESC"
            ).fetchall()
            for row in rows:
                label = row["label"] if isinstance(row, sqlite3.Row) else row[0]
                count = row["entry_count"] if isinstance(row, sqlite3.Row) else row[1]
                self._labels[label] = count
                # Index individual keywords from the label
                for word in label.split():
                    word_clean = word.strip().lower()
                    if len(word_clean) >= 3:
                        # Keep the topic with the highest entry count for each keyword
                        if word_clean not in self._keywords or count > self._labels.get(self._keywords[word_clean], 0):
                            self._keywords[word_clean] = label
        except Exception:
            pass
        self._loaded = True

    def match(self, text: str) -> List[Tuple[str, int, float]]:
        """Match text against cached topics. Returns (label, entry_count, confidence)."""
        if not self._loaded:
            return []

        text_lower = text.lower()
        text_words = set(text_lower.split())
        matches = []
        seen_labels = set()

        # Direct keyword matching
        for word in text_words:
            word_clean = re.sub(r'[^a-z0-9_\-]', '', word)
            if word_clean in self._keywords:
                label = self._keywords[word_clean]
                if label not in seen_labels:
                    seen_labels.add(label)
                    count = self._labels[label]
                    # Confidence based on entry count (more entries = more confident)
                    confidence = min(0.9, 0.4 + (count / 100))
                    matches.append((label, count, confidence))

        # Sort by confidence descending
        matches.sort(key=lambda m: m[2], reverse=True)
        return matches[:5]  # Cap at 5 topic matches


# ─── Project Cluster Cache ───────────────────────────────────────────────────

class _ProjectCache:
    """Cache of project clusters for name matching."""

    def __init__(self):
        self._projects: Dict[str, Dict] = {}  # name → {id, keywords, entry_count}
        self._loaded = False

    def load(self, conn: sqlite3.Connection):
        if self._loaded:
            return
        try:
            rows = conn.execute(
                "SELECT id, name, keywords, entry_count FROM project_clusters WHERE entry_count >= 1"
            ).fetchall()
            for row in rows:
                name = row["name"] if isinstance(row, sqlite3.Row) else row[1]
                keywords_raw = row["keywords"] if isinstance(row, sqlite3.Row) else row[2]
                count = row["entry_count"] if isinstance(row, sqlite3.Row) else row[3]
                keywords = []
                if keywords_raw:
                    try:
                        import json
                        keywords = json.loads(keywords_raw) if isinstance(keywords_raw, str) else keywords_raw
                    except Exception:
                        keywords = keywords_raw.split(',') if isinstance(keywords_raw, str) else []
                self._projects[name.lower()] = {
                    "name": name,
                    "keywords": [k.lower().strip() for k in keywords if k],
                    "entry_count": count,
                }
        except Exception:
            pass
        self._loaded = True

    def match(self, text: str) -> List[Tuple[str, float]]:
        """Match text against project names/keywords. Returns (project_name, confidence)."""
        if not self._loaded:
            return []
        text_lower = text.lower()
        matches = []
        for key, info in self._projects.items():
            # Check project name
            if key in text_lower:
                matches.append((info["name"], 0.9))
                continue
            # Check keywords
            for kw in info["keywords"]:
                if kw and kw in text_lower:
                    matches.append((info["name"], 0.7))
                    break
        return matches[:3]


# ─── Main Trigger Class ─────────────────────────────────────────────────────

class RecallTrigger:
    """
    Analyzes user messages and generates proactive recall queries.

    Usage:
        trigger = RecallTrigger(conn)
        result = trigger.analyze("What was that client issue we discussed?")
        for q in result.queries:
            # Fire recall with q.query
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._entity_extractor = EntityExtractor()
        self._topic_cache = _TopicCache()
        self._project_cache = _ProjectCache()
        self._topic_cache.load(conn)
        self._project_cache.load(conn)

    def analyze(self, message: str) -> TriggerResult:
        """
        Analyze a user message and generate recall queries.

        Returns TriggerResult with prioritized queries and signal metadata.
        """
        result = TriggerResult()

        # Check skip patterns first
        message_stripped = message.strip()
        for pattern in _SKIP_PATTERNS:
            if re.match(pattern, message_stripped, re.IGNORECASE):
                result.skip_recall = True
                return result

        # Very short messages: single words under 3 chars are noise.
        # Single meaningful words like "Solitaire" or "Librarian" should
        # still trigger recall (skip patterns already catch greetings/acks).
        words = message_stripped.split()
        if len(words) == 1 and len(words[0]) < 3:
            result.skip_recall = True
            return result

        # ── Signal 0: Temporal intent detection ──────────────────────────
        # Pure recency queries ("last session", "most recent session", "what
        # did we just work on") should bypass FTS entirely and route to the
        # temporal query path in auto-recall. We detect these early and set
        # temporal_only=True so the caller knows to use session digests
        # instead of keyword search.
        from solitaire.retrieval.query_expander import QueryExpander, QueryIntent
        _temporal_expander = QueryExpander()
        _temporal_check = _temporal_expander._detect_intent(message_stripped.lower())
        if _temporal_check == QueryIntent.TEMPORAL:
            result.temporal_only = True
            result.signals_detected.append("temporal_intent")
            # Return with no queries -- the caller handles temporal lookups directly
            return result

        # ── Signal 1: Entity extraction ──────────────────────────────────
        entities = self._entity_extractor.extract_from_query(message)
        result.entities = entities

        if entities.proper_nouns:
            result.signals_detected.append("entities")
            # Generate entity-focused query
            entity_query = " ".join(entities.proper_nouns[:3])
            result.queries.append(RecallQuery(
                query=entity_query,
                priority=0.8,
                signal="entity_match",
            ))

        if entities.technical_terms:
            result.signals_detected.append("technical_terms")
            for term in entities.technical_terms[:2]:
                result.queries.append(RecallQuery(
                    query=term,
                    priority=0.7,
                    signal="technical_term",
                ))

        # ── Signal 2: Back-reference detection ───────────────────────────
        for pattern, signal_type, priority in _BACK_REFERENCE_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                result.signals_detected.append(signal_type)

                if signal_type in ('temporal_back_ref', 'continuation'):
                    # For temporal refs, recall recent context
                    # Extract the topic words around the back-reference
                    topic_words = self._extract_topic_around_pattern(message, pattern)
                    if topic_words:
                        result.queries.append(RecallQuery(
                            query=topic_words,
                            priority=priority,
                            signal=signal_type,
                            fresh=True,
                        ))
                    else:
                        # Generic recent context recall
                        result.queries.append(RecallQuery(
                            query=message,
                            priority=priority * 0.8,
                            signal=signal_type,
                            fresh=True,
                        ))
                elif signal_type in ('explicit_recall', 'discussion_back_ref', 'question_back_ref'):
                    # All back-reference types should bias toward recent content.
                    # "What were we working on?" wants today's entries, not last week's.
                    result.queries.append(RecallQuery(
                        query=message,
                        priority=priority,
                        signal=signal_type,
                        fresh=True,
                    ))
                elif signal_type == 'demonstrative_ref':
                    # "that issue" — extract the noun and search
                    match = re.search(pattern, message, re.IGNORECASE)
                    if match:
                        result.queries.append(RecallQuery(
                            query=match.group(0) + " " + self._extract_context_words(message),
                            priority=priority,
                            signal=signal_type,
                            fresh=True,
                        ))
                break  # One back-reference signal is enough

        # ── Signal 3: Topic matching ─────────────────────────────────────
        topic_matches = self._topic_cache.match(message)
        if topic_matches:
            result.signals_detected.append("topic_match")
            result.matched_topics = topic_matches
            for label, count, confidence in topic_matches[:2]:
                result.queries.append(RecallQuery(
                    query=label,
                    priority=confidence,
                    signal=f"topic:{label}",
                ))

        # ── Signal 4: Project matching ───────────────────────────────────
        project_matches = self._project_cache.match(message)
        if project_matches:
            result.signals_detected.append("project_match")
            result.matched_projects = project_matches
            for name, confidence in project_matches[:2]:
                result.queries.append(RecallQuery(
                    query=name,
                    priority=confidence,
                    signal=f"project:{name}",
                ))

        # ── Signal 5: Content-heavy messages ─────────────────────────────
        # If the message is long (user explaining something), extract
        # key phrases for context recall
        word_count = len(message.split())
        if word_count >= 20 and not result.queries:
            result.signals_detected.append("content_heavy")
            # Extract the most distinctive 3-4 words
            key_words = self._extract_key_words(message)
            if key_words:
                result.queries.append(RecallQuery(
                    query=key_words,
                    priority=0.5,
                    signal="content_key_words",
                ))

        # ── Dedup and sort ───────────────────────────────────────────────
        result.queries = self._dedup_queries(result.queries)
        result.queries.sort(key=lambda q: q.priority, reverse=True)

        # Cap at 4 queries — more than that burns budget for diminishing returns
        result.queries = result.queries[:4]

        return result

    def _extract_topic_around_pattern(self, message: str, pattern: str) -> str:
        """Extract meaningful topic words near a back-reference pattern."""
        # Remove the pattern match and stop words to get the topic
        cleaned = re.sub(pattern, '', message, flags=re.IGNORECASE)
        return self._extract_context_words(cleaned)

    def _extract_context_words(self, text: str) -> str:
        """Extract meaningful content words from text."""
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'about', 'into', 'through', 'during', 'before', 'after',
            'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both',
            'that', 'this', 'these', 'those', 'it', 'its', 'we', 'you',
            'i', 'me', 'my', 'your', 'our', 'they', 'them', 'their',
            'what', 'which', 'who', 'whom', 'how', 'when', 'where', 'why',
            'if', 'then', 'also', 'just', 'very', 'really',
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        meaningful = [w for w in words if w not in stop_words]
        return ' '.join(meaningful[:6])

    def _extract_key_words(self, message: str) -> str:
        """Extract the most distinctive words from a long message."""
        context = self._extract_context_words(message)
        # Return first 4 meaningful words
        words = context.split()[:4]
        return ' '.join(words) if words else ''

    def _dedup_queries(self, queries: List[RecallQuery]) -> List[RecallQuery]:
        """Remove near-duplicate queries, keeping the highest priority."""
        seen_normalized = {}
        deduped = []
        for q in queries:
            normalized = ' '.join(sorted(q.query.lower().split()))
            if normalized not in seen_normalized:
                seen_normalized[normalized] = q
                deduped.append(q)
            elif q.priority > seen_normalized[normalized].priority:
                # Replace with higher-priority version
                deduped = [x for x in deduped if x is not seen_normalized[normalized]]
                deduped.append(q)
                seen_normalized[normalized] = q
        return deduped
