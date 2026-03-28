"""
The Librarian — Recall Orchestrator

Encapsulates the full recall pipeline: trigger analysis, graph expansion,
query expansion, variant firing, reranking, and tiered confidence waterfall.

Previously this logic lived only in the CLI's cmd_auto_recall (850+ lines
of inline orchestration). This class extracts it into a reusable component
that both the CLI and Solitaire's engine can call.

The pipeline:
1. RecallTrigger.analyze() generates targeted queries from the message
2. Knowledge graph expansion adds neighbor entities
3. QueryExpander produces variants per query
4. FTS fires variants, collects candidates with position-weighted scoring
5. Reranker scores the merged pool on 6 signals
6. Tiered confidence waterfall widens search if results are weak:
   - Tier 1: Standard pipeline (above)
   - Tier 2: Broadened FTS with category-targeted queries
   - Tier 3: Force-injection of corrections, user_facts, user_knowledge (oldest-first)
"""

import sqlite3
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

from .recall_trigger import RecallTrigger, RecallQuery, TriggerResult
from .query_expander import QueryExpander
from .reranker import Reranker, ScoredCandidate
from .recall_confidence import assess_confidence
from .entity_extractor import ExtractedEntities, load_proper_nouns_from_db


@dataclass
class RecallResult:
    """Output of the recall orchestrator."""
    entries: List[Any]                      # RolodexEntry objects
    scored: List[ScoredCandidate]           # Full scored candidates
    recall_tier: int = 1                    # Which tier resolved the query
    candidates_total: int = 0               # Total candidates across all tiers
    queries_fired: List[Dict] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    dominant_intent: str = "exploratory"
    tier1_confidence: Optional[Dict] = None
    tier2_confidence: Optional[Dict] = None
    graph_expansion_count: int = 0
    matched_topics: List[Dict] = field(default_factory=list)    # [{label, entry_count, confidence}]
    matched_projects: List[Dict] = field(default_factory=list)  # [{name, confidence}]


class RecallOrchestrator:
    """
    Full recall pipeline: trigger -> expand -> fire -> rerank -> tiered waterfall.

    Dependencies:
        conn: sqlite3 connection to the rolodex database
        rolodex: Rolodex instance (for keyword_search, etc.)

    Usage:
        orchestrator = RecallOrchestrator(conn, rolodex)
        result = orchestrator.run(message)
        # result.entries contains the top entries
        # result.recall_tier indicates which tier resolved
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        rolodex,  # Rolodex instance
        confidence_threshold: float = 0.45,
        topic_router=None,  # Optional TopicRouter for topic-scoped search
    ):
        self.conn = conn
        self.rolodex = rolodex
        self.confidence_threshold = confidence_threshold
        self._topic_router = topic_router

        # Initialize pipeline components
        self.trigger = RecallTrigger(conn)
        self.expander = QueryExpander()
        self.reranker = Reranker()

        # Hydrate proper noun cache (once per session)
        try:
            load_proper_nouns_from_db(conn)
        except Exception:
            pass

    def run(self, message: str) -> RecallResult:
        """
        Run the full recall pipeline on a user message.

        Returns RecallResult with entries, tier info, and diagnostics.
        Skips temporal-only queries (caller should handle those separately).
        """
        # Step 1: Trigger analysis
        trigger_result = self.trigger.analyze(message)

        if trigger_result.skip_recall:
            return RecallResult(entries=[], scored=[], signals=trigger_result.signals_detected)

        # Fallback: if trigger produced no queries but message wasn't skipped,
        # generate a raw FTS query from the message itself. This covers small
        # databases where no topics/projects/entities have been built yet.
        if not trigger_result.queries:
            keywords = self._extract_keywords(message)
            if keywords:
                trigger_result.queries.append(RecallQuery(
                    query=" ".join(keywords),
                    priority=0.5,
                    signal="fallback_keywords",
                ))
                trigger_result.signals_detected.append("fallback_keywords")
            else:
                return RecallResult(entries=[], scored=[], signals=trigger_result.signals_detected)

        if trigger_result.temporal_only:
            # Temporal queries need special handling (session digests, not FTS)
            # Return empty and let the caller handle this path
            return RecallResult(
                entries=[], scored=[],
                signals=trigger_result.signals_detected,
                dominant_intent="temporal",
            )

        # Step 2: Graph expansion
        graph_expansion_count = self._expand_with_graph(trigger_result)

        # Step 3: Fire queries, collect candidates
        all_candidates, seen_ids, queries_fired, merged_category_bias = (
            self._fire_queries(trigger_result)
        )

        # Step 3b: Topic-scoped search — supplement with topic-matched entries
        if self._topic_router and trigger_result.matched_topics:
            try:
                for label, count, conf in trigger_result.matched_topics[:2]:
                    # Find the topic ID from the router's cache
                    topic_id = self._resolve_topic_id(label)
                    if topic_id:
                        topic_group = self._topic_router.get_topic_group(topic_id)
                        for tid in topic_group:
                            topic_results = self.rolodex.keyword_search_by_topic(
                                message, tid, limit=5
                            )
                            for entry, score in topic_results:
                                if entry.id not in seen_ids:
                                    seen_ids.add(entry.id)
                                    all_candidates.append((entry, score * conf))
                        queries_fired.append({
                            "query": f"topic:{label}",
                            "signal": "topic_scoped",
                            "priority": conf,
                            "fresh": False,
                        })
            except Exception:
                pass  # Topic search is supplementary

        if not all_candidates:
            return RecallResult(
                entries=[], scored=[],
                queries_fired=queries_fired,
                signals=trigger_result.signals_detected,
                graph_expansion_count=graph_expansion_count,
            )

        # Step 4: Determine dominant intent (check original message first)
        query_entities = trigger_result.entities if trigger_result.entities else None
        any_fresh = any(rq.fresh for rq in trigger_result.queries)
        dominant_intent = self._compute_dominant_intent(message, trigger_result)

        # Step 5: Rerank (Tier 1)
        scored = self.reranker.rerank(
            candidates=all_candidates,
            query=message,
            query_entities=query_entities,
            category_bias=merged_category_bias,
            limit=8,
            fresh_mode=any_fresh,
            query_intent=dominant_intent,
        )

        # Step 6: Tiered confidence waterfall
        recall_tier = 1
        tier1_conf_dict = None
        tier2_conf_dict = None

        tier1_confidence = assess_confidence(
            scored_results=scored,
            query_entities=query_entities,
            threshold=self.confidence_threshold,
            tier=1,
            original_message=message,
        )

        if not tier1_confidence.confident:
            # Tier 2: Broadened FTS sweep
            recall_tier = 2
            tier1_conf_dict = self._confidence_to_dict(tier1_confidence)

            scored, all_candidates, seen_ids, t2_queries = self._run_tier2(
                message=message,
                trigger_result=trigger_result,
                all_candidates=all_candidates,
                seen_ids=seen_ids,
                query_entities=query_entities,
                merged_category_bias=merged_category_bias,
                any_fresh=any_fresh,
                dominant_intent=dominant_intent,
            )
            queries_fired.extend(t2_queries)

            tier2_confidence = assess_confidence(
                scored_results=scored,
                query_entities=query_entities,
                threshold=self.confidence_threshold,
                tier=2,
                original_message=message,
            )

            if not tier2_confidence.confident and dominant_intent in ("factual", "retrospective"):
                # Tier 3: Deep biographical / oldest-first sweep
                recall_tier = 3
                tier2_conf_dict = self._confidence_to_dict(tier2_confidence)

                scored, all_candidates, t3_queries = self._run_tier3(
                    message=message,
                    trigger_result=trigger_result,
                    all_candidates=all_candidates,
                    seen_ids=seen_ids,
                    query_entities=query_entities,
                    merged_category_bias=merged_category_bias,
                    any_fresh=any_fresh,
                    dominant_intent=dominant_intent,
                )
                queries_fired.extend(t3_queries)

        # Step 7: Fresh mode filtering
        if any_fresh:
            scored = self._apply_fresh_filter(scored)

        entries = [sc.entry for sc in scored[:8]]

        # Build topic/project match dicts for output
        matched_topics = [
            {"label": label, "entry_count": count, "confidence": round(conf, 3)}
            for label, count, conf in trigger_result.matched_topics
        ]
        matched_projects = [
            {"name": name, "confidence": round(conf, 3)}
            for name, conf in trigger_result.matched_projects
        ]

        return RecallResult(
            entries=entries,
            scored=scored[:8],
            recall_tier=recall_tier,
            candidates_total=len(all_candidates),
            queries_fired=queries_fired,
            signals=trigger_result.signals_detected,
            dominant_intent=dominant_intent,
            tier1_confidence=tier1_conf_dict,
            tier2_confidence=tier2_conf_dict,
            graph_expansion_count=graph_expansion_count,
            matched_topics=matched_topics,
            matched_projects=matched_projects,
        )

    # ── Internal pipeline stages ─────────────────────────────────────────

    def _expand_with_graph(self, result: TriggerResult) -> int:
        """Enrich recall queries with knowledge graph neighbors."""
        count = 0
        try:
            from ..storage.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph(self.conn)
            kg.ensure_schema()

            if result.entities and result.entities.proper_nouns:
                for noun in result.entities.proper_nouns[:3]:
                    neighbors = kg.get_neighbors(noun, depth=1)
                    for neighbor in neighbors[:3]:
                        result.queries.append(RecallQuery(
                            query=f"{noun} {neighbor.entity.name}",
                            priority=0.5 * neighbor.weight,
                            signal=f"graph:{neighbor.relationship}",
                        ))
                        count += 1

            if count > 0:
                seen_normalized = {}
                deduped = []
                for q in result.queries:
                    normalized = ' '.join(sorted(q.query.lower().split()))
                    if normalized not in seen_normalized or q.priority > seen_normalized[normalized].priority:
                        if normalized in seen_normalized:
                            deduped = [x for x in deduped if x is not seen_normalized[normalized]]
                        seen_normalized[normalized] = q
                        deduped.append(q)
                result.queries = sorted(deduped, key=lambda q: q.priority, reverse=True)[:6]
        except Exception:
            pass  # Graph expansion is additive, never blocks recall

        return count

    def _resolve_topic_id(self, label: str) -> Optional[str]:
        """Resolve a topic label to its ID via the topic router's cache."""
        if not self._topic_router:
            return None
        self._topic_router._ensure_cache_loaded()
        label_lower = label.lower()
        for topic_id, topic_data in self._topic_router._topic_cache.items():
            if topic_data.get("label", "").lower() == label_lower:
                return topic_id
        return None

    def _fire_queries(
        self, result: TriggerResult
    ) -> Tuple[List[Tuple], Set[str], List[Dict], List[str]]:
        """Fire each recall query through the expander, collect candidates."""
        all_candidates = []
        seen_ids: Set[str] = set()
        queries_fired = []
        merged_category_bias = []

        for rq in result.queries:
            expanded = self.expander.expand(rq.query)

            if expanded.category_bias:
                for cat in expanded.category_bias:
                    if cat not in merged_category_bias:
                        merged_category_bias.append(cat)

            for variant in expanded.variants[:3]:
                results = self.rolodex.keyword_search(variant, limit=10)
                for i, (entry, score) in enumerate(results):
                    if entry.id not in seen_ids:
                        seen_ids.add(entry.id)
                        position_score = max(0.1, 1.0 - (i * 0.1))
                        weighted_score = position_score * rq.priority
                        all_candidates.append((entry, weighted_score))

            queries_fired.append({
                "query": rq.query,
                "signal": rq.signal,
                "priority": rq.priority,
                "fresh": rq.fresh,
            })

        return all_candidates, seen_ids, queries_fired, merged_category_bias

    def _compute_dominant_intent(self, message: str, result: TriggerResult) -> str:
        """Determine the dominant intent from the message and queries."""
        dominant = "exploratory"
        priority_map = {"factual": 4, "retrospective": 3, "experiential": 2, "exploratory": 1}

        # Check the original message first (highest priority signal)
        msg_expanded = self.expander.expand(message)
        if priority_map.get(msg_expanded.intent, 0) > priority_map.get(dominant, 0):
            dominant = msg_expanded.intent

        # Then check individual queries
        for rq in result.queries:
            rq_expanded = self.expander.expand(rq.query)
            if priority_map.get(rq_expanded.intent, 0) > priority_map.get(dominant, 0):
                dominant = rq_expanded.intent

        return dominant

    def _run_tier2(
        self,
        message: str,
        trigger_result: TriggerResult,
        all_candidates: List[Tuple],
        seen_ids: Set[str],
        query_entities: Optional[ExtractedEntities],
        merged_category_bias: List[str],
        any_fresh: bool,
        dominant_intent: str,
    ) -> Tuple[List[ScoredCandidate], List[Tuple], Set[str], List[Dict]]:
        """Tier 2: Broadened FTS sweep with category-targeted queries."""
        queries_fired = []

        # 2a: Plain proper nouns without graph neighbors
        raw_entities = []
        if trigger_result.entities and trigger_result.entities.proper_nouns:
            raw_entities = trigger_result.entities.proper_nouns[:4]
        else:
            raw_entities = self._extract_keywords(message)

        # 2b: Category routing based on dominant intent
        category_map = {
            "factual": ["correction", "user_knowledge", "user_facts", "fact", "reference"],
            "retrospective": ["user_knowledge", "note", "decision", "preference"],
            "experiential": ["correction", "friction", "breakthrough", "pivot"],
            "exploratory": [],
        }
        tier2_categories = category_map.get(dominant_intent, [])

        # 2c: Fire Tier 2 queries with category filter
        for entity_term in raw_entities:
            t2_results = self.rolodex.keyword_search(
                entity_term, limit=10,
                category_filter=tier2_categories if tier2_categories else None,
            )
            for i, (entry, score) in enumerate(t2_results):
                if entry.id not in seen_ids:
                    seen_ids.add(entry.id)
                    position_score = max(0.1, 1.0 - (i * 0.1))
                    all_candidates.append((entry, position_score * 0.8))
            queries_fired.append({
                "query": entity_term, "tier": 2,
                "categories": tier2_categories,
            })

        # 2d: Raw message keywords without category filter
        msg_keywords = self._extract_keywords(message)
        if msg_keywords:
            kw_query = " ".join(msg_keywords[:6])
            t2_kw_results = self.rolodex.keyword_search(kw_query, limit=10)
            for i, (entry, score) in enumerate(t2_kw_results):
                if entry.id not in seen_ids:
                    seen_ids.add(entry.id)
                    position_score = max(0.1, 1.0 - (i * 0.1))
                    all_candidates.append((entry, position_score * 0.7))
            queries_fired.append({"query": kw_query, "tier": 2, "categories": []})

        # Re-rerank combined pool
        scored = self.reranker.rerank(
            candidates=all_candidates,
            query=message,
            query_entities=query_entities,
            category_bias=merged_category_bias,
            limit=8,
            fresh_mode=any_fresh,
            query_intent=dominant_intent,
        )

        return scored, all_candidates, seen_ids, queries_fired

    def _run_tier3(
        self,
        message: str,
        trigger_result: TriggerResult,
        all_candidates: List[Tuple],
        seen_ids: Set[str],
        query_entities: Optional[ExtractedEntities],
        merged_category_bias: List[str],
        any_fresh: bool,
        dominant_intent: str,
    ) -> Tuple[List[ScoredCandidate], List[Tuple], List[Dict]]:
        """Tier 3: Deep biographical / oldest-first sweep with force-injection."""
        queries_fired = []
        tier3_injected = []

        # Build entity terms for matching
        raw_entities = []
        if trigger_result.entities and trigger_result.entities.proper_nouns:
            raw_entities = trigger_result.entities.proper_nouns[:4]
        if not raw_entities:
            raw_entities = self._extract_keywords(message)

        # 3a: Query user_facts table directly
        try:
            from ..core.user_facts import UserFactsStore
            ufs = UserFactsStore(self.conn)
            relevant_facts = ufs.query_facts(message)
            for fact in relevant_facts[:5]:
                fact_results = self.rolodex.keyword_search(fact.value, limit=3)
                for entry, score in fact_results:
                    tier3_injected.append((entry, 2.5))
            queries_fired.append({"query": "user_facts_table", "tier": 3})
        except Exception:
            pass

        # 3b: Search correction entries oldest-first
        try:
            from ..storage.rolodex import deserialize_entry
            correction_rows = self.conn.execute("""
                SELECT * FROM rolodex_entries
                WHERE category = 'correction'
                AND archived_at IS NULL
                AND superseded_by IS NULL
                ORDER BY created_at ASC
                LIMIT 10
            """).fetchall()
            for row in correction_rows:
                entry = deserialize_entry(row)
                content_lower = (entry.content or "").lower()
                if any(e.lower() in content_lower for e in raw_entities):
                    tier3_injected.append((entry, 2.5))
            queries_fired.append({"query": "corrections_oldest_first", "tier": 3})
        except Exception:
            pass

        # 3c: Search user_knowledge sorted by created_at ASC
        try:
            from ..storage.rolodex import deserialize_entry as _deser
            uk_rows = self.conn.execute("""
                SELECT * FROM rolodex_entries
                WHERE category = 'user_knowledge'
                AND archived_at IS NULL
                AND superseded_by IS NULL
                ORDER BY created_at ASC
                LIMIT 20
            """).fetchall()
            for row in uk_rows:
                entry = _deser(row)
                content_lower = (entry.content or "").lower()
                if any(e.lower() in content_lower for e in raw_entities):
                    tier3_injected.append((entry, 2.0))
        except Exception:
            pass

        # Force-inject: remove lower-scored duplicates, add boosted versions
        if tier3_injected:
            t3_ids = {entry.id for entry, _ in tier3_injected}
            all_candidates = [
                (e, s) for e, s in all_candidates if e.id not in t3_ids
            ]
            all_candidates.extend(tier3_injected)

        # Final rerank
        scored = self.reranker.rerank(
            candidates=all_candidates,
            query=message,
            query_entities=query_entities,
            category_bias=merged_category_bias,
            limit=8,
            fresh_mode=any_fresh,
            query_intent=dominant_intent,
        )

        return scored, all_candidates, queries_fired

    def _apply_fresh_filter(self, scored: List[ScoredCandidate]) -> List[ScoredCandidate]:
        """Apply recency filter for fresh-mode queries."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        fresh_cutoff = 48 * 3600  # 48 hours

        fresh_scored = [
            sc for sc in scored
            if hasattr(sc.entry, 'created_at') and sc.entry.created_at
            and (now - sc.entry.created_at).total_seconds() < fresh_cutoff
        ]
        if fresh_scored:
            non_fresh = [sc for sc in scored if sc not in fresh_scored]
            return fresh_scored[:5] + non_fresh[:3]
        return scored

    @staticmethod
    def _extract_keywords(message: str) -> List[str]:
        """Extract stop-word-stripped keywords from a message."""
        _stop = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'who', 'how',
            'when', 'where', 'why', 'do', 'does', 'did', 'my', 'your', 'our',
            'this', 'that', 'it', 'i', 'me', 'we', 'us', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'and', 'or', 'but', 'not', 'can',
            'will', 'would', 'could', 'should', 'has', 'have', 'had', 'be', 'been',
        }
        return [w for w in message.lower().split() if w not in _stop and len(w) > 2][:4]

    @staticmethod
    def _confidence_to_dict(conf) -> Dict:
        """Convert a RecallConfidence to a serializable dict."""
        return {
            "top_score": round(conf.top_score, 4),
            "score_gap": round(conf.score_gap, 4),
            "entity_hit_rate": round(conf.entity_hit_rate, 4),
            "answer_signal": round(conf.answer_signal, 4),
            "confident": conf.confident,
        }
