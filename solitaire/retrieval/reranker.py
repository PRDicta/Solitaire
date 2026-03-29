"""
The Librarian — Re-ranker

Takes a wide pool of candidate entries and re-ranks them using multiple
signals beyond embedding similarity. This is the "narrow" phase of the
wide-net-then-narrow search pattern.

Scoring signals:
1. Semantic similarity (from the original search score)
2. Entity match (exact string match of extracted entities in entry content)
3. Category match (from intent detection — biases toward relevant categories)
4. Recency (newer entries get a mild boost)
5. Access frequency (well-worn book principle — frequently accessed entries rank higher)
6. Confidence (Hindsight reinforcement — reinforced entries rank higher)
7. Identity resonance (entries touching active identity nodes rank higher)

Post-filters:
- Contradiction resolution (Phase 2): when two scored entries share entity
  overlap but contain conflicting claims, the older one is suppressed.
  This prevents stale facts from surviving retrieval when a newer correction
  exists in the same result set.

No LLM calls — purely heuristic, runs in microseconds.
"""
import time
import re
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from ..core.types import RolodexEntry
from .entity_extractor import EntityExtractor, ExtractedEntities
from ..core.confidence import extract_confidence_from_metadata, compute_effective
from .conflict_utils import (
    extract_claim_entities,
    detect_claim_conflict,
    numeric_conflict,
    preference_conflict,
    negation_conflict,
)


@dataclass
class RerankerConfig:
    """Weights for each scoring signal. All should sum to ~1.0 for interpretability."""
    semantic_weight: float = 0.275
    entity_weight: float = 0.25
    category_weight: float = 0.125
    recency_weight: float = 0.10
    frequency_weight: float = 0.10
    confidence_weight: float = 0.10  # Hindsight: reinforced entries rank higher
    identity_weight: float = 0.05   # Identity resonance: entries touching active identity nodes

    # Verbatim boost: additive bonus applied to entries with verbatim_source=True.
    # Ensures original user/assistant text is preferred over assistant summaries,
    # but doesn't let a mediocre verbatim entry dominate a highly-relevant summary.
    # Changed from multiplicative (1.5×) to additive (0.15) in v1.3 to prevent
    # score distortion.
    verbatim_boost: float = 0.15

    # Provenance boost: graduated by source authority.
    # user-stated content is the user's own words (highest trust).
    # inferred content gets slight skepticism (enrichment pipeline output).
    # Legacy field user_stated_boost kept for backward compat but unused in scoring.
    user_stated_boost: float = 0.10  # Deprecated: use provenance_boosts

    provenance_boosts: dict = field(default_factory=lambda: {
        "user-stated": 0.10,       # User's own words
        "system": 0.07,            # System-level entries
        "external-import": 0.03,   # Imported, not verified in-session
        "assistant-inferred": 0.02,
        "observed": 0.01,          # Observer/scanner patterns
        "inferred": -0.02,         # Enrichment pipeline, slight skepticism
        "unknown": 0.0,            # Unclassified, no boost
    })

    # Recency decay: entries older than this many days get no recency boost
    recency_horizon_days: float = 30.0

    # Frequency cap: beyond this many accesses, no additional boost
    frequency_cap: int = 50

    # Length normalization: penalizes entries above a character threshold.
    # BM25 inherently favors longer documents with more term occurrences.
    # A 42,000-char continuation summary mentioning key terms 3x each will
    # outscore a 345-char entry that IS the answer. The penalty is applied
    # as a multiplier on the semantic score: entries below the threshold
    # get 1.0 (no change), entries above get a decaying multiplier that
    # approaches length_penalty_floor for very long entries.
    #
    # Calibrated against real corpus: median entry is ~200 chars, mean is
    # ~750 chars, ground truth entries average 200-500 chars. Entries above
    # 1500 chars are almost always continuation summaries or reingestion
    # artifacts. The penalty should be gentle at 1500 and firm by 5000.
    #
    # Fix: 2026-03-17. Addresses continuation summary dominance (Sprint
    # Track 1 Q5 and the 90%→40% regression when corpus grew 40%).
    length_penalty_threshold: int = 1500   # chars; no penalty below this
    length_penalty_ceiling: int = 10000    # chars; max penalty at this point
    length_penalty_floor: float = 0.4      # minimum multiplier for very long entries


@dataclass
class ScoredCandidate:
    """An entry with its composite re-ranking score and signal breakdown."""
    entry: RolodexEntry
    composite_score: float = 0.0
    semantic_score: float = 0.0
    entity_score: float = 0.0
    category_score: float = 0.0
    recency_score: float = 0.0
    frequency_score: float = 0.0
    confidence_score: float = 0.0  # Hindsight: entry confidence (0-1)
    identity_resonance_score: float = 0.0  # Identity: overlap with active identity nodes


class Reranker:
    """
    Multi-signal re-ranker for search candidates.
    Takes a wide pool and narrows to the most relevant entries.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self.entity_extractor = EntityExtractor()

    def rerank(
        self,
        candidates: List[Tuple[RolodexEntry, float]],
        query: str,
        query_entities: Optional[ExtractedEntities] = None,
        category_bias: Optional[List[str]] = None,
        limit: int = 5,
        fresh_mode: bool = False,
        query_intent: Optional[str] = None,
        identity_context: Optional[dict] = None,
    ) -> List[ScoredCandidate]:
        """
        Re-rank a pool of candidates using multiple signals.

        Args:
            candidates: List of (entry, search_score) from initial wide search
            query: The original query string
            query_entities: Pre-extracted entities from the query (or will extract)
            category_bias: Categories to boost (from intent detection)
            limit: Number of results to return
            query_intent: Intent type from QueryExpander (factual, retrospective,
                         experiential, exploratory). Used to scale recency weight.
            identity_context: Optional dict with active identity nodes for resonance
                             scoring. Keys: growth_edges, commitments, patterns, north_star.
                             Each value is a list of identity node objects (or None).

        Returns:
            Top N entries re-ranked by composite score
        """
        if not candidates:
            return []

        # In fresh mode (back-references, continuations), dramatically boost recency.
        # Rebalance weights: recency 0.10 → 0.40, semantic 0.35 → 0.20, entity 0.30 → 0.15.
        # This ensures "what were we working on?" returns today's entries, not last week's.
        cfg = self.config
        if fresh_mode:
            # Fresh mode: recency-dominant scoring for back-references.
            # Two changes from normal mode:
            # 1. Weight shift: recency 0.10 → 0.40 (dominant signal)
            # 2. Horizon shift: 30 days → 2 days. This is the critical fix.
            #    With a 30-day horizon, entries 5 minutes old and 6 hours old
            #    score ~0.9999 vs ~0.9917 — a 0.008 difference that gets
            #    swamped by entity/semantic scores. With a 2-day horizon:
            #    5 min = 0.998, 4h = 0.917, 24h = 0.500, 48h = 0.0.
            #    Now recency creates real differentiation at human timescales.
            cfg = RerankerConfig(
                semantic_weight=0.20,
                entity_weight=0.15,
                category_weight=0.10,
                recency_weight=0.40,
                frequency_weight=0.05,
                confidence_weight=0.05,  # Reduced in fresh mode (recency dominates)
                identity_weight=0.03,    # Reduced in fresh mode
                verbatim_boost=cfg.verbatim_boost,
                user_stated_boost=cfg.user_stated_boost,
                recency_horizon_days=2.0,  # Was cfg.recency_horizon_days (30d). Tightened for fresh mode.
                frequency_cap=cfg.frequency_cap,
            )

        # Intent-based recency scaling (Phase 12).
        # Factual queries ("how does X work") should not be biased toward
        # recent entries — the answer to "how does the boot manifest work"
        # doesn't change because newer entries were added. Retrospective
        # queries ("what did we decide") benefit from mild recency but less
        # than the default. Only temporal/continuation queries (handled by
        # fresh_mode above) should have strong recency.
        #
        # Fix: 2026-03-17. Prevents slow erosion of older foundational
        # entries as the rolodex grows and newer entries accumulate
        # recency advantage.
        if not fresh_mode and query_intent:
            intent_recency_map = {
                "factual": 0.03,        # Almost no recency bias
                "retrospective": 0.05,  # Mild recency
                "experiential": 0.10,   # Default (experience is time-sensitive)
                "exploratory": 0.08,    # Slightly below default
            }
            adjusted_recency = intent_recency_map.get(query_intent, cfg.recency_weight)
            if adjusted_recency != cfg.recency_weight:
                # Redistribute the freed weight to semantic scoring,
                # since factual queries should prioritize content match.
                freed = cfg.recency_weight - adjusted_recency
                cfg = RerankerConfig(
                    semantic_weight=cfg.semantic_weight + freed,
                    entity_weight=cfg.entity_weight,
                    category_weight=cfg.category_weight,
                    recency_weight=adjusted_recency,
                    frequency_weight=cfg.frequency_weight,
                    confidence_weight=cfg.confidence_weight,
                    identity_weight=cfg.identity_weight,
                    verbatim_boost=cfg.verbatim_boost,
                    user_stated_boost=cfg.user_stated_boost,
                    recency_horizon_days=cfg.recency_horizon_days,
                    frequency_cap=cfg.frequency_cap,
                    length_penalty_threshold=cfg.length_penalty_threshold,
                    length_penalty_ceiling=cfg.length_penalty_ceiling,
                    length_penalty_floor=cfg.length_penalty_floor,
                )

        # Extract entities from query if not provided
        if query_entities is None:
            query_entities = self.entity_extractor.extract_from_query(query)

        # Normalize search scores to 0-1 range
        max_search_score = max(score for _, score in candidates) if candidates else 1.0
        if max_search_score == 0:
            max_search_score = 1.0

        # Find the most recent timestamp for recency normalization
        now = time.time()

        # Score each candidate
        scored: List[ScoredCandidate] = []
        for entry, search_score in candidates:
            sc = ScoredCandidate(entry=entry)

            # Signal 1: Semantic similarity (normalized from original search)
            # Apply length penalty: long entries get their semantic score dampened
            # because BM25 inflates scores for long documents with many term hits.
            length_multiplier = self._score_length_penalty(entry)
            sc.semantic_score = (search_score / max_search_score) * length_multiplier

            # Signal 2: Entity match
            sc.entity_score = self._score_entity_match(entry, query_entities)

            # Signal 3: Category match
            sc.category_score = self._score_category_match(entry, category_bias)

            # Signal 4: Recency
            sc.recency_score = self._score_recency(entry, now)

            # Signal 5: Access frequency
            sc.frequency_score = self._score_frequency(entry)

            # Signal 6: Confidence (Hindsight)
            sc.confidence_score = self._score_confidence(entry)

            # Signal 7: Identity resonance
            sc.identity_resonance_score = self._score_identity_resonance(
                entry, identity_context
            )

            # Composite score
            sc.composite_score = (
                cfg.semantic_weight * sc.semantic_score
                + cfg.entity_weight * sc.entity_score
                + cfg.category_weight * sc.category_score
                + cfg.recency_weight * sc.recency_score
                + cfg.frequency_weight * sc.frequency_score
                + cfg.confidence_weight * sc.confidence_score
                + cfg.identity_weight * sc.identity_resonance_score
            )

            # Verbatim boost: original text preferred over summaries (additive)
            if getattr(entry, "verbatim_source", True):
                sc.composite_score += cfg.verbatim_boost

            # Provenance boost: graduated by source authority
            provenance = getattr(entry, "provenance", "unknown")
            sc.composite_score += cfg.provenance_boosts.get(provenance, 0.0)

            scored.append(sc)

        # Sort by composite score descending
        scored.sort(key=lambda s: s.composite_score, reverse=True)

        # Phase 2: Contradiction resolution post-filter.
        # Scan the top results for entry pairs that share entity overlap
        # but contain conflicting claims. When found, suppress the older one.
        scored = self._resolve_contradictions(scored, limit)

        return scored[:limit]

    def _score_entity_match(
        self, entry: RolodexEntry, query_entities: ExtractedEntities
    ) -> float:
        """
        Score based on exact entity matches in entry content.
        Returns 0-1 where 1 means all query entities were found.
        """
        if not query_entities.all_entities:
            return 0.0

        content_lower = entry.content.lower()
        tags_lower = " ".join(entry.tags).lower() if entry.tags else ""
        search_text = content_lower + " " + tags_lower

        matches = 0
        for entity in query_entities.all_entities:
            if entity.lower() in search_text:
                matches += 1

        # Also check attribution match
        if query_entities.attribution:
            # If query asks about "what I said" (user attribution)
            # boost entries that have user attribution markers
            if query_entities.attribution == "user":
                user_markers = ["user", "philip", "i said", "i asked", "my "]
                if any(m in content_lower for m in user_markers):
                    matches += 0.5
            elif query_entities.attribution == "assistant":
                assistant_markers = ["assistant", "claude", "you said", "suggested"]
                if any(m in content_lower for m in assistant_markers):
                    matches += 0.5

        total_entities = len(query_entities.all_entities) + (
            0.5 if query_entities.attribution else 0
        )

        return min(matches / total_entities, 1.0) if total_entities > 0 else 0.0

    def _score_category_match(
        self, entry: RolodexEntry, category_bias: Optional[List[str]]
    ) -> float:
        """
        Score based on whether entry category matches the intent bias.

        Uses a soft bias (0.3 baseline for non-matching, 1.0 for matching)
        instead of a hard gate (0.0/1.0). This prevents valid FTS hits in
        unexpected categories from being silently killed.

        Fix: 2026-03-16. Prior behavior was hard 0.0 for non-matching categories,
        which caused category_weight (0.15) to zero out entries that were perfect
        FTS matches but stored in a different category than the intent detector
        predicted. The 0.3 floor ensures non-matching entries lose ~10.5% of their
        composite score (0.15 × 0.7) instead of the full 15%.
        """
        if not category_bias:
            return 0.5  # Neutral — no bias means no penalty or boost

        cat = entry.category
        cat_str = cat.value if hasattr(cat, "value") else str(cat)

        return 1.0 if cat_str in category_bias else 0.3

    def _score_recency(self, entry: RolodexEntry, now: float) -> float:
        """
        Score based on how recently the entry was created.
        Linear decay over the recency horizon.
        """
        if not hasattr(entry, "created_at") or entry.created_at is None:
            return 0.5  # Unknown age — neutral

        try:
            # created_at might be a datetime or a timestamp
            if hasattr(entry.created_at, "timestamp"):
                entry_time = entry.created_at.timestamp()
            else:
                entry_time = float(entry.created_at)

            age_seconds = now - entry_time
            horizon_seconds = self.config.recency_horizon_days * 86400

            if age_seconds <= 0:
                return 1.0
            elif age_seconds >= horizon_seconds:
                return 0.0
            else:
                return 1.0 - (age_seconds / horizon_seconds)
        except (TypeError, ValueError):
            return 0.5

    def _score_length_penalty(self, entry: RolodexEntry) -> float:
        """
        Compute a length-based penalty multiplier for the semantic score.
        Returns 1.0 for entries at or below threshold, decays linearly to
        floor for entries at or above ceiling.

        This counteracts BM25's bias toward long documents: a 40,000-char
        continuation summary that mentions "Dicta" and "pipeline" incidentally
        shouldn't outscore a 345-char entry that IS the routing rule.
        """
        content_len = len(entry.content) if entry.content else 0
        threshold = self.config.length_penalty_threshold
        ceiling = self.config.length_penalty_ceiling
        floor = self.config.length_penalty_floor

        if content_len <= threshold:
            return 1.0
        elif content_len >= ceiling:
            return floor
        else:
            # Linear decay from 1.0 to floor between threshold and ceiling
            ratio = (content_len - threshold) / (ceiling - threshold)
            return 1.0 - ratio * (1.0 - floor)

    def _score_confidence(self, entry: RolodexEntry) -> float:
        """
        Score based on entry confidence (Hindsight reinforcement system).

        Reads the confidence score from entry metadata. Entries with no
        confidence data get a neutral 0.5 (pre-Hindsight entries shouldn't
        be penalized). Entries with high reinforcement count and recent
        reinforcement score close to 1.0. Stale, unreinforced entries
        score lower.

        The effective score already accounts for reinforcement bonus and
        time decay, so we use it directly.
        """
        metadata = getattr(entry, 'metadata', None)
        if not metadata or not isinstance(metadata, dict):
            return 0.5  # Neutral for entries without metadata

        score = extract_confidence_from_metadata(metadata)
        if score is None:
            return 0.5  # Neutral for pre-Hindsight entries

        return score.effective

    def _score_identity_resonance(
        self,
        entry: RolodexEntry,
        identity_context: Optional[dict],
    ) -> float:
        """
        Score how much an entry resonates with the agent's active identity.

        Checks whether the entry's content overlaps with keywords from active
        identity nodes (growth edges, commitments, patterns, north star).
        Entries that touch active identity content are more personally
        meaningful and should rank higher.

        Returns 0.0-1.0 where:
          1.0 = strong overlap with growth edges or commitments (highest priority)
          0.7 = overlaps with active behavioral patterns
          0.5 = touches north star themes
          0.0 = no identity relevance (neutral, not penalized)
        """
        if not identity_context:
            return 0.0

        content_lower = (entry.content or "").lower()
        tags_lower = " ".join(entry.tags).lower() if entry.tags else ""
        search_text = content_lower + " " + tags_lower

        if len(search_text.strip()) < 10:
            return 0.0

        best_score = 0.0

        # Growth edges and commitments: highest identity relevance
        for node_list_key in ("growth_edges", "commitments"):
            nodes = identity_context.get(node_list_key)
            if not nodes:
                continue
            for node in nodes:
                node_content = getattr(node, "content", "") or ""
                if not node_content:
                    continue
                keywords = self._extract_identity_keywords(node_content)
                if keywords and self._keyword_overlap(keywords, search_text):
                    best_score = max(best_score, 1.0)

        if best_score >= 1.0:
            return 1.0

        # Patterns: behavioral relevance
        patterns = identity_context.get("patterns")
        if patterns:
            for node in patterns:
                node_content = getattr(node, "content", "") or ""
                if not node_content:
                    continue
                keywords = self._extract_identity_keywords(node_content)
                if keywords and self._keyword_overlap(keywords, search_text):
                    best_score = max(best_score, 0.7)

        if best_score >= 0.7:
            return best_score

        # North star: thematic relevance
        north_star = identity_context.get("north_star")
        if north_star:
            ns_content = getattr(north_star, "content", "") or ""
            if ns_content:
                keywords = self._extract_identity_keywords(ns_content)
                if keywords and self._keyword_overlap(keywords, search_text):
                    best_score = max(best_score, 0.5)

        return best_score

    @staticmethod
    def _extract_identity_keywords(text: str) -> Set[str]:
        """Extract significant keywords from an identity node's content."""
        _stop = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'who', 'how',
            'when', 'where', 'why', 'do', 'does', 'did', 'my', 'your', 'our',
            'this', 'that', 'it', 'i', 'me', 'we', 'us', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'and', 'or', 'but', 'not', 'can',
            'will', 'would', 'could', 'should', 'has', 'have', 'had', 'be', 'been',
            'being', 'about', 'into', 'through', 'during', 'before', 'after',
            'also', 'just', 'very', 'really', 'than', 'then', 'some', 'only',
            'its', 'they', 'them', 'their', 'which', 'these', 'those', 'other',
            'more', 'most', 'such', 'each', 'every', 'both', 'between',
            'practice', 'status', 'active', 'identified', 'practicing',
            'not', 'instead', 'rather', 'whether',
        }
        words = set(re.findall(r'\b[a-z]{4,}\b', text.lower()))
        return words - _stop

    @staticmethod
    def _keyword_overlap(keywords: Set[str], search_text: str) -> bool:
        """Check if enough identity keywords appear in the entry text."""
        if not keywords:
            return False
        hits = sum(1 for kw in keywords if kw in search_text)
        # Require at least 2 keyword hits, or 1 if the keyword set is small
        threshold = 2 if len(keywords) >= 4 else 1
        return hits >= threshold

    def _score_frequency(self, entry: RolodexEntry) -> float:
        """
        Score based on access frequency (the well-worn book principle).
        More accesses = higher score, capped at frequency_cap.

        Entries younger than 24 hours get a frequency floor of 0.5 (neutral)
        instead of 0.0, since they haven't had time to accumulate accesses.
        """
        access_count = getattr(entry, "access_count", 0) or 0

        if access_count <= 0:
            # New entries shouldn't be penalized for having zero accesses.
            # Check age: if < 24 hours old, return neutral instead of zero.
            try:
                if hasattr(entry, "created_at") and entry.created_at is not None:
                    import time as _time
                    if hasattr(entry.created_at, "timestamp"):
                        entry_time = entry.created_at.timestamp()
                    else:
                        entry_time = float(entry.created_at)
                    age_hours = (_time.time() - entry_time) / 3600
                    if age_hours < 24:
                        return 0.5  # Neutral -- too new to judge
            except (TypeError, ValueError):
                pass
            return 0.0

        return min(access_count / self.config.frequency_cap, 1.0)

    # ─── Phase 2: Contradiction Resolution ──────────────────────────────────

    def _resolve_contradictions(
        self,
        scored: List[ScoredCandidate],
        limit: int,
    ) -> List[ScoredCandidate]:
        """
        Post-filter: detect and resolve contradictions in scored results.

        When two entries share significant entity overlap but contain
        conflicting claims (numeric values, status assertions, or negation
        patterns), suppress the older one. "Suppress" means removing it
        from the result set entirely, not just demoting its score. A
        contradicted entry is wrong, not just less relevant.

        Only scans the top `limit * 2` entries to keep this fast. Pairwise
        comparison is O(n^2) but n is capped at ~16, so it's negligible.

        Returns the filtered list (may be shorter than input).
        """
        # Only worth checking if we have at least 2 results
        if len(scored) < 2:
            return scored

        # Scan window: check more than `limit` entries so we can backfill
        # if a suppressed entry was in the top N
        scan_window = min(len(scored), limit * 2)
        scan_set = scored[:scan_window]

        # Build per-entry entity sets for pairwise comparison
        entry_entities: List[Set[str]] = []
        for sc in scan_set:
            entities = self._extract_claim_entities(sc.entry)
            entry_entities.append(entities)

        suppressed_ids: Set[str] = set()

        for i in range(len(scan_set)):
            if scan_set[i].entry.id in suppressed_ids:
                continue
            for j in range(i + 1, len(scan_set)):
                if scan_set[j].entry.id in suppressed_ids:
                    continue

                # Check entity overlap: need shared context to compare claims
                overlap = entry_entities[i] & entry_entities[j]
                if len(overlap) < 1:
                    continue

                # Check for conflicting claims
                conflict = self._detect_claim_conflict(
                    scan_set[i].entry, scan_set[j].entry
                )
                if not conflict:
                    continue

                # Conflict found: suppress the older entry
                entry_i = scan_set[i].entry
                entry_j = scan_set[j].entry
                older_idx = i if self._entry_is_older(entry_i, entry_j) else j
                suppressed_ids.add(scan_set[older_idx].entry.id)

        if not suppressed_ids:
            return scored

        return [sc for sc in scored if sc.entry.id not in suppressed_ids]

    @staticmethod
    def _extract_claim_entities(entry: RolodexEntry) -> Set[str]:
        """Delegate to shared conflict_utils."""
        return extract_claim_entities(entry.content or "", entry.tags)

    @staticmethod
    def _detect_claim_conflict(entry_a: RolodexEntry, entry_b: RolodexEntry) -> bool:
        """Delegate to shared conflict_utils. Returns True if conflict detected."""
        return detect_claim_conflict(
            entry_a.content or "", entry_b.content or ""
        ) is not None

    @staticmethod
    def _entry_is_older(a: RolodexEntry, b: RolodexEntry) -> bool:
        """Return True if entry a is older than entry b."""
        try:
            time_a = a.created_at.timestamp() if hasattr(a.created_at, 'timestamp') else float(a.created_at)
            time_b = b.created_at.timestamp() if hasattr(b.created_at, 'timestamp') else float(b.created_at)
            return time_a < time_b
        except (TypeError, ValueError, AttributeError):
            return False  # Can't determine; don't suppress either
