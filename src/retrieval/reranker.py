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
6. Confidence (Hindsight reinforcement — entries confirmed by re-observation rank higher)

No LLM calls — purely heuristic, runs in microseconds.
"""
import time
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from ..core.types import RolodexEntry
from .entity_extractor import EntityExtractor, ExtractedEntities
from ..core.confidence import extract_confidence_from_metadata


@dataclass
class RerankerConfig:
    """Weights for each scoring signal. All should sum to ~1.0 for interpretability."""
    semantic_weight: float = 0.30
    entity_weight: float = 0.25
    category_weight: float = 0.15
    recency_weight: float = 0.10
    frequency_weight: float = 0.10
    confidence_weight: float = 0.10  # Hindsight: reinforced entries rank higher

    # Verbatim boost: additive bonus applied to entries with verbatim_source=True.
    # Ensures original user/assistant text is preferred over assistant summaries,
    # but doesn't let a mediocre verbatim entry dominate a highly-relevant summary.
    # Changed from multiplicative (1.5×) to additive (0.15) in v1.3 to prevent
    # score distortion.
    verbatim_boost: float = 0.15

    # Provenance boost: user-stated content ranks higher than assistant-inferred.
    # The user's own words are more authoritative than the assistant's interpretation.
    user_stated_boost: float = 0.10

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

            # Composite score
            sc.composite_score = (
                cfg.semantic_weight * sc.semantic_score
                + cfg.entity_weight * sc.entity_score
                + cfg.category_weight * sc.category_score
                + cfg.recency_weight * sc.recency_score
                + cfg.frequency_weight * sc.frequency_score
                + cfg.confidence_weight * sc.confidence_score
            )

            # Verbatim boost: original text preferred over summaries (additive)
            if getattr(entry, "verbatim_source", True):
                sc.composite_score += cfg.verbatim_boost

            # Provenance boost: user-stated content ranks higher
            if getattr(entry, "provenance", "unknown") == "user-stated":
                sc.composite_score += cfg.user_stated_boost

            scored.append(sc)

        # Sort by composite score descending
        scored.sort(key=lambda s: s.composite_score, reverse=True)

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
                user_markers = ["user", "owner", "i said", "i asked", "my "]
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
        continuation summary that mentions key terms incidentally
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

    def _score_frequency(self, entry: RolodexEntry) -> float:
        """
        Score based on access frequency (the well-worn book principle).
        More accesses = higher score, capped at frequency_cap.
        """
        access_count = getattr(entry, "access_count", 0) or 0

        if access_count <= 0:
            return 0.0

        return min(access_count / self.config.frequency_cap, 1.0)
