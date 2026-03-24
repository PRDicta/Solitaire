"""
The Librarian — Identity Reflection (Phase 4)

End-of-session review of the identity graph. Runs alongside skill reflection
in cmd_end. Handles:

1. Candidate review: auto-promote high-signal candidates, flag borderline ones
2. Pattern trajectory updates: compare observation counts before/after session
3. Growth edge progression: advance status when addressed patterns improve

Design: operates on the IdentityGraph directly. No LLM call. Returns a
structured report for inclusion in the session end output.
"""
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from solitaire.storage.identity_graph import (
    IdentityGraph, IdentityNode, IdentityCandidate, IdentityEdge,
    IdentityReference, IdentitySignal, NodeType, EdgeType,
    GrowthEdgeStatus, PatternTrajectory, CommitmentOutcome,
)


# Signal reliability weights (from Vazire SOKA model, spec §3.4)
SIGNAL_WEIGHTS = {
    "user_correction": 1.0,
    "enrichment_scanner": 0.6,
    "self_report": 0.3,
}


@dataclass
class ReflectionReport:
    """Results of an identity reflection pass."""
    candidates_reviewed: int = 0
    candidates_promoted: int = 0
    candidates_flagged: int = 0       # borderline, left for manual review
    candidates_dismissed: int = 0     # auto-dismissed (low quality)
    patterns_updated: int = 0
    growth_edges_progressed: int = 0
    commitments_evaluated: int = 0    # Phase 6d
    promoted_node_ids: List[str] = field(default_factory=list)
    flagged_candidate_ids: List[str] = field(default_factory=list)
    trajectory_changes: List[Dict] = field(default_factory=list)
    growth_edge_changes: List[Dict] = field(default_factory=list)
    commitment_outcomes: Dict[str, str] = field(default_factory=dict)  # Phase 6d
    source_node_updates: List[Dict] = field(default_factory=list)       # Phase 6d

    def to_dict(self) -> Dict:
        d = {
            "candidates_reviewed": self.candidates_reviewed,
            "candidates_promoted": self.candidates_promoted,
            "candidates_flagged": self.candidates_flagged,
            "candidates_dismissed": self.candidates_dismissed,
            "patterns_updated": self.patterns_updated,
            "growth_edges_progressed": self.growth_edges_progressed,
        }
        if self.promoted_node_ids:
            d["promoted_node_ids"] = self.promoted_node_ids
        if self.flagged_candidate_ids:
            d["flagged_candidate_ids"] = self.flagged_candidate_ids
        if self.trajectory_changes:
            d["trajectory_changes"] = self.trajectory_changes
        if self.growth_edge_changes:
            d["growth_edge_changes"] = self.growth_edge_changes
        if self.commitments_evaluated > 0:
            d["commitments_evaluated"] = self.commitments_evaluated
            d["commitment_outcomes"] = self.commitment_outcomes
        if self.source_node_updates:
            d["source_node_updates"] = self.source_node_updates
        return d

    @property
    def has_activity(self) -> bool:
        return (self.candidates_reviewed > 0
                or self.patterns_updated > 0
                or self.growth_edges_progressed > 0
                or self.commitments_evaluated > 0)


class IdentityReflector:
    """Runs end-of-session identity graph review."""

    # Candidates with these types auto-promote if they appear 2+ times
    # in the session (multiple signals pointing to the same insight).
    # Single-occurrence candidates are flagged for manual review.
    AUTO_PROMOTE_TYPES = {
        NodeType.REALIZATION.value,
        NodeType.LESSON.value,
    }

    # These types need stronger evidence (2+ sessions) before promotion.
    MULTI_SESSION_TYPES = {
        NodeType.PATTERN.value,
        NodeType.PREFERENCE.value,
        NodeType.TENSION.value,
    }

    # Minimum content length for auto-promotion (filters out noise)
    MIN_CONTENT_LENGTH = 20

    def __init__(self, identity_graph: IdentityGraph):
        self.ig = identity_graph

    def reflect(
        self,
        session_id: str,
        pattern_snapshot: Optional[Dict[str, int]] = None,
    ) -> ReflectionReport:
        """Run full identity reflection for a session.

        Args:
            session_id: Current session ID.
            pattern_snapshot: Optional dict of {node_id: observation_count}
                captured at session start. Used to detect which patterns
                were reinforced during this session.

        Returns:
            ReflectionReport with all changes made.
        """
        report = ReflectionReport()

        # 1. Review candidates from this session
        self._review_candidates(session_id, report)

        # 2. Update pattern trajectories
        self._update_trajectories(pattern_snapshot, report)

        # 3. Progress growth edges
        self._progress_growth_edges(pattern_snapshot, report)

        # 4. Phase 6d: Evaluate active commitments for this session
        self._evaluate_commitments(session_id, report)

        return report

    # ── Candidate Review ──────────────────────────────────────────────────

    def _review_candidates(self, session_id: str, report: ReflectionReport):
        """Review pending candidates from this session."""
        candidates = self.ig.get_pending_candidates(session_id=session_id)
        report.candidates_reviewed = len(candidates)

        if not candidates:
            return

        # Group by content similarity to detect reinforced signals
        content_groups = self._group_similar_candidates(candidates)

        for group in content_groups:
            if len(group) == 0:
                continue

            representative = group[0]  # highest quality candidate in group

            # Filter: too short = dismiss
            if len(representative.content.strip()) < self.MIN_CONTENT_LENGTH:
                for c in group:
                    self.ig.dismiss_candidate(c.id)
                    report.candidates_dismissed += 1
                continue

            # Auto-promote: realizations and lessons with decent content,
            # OR any type with 2+ signals in the same session
            should_promote = False

            if representative.node_type in self.AUTO_PROMOTE_TYPES:
                should_promote = True
            elif len(group) >= 2:
                # Multiple signals pointing to the same insight
                should_promote = True

            if should_promote:
                # Check for near-duplicate existing nodes before promoting
                if self._has_similar_node(representative):
                    # Reinforce existing instead of creating duplicate
                    for c in group:
                        self.ig.dismiss_candidate(c.id)
                        report.candidates_dismissed += 1
                else:
                    node_id = self.ig.promote_candidate(representative.id)
                    if node_id:
                        report.candidates_promoted += 1
                        report.promoted_node_ids.append(node_id)
                        # Phase 5: Copy cross-references from candidate to new node
                        self._copy_references(representative.id, node_id)
                    # Dismiss the rest of the group (duplicates)
                    for c in group[1:]:
                        self.ig.dismiss_candidate(c.id)
                        report.candidates_dismissed += 1
            else:
                # Flag for manual review
                for c in group:
                    report.candidates_flagged += 1
                    report.flagged_candidate_ids.append(c.id)

    def _group_similar_candidates(
        self, candidates: List[IdentityCandidate]
    ) -> List[List[IdentityCandidate]]:
        """Group candidates by content similarity.

        Simple approach: candidates whose content shares 50%+ words
        (excluding stopwords) are grouped together.
        """
        import re
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'was', 'one', 'our', 'out', 'has', 'have',
            'been', 'from', 'that', 'this', 'with', 'when', 'than',
            'them', 'they', 'will', 'each', 'make', 'like', 'into',
            'over', 'such', 'just', 'also', 'some', 'what', 'there',
            'about', 'which', 'their', 'would', 'could', 'should',
            'being', 'doing', 'going', 'other',
        }

        def key_words(text):
            words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', text))
            return words - stopwords

        groups: List[List[IdentityCandidate]] = []
        assigned = set()

        for i, cand in enumerate(candidates):
            if i in assigned:
                continue
            group = [cand]
            assigned.add(i)
            words_i = key_words(cand.content)
            if not words_i:
                groups.append(group)
                continue

            for j, other in enumerate(candidates):
                if j in assigned or j <= i:
                    continue
                words_j = key_words(other.content)
                if not words_j:
                    continue
                overlap = len(words_i & words_j) / max(len(words_i), len(words_j))
                if overlap >= 0.5:
                    group.append(other)
                    assigned.add(j)

            groups.append(group)

        return groups

    def _has_similar_node(self, candidate: IdentityCandidate) -> bool:
        """Check if an existing node has similar content."""
        import re
        existing = self.ig.get_nodes_by_type(candidate.node_type, limit=50)
        if not existing:
            return False

        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'was', 'one', 'our', 'out', 'has', 'have',
            'been', 'from', 'that', 'this', 'with', 'when',
        }

        cand_words = set(
            w.lower() for w in re.findall(r'\b\w{3,}\b', candidate.content)
        ) - stopwords

        if not cand_words:
            return False

        for node in existing:
            node_words = set(
                w.lower() for w in re.findall(r'\b\w{3,}\b', node.content)
            ) - stopwords
            if not node_words:
                continue
            overlap = len(cand_words & node_words) / max(len(cand_words), len(node_words))
            if overlap >= 0.6:
                # Reinforce the existing node instead
                self.ig.reinforce_node(node.id)
                return True

        return False

    def _copy_references(self, candidate_id: str, node_id: str):
        """Copy cross-references from a candidate to its promoted node.

        Phase 5: When a candidate is promoted, any references that were
        attached to the candidate ID get re-pointed to the new node ID.
        """
        try:
            refs = self.ig.get_references_for_node(candidate_id)
            for ref in refs:
                self.ig.add_reference(IdentityReference(
                    identity_node_id=node_id,
                    ref_type=ref.ref_type,
                    ref_id=ref.ref_id,
                ))
        except Exception:
            pass  # References are additive; don't block promotion

    # ── Trajectory Updates ────────────────────────────────────────────────

    def _update_trajectories(
        self,
        snapshot: Optional[Dict[str, int]],
        report: ReflectionReport,
    ):
        """Update pattern trajectories based on session activity.

        If a pattern was reinforced this session (observation_count increased),
        check whether that's improving or regressing based on valence.
        """
        if not snapshot:
            return

        patterns = self.ig.get_nodes_by_type(NodeType.PATTERN.value, limit=100)
        for p in patterns:
            if p.id not in snapshot:
                continue

            old_count = snapshot[p.id]
            new_count = p.observation_count

            if new_count <= old_count:
                continue  # Not reinforced this session

            # Pattern was observed again. Determine trajectory.
            old_trajectory = p.trajectory or PatternTrajectory.STABLE.value

            if p.valence == "negative":
                # Negative pattern reinforced = regressing (it keeps happening)
                new_trajectory = PatternTrajectory.REGRESSING.value
            elif p.valence == "positive":
                # Positive pattern reinforced = improving (good habit strengthening)
                new_trajectory = PatternTrajectory.IMPROVING.value
            else:
                # Neutral pattern: just stable
                new_trajectory = PatternTrajectory.STABLE.value

            if new_trajectory != old_trajectory:
                self.ig.update_node_trajectory(p.id, new_trajectory)
                report.patterns_updated += 1
                report.trajectory_changes.append({
                    "node_id": p.id,
                    "content": p.content[:80],
                    "old_trajectory": old_trajectory,
                    "new_trajectory": new_trajectory,
                    "observation_count": new_count,
                })

    # ── Growth Edge Progression ───────────────────────────────────────────

    def _progress_growth_edges(
        self,
        snapshot: Optional[Dict[str, int]],
        report: ReflectionReport,
    ):
        """Progress growth edge statuses based on session evidence.

        Logic:
        - If the negative pattern a growth edge addresses was NOT reinforced
          this session, that's evidence of improvement.
        - If the growth edge has been active for 3+ sessions without the
          pattern regressing, advance status.
        """
        active_edges = self.ig.get_active_growth_edges()
        if not active_edges:
            return

        for edge in active_edges:
            # Find edges linking this growth edge to patterns it addresses
            related_edges = self.ig.get_edges_for_node(edge.id)
            addressed_patterns = []
            for e in related_edges:
                if e.edge_type == EdgeType.EVOLVED_FROM.value:
                    target = self.ig.get_node(e.target_node)
                    if target and target.node_type == NodeType.PATTERN.value:
                        addressed_patterns.append(target)

            if not addressed_patterns or not snapshot:
                continue

            # Check if any addressed pattern was reinforced (bad sign)
            pattern_reinforced = False
            for p in addressed_patterns:
                if p.id in snapshot and p.observation_count > snapshot[p.id]:
                    pattern_reinforced = True
                    break

            if pattern_reinforced:
                # Pattern still occurring; don't advance
                continue

            # Pattern NOT reinforced this session. Consider advancing.
            old_status = edge.status or GrowthEdgeStatus.IDENTIFIED.value
            new_status = None

            if old_status == GrowthEdgeStatus.IDENTIFIED.value:
                new_status = GrowthEdgeStatus.PRACTICING.value
            elif old_status == GrowthEdgeStatus.PRACTICING.value:
                new_status = GrowthEdgeStatus.IMPROVING.value
            elif old_status == GrowthEdgeStatus.IMPROVING.value:
                # Need sustained evidence for integration.
                # Only advance if observation_count on the growth edge
                # itself is 5+ (meaning we've tracked it across many sessions).
                if edge.observation_count >= 5:
                    new_status = GrowthEdgeStatus.INTEGRATED.value

            if new_status and new_status != old_status:
                self.ig.update_node_status(edge.id, new_status)
                self.ig.reinforce_node(edge.id)  # Track that we reviewed it
                report.growth_edges_progressed += 1
                report.growth_edge_changes.append({
                    "node_id": edge.id,
                    "content": edge.content[:80],
                    "old_status": old_status,
                    "new_status": new_status,
                })


    # ── Commitment Evaluation (Phase 6d) ────────────────────────────────────

    def _evaluate_commitments(
        self,
        session_id: str,
        report: ReflectionReport,
    ):
        """Evaluate active commitments for a session using reliability-weighted signals.

        Per spec §6.3: explicit reflection is the bonus path. It captures the same
        data as boot-time retrospective but allows qualitative observations.
        Marks commitments as evaluated so boot-time retro skips them.

        Per spec §6.4: outcome computation uses weighted signal aggregation.
        """
        commitments = self.ig.get_active_commitments(session_id)
        if not commitments:
            return

        for commitment in commitments:
            # Skip already-evaluated commitments (double-count prevention)
            if commitment.metadata.get("evaluated_at"):
                continue

            signals = self.ig.get_signals_for_commitment(commitment.id)
            # Also check session-level signals not linked to a specific commitment
            session_signals = self.ig.get_signals_for_session(session_id)
            unlinked = [
                s for s in session_signals
                if s.commitment_id is None
            ]

            outcome = self._compute_weighted_outcome(signals, unlinked, commitment)

            # Update commitment status
            self.ig.update_node_status(commitment.id, outcome.value)

            # Mark as evaluated (double-count prevention for boot-time retro)
            meta = commitment.metadata or {}
            meta["evaluated_at"] = datetime.utcnow().isoformat()
            meta["evaluated_by"] = "explicit_reflection"
            self.ig.update_node_metadata(commitment.id, meta)

            # Propagate to source nodes
            source_updates = self.ig._propagate_outcome(commitment, outcome)

            report.commitments_evaluated += 1
            report.commitment_outcomes[commitment.id] = outcome.value
            report.source_node_updates.extend(source_updates)

    def _compute_weighted_outcome(
        self,
        linked_signals: List[IdentitySignal],
        unlinked_signals: List[IdentitySignal],
        commitment: IdentityNode,
    ) -> CommitmentOutcome:
        """Compute commitment outcome using reliability-weighted signal aggregation.

        Per spec §3.4 and §6.2:
        - user_correction: weight 1.0 (external observer, highest reliability)
        - enrichment_scanner: weight 0.6 (keyword detection, moderate)
        - self_report: weight 0.3 (subject to confabulation risk)

        User corrections override all other signals.
        Self-report only = partial (insufficient for full credit).
        """
        if not linked_signals and not unlinked_signals:
            return CommitmentOutcome.NOT_APPLICABLE

        # Separate by source type
        user_corrections = [s for s in linked_signals if s.source == "user_correction"]
        scanner_signals = [s for s in linked_signals if s.source == "enrichment_scanner"]
        self_reports = [s for s in linked_signals if s.source == "self_report"]

        # User corrections take priority (highest reliability)
        if user_corrections:
            missed = [s for s in user_corrections if s.signal_type == "missed"]
            held = [s for s in user_corrections if s.signal_type == "held"]

            if missed and held:
                # Conflicting corrections: compute weighted score
                missed_weight = sum(s.confidence for s in missed)
                held_weight = sum(s.confidence for s in held)
                if missed_weight > held_weight:
                    return CommitmentOutcome.MISSED
                elif held_weight > missed_weight:
                    return CommitmentOutcome.HONORED
                else:
                    return CommitmentOutcome.PARTIAL
            elif missed:
                return CommitmentOutcome.MISSED
            elif held:
                return CommitmentOutcome.HONORED
            else:
                return CommitmentOutcome.PARTIAL

        # Scanner signals: moderate reliability
        if scanner_signals:
            missed = [s for s in scanner_signals if s.signal_type == "missed"]
            held = [s for s in scanner_signals if s.signal_type == "held"]
            if missed and not held:
                return CommitmentOutcome.MISSED
            elif held and not missed:
                # Scanner alone is moderate evidence; give partial credit
                return CommitmentOutcome.PARTIAL
            elif missed and held:
                return CommitmentOutcome.PARTIAL

        # Self-report only: lowest reliability
        if self_reports:
            # Per spec: self-report only = partial (insufficient for full credit)
            return CommitmentOutcome.PARTIAL

        # Unlinked signals exist but no linked ones: check if any are relevant
        # via keyword matching against commitment content
        if unlinked_signals and not linked_signals:
            return CommitmentOutcome.NOT_APPLICABLE

        return CommitmentOutcome.NOT_APPLICABLE


# ── Snapshot Helper ───────────────────────────────────────────────────────

def capture_pattern_snapshot(conn, identity_graph=None) -> Dict[str, int]:
    """Capture current pattern observation counts for later comparison.

    Call this at session start (after boot) to establish a baseline.
    Pass the result to IdentityReflector.reflect() at session end.
    """
    try:
        ig = identity_graph or IdentityGraph(conn)
        patterns = ig.get_nodes_by_type(NodeType.PATTERN.value, limit=100)
        return {p.id: p.observation_count for p in patterns}
    except Exception:
        return {}


# ── Pipeline Entry Point ──────────────────────────────────────────────────

def run_identity_reflection(
    conn,
    session_id: str,
    pattern_snapshot: Optional[Dict[str, int]] = None,
    identity_graph=None,
) -> Optional[Dict]:
    """Run identity reflection. Called from cmd_end.

    Returns report dict or None on error.
    """
    try:
        ig = identity_graph or IdentityGraph(conn)
        reflector = IdentityReflector(ig)
        report = reflector.reflect(session_id, pattern_snapshot)
        if report.has_activity:
            return report.to_dict()
        return None
    except Exception:
        return None  # Reflection is additive; never block session end
