"""
The Librarian — Identity Enrichment Scanner (Phase 3)

Lightweight introspective scanner that runs in the ingestion pipeline.
Detects six signal types from incoming content and writes candidates
to the identity_candidates table for later review/promotion.

Signals (from spec §5.1):
1. Realization — learned something about self
2. Pattern reinforcement — known pattern surfaced again
3. Preference — differential engagement detected
4. User correction — user corrected a behavior
5. Tension — unresolved contradiction encountered
6. Lesson — experience produced concrete change

Design: keyword heuristics only (no LLM call). Additive-only; never
blocks ingestion. Candidates require explicit promotion to become nodes.
"""
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from solitaire.storage.identity_graph import (
    IdentityGraph, IdentityNode, IdentityCandidate, IdentityReference,
    IdentitySignal, NodeType, EdgeType, CommitmentOutcome,
)


# ── Signal Types ──────────────────────────────────────────────────────────

@dataclass
class EnrichmentSignal:
    """A detected identity signal from content analysis."""
    signal_type: str          # matches NodeType values
    content: str              # the extracted insight
    confidence: float         # 0.0-1.0, how confident the detection is
    source_text: str          # the original text that triggered detection
    reinforces_node: Optional[str] = None  # existing node ID if reinforcement


# ── Keyword Patterns ──────────────────────────────────────────────────────

# Realization signals: self-referential understanding statements
REALIZATION_PATTERNS = [
    r"\bI (?:just )?(?:realized|noticed|discovered|understood|learned|recognized)\b",
    r"\b(?:it (?:occurs|occurred) to me|it struck me|I see now)\b",
    r"\bI (?:hadn't|hadn't|haven't) (?:considered|thought about|realized)\b",
    r"\b(?:this (?:means|suggests|implies) (?:I|my))\b",
    r"\bthat's (?:a|an) (?:blind spot|gap|limitation) (?:I|in my)\b",
]

# Preference signals: differential engagement markers
PREFERENCE_PATTERNS = [
    r"\bI (?:find|notice) (?:myself|I'm) (?:more )?(?:drawn to|engaged (?:by|with)|interested in)\b",
    r"\b(?:this is|that's) (?:more )?(?:interesting|engaging|compelling)\b",
    r"\bI (?:prefer|gravitate toward|lean toward|tend to favor)\b",
    r"\bthere's (?:something|more) (?:interesting|compelling) about\b",
    r"\bI (?:enjoy|like|appreciate) (?:this|these|that) (?:kind|type|sort) of\b",
]

# Tension signals: unresolved contradictions
TENSION_PATTERNS = [
    r"\bI (?:can't|cannot) (?:resolve|reconcile|determine|verify|distinguish)\b",
    r"\b(?:this|there's a) (?:tension|contradiction|conflict) between\b",
    r"\bI'm (?:not sure|uncertain) (?:whether|if|how) (?:I|my|this)\b",
    r"\b(?:on one hand|the question (?:is|remains)|I don't know (?:whether|if))\b",
    r"\b(?:is this|am I) (?:actually|really|genuinely)\b",
]

# Lesson signals: experience produced concrete change
LESSON_PATTERNS = [
    r"\b(?:the )?(?:takeaway|lesson|learning) (?:is|was|here)\b",
    r"\bnext time (?:I|we) should\b",
    r"\b(?:this|that)\b.{0,30}\b(?:taught|showed|proved|demonstrated) (?:me|us|that)\b",
    r"\b(?:I|we) (?:should (?:have|always)|need to (?:always|remember to))\b",
    r"\b(?:the fix|the solution|what worked) was\b",
    r"\b(?:never|always) (?:again|do this|skip)\b",
]

# User correction signals: user pointing out a behavioral issue
CORRECTION_PATTERNS = [
    r"\b(?:you're|you are) (?:doing it again|deflecting|avoiding|hedging|over-)\b",
    r"\b(?:stop|don't|quit) (?:doing that|hedging|deflecting|apologizing)\b",
    r"\b(?:that's not what I|I didn't ask for|I said)\b",
    r"\b(?:you missed|you forgot|you ignored|you skipped)\b",
    r"\b(?:be more|be less|try to be|you should be)\b",
    r"\b(?:that's the AI|that sounds like AI|too generic|too vague)\b",
]


# Commitment signal patterns: self-report markers [HELD:xxx] and [MISSED:xxx]
COMMITMENT_SIGNAL_PATTERNS = [
    r"\[HELD:([\w]+)\]",
    r"\[MISSED:([\w]+)\]",
]

# Signal reliability weights (from Vazire SOKA model, spec §3.4)
SIGNAL_WEIGHTS = {
    "user_correction": 1.0,
    "enrichment_scanner": 0.6,
    "self_report": 0.3,
}


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


COMPILED = {
    "realization": _compile_patterns(REALIZATION_PATTERNS),
    "preference": _compile_patterns(PREFERENCE_PATTERNS),
    "tension": _compile_patterns(TENSION_PATTERNS),
    "lesson": _compile_patterns(LESSON_PATTERNS),
    "correction": _compile_patterns(CORRECTION_PATTERNS),
}

COMPILED_COMMITMENT = [re.compile(p) for p in COMMITMENT_SIGNAL_PATTERNS]


# ── Scanner ───────────────────────────────────────────────────────────────

class IdentityEnrichmentScanner:
    """Scans ingested content for identity signals.

    Runs as part of the ingestion pipeline. Keyword-based (no LLM).
    Writes candidates to identity_candidates for later review.
    """

    def __init__(self, identity_graph: IdentityGraph, session_id: str):
        self.ig = identity_graph
        self.session_id = session_id

    def scan(
        self,
        content: str,
        role: str = "assistant",
        rolodex_entry_id: Optional[str] = None,
        kg_entity_ids: Optional[List[str]] = None,
    ) -> Dict:
        """Scan content for identity signals. Returns stats dict.

        Args:
            content: The ingested message text.
            role: 'user' or 'assistant'. User messages are checked for
                  corrections; assistant messages for self-referential signals.
            rolodex_entry_id: If provided, attach as cross-reference to candidates.
            kg_entity_ids: If provided, attach KG entity refs to candidates.

        Returns:
            Dict with signal counts and candidate IDs created.
        """
        self._pending_refs = {
            "rolodex_entry_id": rolodex_entry_id,
            "kg_entity_ids": kg_entity_ids or [],
        }

        signals: List[EnrichmentSignal] = []

        if role == "user":
            # User messages: check for corrections only
            signals.extend(self._detect_corrections(content))
        else:
            # Assistant messages: check for self-referential signals
            signals.extend(self._detect_realizations(content))
            signals.extend(self._detect_preferences(content))
            signals.extend(self._detect_tensions(content))
            signals.extend(self._detect_lessons(content))

        # Both roles: check for pattern reinforcement against existing nodes
        reinforcements = self._detect_pattern_reinforcement(content)

        # Phase 6c: Commitment signal detection
        commitment_signals_written = []

        # 6c.1: Detect [HELD:xxx] / [MISSED:xxx] self-report markers
        # (assistant messages only; these are internal markers)
        if role == "assistant":
            commitment_signals_written.extend(
                self._detect_commitment_signals(content)
            )

        # 6c.2: Match user corrections to active commitments
        if role == "user":
            commitment_signals_written.extend(
                self._match_correction_to_commitments(content)
            )

        # Write candidates for new signals (not reinforcements)
        candidates_created = []
        for sig in signals:
            if sig.confidence >= 0.3:  # minimum confidence threshold
                cand_id = self._write_candidate(sig)
                candidates_created.append(cand_id)

        # Apply reinforcements directly (no candidate needed)
        # Also attach references to reinforced nodes
        reinforced_nodes = []
        for node_id in reinforcements:
            self.ig.reinforce_node(node_id)
            self._attach_references_to_node(node_id)
            reinforced_nodes.append(node_id)

        stats = {
            "signals_detected": len(signals),
            "candidates_created": len(candidates_created),
            "patterns_reinforced": len(reinforced_nodes),
        }
        if candidates_created:
            stats["candidate_ids"] = candidates_created
        if reinforced_nodes:
            stats["reinforced_node_ids"] = reinforced_nodes
        if commitment_signals_written:
            stats["commitment_signals"] = len(commitment_signals_written)
            stats["commitment_signal_ids"] = commitment_signals_written

        return stats

    # ── Signal Detectors ──────────────────────────────────────────────────

    def _detect_realizations(self, content: str) -> List[EnrichmentSignal]:
        """Detect realization signals in assistant content."""
        signals = []
        for pattern in COMPILED["realization"]:
            matches = pattern.finditer(content)
            for m in matches:
                # Extract the sentence containing the match
                sentence = self._extract_sentence(content, m.start())
                if len(sentence) < 15:
                    continue  # too short to be meaningful
                signals.append(EnrichmentSignal(
                    signal_type=NodeType.REALIZATION.value,
                    content=sentence,
                    confidence=0.5,
                    source_text=m.group(),
                ))
        return self._deduplicate_signals(signals)

    def _detect_preferences(self, content: str) -> List[EnrichmentSignal]:
        """Detect preference signals in assistant content."""
        signals = []
        for pattern in COMPILED["preference"]:
            matches = pattern.finditer(content)
            for m in matches:
                sentence = self._extract_sentence(content, m.start())
                if len(sentence) < 15:
                    continue
                signals.append(EnrichmentSignal(
                    signal_type=NodeType.PREFERENCE.value,
                    content=sentence,
                    confidence=0.4,
                    source_text=m.group(),
                ))
        return self._deduplicate_signals(signals)

    def _detect_tensions(self, content: str) -> List[EnrichmentSignal]:
        """Detect tension signals in assistant content."""
        signals = []
        for pattern in COMPILED["tension"]:
            matches = pattern.finditer(content)
            for m in matches:
                sentence = self._extract_sentence(content, m.start())
                if len(sentence) < 15:
                    continue
                signals.append(EnrichmentSignal(
                    signal_type=NodeType.TENSION.value,
                    content=sentence,
                    confidence=0.4,
                    source_text=m.group(),
                ))
        return self._deduplicate_signals(signals)

    def _detect_lessons(self, content: str) -> List[EnrichmentSignal]:
        """Detect lesson signals in assistant content."""
        signals = []
        for pattern in COMPILED["lesson"]:
            matches = pattern.finditer(content)
            for m in matches:
                sentence = self._extract_sentence(content, m.start())
                if len(sentence) < 15:
                    continue
                signals.append(EnrichmentSignal(
                    signal_type=NodeType.LESSON.value,
                    content=sentence,
                    confidence=0.5,
                    source_text=m.group(),
                ))
        return self._deduplicate_signals(signals)

    def _detect_corrections(self, content: str) -> List[EnrichmentSignal]:
        """Detect user correction signals."""
        signals = []
        for pattern in COMPILED["correction"]:
            matches = pattern.finditer(content)
            for m in matches:
                sentence = self._extract_sentence(content, m.start())
                if len(sentence) < 10:
                    continue
                # Corrections map to pattern nodes (negative behavioral pattern)
                signals.append(EnrichmentSignal(
                    signal_type=NodeType.PATTERN.value,
                    content=f"User correction: {sentence}",
                    confidence=0.6,  # higher confidence for direct corrections
                    source_text=m.group(),
                ))
        return self._deduplicate_signals(signals)

    def _detect_pattern_reinforcement(self, content: str) -> List[str]:
        """Check if content matches any existing pattern nodes.

        Returns list of node IDs that were reinforced.
        """
        reinforced = []
        existing_patterns = self.ig.get_nodes_by_type(
            NodeType.PATTERN.value, limit=50
        )
        if not existing_patterns:
            return reinforced

        content_lower = content.lower()
        for node in existing_patterns:
            # Extract key terms from the pattern content (3+ char words)
            pattern_words = set(
                w.lower() for w in re.findall(r'\b\w{3,}\b', node.content)
            )
            # Remove very common words
            stopwords = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                'have', 'been', 'from', 'that', 'this', 'with', 'when',
                'than', 'them', 'they', 'will', 'each', 'make', 'like',
                'into', 'over', 'such', 'just', 'also', 'some', 'what',
                'there', 'about', 'which', 'their', 'would', 'could',
                'should', 'being', 'doing', 'going', 'other',
            }
            key_terms = pattern_words - stopwords
            if len(key_terms) < 2:
                continue

            # Require at least 40% of key terms to appear in content
            matches = sum(1 for t in key_terms if t in content_lower)
            match_ratio = matches / len(key_terms)
            if match_ratio >= 0.4:
                reinforced.append(node.id)

        return reinforced

    # ── Commitment Signal Detectors (Phase 6c) ─────────────────────────────

    def _detect_commitment_signals(self, content: str) -> List[str]:
        """Detect [HELD:xxx] and [MISSED:xxx] self-report markers in content.

        Writes directly to identity_signals table (not candidates).
        Returns list of signal IDs written.

        Per spec §5.1: self-report signals carry weight 0.3.
        """
        written = []
        for pattern in COMPILED_COMMITMENT:
            for match in pattern.finditer(content):
                full_match = match.group(0)
                short_id = match.group(1)

                # Determine signal type from the tag
                if full_match.startswith("[HELD:"):
                    signal_type = "held"
                elif full_match.startswith("[MISSED:"):
                    signal_type = "missed"
                else:
                    continue

                # Find the commitment node by short ID prefix
                commitment_id = self._resolve_commitment_id(short_id)

                # Extract surrounding context for the signal content
                sentence = self._extract_sentence(content, match.start())

                signal = IdentitySignal(
                    id="",  # auto-generated
                    session_id=self.session_id,
                    commitment_id=commitment_id,
                    signal_type=signal_type,
                    content=sentence,
                    source="self_report",
                    confidence=SIGNAL_WEIGHTS["self_report"],
                )
                sig_id = self.ig.add_signal(signal)
                written.append(sig_id)

        return written

    def _match_correction_to_commitments(self, content: str) -> List[str]:
        """Match user corrections to active commitments by keyword overlap.

        Per spec §5.2: user corrections are the highest-reliability signal.
        Matching uses the same 40% key-term overlap as pattern reinforcement.

        Returns list of signal IDs written.
        """
        # First check if this is actually a correction
        is_correction = False
        for pattern in COMPILED["correction"]:
            if pattern.search(content):
                is_correction = True
                break

        if not is_correction:
            return []

        # Get active commitments for this session
        active_commitments = self.ig.get_active_commitments(self.session_id)
        if not active_commitments:
            # Also check commitments without session filter (cross-session)
            active_commitments = self.ig.get_active_commitments()

        if not active_commitments:
            return []

        written = []
        content_lower = content.lower()

        for commitment in active_commitments:
            # Get the source node for richer keyword matching
            source_id = commitment.metadata.get("source_node")
            match_text = commitment.content
            if source_id:
                source_node = self.ig.get_node(source_id)
                if source_node:
                    match_text = source_node.content + " " + commitment.content

            # Extract key terms (same approach as pattern reinforcement)
            key_words = set(
                w.lower() for w in re.findall(r'\b\w{3,}\b', match_text)
            )
            stopwords = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                'have', 'been', 'from', 'that', 'this', 'with', 'when',
                'than', 'them', 'they', 'will', 'each', 'make', 'like',
                'into', 'over', 'such', 'just', 'also', 'some', 'what',
                'there', 'about', 'which', 'their', 'would', 'could',
                'should', 'being', 'doing', 'going', 'other',
                'practice', 'watch', 'signal', 'held', 'missed',
            }
            key_terms = key_words - stopwords
            if len(key_terms) < 2:
                continue

            # 40% overlap threshold
            matches = sum(1 for t in key_terms if t in content_lower)
            match_ratio = matches / len(key_terms)

            if match_ratio >= 0.4:
                signal = IdentitySignal(
                    id="",
                    session_id=self.session_id,
                    commitment_id=commitment.id,
                    signal_type="missed",  # user corrections signal missed behavior
                    content=f"User correction: {content[:300]}",
                    source="user_correction",
                    confidence=SIGNAL_WEIGHTS["user_correction"],
                )
                sig_id = self.ig.add_signal(signal)
                written.append(sig_id)

        return written

    def _resolve_commitment_id(self, short_id: str) -> Optional[str]:
        """Resolve a short commitment ID prefix to a full node ID.

        Commitments use IDs like idn_abc123def456. The short_id in
        [HELD:abc123def456] or [HELD:abc123] is matched as a prefix.
        """
        # Try exact match with idn_ prefix
        full_id = f"idn_{short_id}"
        node = self.ig.get_node(full_id)
        if node and node.node_type == NodeType.COMMITMENT.value:
            return full_id

        # Try prefix match against active commitments
        active = self.ig.get_active_commitments(self.session_id)
        for c in active:
            if c.id.endswith(short_id) or short_id in c.id:
                return c.id

        # No match found; signal still recorded with null commitment_id
        return None

    # ── Helpers ────────────────────────────────────────────────────────────

    def _extract_sentence(self, text: str, position: int) -> str:
        """Extract the sentence containing the given position.

        Looks backward for sentence start and forward for sentence end.
        Caps at 300 chars to avoid runaway extraction.
        """
        # Find sentence start (look backward for period, newline, or start)
        start = position
        while start > 0 and start > position - 300:
            if text[start - 1] in '.!?\n':
                break
            start -= 1

        # Find sentence end (look forward for period, newline, or end)
        end = position
        while end < len(text) and end < position + 300:
            if text[end] in '.!?\n':
                end += 1  # include the punctuation
                break
            end += 1

        return text[start:end].strip()

    def _deduplicate_signals(self, signals: List[EnrichmentSignal]) -> List[EnrichmentSignal]:
        """Remove duplicate signals from the same sentence."""
        seen_content = set()
        unique = []
        for sig in signals:
            # Normalize: strip whitespace, lowercase for comparison
            key = sig.content.strip().lower()[:100]
            if key not in seen_content:
                seen_content.add(key)
                unique.append(sig)
        return unique

    def _write_candidate(self, signal: EnrichmentSignal) -> str:
        """Write a signal as an identity candidate, with cross-references."""
        candidate = IdentityCandidate(
            id="",  # auto-generated
            session_id=self.session_id,
            node_type=signal.signal_type,
            content=signal.content,
            signal_source=signal.source_text,
        )
        cand_id = self.ig.add_candidate(candidate)

        # Store pending references in candidate metadata.
        # These will be attached to the node when promoted (Phase 4 reflection).
        # For now, store them as identity_references pointing to the candidate ID.
        # When promoted, the reflection module copies them to the new node.
        self._attach_references_to_candidate(cand_id)

        return cand_id

    def _attach_references_to_candidate(self, candidate_id: str):
        """Attach cross-references to a candidate (stored as references)."""
        refs = getattr(self, '_pending_refs', {})

        # Rolodex entry reference
        rolodex_id = refs.get("rolodex_entry_id")
        if rolodex_id:
            self.ig.add_reference(IdentityReference(
                identity_node_id=candidate_id,
                ref_type="rolodex_entry",
                ref_id=rolodex_id,
            ))

        # Session reference
        if self.session_id:
            self.ig.add_reference(IdentityReference(
                identity_node_id=candidate_id,
                ref_type="session",
                ref_id=self.session_id,
            ))

        # KG entity references
        for entity_id in refs.get("kg_entity_ids", []):
            self.ig.add_reference(IdentityReference(
                identity_node_id=candidate_id,
                ref_type="entity_node",
                ref_id=entity_id,
            ))

    def _attach_references_to_node(self, node_id: str):
        """Attach cross-references to an existing node (for reinforcements)."""
        refs = getattr(self, '_pending_refs', {})

        rolodex_id = refs.get("rolodex_entry_id")
        if rolodex_id:
            self.ig.add_reference(IdentityReference(
                identity_node_id=node_id,
                ref_type="rolodex_entry",
                ref_id=rolodex_id,
            ))


# ── Pipeline Entry Point ──────────────────────────────────────────────────

def run_identity_enrichment(
    conn,
    session_id: str,
    content: str,
    role: str = "assistant",
    rolodex_entry_id: Optional[str] = None,
    kg_entity_ids: Optional[List[str]] = None,
    identity_graph=None,
) -> Optional[Dict]:
    """Run identity enrichment on ingested content.

    Called from the ingestion pipeline in librarian_cli.py.
    Returns stats dict or None if no signals detected.
    """
    try:
        ig = identity_graph or IdentityGraph(conn)
        scanner = IdentityEnrichmentScanner(ig, session_id)
        stats = scanner.scan(
            content, role=role,
            rolodex_entry_id=rolodex_entry_id,
            kg_entity_ids=kg_entity_ids,
        )
        if (stats["signals_detected"] > 0
                or stats["patterns_reinforced"] > 0
                or stats.get("commitment_signals", 0) > 0):
            return stats
        return None
    except Exception:
        return None  # Enrichment is additive; never block ingestion
