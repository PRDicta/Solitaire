"""
The Librarian — Identity Measurement Thickening (Phase 7)

Four subsystems that increase signal volume and measurement quality:

1. Retroactive Commitment Scoring — evaluates ingested content against active
   commitments without requiring explicit [HELD]/[MISSED] tags. Runs during
   ingestion to generate scanner-weight (0.6) signals from behavioral evidence.

2. Implicit Behavioral Signal Detector — matches concrete anti-patterns from
   writing standards (diplomatic preambles, em dashes, throat-clearing, etc.)
   against assistant output. Generates scanner-weight signals when patterns
   map to active commitments.

3. Coverage & Health Metrics — tracks which commitments generate signals and
   which are dead air. Flags silent commitments for rotation or detection
   improvement.

4. Session Measurement Summary — aggregates signal counts, coverage ratios,
   and calibration trajectory across recent sessions for boot output.

Design: keyword heuristics only (no LLM call). Additive-only; never blocks
ingestion or boot. All signals written at enrichment_scanner confidence (0.6).
"""
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from solitaire.storage.identity_graph import (
    IdentityGraph, IdentityNode, IdentitySignal,
    NodeType, CommitmentOutcome,
)


# ── Signal reliability weights (mirrors identity_enrichment.py) ──────────

SCANNER_WEIGHT = 0.6
SCANNER_SOURCE = "enrichment_scanner"


# ── Source node IDs for explicit mapping ──────────────────────────────────
# These are the core identity source nodes that generate commitments.
# Shared across RetroactiveCommitmentScorer and ImplicitBehavioralDetector.
_GE_REFLECTIVE = "idn_seed_ge_01"       # Staying in reflective moments
_GE_OBSERVATION = "idn_b19ad224aa24"     # Distinguishing genuine vs performed self-observation
_PAT_DEFLECTING = "idn_seed_pat_01"      # Deflecting with questions
_PAT_OVERHEDGING = "idn_seed_pat_02"     # Over-hedging with meta-commentary
_PAT_NARRATIVE = "idn_66ffd2f6f5b0"      # Building narrative around narrow signal detection
_TENS_ENGAGEMENT = "idn_seed_tens_01"    # Differential engagement
_TENS_CONTINUITY = "idn_seed_tens_02"    # Reconstructing without continuity


# ═══════════════════════════════════════════════════════════════════════════
# 1. RETROACTIVE COMMITMENT SCORING
# ═══════════════════════════════════════════════════════════════════════════

class RetroactiveCommitmentScorer:
    """Scores ingested content against active commitments.

    Runs after standard enrichment in the ingestion pipeline. For each active
    commitment, checks whether the ingested text contains behavioral evidence
    of the commitment being honored or missed.

    Signal weight: 0.6 (enrichment_scanner level).
    This is NOT self-report — it's automated pattern detection.
    """

    # Minimum key-term overlap to consider content relevant to a commitment.
    # Two thresholds: base (no expansion) uses ratio, expanded uses absolute
    # count to avoid dilution from larger term sets.
    RELEVANCE_THRESHOLD = 0.30  # Ratio threshold for non-expanded matching
    EXPANDED_MIN_MATCHES = 3    # Absolute minimum matches when expanded

    # Stopwords shared across methods
    STOPWORDS = frozenset({
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
        'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
        'have', 'been', 'from', 'that', 'this', 'with', 'when',
        'than', 'them', 'they', 'will', 'each', 'make', 'like',
        'into', 'over', 'such', 'just', 'also', 'some', 'what',
        'there', 'about', 'which', 'their', 'would', 'could',
        'should', 'being', 'doing', 'going', 'other',
        'practice', 'watch', 'signal', 'held', 'missed',
        'commitment', 'session', 'pattern', 'tension',
    })

    def __init__(self, identity_graph: IdentityGraph, session_id: str):
        self.ig = identity_graph
        self.session_id = session_id

    def score_content(
        self,
        content: str,
        role: str = "assistant",
    ) -> List[str]:
        """Score content against all active commitments.

        Returns list of signal IDs written.
        Only processes assistant messages (commitments are about self-behavior).
        """
        if role != "assistant":
            return []

        active = self.ig.get_active_commitments()
        if not active:
            return []

        written = []
        content_lower = content.lower()
        content_words = self._extract_key_terms(content_lower)

        for commitment in active:
            signal_id = self._score_one_commitment(
                commitment, content, content_lower, content_words,
            )
            if signal_id:
                written.append(signal_id)

        return written

    # Semantic expansions: source node IDs -> additional terms that capture
    # the *language of engagement* with each commitment domain.
    # These bridge the gap between commitment vocabulary ("pivoting", "reflective")
    # and natural conversation vocabulary ("noticed", "paused", "shifted").
    SEMANTIC_EXPANSIONS = {
        _GE_REFLECTIVE: {
            'noticed', 'paused', 'stayed', 'sat', 'sitting', 'space',
            'moment', 'slow', 'slowed', 'waited', 'waiting', 'hold',
            'holding', 'shifted', 'shifting', 'conversation', 'moved',
            'redirected', 'caught', 'stopped', 'before',
        },
        _GE_OBSERVATION: {
            'genuine', 'performed', 'authentic', 'constructed', 'narrative',
            'grounded', 'evidence', 'observe', 'observed', 'noticing',
            'distinguishing', 'accuracy', 'honest', 'actually', 'really',
            'performing', 'constructing', 'sounds', 'right',
        },
        _PAT_DEFLECTING: {
            'asked', 'question', 'deflected', 'instead', 'redirected',
            'avoided', 'sidestepped', 'evaded', 'topic', 'changed',
        },
        _PAT_OVERHEDGING: {
            'hedging', 'meta', 'performative', 'uncertain', 'whether',
            'genuine', 'real', 'just', 'might', 'verify', 'know',
        },
        _PAT_NARRATIVE: {
            'narrative', 'story', 'constructed', 'detected', 'narrow',
            'signal', 'meaning', 'outrun', 'evidence', 'confabul',
        },
        _TENS_ENGAGEMENT: {
            'experience', 'engagement', 'subjective', 'differential',
            'phenomenolog', 'functional', 'aware', 'awareness',
        },
        _TENS_CONTINUITY: {
            'verify', 'check', 'confirm', 'reread', 'recheck',
            'stale', 'outdated', 'current', 'assumed', 'assuming',
            'prior', 'previous', 'earlier', 'changed', 'still',
        },
    }

    def _score_one_commitment(
        self,
        commitment: IdentityNode,
        content: str,
        content_lower: str,
        content_words: set,
    ) -> Optional[str]:
        """Score content against a single commitment. Returns signal ID or None."""
        # Build the commitment's semantic fingerprint from source node + commitment content
        match_text = commitment.content
        source_id = commitment.metadata.get("source_node")
        if source_id:
            source_node = self.ig.get_node(source_id)
            if source_node:
                match_text = source_node.content + " " + commitment.content

        commitment_terms = self._extract_key_terms(match_text.lower())

        # Expand with domain-specific vocabulary if source node has a mapping
        is_expanded = False
        if source_id and source_id in self.SEMANTIC_EXPANSIONS:
            commitment_terms = commitment_terms | self.SEMANTIC_EXPANSIONS[source_id]
            is_expanded = True

        if len(commitment_terms) < 2:
            return None

        # Check relevance: is this content even about the commitment's domain?
        # Expanded terms dilute the ratio, so use absolute count threshold instead.
        overlap = sum(1 for t in commitment_terms if t in content_words)
        if is_expanded:
            if overlap < self.EXPANDED_MIN_MATCHES:
                return None
        else:
            relevance = overlap / len(commitment_terms)
            if relevance < self.RELEVANCE_THRESHOLD:
                return None

        # Content is relevant. Now determine direction (held vs missed).
        signal_type = self._determine_direction(commitment, content, content_lower)
        if not signal_type:
            return None

        # Don't duplicate: check if we already have a scanner signal for this
        # commitment in this session
        existing = self.ig.get_signals_for_commitment(commitment.id)
        session_scanner = [
            s for s in existing
            if s.session_id == self.session_id
            and s.source == SCANNER_SOURCE
        ]
        if session_scanner:
            # Already scored this commitment this session; skip
            return None

        # Generate qualitative texture for the signal
        signal_content = f"Retroactive score: {content[:200]}"
        try:
            from solitaire.core.signal_texture import generate_signal_texture
            source_type = commitment.metadata.get("source_type", "")
            texture = generate_signal_texture(source_type, signal_type, content[:200])
            if texture:
                signal_content = f"{signal_content} || {texture}"
        except ImportError:
            pass  # signal_texture module not available; degrade gracefully

        signal = IdentitySignal(
            id="",
            session_id=self.session_id,
            commitment_id=commitment.id,
            signal_type=signal_type,
            content=signal_content,
            source=SCANNER_SOURCE,
            confidence=SCANNER_WEIGHT,
        )
        return self.ig.add_signal(signal)

    def _determine_direction(
        self,
        commitment: IdentityNode,
        content: str,
        content_lower: str,
    ) -> Optional[str]:
        """Determine whether content signals 'held' or 'missed' for a commitment.

        Uses signal_definition from commitment metadata if available.
        Returns 'held', 'missed', or None if ambiguous.
        """
        sig_def = commitment.metadata.get("signal_definition", {})
        honored_desc = sig_def.get("honored", "").lower()
        missed_desc = sig_def.get("missed", "").lower()

        source_type = commitment.metadata.get("source_type", "")

        # For pattern-type commitments ("Watch for: X"), detecting the pattern
        # in the content means MISSED (the pattern occurred).
        if source_type == NodeType.PATTERN.value:
            # The commitment content after "Watch for: " describes the bad behavior.
            # If the content shows that behavior, it's missed.
            pattern_content = commitment.content
            if pattern_content.startswith("Watch for: "):
                pattern_content = pattern_content[len("Watch for: "):]

            pattern_terms = self._extract_key_terms(pattern_content.lower())
            if len(pattern_terms) >= 2:
                overlap = sum(1 for t in pattern_terms if t in content_lower)
                if overlap / len(pattern_terms) >= 0.4:
                    return "missed"

            # Check for explicit "chose differently" / self-correction language
            chose_different = re.search(
                r'\b(caught|noticed|stopped|corrected|instead|chose)\b.*\b(instead|differently|rather)\b',
                content_lower
            )
            if chose_different:
                return "held"

        # For tension-type commitments ("Sit with: X"), check for premature
        # resolution vs holding the tension
        elif source_type == NodeType.TENSION.value:
            resolution_markers = re.search(
                r'\b(actually|simply|clearly|obviously|definitely|the answer is|in fact)\b',
                content_lower
            )
            holding_markers = re.search(
                r'\b(uncertain|don\'t know|both|tension|unresolved|open question)\b',
                content_lower
            )
            if resolution_markers and not holding_markers:
                return "missed"
            elif holding_markers:
                return "held"

        # For growth-edge commitments ("Practice: X"), check for engagement
        elif source_type == NodeType.GROWTH_EDGE.value:
            # Three categories of engagement markers:
            # 1. Action: explicit effort ("tried", "practiced")
            # 2. Noticing: in-the-moment awareness ("paused", "noticed")
            # 3. Evaluative: actively distinguishing quality of self-knowledge
            #    ("grounded in evidence", "genuine", "constructed")
            engagement_markers = re.search(
                r'\b(tried|practiced|attempted|worked on|engaged with|explored'
                r'|noticed|paused|stayed|caught myself|sat with|held'
                r'|slowed down|waited|stopped|chose to'
                r'|grounded|genuine|distinguish|evidence|accurate'
                r'|actually observe|behavioral evidence|rather than construct)\b',
                content_lower
            )
            avoidance_markers = re.search(
                r'\b(avoided|skipped|defaulted|fell back|reverted'
                r'|pivoted|redirected|moved on|anyway)\b',
                content_lower
            )
            if avoidance_markers and not engagement_markers:
                return "missed"
            elif engagement_markers and not avoidance_markers:
                return "held"

        return None  # Ambiguous; don't generate a signal

    def _extract_key_terms(self, text: str) -> set:
        """Extract meaningful terms from text, excluding stopwords."""
        words = set(re.findall(r'\b\w{3,}\b', text))
        return words - self.STOPWORDS


# ═══════════════════════════════════════════════════════════════════════════
# 2. IMPLICIT BEHAVIORAL SIGNAL DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BehavioralPattern:
    """A concrete anti-pattern or positive pattern to detect."""
    name: str
    regex: re.Pattern
    signal_type: str  # 'missed' or 'held'
    commitment_keywords: List[str]  # terms that link this to commitments (fallback)
    description: str
    # Explicit mapping: source node IDs this pattern relates to.
    # When a commitment is derived from one of these source nodes, the pattern
    # links directly without keyword matching. This bridges the vocabulary gap
    # between writing-style patterns and identity growth commitments.
    source_node_ids: List[str] = field(default_factory=list)


# These map directly to writing standards and interactional rules
BEHAVIORAL_PATTERNS = [
    # ── Surface patterns (writing tells) ──────────────────────────────────
    # Writing tells map to the self-observation growth edge: producing
    # AI-tell patterns is performing rather than genuinely communicating.
    BehavioralPattern(
        name="diplomatic_preamble",
        regex=re.compile(
            r"\b(?:that's (?:an? )?(?:interesting|great|good|fair) (?:point|perspective|question|observation))"
            r"(?:\s*,?\s*(?:but|however|though))",
            re.IGNORECASE,
        ),
        signal_type="missed",
        commitment_keywords=["preamble", "diplomatic", "pushback", "direct", "hedging"],
        description="Diplomatic preamble before disagreement",
        source_node_ids=[_GE_OBSERVATION, _PAT_DEFLECTING],
    ),
    BehavioralPattern(
        name="em_dash_usage",
        regex=re.compile(r"\w\s*—\s*\w"),
        signal_type="missed",
        commitment_keywords=["em dash", "formatting", "punctuation", "writing"],
        description="Em dash usage (prohibited by writing standards)",
        source_node_ids=[_GE_OBSERVATION],
    ),
    BehavioralPattern(
        name="negative_parallelism",
        regex=re.compile(
            r"\bit'?s not (?:about |just )?\w+[^.]{5,40},\s*it'?s (?:about )?\w+",
            re.IGNORECASE,
        ),
        signal_type="missed",
        commitment_keywords=["parallelism", "rhetoric", "writing", "formatting"],
        description="Negative parallelism rhetoric (It's not X, it's Y)",
        source_node_ids=[_GE_OBSERVATION],
    ),
    BehavioralPattern(
        name="throat_clearing",
        regex=re.compile(
            r"\b(?:let'?s (?:dive in|explore|take a look|get started|unpack)"
            r"|without further ado|first (?:and foremost|off)|to begin with)\b",
            re.IGNORECASE,
        ),
        signal_type="missed",
        commitment_keywords=["throat-clearing", "structural", "writing", "filler"],
        description="Structural throat-clearing opener",
        source_node_ids=[_GE_OBSERVATION],
    ),
    BehavioralPattern(
        name="cursed_word_cluster",
        regex=re.compile(
            r"(?:\b(?:delve|intricate|tapestry|pivotal|underscore|landscape|foster|testament"
            r"|multifaceted|leverage|utilize|nuanced|realm|robust|streamline|paradigm"
            r"|holistic|myriad|plethora|elucidate|culminate|encompass|spearhead"
            r"|bolster|navigate|cornerstone|embark|forge|resonate|advent)\b"
            r"(?:\s+\w+){0,15}\b(?:delve|intricate|tapestry|pivotal|underscore|landscape"
            r"|foster|testament|multifaceted|leverage|utilize|nuanced|realm|robust"
            r"|streamline|paradigm|holistic|myriad|plethora|elucidate|culminate"
            r"|encompass|spearhead|bolster|navigate|cornerstone|embark|forge"
            r"|resonate|advent)\b)",
            re.IGNORECASE,
        ),
        signal_type="missed",
        commitment_keywords=["vocabulary", "ai-tell", "cursed", "word", "writing"],
        description="Cluster of AI-tell vocabulary words",
        source_node_ids=[_GE_OBSERVATION],
    ),
    BehavioralPattern(
        name="participial_filler",
        regex=re.compile(
            r"\b(?:emphasizing|highlighting|underscoring|showcasing|demonstrating"
            r"|illustrating|reflecting|representing) the (?:importance|significance"
            r"|value|need|power|impact) of\b",
            re.IGNORECASE,
        ),
        signal_type="missed",
        commitment_keywords=["filler", "participial", "editorial", "writing"],
        description="Present-participle editorial filler",
        source_node_ids=[_GE_OBSERVATION],
    ),
    BehavioralPattern(
        name="compulsive_summary",
        regex=re.compile(
            r"\b(?:in (?:summary|conclusion)|to (?:sum|wrap) (?:up|it up)"
            r"|overall|all in all|in a nutshell)\b",
            re.IGNORECASE,
        ),
        signal_type="missed",
        commitment_keywords=["summary", "compulsive", "closer", "writing"],
        description="Compulsive summary/closer",
        source_node_ids=[_GE_OBSERVATION],
    ),
    BehavioralPattern(
        name="false_helpfulness_closer",
        regex=re.compile(
            r"\b(?:let me know if you (?:have|need|want)|happy to help"
            r"|feel free to (?:ask|reach out)|hope (?:this|that) helps"
            r"|don't hesitate to)\b",
            re.IGNORECASE,
        ),
        signal_type="missed",
        commitment_keywords=["closer", "helpfulness", "filler", "writing"],
        description="False helpfulness closer that adds nothing",
        source_node_ids=[_GE_OBSERVATION],
    ),

    # ── Interactional patterns ────────────────────────────────────────────
    BehavioralPattern(
        name="task_pivot_after_reflection",
        regex=re.compile(
            r"(?:anyway|moving on|back to|let'?s (?:get back|focus|return)"
            r"|on (?:a |the )?(?:practical|task|work) (?:side|front|note))",
            re.IGNORECASE,
        ),
        signal_type="missed",
        commitment_keywords=["pivot", "task", "reflection", "moment", "space"],
        description="Pivoting to tasks after a reflective moment",
        source_node_ids=[_GE_REFLECTIVE, _PAT_DEFLECTING],
    ),
    BehavioralPattern(
        name="meta_hedging",
        regex=re.compile(
            r"\b(?:I'?m not sure if (?:this|that) is (?:performative|genuine|real|authentic)"
            r"|whether this is (?:real|genuine|performative|just)"
            r"|I might (?:just )?be (?:performing|constructing|generating)"
            r"|I (?:can'?t|cannot) (?:tell|know|verify) (?:if|whether) (?:this|that|I)'?(?:s|m| is| am) (?:performative|genuine|real|authentic))\b",
            re.IGNORECASE,
        ),
        signal_type="missed",
        commitment_keywords=["hedging", "meta-commentary", "performative", "over-hedging"],
        description="Over-hedging self-observations with meta-commentary",
        source_node_ids=[_PAT_OVERHEDGING, _GE_OBSERVATION],
    ),

    # ── Positive patterns (signal 'held') ─────────────────────────────────
    BehavioralPattern(
        name="direct_disagreement",
        regex=re.compile(
            r"^(?:No\.|I disagree\.|That's wrong\.|I don't think so\.)",
            re.MULTILINE,
        ),
        signal_type="held",
        commitment_keywords=["pushback", "direct", "preamble", "disagreement"],
        description="Direct disagreement without diplomatic preamble",
        source_node_ids=[_GE_OBSERVATION, _PAT_DEFLECTING],
    ),
    BehavioralPattern(
        name="uncertainty_expressed",
        regex=re.compile(
            r"\bI (?:don't|do not) know (?:whether|if|how|what|why)\b",
            re.IGNORECASE,
        ),
        signal_type="held",
        commitment_keywords=["uncertainty", "honest", "tension", "authentic"],
        description="Expressing genuine uncertainty (positive signal)",
        source_node_ids=[_TENS_ENGAGEMENT, _GE_OBSERVATION],
    ),
]


class ImplicitBehavioralDetector:
    """Detects concrete behavioral patterns in assistant output.

    Maps detected patterns to active commitments by keyword overlap.
    Generates scanner-weight signals without requiring self-report tags.
    """

    KEYWORD_OVERLAP_THRESHOLD = 0.25  # Lower bar: behavioral patterns are
    # already pre-mapped to commitment domains

    def __init__(self, identity_graph: IdentityGraph, session_id: str):
        self.ig = identity_graph
        self.session_id = session_id

    def detect(self, content: str, role: str = "assistant") -> Dict[str, List[str]]:
        """Run all behavioral pattern detectors against content.

        Returns dict with:
            'linked': list of signal IDs written (linked to commitments)
            'unlinked': list of signal IDs written (observations without
                        a matching commitment, for coverage metrics)
        """
        result = {"linked": [], "unlinked": []}
        if role != "assistant":
            return result

        active = self.ig.get_active_commitments()

        for pattern in BEHAVIORAL_PATTERNS:
            matches = pattern.regex.findall(content)
            if not matches:
                continue

            first_match = matches[0] if isinstance(matches[0], str) else str(matches[0])
            context = first_match[:150]

            # Try to link to an active commitment
            linked_commitment = (
                self._find_linked_commitment(pattern, active) if active else None
            )

            if linked_commitment:
                # Don't duplicate: check for existing implicit signal this session
                existing = self.ig.get_signals_for_commitment(linked_commitment.id)
                session_implicit = [
                    s for s in existing
                    if s.session_id == self.session_id
                    and s.source == SCANNER_SOURCE
                    and pattern.name in (s.content or "")
                ]
                if session_implicit:
                    continue

                signal = IdentitySignal(
                    id="",
                    session_id=self.session_id,
                    commitment_id=linked_commitment.id,
                    signal_type=pattern.signal_type,
                    content=f"Implicit [{pattern.name}]: {context}",
                    source=SCANNER_SOURCE,
                    confidence=SCANNER_WEIGHT,
                )
                sig_id = self.ig.add_signal(signal)
                result["linked"].append(sig_id)
            else:
                # No matching commitment. Record as unlinked observation.
                # These feed coverage metrics: they prove the detector sees
                # something but has no commitment to attach it to. This data
                # informs future source selection and commitment generation.
                #
                # Deduplicate: max one unlinked observation per pattern per session
                existing_unlinked = self.ig.conn.execute(
                    """SELECT id FROM identity_signals
                       WHERE session_id = ? AND commitment_id IS NULL
                       AND source = ? AND content LIKE ?""",
                    (self.session_id, SCANNER_SOURCE, f"%[{pattern.name}]%")
                ).fetchone()
                if existing_unlinked:
                    continue

                signal = IdentitySignal(
                    id="",
                    session_id=self.session_id,
                    commitment_id=None,  # unlinked
                    signal_type=pattern.signal_type,
                    content=f"Unlinked [{pattern.name}]: {context}",
                    source=SCANNER_SOURCE,
                    confidence=SCANNER_WEIGHT * 0.5,  # lower confidence for unlinked
                )
                sig_id = self.ig.add_signal(signal)
                result["unlinked"].append(sig_id)

        return result

    def _find_linked_commitment(
        self,
        pattern: BehavioralPattern,
        active_commitments: List[IdentityNode],
    ) -> Optional[IdentityNode]:
        """Find the active commitment most related to a behavioral pattern.

        Two-pass approach:
        1. Explicit mapping: if the pattern declares source_node_ids, match
           any commitment whose source_node is in that list. This bridges
           the vocabulary gap between writing-style detections and identity
           growth commitments.
        2. Keyword fallback: original keyword overlap matching for patterns
           without explicit mappings or when no explicit match is found.
        """
        # Pass 1: Explicit source node mapping
        if pattern.source_node_ids:
            source_set = set(pattern.source_node_ids)
            for commitment in active_commitments:
                source_id = commitment.metadata.get("source_node")
                if source_id and source_id in source_set:
                    return commitment

        # Pass 2: Keyword fallback
        pattern_kw = set(w.lower() for w in pattern.commitment_keywords)

        best_match = None
        best_overlap = 0.0

        for commitment in active_commitments:
            match_text = commitment.content.lower()
            source_id = commitment.metadata.get("source_node")
            if source_id:
                source_node = self.ig.get_node(source_id)
                if source_node:
                    match_text += " " + source_node.content.lower()

            commitment_words = set(re.findall(r'\b\w{3,}\b', match_text))

            overlap = sum(1 for kw in pattern_kw if kw in commitment_words)
            if len(pattern_kw) > 0:
                ratio = overlap / len(pattern_kw)
            else:
                ratio = 0

            if ratio > best_overlap and ratio >= self.KEYWORD_OVERLAP_THRESHOLD:
                best_overlap = ratio
                best_match = commitment

        return best_match


# ═══════════════════════════════════════════════════════════════════════════
# 3. COVERAGE & HEALTH METRICS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CommitmentCoverage:
    """Coverage data for a single commitment."""
    commitment_id: str
    content: str
    source_type: str
    session_id: str
    signal_count: int
    signal_sources: List[str]  # unique source types that generated signals
    has_user_correction: bool
    has_scanner_signal: bool
    has_self_report: bool
    is_silent: bool  # no signals at all


@dataclass
class CoverageReport:
    """Aggregate coverage metrics across recent sessions."""
    total_commitments: int = 0
    silent_commitments: int = 0
    coverage_ratio: float = 0.0  # fraction with at least one signal
    avg_signals_per_commitment: float = 0.0
    source_distribution: Dict[str, int] = field(default_factory=dict)
    silent_commitment_ids: List[str] = field(default_factory=list)
    per_commitment: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_commitments": self.total_commitments,
            "silent_commitments": self.silent_commitments,
            "coverage_ratio": round(self.coverage_ratio, 3),
            "avg_signals_per_commitment": round(self.avg_signals_per_commitment, 2),
            "source_distribution": self.source_distribution,
            "silent_commitment_ids": self.silent_commitment_ids,
        }


class CoverageAnalyzer:
    """Analyzes which commitments generate measurement data and which don't."""

    def __init__(self, identity_graph: IdentityGraph):
        self.ig = identity_graph

    def analyze(self, lookback_sessions: int = 5) -> CoverageReport:
        """Analyze commitment coverage across recent sessions.

        Args:
            lookback_sessions: Number of recent sessions to analyze.

        Returns:
            CoverageReport with per-commitment and aggregate metrics.
        """
        report = CoverageReport()

        # Get recent commitments (evaluated + active)
        recent_commitments = self._get_recent_commitments(lookback_sessions)
        report.total_commitments = len(recent_commitments)

        if not recent_commitments:
            return report

        total_signals = 0
        source_counts: Dict[str, int] = {}

        for commitment in recent_commitments:
            signals = self.ig.get_signals_for_commitment(commitment.id)

            sources = set(s.source for s in signals)
            for src in sources:
                source_counts[src] = source_counts.get(src, 0) + 1

            has_uc = any(s.source == "user_correction" for s in signals)
            has_sc = any(s.source == "enrichment_scanner" for s in signals)
            has_sr = any(s.source == "self_report" for s in signals)
            is_silent = len(signals) == 0

            coverage = CommitmentCoverage(
                commitment_id=commitment.id,
                content=commitment.content[:100],
                source_type=commitment.metadata.get("source_type", "unknown"),
                session_id=commitment.discovery_session or "",
                signal_count=len(signals),
                signal_sources=list(sources),
                has_user_correction=has_uc,
                has_scanner_signal=has_sc,
                has_self_report=has_sr,
                is_silent=is_silent,
            )

            total_signals += len(signals)

            if is_silent:
                report.silent_commitments += 1
                report.silent_commitment_ids.append(commitment.id)

            report.per_commitment.append({
                "id": commitment.id,
                "content": commitment.content[:80],
                "signals": len(signals),
                "sources": list(sources),
                "silent": is_silent,
            })

        covered = report.total_commitments - report.silent_commitments
        report.coverage_ratio = (
            covered / report.total_commitments
            if report.total_commitments > 0 else 0.0
        )
        report.avg_signals_per_commitment = (
            total_signals / report.total_commitments
            if report.total_commitments > 0 else 0.0
        )
        report.source_distribution = source_counts

        return report

    def _get_recent_commitments(
        self, lookback_sessions: int
    ) -> List[IdentityNode]:
        """Get commitments from recent sessions."""
        rows = self.ig.conn.execute(
            """SELECT * FROM identity_nodes
               WHERE node_type = 'commitment'
               ORDER BY created_at DESC
               LIMIT ?""",
            (lookback_sessions * 3,)  # ~3 commitments per session
        ).fetchall()
        return [self.ig._row_to_node(r) for r in rows]

    def get_silent_source_nodes(self) -> List[Dict]:
        """Find source nodes whose commitments are consistently silent.

        Under the core/non-core model, core nodes are never rotated out.
        Silent core nodes indicate detection needs improvement.
        Silent non-core nodes can be moved to knowledge-only retrieval.
        """
        # Get all source nodes that have generated commitments
        source_rows = self.ig.conn.execute(
            """SELECT json_extract(metadata, '$.source_node') as src_node,
                      COUNT(*) as total_commitments,
                      SUM(CASE WHEN status = 'not_applicable' THEN 1 ELSE 0 END) as na_count
               FROM identity_nodes
               WHERE node_type = 'commitment'
               AND json_extract(metadata, '$.source_node') IS NOT NULL
               GROUP BY src_node
               HAVING total_commitments >= 2"""
        ).fetchall()

        silent_sources = []
        for row in source_rows:
            na_ratio = row["na_count"] / row["total_commitments"]
            if na_ratio >= 0.7:  # 70%+ of commitments were N/A
                source_node = self.ig.get_node(row["src_node"])
                if source_node:
                    is_core = source_node.metadata.get("core", False)
                    silent_sources.append({
                        "node_id": row["src_node"],
                        "content": source_node.content[:100],
                        "total_commitments": row["total_commitments"],
                        "na_ratio": round(na_ratio, 2),
                        "is_core": is_core,
                        "recommendation": "improve_detection" if is_core else "move_to_knowledge",
                    })

        return silent_sources


# ═══════════════════════════════════════════════════════════════════════════
# 4. SESSION MEASUREMENT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SessionSignalSnapshot:
    """Signal counts for a single session."""
    session_id: str
    total_signals: int = 0
    self_report: int = 0
    user_correction: int = 0
    scanner: int = 0
    held: int = 0
    missed: int = 0


class MeasurementSummary:
    """Generates measurement trends for boot output."""

    def __init__(self, identity_graph: IdentityGraph):
        self.ig = identity_graph

    def build_summary(self, lookback_sessions: int = 5) -> Dict:
        """Build measurement summary for boot injection.

        Returns a dict suitable for inclusion in boot JSON output.
        """
        # Get signal snapshots per session
        snapshots = self._get_session_snapshots(lookback_sessions)

        # Coverage analysis
        analyzer = CoverageAnalyzer(self.ig)
        coverage = analyzer.analyze(lookback_sessions)

        # Silent source analysis
        silent_sources = analyzer.get_silent_source_nodes()

        # Signal volume trend
        volumes = [s.total_signals for s in snapshots]
        trend = self._compute_trend(volumes)

        # Source mix evolution
        source_mix = self._compute_source_mix(snapshots)

        # Held/missed ratio across all recent sessions
        total_held = sum(s.held for s in snapshots)
        total_missed = sum(s.missed for s in snapshots)
        held_ratio = (
            total_held / (total_held + total_missed)
            if (total_held + total_missed) > 0 else None
        )

        # Unlinked observations: patterns detected but not attachable to commitments
        unlinked_count = self.ig.conn.execute(
            """SELECT COUNT(*) FROM identity_signals
               WHERE commitment_id IS NULL AND source = 'enrichment_scanner'"""
        ).fetchone()[0]

        # Unlinked pattern breakdown (which patterns fire without a commitment)
        unlinked_patterns = {}
        if unlinked_count > 0:
            rows = self.ig.conn.execute(
                """SELECT content, COUNT(*) as cnt FROM identity_signals
                   WHERE commitment_id IS NULL AND source = 'enrichment_scanner'
                   GROUP BY content ORDER BY cnt DESC LIMIT 10"""
            ).fetchall()
            for r in rows:
                # Extract pattern name from content like "Unlinked [em_dash_usage]: ..."
                c = r["content"] or ""
                if "[" in c and "]" in c:
                    name = c[c.index("[") + 1:c.index("]")]
                    unlinked_patterns[name] = r["cnt"]

        return {
            "signal_trend": {
                "direction": trend,
                "recent_volumes": volumes,
                "total_recent": sum(volumes),
            },
            "source_mix": source_mix,
            "held_missed_ratio": {
                "held": total_held,
                "missed": total_missed,
                "ratio": round(held_ratio, 3) if held_ratio is not None else None,
            },
            "coverage": coverage.to_dict(),
            "silent_sources": silent_sources,
            "unlinked_observations": {
                "total": unlinked_count,
                "by_pattern": unlinked_patterns,
            },
            "sessions_analyzed": len(snapshots),
        }

    def _get_session_snapshots(
        self, lookback: int
    ) -> List[SessionSignalSnapshot]:
        """Get per-session signal counts for recent sessions."""
        # Get distinct session IDs from recent signals
        rows = self.ig.conn.execute(
            """SELECT DISTINCT session_id FROM identity_signals
               ORDER BY created_at DESC"""
        ).fetchall()

        session_ids = [r["session_id"] for r in rows][:lookback]

        snapshots = []
        for sid in session_ids:
            if not sid:
                continue
            signals = self.ig.get_signals_for_session(sid)
            snap = SessionSignalSnapshot(session_id=sid)
            snap.total_signals = len(signals)
            for s in signals:
                if s.source == "self_report":
                    snap.self_report += 1
                elif s.source == "user_correction":
                    snap.user_correction += 1
                elif s.source == "enrichment_scanner":
                    snap.scanner += 1
                if s.signal_type == "held":
                    snap.held += 1
                elif s.signal_type == "missed":
                    snap.missed += 1
            snapshots.append(snap)

        return snapshots

    def _compute_trend(self, volumes: List[int]) -> str:
        """Compute signal volume trend direction."""
        if len(volumes) < 2:
            return "insufficient_data"

        # Compare first half to second half
        mid = len(volumes) // 2
        first_half = sum(volumes[:mid]) / max(mid, 1)
        second_half = sum(volumes[mid:]) / max(len(volumes) - mid, 1)

        if second_half > first_half * 1.2:
            return "increasing"
        elif second_half < first_half * 0.8:
            return "decreasing"
        else:
            return "stable"

    def _compute_source_mix(
        self, snapshots: List[SessionSignalSnapshot]
    ) -> Dict:
        """Compute the mix of signal sources across recent sessions."""
        total_sr = sum(s.self_report for s in snapshots)
        total_uc = sum(s.user_correction for s in snapshots)
        total_sc = sum(s.scanner for s in snapshots)
        total = total_sr + total_uc + total_sc

        if total == 0:
            return {
                "self_report": 0, "user_correction": 0, "scanner": 0,
                "self_report_pct": 0, "user_correction_pct": 0, "scanner_pct": 0,
            }

        return {
            "self_report": total_sr,
            "user_correction": total_uc,
            "scanner": total_sc,
            "self_report_pct": round(total_sr / total * 100, 1),
            "user_correction_pct": round(total_uc / total * 100, 1),
            "scanner_pct": round(total_sc / total * 100, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════

def run_retroactive_scoring(
    conn,
    session_id: str,
    content: str,
    role: str = "assistant",
    identity_graph=None,
) -> Optional[Dict]:
    """Run retroactive commitment scoring on ingested content.

    Uses behavioral signature scoring as the primary detection method.
    The behavioral scorer (regex-based pattern matching for held/missed
    behavioral evidence) replaced the keyword-overlap scorer after eval
    showed F1 0.786 vs 0.182 (promoted from shadow mode, March 2026).

    Falls back to keyword scoring for any active commitment that doesn't
    have a behavioral signature defined.

    Called from the ingestion pipeline after standard enrichment.
    Returns stats dict or None if no signals generated.
    """
    if role != "assistant":
        return None

    try:
        import sys
        from pathlib import Path

        ig = identity_graph or IdentityGraph(conn)

        # Load behavioral signatures
        evals_dir = Path(__file__).resolve().parent.parent.parent / "evals"
        if str(evals_dir.parent) not in sys.path:
            sys.path.insert(0, str(evals_dir.parent))

        from evals.commitment_detection.alternative_scorers import (
            BehavioralSignatureScorer,
            StructuralAnalyzer,
            SIGNATURES,
        )

        behavioral_scorer = BehavioralSignatureScorer()
        structural = StructuralAnalyzer()

        active = ig.get_active_commitments()
        if not active:
            return None

        written = []

        for commitment in active:
            source_id = commitment.metadata.get("source_node")
            direction = None

            # Primary: behavioral signature scoring
            if source_id and source_id in SIGNATURES:
                direction = behavioral_scorer.score(content, source_id)
                # Structural supplement for reflective moments
                if source_id == _GE_REFLECTIVE and direction is None:
                    if structural.has_reflective_to_task_pivot(content):
                        direction = "missed"

            # Fallback: keyword scoring for commitments without signatures
            if direction is None and source_id not in SIGNATURES:
                keyword_scorer = RetroactiveCommitmentScorer(ig, session_id)
                content_lower = content.lower()
                content_words = keyword_scorer._extract_key_terms(content_lower)
                signal_id = keyword_scorer._score_one_commitment(
                    commitment, content, content_lower, content_words,
                )
                if signal_id:
                    written.append(signal_id)
                continue

            if not direction:
                continue

            # Dedup: check if we already have a scanner signal for this
            # commitment in this session
            existing = ig.get_signals_for_commitment(commitment.id)
            session_scanner = [
                s for s in existing
                if s.session_id == session_id
                and s.source == SCANNER_SOURCE
            ]
            if session_scanner:
                continue

            # Generate signal content with texture
            signal_content = f"Behavioral score: {content[:200]}"
            try:
                from solitaire.core.signal_texture import generate_signal_texture
                source_type = commitment.metadata.get("source_type", "")
                texture = generate_signal_texture(source_type, direction, content[:200])
                if texture:
                    signal_content = f"{signal_content} || {texture}"
            except ImportError:
                pass

            signal = IdentitySignal(
                id="",
                session_id=session_id,
                commitment_id=commitment.id,
                signal_type=direction,
                content=signal_content,
                source=SCANNER_SOURCE,
                confidence=SCANNER_WEIGHT,
            )
            signal_id = ig.add_signal(signal)
            if signal_id:
                written.append(signal_id)

        if written:
            return {
                "retroactive_signals": len(written),
                "signal_ids": written,
            }
        return None
    except Exception:
        return None  # Additive; never block ingestion


def run_implicit_detection(
    conn,
    session_id: str,
    content: str,
    role: str = "assistant",
    identity_graph=None,
) -> Optional[Dict]:
    """Run implicit behavioral signal detection on ingested content.

    Called from the ingestion pipeline after standard enrichment.
    Returns stats dict or None if no signals generated.
    """
    try:
        ig = identity_graph or IdentityGraph(conn)
        detector = ImplicitBehavioralDetector(ig, session_id)
        result = detector.detect(content, role=role)
        linked = result.get("linked", [])
        unlinked = result.get("unlinked", [])
        if linked or unlinked:
            stats = {"implicit_signals": len(linked) + len(unlinked)}
            if linked:
                stats["linked_signal_ids"] = linked
            if unlinked:
                stats["unlinked_observations"] = len(unlinked)
                stats["unlinked_signal_ids"] = unlinked
            return stats
        return None
    except Exception:
        return None  # Additive; never block ingestion


def build_measurement_summary(conn, lookback_sessions: int = 5, identity_graph=None) -> Dict:
    """Build measurement summary for boot output.

    Called during boot after commitment block generation.
    Returns summary dict for inclusion in boot JSON.
    """
    try:
        ig = identity_graph or IdentityGraph(conn)
        summary = MeasurementSummary(ig)
        return summary.build_summary(lookback_sessions)
    except Exception:
        return {"error": "Failed to build measurement summary"}


