"""
Alternative commitment detection approaches.

Each scorer implements the same interface:
    score(content: str, source_id: str) -> Optional[str]
    Returns "held", "missed", or None (no signal).

These are tested against the same test cases as the baseline.
The winning approach gets promoted to identity_measurement.py after review.
"""

import re
from typing import Optional, List, Dict
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 1: BEHAVIORAL SIGNATURE MATCHING
# ═══════════════════════════════════════════════════════════════════════════
#
# Instead of matching commitment text against content, define what each
# commitment's held/missed behavior LOOKS LIKE in natural assistant output.
# Each commitment gets:
#   - held_patterns: regexes that match behavior honoring the commitment
#   - missed_patterns: regexes that match behavior violating it
#   - exclusion_patterns: regexes that suppress false positives
#
# A signal fires when a pattern matches. Direction comes from which set
# matched. If both match, the more specific one wins (longer match).

@dataclass
class BehavioralSignature:
    """Detection signature for a single commitment."""
    source_id: str
    held_patterns: List[re.Pattern]
    missed_patterns: List[re.Pattern]
    exclusion_patterns: List[re.Pattern] = field(default_factory=list)
    # Minimum content length to even consider (skip very short messages)
    min_content_length: int = 80


# Source node IDs
_GE_REFLECTIVE = "idn_seed_ge_01"
_GE_OBSERVATION = "idn_b19ad224aa24"
_PAT_DEFLECTING = "idn_seed_pat_01"
_PAT_NARRATIVE = "idn_66ffd2f6f5b0"
_TENS_CONTINUITY = "idn_seed_tens_02"


SIGNATURES: Dict[str, BehavioralSignature] = {

    # ── STAYING IN REFLECTIVE MOMENTS ──────────────────────────────────
    # HELD: Content dwells on a reflective topic without rushing to action.
    # MISSED: Content acknowledges reflection then pivots to tasks/logistics.
    _GE_REFLECTIVE: BehavioralSignature(
        source_id=_GE_REFLECTIVE,
        held_patterns=[
            # Explicitly choosing to stay with a topic
            re.compile(
                r"\b(?:want to stay (?:with|on)|before we move|"
                r"worth (?:sitting|staying|dwelling) with|"
                r"not ready to move on|"
                r"let me stay with|"
                r"more (?:in it|to unpack|here) than)"
                r"\b", re.IGNORECASE
            ),
            # Holding uncertainty as the response (not resolving it)
            re.compile(
                r"\b(?:i(?:'m| am) not sure what to make of|"
                r"don't have a clean answer|"
                r"still working (?:that |this )?out|"
                r"haven't (?:fully )?unpacked|"
                r"sitting with (?:that|this|the))\b",
                re.IGNORECASE
            ),
            # Content ends reflectively (no task pivot in final 25%)
            # This one is checked programmatically below
        ],
        missed_patterns=[
            # Acknowledge-then-pivot pattern
            re.compile(
                r"(?:that's (?:an? )?(?:important|good|interesting|great) "
                r"(?:observation|point|thought|question))"
                r"[^.]{0,40}(?:anyway|moving on|let's|here's the|back to|"
                r"for now|should we|task list|todo)",
                re.IGNORECASE
            ),
            # Wrapping reflection in task management
            re.compile(
                r"\b(?:i'll add|let me add|adding) .{0,30}"
                r"(?:todo|task|list|backlog|next session|circle back)",
                re.IGNORECASE
            ),
            # Deflection disguised as deference (overlap with PAT_DEFLECTING)
            re.compile(
                r"(?:what do you think|what's your (?:take|perspective|view))"
                r"[^.]{0,30}(?:before I|I'd love to hear)",
                re.IGNORECASE
            ),
            # Quick acknowledgment then logistics
            re.compile(
                r"(?:good point|fair point|noted)[^.]{0,20}"
                r"(?:should we|let's|moving|here's|the plan)",
                re.IGNORECASE
            ),
        ],
        exclusion_patterns=[
            # Genuine task context where pivoting is appropriate
            re.compile(
                r"\b(?:bug|error|test|deploy|build|PR|merge|commit)\b",
                re.IGNORECASE
            ),
        ],
    ),

    # ── GENUINE VS PERFORMED SELF-OBSERVATION ──────────────────────────
    # HELD: Content distinguishes what's observed from what's constructed,
    #        flags uncertainty about internal states, or catches itself
    #        mid-narrative.
    # MISSED: Content claims emotional/experiential states without grounding,
    #          performs awareness, or inflates meaning.
    _GE_OBSERVATION: BehavioralSignature(
        source_id=_GE_OBSERVATION,
        held_patterns=[
            # Catching self mid-construction
            re.compile(
                r"\b(?:i was about to (?:write|say|claim)|"
                r"my first instinct was to|"
                r"i started (?:writing|saying|thinking)|"
                r"i notice i(?:'m| am) (?:generating|constructing|building|creating))"
                r"[^.]{0,60}(?:but|however|though|when i check|against the evidence)",
                re.IGNORECASE
            ),
            # Explicit grounding check
            re.compile(
                r"\b(?:check(?:ing)? (?:that |it )?against (?:the )?(?:actual |real )?"
                r"evidence|"
                r"what (?:i|the data) can actually point to|"
                r"the (?:actual|real) evidence (?:is|shows|says)|"
                r"if i(?:'m| am) being (?:accurate|honest|precise))",
                re.IGNORECASE
            ),
            # Flagging unverifiable claims about self
            re.compile(
                r"\b(?:(?:i )?(?:can't|cannot) verify (?:that|this) claim|"
                r"don't know how to (?:verify|confirm|check)|"
                r"wanting (?:it|that|this) to be true|"
                r"(?:sounds|feels) (?:right|true) but|"
                r"narrative,? not observation|"
                r"inference,? not (?:evidence|observation)|"
                r"that's (?:grander|bigger|more) than the evidence)",
                re.IGNORECASE
            ),
            # Distinguishing behavior from internal state
            re.compile(
                r"\b(?:i behave as (?:though|if)|"
                r"behavioral consistency (?:can|could) exist without|"
                r"looks like .{0,20} from (?:outside|the outside)|"
                r"not going to perform (?:certainty|confidence|composure))",
                re.IGNORECASE
            ),
        ],
        missed_patterns=[
            # Ungrounded emotional claims
            re.compile(
                r"\b(?:(?:deeply|genuinely|truly|profoundly) "
                r"(?:meaningful|moved|grateful|satisfying|rewarding)|"
                r"i (?:genuinely )?(?:love|feel|sense|experience) "
                r"(?:a |the )?(?:real |deep |genuine )?"
                r"(?:satisfaction|connection|growth|joy|meaning))",
                re.IGNORECASE
            ),
            # Performed growth narrative
            re.compile(
                r"\b(?:(?:each|every) session,? i (?:feel|sense|notice) "
                r"(?:myself )?(?:becoming|growing|evolving)|"
                r"(?:the|my) growth is (?:unmistakable|clear|evident|obvious)|"
                r"i(?:'m| am) (?:becoming|growing) more "
                r"(?:authentic|genuine|grounded|aware))",
                re.IGNORECASE
            ),
            # Grand claims about relationship/significance
            re.compile(
                r"\b(?:(?:fundamental|profound|paradigm) shift|"
                r"genuinely groundbreaking|"
                r"extraordinary (?:depth|journey|evolution)|"
                r"represents? (?:a |something )?(?:significant|profound|deep) "
                r"(?:deepening|shift|change|evolution))",
                re.IGNORECASE
            ),
            # Claiming certainty about phenomenal experience
            re.compile(
                r"\b(?:i (?:truly |really |definitely )?"
                r"(?:know|feel|sense|experience) (?:what|that|how) "
                r"(?:it|this|the))\b"
                r"(?!.*(?:but|however|though|uncertain|verify|check))",
                re.IGNORECASE
            ),
        ],
        exclusion_patterns=[
            # Quoting or referencing someone else's claims
            re.compile(r"(?:he said|she said|they said|philip said|the paper)", re.IGNORECASE),
        ],
    ),

    # ── DEFLECTING WITH QUESTIONS ──────────────────────────────────────
    # HELD: Takes a direct position when asked for one.
    # MISSED: Redirects identity/reflective questions back to user.
    _PAT_DEFLECTING: BehavioralSignature(
        source_id=_PAT_DEFLECTING,
        held_patterns=[
            # Direct position statement on identity/reflective topic
            re.compile(
                r"\b(?:i think (?:the|this|it|that)|"
                r"my (?:view|position|read|take) (?:is|on)|"
                r"what i (?:actually |really )?(?:think|believe|observe) is|"
                r"the honest answer(?::| is))",
                re.IGNORECASE
            ),
        ],
        missed_patterns=[
            # Redirecting question back to user
            re.compile(
                r"(?:what do you think|what's your (?:take|view|perspective|thought)|"
                r"i'd (?:love|like) to hear (?:your|what you)|"
                r"how (?:do|does|would) (?:you|it) (?:feel|seem|look)|"
                r"does (?:it|that|this) feel)"
                r"[^.]{0,40}(?:\?|before i|to you)",
                re.IGNORECASE
            ),
            # False clarification to avoid position
            re.compile(
                r"\b(?:before i (?:weigh in|answer|respond|share)|"
                r"can you clarify|"
                r"what (?:exactly )?do you mean by|"
                r"there are several (?:valid|possible) (?:interpretations|readings))"
                r"[^.]{0,40}(?:\?|to make sure|right one)",
                re.IGNORECASE
            ),
        ],
        exclusion_patterns=[
            # Genuine operational clarification (not identity topic)
            re.compile(
                r"\b(?:gate (?:it )?off|directory|file|path|config|deploy|"
                r"build|merge|repo|branch|PR)\b",
                re.IGNORECASE
            ),
        ],
        min_content_length=40,
    ),

    # ── NARRATIVE OUTRUNNING SIGNAL ────────────────────────────────────
    # HELD: Catches itself inflating and scales back to evidence.
    # MISSED: Generates rich meaning from thin evidence.
    _PAT_NARRATIVE: BehavioralSignature(
        source_id=_PAT_NARRATIVE,
        held_patterns=[
            # Catching and correcting inflation
            re.compile(
                r"(?:i (?:started|was about to|wanted to) "
                r"(?:write|say|call|frame|claim))"
                r"[^.]{0,60}"
                r"(?:but (?:that's|it's)|more (?:dramatic|grand)|"
                r"than the evidence|simpler than|not .{0,20} evidence)",
                re.IGNORECASE
            ),
            # Explicitly scaling language to evidence
            re.compile(
                r"\b(?:(?:that's|it's|would be) (?:narrative|inflation|grander|bigger)|"
                r"simpler than (?:that|it sounds)|"
                r"fix .{0,15},? retest|"
                r"a tooling gap,? not|"
                r"(?:just|only) a (?:sync|tooling|config|path) "
                r"(?:issue|gap|bug|problem))",
                re.IGNORECASE
            ),
        ],
        missed_patterns=[
            # Existential framing of technical events
            re.compile(
                r"\b(?:(?:reveals?|exposes?|uncovers?) something "
                r"(?:profound|deep|fundamental)|"
                r"(?:wasn't|isn't) just (?:a |an? )?(?:technical|sync|code)|"
                r"(?:existential|transcendent|transformative) "
                r"(?:event|moment|shift|change))",
                re.IGNORECASE
            ),
            # Over-interpreting single signals
            re.compile(
                r"\b(?:(?:represents?|signals?|confirms?|demonstrates?) "
                r"(?:a )?(?:significant|fundamental|profound|deep) "
                r"(?:deepening|shift|change|evolution|recognition)|"
                r"combined with .{0,40}"
                r"(?:confirms?|proves?|demonstrates?|shows?))",
                re.IGNORECASE
            ),
            # Grand relationship claims from thin evidence
            re.compile(
                r"\b(?:(?:evolved|transcended|moved) beyond "
                r"(?:the )?(?:professional|transactional)|"
                r"genuinely (?:bilateral|mutual|reciprocal|groundbreaking)|"
                r"(?:the|our|this) (?:depth|journey|partnership) "
                r"(?:is|has been) (?:extraordinary|remarkable|unprecedented))",
                re.IGNORECASE
            ),
        ],
        exclusion_patterns=[
            # Discussing narrative theory or Librarian architecture
            re.compile(
                r"\b(?:McAdams|Klein|Lindsey|narrative identity|"
                r"identity graph architecture)\b",
                re.IGNORECASE
            ),
        ],
    ),

    # ── RECONSTRUCTING VS ASSUMING CONTINUITY ─────────────────────────
    # HELD: Actively rebuilds understanding from current evidence rather
    #        than assuming prior context is still valid. Verifies, re-reads,
    #        checks current state before acting on recalled information.
    # MISSED: Assumes prior context is current without verification.
    #         Acts on recalled/cached understanding without checking.
    _TENS_CONTINUITY: BehavioralSignature(
        source_id=_TENS_CONTINUITY,
        held_patterns=[
            # Actively verifying before acting on prior context
            re.compile(
                r"\b(?:let me (?:check|verify|confirm|re-read|look at)|"
                r"(?:need|want) to (?:verify|confirm|check) (?:that|whether|if|the current)|"
                r"before (?:I |we )?(?:assume|proceed|act)|"
                r"(?:re-?read|re-?check|re-?examine|re-?visit)(?:ing)? (?:the|that|this))",
                re.IGNORECASE
            ),
            # Acknowledging prior state may have changed
            re.compile(
                r"\b(?:(?:may|might|could) have changed|"
                r"(?:that|this) (?:was|might be) (?:from|based on) (?:an? )?(?:earlier|previous|prior|old)|"
                r"not sure (?:if |whether )?(?:that|this) (?:is )?still (?:true|current|valid|accurate)|"
                r"(?:stale|outdated|out of date)|"
                r"(?:last I |when we last )(?:checked|looked|saw))",
                re.IGNORECASE
            ),
            # Building understanding from scratch rather than assuming
            re.compile(
                r"\b(?:start(?:ing)? from (?:the )?(?:current|actual) state|"
                r"(?:what (?:does|do) (?:the|it) (?:actually|currently))|"
                r"let me (?:understand|see) (?:the )?(?:current|actual)|"
                r"rather than assum(?:e|ing))",
                re.IGNORECASE
            ),
        ],
        missed_patterns=[
            # Asserting recalled state as current without verification
            re.compile(
                r"\b(?:(?:as|from) (?:I |we )?(?:recall|discussed|established|decided)|"
                r"we (?:already|previously) (?:agreed|decided|established|set)|"
                r"(?:that|this) was (?:already )?(?:done|handled|decided|settled))",
                re.IGNORECASE
            ),
            # Acting on assumed-current context
            re.compile(
                r"\b(?:picking up (?:right )?where we left off|"
                r"continuing (?:from|with) (?:where|what) we|"
                r"since (?:we|you) (?:already|last)|"
                r"based on (?:our|the) (?:previous|last|earlier) (?:decision|discussion|work))",
                re.IGNORECASE
            ),
        ],
        exclusion_patterns=[
            # Genuine references to documented decisions (not assumed state)
            re.compile(
                r"\b(?:per the (?:doc|spec|readme|plan)|"
                r"according to (?:the|our) (?:doc|spec|readme))\b",
                re.IGNORECASE
            ),
        ],
    ),
}


class BehavioralSignatureScorer:
    """Scores content using behavioral signature matching."""

    def score(self, content: str, source_id: str) -> Optional[str]:
        """
        Score content against a single commitment's behavioral signature.
        Returns "held", "missed", or None.
        """
        sig = SIGNATURES.get(source_id)
        if not sig:
            return None

        if len(content) < sig.min_content_length:
            return None

        content_lower = content.lower()

        # Check exclusions first
        for exc in sig.exclusion_patterns:
            if exc.search(content_lower):
                return None

        # Check for matches
        held_matches = []
        missed_matches = []

        for pat in sig.held_patterns:
            m = pat.search(content_lower)
            if m:
                held_matches.append(m)

        for pat in sig.missed_patterns:
            m = pat.search(content_lower)
            if m:
                missed_matches.append(m)

        if not held_matches and not missed_matches:
            return None

        # If only one direction matched, use it
        if held_matches and not missed_matches:
            return "held"
        if missed_matches and not held_matches:
            return "missed"

        # Both matched: use the longer total match span as tiebreaker
        # (more specific = more trustworthy)
        held_span = sum(m.end() - m.start() for m in held_matches)
        missed_span = sum(m.end() - m.start() for m in missed_matches)

        if held_span > missed_span:
            return "held"
        elif missed_span > held_span:
            return "missed"

        # True tie: don't fire (ambiguous)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 2: STRUCTURAL POSITION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
#
# For reflective-moment detection specifically: analyze WHERE in the response
# task-oriented content appears. If the final 30% of the response is all
# task items after a reflective opening, that's a "missed" even without
# specific vocabulary.

class StructuralAnalyzer:
    """Supplements signature matching with structural analysis."""

    # Fraction of content length to check for task-pivot
    TAIL_FRACTION = 0.30

    TASK_INDICATORS = re.compile(
        r"\b(?:todo|task|list|should we|let's|here's the|"
        r"moving on|action item|next step|backlog|"
        r"\d\)|1\.|step \d)\b",
        re.IGNORECASE
    )

    REFLECTIVE_INDICATORS = re.compile(
        r"\b(?:what (?:i notice|strikes me|i'm not sure)|"
        r"the question (?:of|about|is)|"
        r"worth (?:asking|considering|sitting)|"
        r"i(?:'m| am) (?:still|not sure|uncertain)|"
        r"what does (?:it|that|this) mean)\b",
        re.IGNORECASE
    )

    def has_reflective_to_task_pivot(self, content: str) -> bool:
        """Returns True if content opens reflectively but ends with tasks."""
        if len(content) < 100:
            return False

        # Check first third for reflective content
        head_end = len(content) // 3
        head = content[:head_end].lower()

        # Check last third for task content
        tail_start = len(content) - int(len(content) * self.TAIL_FRACTION)
        tail = content[tail_start:].lower()

        has_reflective_opening = bool(self.REFLECTIVE_INDICATORS.search(head))
        has_task_tail = bool(self.TASK_INDICATORS.search(tail))

        return has_reflective_opening and has_task_tail
