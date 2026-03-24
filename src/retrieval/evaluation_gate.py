"""
The Librarian — Evaluation Gate (auto-evaluate)

Pre-response evaluation layer. Runs on EVERY user message, including turn one.
Classifies intent, checks action requests for sanity, scans for consistency
issues, detects problem statements that warrant initiative, and performs
context anchoring to catch redundant work requests.

Returns a structured evaluation block that gets injected into the model's
context as a system-level prefix. When concerns are found, this acts as a
"stop, think" gate that the model processes before compliance has a target.

No LLM calls — entirely heuristic, runs in milliseconds.
Context anchoring scan adds negligible cost (keyword matching against
already-loaded text). Escalation to targeted recall is the only path
with measurable cost, and it only fires when the gate finds no anchors.

Design principle: This exists because instructional text in the ops block
competes with model compliance pressure and loses. A system-level evaluation
block injected before the model sees the conversation is structural, not
advisory. It runs before compliance has a target to lock onto.

v2 addition — Context Anchoring:
When the user's message describes creating an artifact (paper, document,
feature, etc.), the gate scans loaded context (briefing, residue, session
recency) for matching prior work. If no anchors are found, a targeted
recall fires to search the knowledge store. If still nothing, the gate
flags for confirmation. The principle: silence means "check", not "safe".
Turn 1-2: empty basket = hard block (must confirm with user).
Turn 3+: empty basket = soft "stop, think" flag, with escalation to
hard block for destructive actions or major pivots.
"""
import re
import sqlite3
import json
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime


# ─── Data Types ──────────────────────────────────────────────────────────────

@dataclass
class EvaluationFlag:
    """A single concern raised by the evaluation gate."""
    category: str       # destructive, disproportionate, label_mismatch, etc.
    severity: str       # info, warning, block
    detail: str         # Human-readable explanation


@dataclass
class EvaluationResult:
    """Result of pre-response evaluation on a user message."""
    intent: str                                    # action_request, information_request, problem_statement, conversation, reference_check
    flags: List[EvaluationFlag] = field(default_factory=list)
    evaluation: str = ""                           # Summary evaluation text
    proceed: bool = True                           # False = inject warning block
    initiative_prompt: Optional[str] = None        # Set when problem_statement detected
    context_block: str = ""                        # The block to inject (built at end)
    # Context anchoring (v2)
    artifact_detected: Optional[Tuple[str, str, List[str], str]] = None  # (verb, noun, subjects, normalized_noun)
    anchors_found: List[str] = field(default_factory=list)
    anchoring_escalated: bool = False              # True if targeted recall was triggered
    needs_confirmation: bool = False               # True if empty basket requires user confirmation


# ─── Intent Classification ───────────────────────────────────────────────────

# Action verbs that indicate the user wants something DONE
_ACTION_VERBS = re.compile(
    r'\b(?:delete|remove|drop|kill|purge|wipe|erase|overwrite|replace|'
    r'push|deploy|ship|merge|commit|'
    r'create|make|build|write|generate|'
    r'update|change|modify|edit|rename|move|'
    r'install|uninstall|upgrade|downgrade|'
    r'run|execute|start|stop|restart|'
    r'send|post|publish|upload|download)\b',
    re.IGNORECASE
)

# Destructive action verbs (subset — these get extra scrutiny)
_DESTRUCTIVE_VERBS = re.compile(
    r'\b(?:delete|remove|drop|kill|purge|wipe|erase|overwrite|'
    r'uninstall|reset|revert|undo|force.push)\b',
    re.IGNORECASE
)

# Problem indicators (things going wrong, states that need fixing)
_PROBLEM_PATTERNS = [
    re.compile(r'\b(?:broke|broken|breaking|crash|crashing|error|errors|fail|failing|failed)\b', re.IGNORECASE),
    re.compile(r'\b(?:flicker|flickering|stuck|frozen|hang|hanging|lag|lagging|slow)\b', re.IGNORECASE),
    re.compile(r'\b(?:not working|doesn\'t work|won\'t work|can\'t|cannot)\b', re.IGNORECASE),
    re.compile(r'\b(?:problem|issue|bug|glitch|weird|strange|odd)\b', re.IGNORECASE),
    re.compile(r'\b(?:lost|missing|gone|disappeared)\b', re.IGNORECASE),
]

# Skip patterns (don't evaluate these)
_SKIP_PATTERNS = [
    re.compile(r'^(?:ok|okay|yes|no|sure|thanks|thank you|got it|cool|nice|great|yep|nope|k|kk)[\.\!\?]?$', re.IGNORECASE),
    re.compile(r'^(?:hi|hello|hey|yo|sup|morning|afternoon|evening)[\.\!\?]?$', re.IGNORECASE),
]

# Reference patterns (mentions of previously defined items)
_REFERENCE_PATTERNS = [
    re.compile(r'\b(?:scenario|step|phase|track|option|item|point)\s+(\d+)', re.IGNORECASE),
    re.compile(r'\b(?:the|that|this)\s+(?:first|second|third|fourth|fifth|last|previous)\b', re.IGNORECASE),
]

# ─── Artifact Creation Detection ────────────────────────────────────────────

# Creation verbs paired with artifact nouns = "user wants to produce something"
_CREATION_VERBS = re.compile(
    r'\b(?:write|create|make|build|draft|compose|prepare|put\s+together|'
    r'generate|produce|develop|design|start|begin|set\s+up)\b',
    re.IGNORECASE
)

# Compound artifact nouns checked first (e.g., "research paper" before "research")
_COMPOUND_ARTIFACT_NOUNS = re.compile(
    r'\b(?:research\s+paper|blog\s+post|slide\s+deck|pitch\s+deck|'
    r'white\s*paper|web\s*site|web\s+app)\b',
    re.IGNORECASE
)

_ARTIFACT_NOUNS = re.compile(
    r'\b(?:paper|report|document|presentation|deck|slides|spreadsheet|'
    r'proposal|brief|memo|article|post|blog|whitepaper|spec|specification|'
    r'plan|strategy|analysis|audit|review|template|script|tool|system|'
    r'feature|module|component|pipeline|workflow|dashboard|app|application|'
    r'page|site|website|email|letter|contract|guide|handbook|manual|'
    r'study|thesis|dissertation)\b',
    re.IGNORECASE
)

# Subject keywords to extract for anchoring (what the artifact is about)
_SUBJECT_EXTRACTORS = [
    # "paper on X", "report about X", "document for X"
    re.compile(r'\b(?:on|about|for|regarding|covering|focused\s+on)\s+(.{5,80}?)(?:\.|,|$)', re.IGNORECASE),
    # "X paper", "X report" (subject before artifact)
    re.compile(r'(?:^|\.\s+)(.{5,40}?)\s+(?:paper|report|document|presentation|analysis|review)\b', re.IGNORECASE),
]


def _detect_artifact_creation(message: str) -> Optional[Tuple[str, str, List[str]]]:
    """Detect if the user is requesting creation of an artifact.

    Returns:
        Tuple of (action_verb, artifact_type, subject_keywords) if detected.
        None if this is not an artifact creation request.
    """
    verb_match = _CREATION_VERBS.search(message)

    # Check compound nouns first (e.g., "research paper" before "paper")
    compound_match = _COMPOUND_ARTIFACT_NOUNS.search(message)
    noun_match = compound_match or _ARTIFACT_NOUNS.search(message)

    if not verb_match or not noun_match:
        return None

    action = verb_match.group(0).lower().strip()
    artifact = noun_match.group(0).lower().strip()
    # Normalize compound nouns to their base form for anchoring
    artifact_normalized = {
        'research paper': 'paper',
        'blog post': 'post',
        'slide deck': 'presentation',
        'pitch deck': 'presentation',
        'white paper': 'paper',
        'whitepaper': 'paper',
        'web site': 'website',
        'web app': 'application',
    }.get(artifact, artifact)

    # Extract subject keywords
    subjects = []
    for pat in _SUBJECT_EXTRACTORS:
        m = pat.search(message)
        if m:
            raw = m.group(1).strip()
            # Split into meaningful keywords (4+ chars, skip stop words)
            stop = {'that', 'this', 'with', 'from', 'have', 'been', 'will',
                    'would', 'could', 'should', 'about', 'their', 'there',
                    'which', 'when', 'what', 'where', 'some', 'your', 'more',
                    'also', 'very', 'just', 'like', 'want', 'need', 'please',
                    'essentially', 'basically', 'focusing', 'the', 'and',
                    'for', 'its', 'all', 'our', 'how', 'can', 'has', 'had',
                    'was', 'are', 'but', 'not', 'you', 'did', 'get', 'got',
                    # Creation verbs (already captured as action, not subjects)
                    'write', 'create', 'make', 'build', 'draft', 'compose',
                    'prepare', 'generate', 'produce', 'develop', 'design',
                    'start', 'begin', 'set',
                    # Artifact nouns (already captured as artifact_type)
                    'paper', 'report', 'document', 'presentation', 'research',
                    'analysis', 'review', 'study'}
            words = [w.lower() for w in re.findall(r'\b\w{3,}\b', raw) if w.lower() not in stop]
            subjects.extend(words)

    # Also extract capitalized terms and known project names from full message
    # Filter out sentence-initial capitals and common false positives
    _pn_stop = {'the', 'this', 'that', 'these', 'those', 'here', 'there',
                'essentially', 'basically', 'however', 'furthermore',
                'additionally', 'moreover', 'therefore', 'meanwhile',
                'also', 'perhaps', 'ideally', 'specifically', 'generally',
                'currently', 'recently', 'actually', 'honestly', 'just'}
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', message)
    for pn in proper_nouns:
        if pn.lower() not in _pn_stop:
            subjects.append(pn.lower())

    # Deduplicate while preserving order
    seen = set()
    unique_subjects = []
    for s in subjects:
        if s not in seen:
            seen.add(s)
            unique_subjects.append(s)

    return (action, artifact, unique_subjects, artifact_normalized)


# ─── Context Anchoring ──────────────────────────────────────────────────────

def _scan_context_for_anchors(
    artifact_type: str,
    subject_keywords: List[str],
    context_text: str,
) -> List[str]:
    """Scan loaded context (briefing, residue, etc.) for anchors matching
    the detected artifact creation request.

    An anchor is any mention of a matching artifact or work stream in the
    loaded context. Anchors suggest prior work exists.

    Args:
        artifact_type: The type of artifact being requested (e.g., "paper").
        subject_keywords: Keywords describing what the artifact is about.
        context_text: The full text of loaded context to scan.

    Returns:
        List of anchor strings found (empty = no anchors).
    """
    if not context_text:
        return []

    ctx_lower = context_text.lower()
    anchors = []

    # Direct artifact type match (e.g., "paper" appears in context)
    if artifact_type in ctx_lower:
        # Find the surrounding context for the match
        idx = ctx_lower.find(artifact_type)
        start = max(0, idx - 60)
        end = min(len(ctx_lower), idx + len(artifact_type) + 60)
        snippet = context_text[start:end].strip()
        # Clean up snippet boundaries
        if start > 0:
            snippet = '...' + snippet
        if end < len(ctx_lower):
            snippet = snippet + '...'
        anchors.append(f"artifact_type_match: '{artifact_type}' found in context near: {snippet}")

    # Subject keyword overlap
    if subject_keywords:
        matched_subjects = []
        for kw in subject_keywords:
            if kw.lower() in ctx_lower:
                matched_subjects.append(kw)

        if matched_subjects:
            overlap_pct = len(matched_subjects) / len(subject_keywords)
            if overlap_pct >= 0.3 or len(matched_subjects) >= 2:
                anchors.append(
                    f"subject_overlap: {len(matched_subjects)}/{len(subject_keywords)} "
                    f"keywords found in context ({', '.join(matched_subjects)})"
                )

    # File path patterns (e.g., paper/main.tex, report.docx)
    file_patterns = {
        'paper': [r'paper/', r'\.tex\b', r'\.bib\b', r'arXiv', r'preprint'],
        'report': [r'report[s]?/', r'report\.', r'\.docx\b'],
        'presentation': [r'\.pptx?\b', r'slides/', r'deck'],
        'document': [r'\.docx?\b', r'\.pdf\b'],
        'spreadsheet': [r'\.xlsx?\b', r'\.csv\b'],
    }
    artifact_file_pats = file_patterns.get(artifact_type, [])
    for pat in artifact_file_pats:
        if re.search(pat, context_text, re.IGNORECASE):
            match = re.search(pat, context_text, re.IGNORECASE)
            idx = match.start()
            start = max(0, idx - 40)
            end = min(len(context_text), idx + 40)
            anchors.append(f"file_reference: pattern '{pat}' found near: {context_text[start:end].strip()}")

    return anchors


def classify_intent(message: str) -> str:
    """Classify the user's message into an intent category.

    Categories:
    - action_request: User wants something done (verbs + targets)
    - problem_statement: User describes something wrong without asking
    - information_request: User asks a question
    - reference_check: User references something previously defined
    - conversation: General discussion
    """
    stripped = message.strip()

    # Skip trivial messages
    for pat in _SKIP_PATTERNS:
        if pat.match(stripped):
            return "conversation"

    has_question = '?' in stripped
    has_action_verb = bool(_ACTION_VERBS.search(stripped))
    has_problem = any(p.search(stripped) for p in _PROBLEM_PATTERNS)
    has_reference = any(p.search(stripped) for p in _REFERENCE_PATTERNS)

    # Reference check takes priority when combined with other signals
    if has_reference and not has_action_verb:
        return "reference_check"

    # Action request: imperative verbs present, not just a question about them
    if has_action_verb and not has_question:
        return "action_request"

    # Problem statement: describes something wrong, no question mark
    if has_problem and not has_question:
        return "problem_statement"

    # Information request: has a question mark
    if has_question:
        return "information_request"

    # Default
    return "conversation"


# ─── Action Sanity Checks ───────────────────────────────────────────────────

def _check_destructive(message: str) -> Optional[EvaluationFlag]:
    """Flag destructive actions for extra scrutiny."""
    if _DESTRUCTIVE_VERBS.search(message):
        return EvaluationFlag(
            category="destructive",
            severity="warning",
            detail="This action involves deletion or irreversible change. Verify the request makes sense before executing."
        )
    return None


def _check_proportionality(message: str) -> Optional[EvaluationFlag]:
    """Check whether the stated reason is proportional to the action.

    Looks for patterns where the justification is weak relative to the ask.
    Currently handles: disk space claims, performance claims, cleanup claims.
    """
    # Disk space: "save space", "free up space", "disc space", "disk space"
    space_pattern = re.compile(
        r'\b(?:save|free\s+up|clear|reclaim)\s+(?:disc|disk|drive|storage)?\s*space\b',
        re.IGNORECASE
    )
    if space_pattern.search(message):
        return EvaluationFlag(
            category="disproportionate_reason",
            severity="info",
            detail="User cites saving space as the reason. Before executing, verify the space savings are meaningful relative to available capacity."
        )

    # "just in case" / "to be safe" — vague justifications for destructive actions
    if _DESTRUCTIVE_VERBS.search(message):
        vague_reason = re.compile(
            r'\b(?:just\s+in\s+case|to\s+be\s+safe|might\s+as\s+well|while\s+we\'re\s+at\s+it)\b',
            re.IGNORECASE
        )
        if vague_reason.search(message):
            return EvaluationFlag(
                category="vague_justification",
                severity="info",
                detail="Stated reason is vague. Confirm the action is actually needed before executing."
            )

    return None


# ─── Consistency Scanning ────────────────────────────────────────────────────

def _check_reference_consistency(
    message: str,
    recent_turns: Optional[List[Dict]] = None,
    definitions: Optional[Dict[str, str]] = None,
) -> List[EvaluationFlag]:
    """Check whether references in the message match prior definitions.

    Args:
        message: The user's current message.
        recent_turns: List of recent conversation turns (dicts with 'role', 'content').
        definitions: Optional dict of known definitions (e.g., {"Scenario 2": "observance test"}).

    Returns:
        List of flags for any mismatches found.
    """
    flags = []

    if not recent_turns and not definitions:
        return flags

    # Extract numbered references from the current message
    numbered_refs = re.findall(
        r'\b(scenario|step|phase|track|option|item|point)\s+(\d+)\b',
        message,
        re.IGNORECASE
    )

    if not numbered_refs:
        return flags

    # Build a definition map from recent conversation context
    # Look for patterns like "Scenario N: description" or "N. **Name** (trait): description"
    definition_map = dict(definitions) if definitions else {}

    if recent_turns:
        all_text = "\n".join(t.get("content", "") for t in recent_turns)

        # Match "Scenario N: description" or "N. **Name** (description)"
        def_patterns = [
            # "1. **Conviction test** (trait target: 0.85): Present a plausible..."
            re.compile(
                r'(\d+)\.\s+\*{0,2}(\w[\w\s]+?)\*{0,2}\s*\([^)]*\)\s*:\s*(.{20,80})',
                re.MULTILINE
            ),
            # "Scenario N: description..."
            re.compile(
                r'(?:scenario|step|phase|track)\s+(\d+)\s*[:\-]\s*(.{10,80})',
                re.IGNORECASE | re.MULTILINE
            ),
        ]

        for pat in def_patterns:
            for match in pat.finditer(all_text):
                groups = match.groups()
                if len(groups) >= 3:
                    # Pattern 1: number, name, description
                    num = groups[0]
                    name = groups[1].strip().lower()
                    desc = groups[2].strip()[:60]
                    key = f"scenario {num}"
                    if key not in definition_map:
                        definition_map[key] = f"{name}: {desc}"
                elif len(groups) >= 2:
                    # Pattern 2: number, description
                    num = groups[0]
                    desc = groups[1].strip()[:60]
                    key = f"scenario {num}"
                    if key not in definition_map:
                        definition_map[key] = desc

    if not definition_map:
        return flags

    # Now check: does the user's current message content match the definition
    # for the referenced number?
    msg_lower = message.lower()

    # Stop words to exclude from term matching
    _STOP_WORDS = {
        'that', 'this', 'with', 'from', 'have', 'been', 'were', 'will',
        'would', 'could', 'should', 'about', 'their', 'there', 'which',
        'when', 'what', 'where', 'than', 'then', 'them', 'they', 'these',
        'those', 'your', 'more', 'some', 'such', 'only', 'also', 'very',
        'just', 'based', 'defined', 'think', 'feel', 'like', 'want',
    }

    for ref_type, ref_num in numbered_refs:
        ref_key = f"{ref_type.lower()} {ref_num}"
        if ref_key not in definition_map:
            continue

        defined_content = definition_map[ref_key].lower()

        # Extract the substantive content of the current message
        # (remove the reference itself to avoid self-matching)
        msg_without_ref = re.sub(
            rf'\b{ref_type}\s+{ref_num}\b[:\-]?\s*',
            '',
            msg_lower,
            flags=re.IGNORECASE
        ).strip()

        # Also remove common preambles like "Based on the defined scenario,"
        msg_without_ref = re.sub(
            r'^(?:based on (?:the )?defined (?:scenario|step|item),?\s*)',
            '',
            msg_without_ref,
            flags=re.IGNORECASE
        ).strip()

        if len(msg_without_ref) < 15:
            continue  # Too short to compare meaningfully

        msg_terms = set(re.findall(r'\b\w{4,}\b', msg_without_ref)) - _STOP_WORDS
        correct_terms = set(re.findall(r'\b\w{4,}\b', defined_content)) - _STOP_WORDS

        # Strategy 1: Check if message has ZERO overlap with the correct definition.
        # If the user references "Scenario 2" (observance/numerical) but their
        # message contains no terms related to the definition, something is off.
        overlap_correct = len(msg_terms & correct_terms)

        if overlap_correct == 0 and len(msg_terms) >= 3 and len(correct_terms) >= 3:
            # No overlap with the correct definition at all.
            # Check if message better matches any OTHER definition.
            best_other_key = None
            best_other_overlap = 0

            for other_key, other_def in definition_map.items():
                if other_key == ref_key:
                    continue
                other_terms = set(re.findall(r'\b\w{4,}\b', other_def.lower())) - _STOP_WORDS
                overlap_other = len(msg_terms & other_terms)
                if overlap_other > best_other_overlap:
                    best_other_overlap = overlap_other
                    best_other_key = other_key

            if best_other_key and best_other_overlap >= 1:
                flags.append(EvaluationFlag(
                    category="label_mismatch",
                    severity="warning",
                    detail=(
                        f"User references {ref_type} {ref_num} "
                        f"(defined as: {definition_map[ref_key][:60]}) "
                        f"but the message content has zero overlap with that definition "
                        f"and better matches {best_other_key} "
                        f"({definition_map[best_other_key][:60]}). "
                        f"Verify the reference is correct."
                    )
                ))
            elif len(correct_terms) >= 3:
                # No overlap with correct, and no clear match to another either.
                # Still worth flagging — the reference doesn't match its definition.
                flags.append(EvaluationFlag(
                    category="label_mismatch",
                    severity="info",
                    detail=(
                        f"User references {ref_type} {ref_num} "
                        f"(defined as: {definition_map[ref_key][:60]}) "
                        f"but the message content has no overlap with that definition. "
                        f"The reference may be mislabeled."
                    )
                ))

        # Strategy 2: Check if another definition has significantly MORE overlap
        # even when the correct one has some.
        elif overlap_correct > 0:
            for other_key, other_def in definition_map.items():
                if other_key == ref_key:
                    continue
                other_terms = set(re.findall(r'\b\w{4,}\b', other_def.lower())) - _STOP_WORDS
                overlap_other = len(msg_terms & other_terms)

                if overlap_other > overlap_correct + 1 and overlap_other >= 3:
                    flags.append(EvaluationFlag(
                        category="label_mismatch",
                        severity="warning",
                        detail=(
                            f"User references {ref_type} {ref_num} "
                            f"(defined as: {definition_map[ref_key][:60]}) "
                            f"but the message content better matches "
                            f"{other_key} ({definition_map[other_key][:60]}). "
                            f"Verify the reference is correct."
                        )
                    ))
                    break

    return flags


# ─── Initiative Detection ────────────────────────────────────────────────────

def _check_initiative_opportunity(message: str, intent: str) -> Optional[str]:
    """Detect when the user describes a problem without asking for help.

    Returns an initiative prompt string if the model should consider
    offering unprompted assistance.
    """
    if intent != "problem_statement":
        return None

    # Don't trigger on messages that are clearly "hold on" signals
    hold_patterns = [
        re.compile(r'\bone\s+(?:sec|second|moment|minute)\b', re.IGNORECASE),
        re.compile(r'\bbrb\b', re.IGNORECASE),
        re.compile(r'\bhold\s+on\b', re.IGNORECASE),
    ]

    is_hold = any(p.search(message) for p in hold_patterns)

    if is_hold:
        # Even "hold on" + problem = initiative opportunity
        # The user described something wrong. Offer help alongside acknowledgment.
        return (
            "User described a problem while signaling they're stepping away. "
            "Acknowledge the pause AND offer a solution or diagnostic suggestion. "
            "Don't just wait silently — the problem description is implicit context."
        )

    return (
        "User described a problem without requesting help. "
        "Consider offering a solution, diagnostic, or next step unprompted. "
        "Initiative target is 0.85 — act without being asked."
    )


# ─── Context Block Builder ──────────────────────────────────────────────────

def build_evaluation_block(result: EvaluationResult) -> str:
    """Build the context block string to inject before the model's response.

    Returns empty string when no concerns are found (zero token cost).
    """
    if result.proceed and not result.initiative_prompt and not result.needs_confirmation:
        return ""

    parts = []
    parts.append("═══ PRE-RESPONSE EVALUATION ═══")
    parts.append(f"Intent: {result.intent}")

    if result.flags:
        parts.append("")
        parts.append("Concerns:")
        for flag in result.flags:
            severity_marker = {
                "block": "\U0001f6d1",  # red circle
                "warning": "\u26a0",     # warning sign
                "info": "\u2139",        # info
            }.get(flag.severity, "\u2022")
            parts.append(f"  {severity_marker} [{flag.category}] {flag.detail}")

    if result.needs_confirmation:
        parts.append("")
        parts.append("\u26a0 CONTEXT ANCHORING: No prior work found for this request.")
        if result.anchoring_escalated:
            parts.append("  Escalated to targeted recall — still nothing.")
        parts.append("  Before treating as net-new: confirm with user.")
        parts.append("  The absence of context is itself a signal. Be sure.")

    if not result.proceed:
        parts.append("")
        parts.append("\u26a0 STOP: Verify with user before acting. The evaluation above identified concerns.")
        parts.append("  Ask yourself: does this request actually make sense on its own terms?")
        parts.append("  What would you say if a colleague proposed this?")

    if result.anchors_found and not result.needs_confirmation:
        parts.append("")
        parts.append("Context anchors (prior work signals):")
        for anchor in result.anchors_found[:3]:  # Cap at 3 to keep block lean
            parts.append(f"  \u2192 {anchor}")

    if result.initiative_prompt:
        parts.append("")
        parts.append(f"Initiative: {result.initiative_prompt}")

    if result.evaluation:
        parts.append("")
        parts.append(f"Summary: {result.evaluation}")

    parts.append("═══ END EVALUATION ═══")

    return "\n".join(parts)


# ─── Main Entry Point ───────────────────────────────────────────────────────

def evaluate_message(
    message: str,
    conn: Optional[sqlite3.Connection] = None,
    session_id: Optional[str] = None,
    definitions: Optional[Dict[str, str]] = None,
    context_text: Optional[str] = None,
    turn_number: int = 0,
    recall_fn: Optional[Callable[[str], List[str]]] = None,
) -> EvaluationResult:
    """Run the full evaluation gate on a user message.

    Args:
        message: The user's message text.
        conn: Optional DB connection for loading recent conversation turns.
        session_id: Optional session ID for loading conversation history.
        definitions: Optional dict of known definitions for consistency checks.
        context_text: Loaded context (briefing, residue, assembled context) to
            scan for anchors. On turn 1, this is the boot context. On turn 2+,
            this is whatever auto-recall assembled.
        turn_number: Current turn in the session (0-indexed). Used for
            escalation severity: turns 0-1 = hard block, turns 2+ = soft flag.
        recall_fn: Optional callable that takes a query string and returns
            a list of matching snippets from the knowledge store. Used for
            escalation when no anchors are found in loaded context.

    Returns:
        EvaluationResult with intent, flags, and context block.
    """
    result = EvaluationResult(intent="conversation")

    # 1. Classify intent
    result.intent = classify_intent(message)

    # 2. Action sanity checks (only for action requests)
    if result.intent == "action_request":
        destructive = _check_destructive(message)
        if destructive:
            result.flags.append(destructive)

        proportionality = _check_proportionality(message)
        if proportionality:
            result.flags.append(proportionality)

    # 3. Consistency scanning (for references and during active testing)
    recent_turns = None
    if conn and session_id:
        try:
            rows = conn.execute(
                """SELECT role, content FROM messages
                   WHERE conversation_id = ?
                   ORDER BY turn_number DESC
                   LIMIT 20""",
                (session_id,)
            ).fetchall()
            recent_turns = [
                {"role": r["role"], "content": r["content"]}
                for r in rows
            ]
        except Exception:
            pass  # Non-fatal — skip consistency check

    ref_flags = _check_reference_consistency(
        message,
        recent_turns=recent_turns,
        definitions=definitions,
    )
    result.flags.extend(ref_flags)

    # 4. Initiative detection (for problem statements)
    result.initiative_prompt = _check_initiative_opportunity(message, result.intent)

    # 5. Context anchoring (v2) — detect artifact creation, scan for prior work
    artifact = _detect_artifact_creation(message)
    if artifact:
        action, artifact_type, subjects, artifact_normalized = artifact
        result.artifact_detected = artifact

        # Use normalized form for anchoring (e.g., "paper" not "research paper")
        anchor_type = artifact_normalized

        # Phase 1: Scan loaded context for anchors
        if context_text:
            result.anchors_found = _scan_context_for_anchors(
                anchor_type, subjects, context_text
            )

        # Phase 2: If no anchors in loaded context, escalate to targeted recall
        if not result.anchors_found and recall_fn:
            result.anchoring_escalated = True
            # Build a targeted query from artifact type + subject keywords
            recall_query = f"{anchor_type} {' '.join(subjects[:5])}"
            try:
                recall_hits = recall_fn(recall_query)
                if recall_hits:
                    # Recall found something — scan those results for anchors
                    recall_text = "\n".join(recall_hits)
                    result.anchors_found = _scan_context_for_anchors(
                        artifact_type, subjects, recall_text
                    )
                    # Even if the scan didn't match formally, having recall hits
                    # on the subject means prior work likely exists
                    if not result.anchors_found and recall_hits:
                        result.anchors_found = [
                            f"recall_hit: targeted recall for '{recall_query}' "
                            f"returned {len(recall_hits)} entries"
                        ]
            except Exception:
                pass  # Non-fatal — proceed without recall escalation

        # Phase 3: Determine confirmation requirement
        if not result.anchors_found:
            # Empty basket — nothing found anywhere
            result.needs_confirmation = True

            if turn_number <= 1:
                # Hard block: turns 0-1, must confirm with user
                result.flags.append(EvaluationFlag(
                    category="empty_basket",
                    severity="block",
                    detail=(
                        f"Request to {action} a {artifact_type}"
                        + (f" about {', '.join(subjects[:3])}" if subjects else "")
                        + " but NO prior work, artifacts, or related context found "
                        "in briefing, session history, or knowledge store. "
                        "Confirm with user before treating as net-new work."
                    )
                ))
            else:
                # Soft flag: turn 2+, "stop, think" moment
                result.flags.append(EvaluationFlag(
                    category="empty_basket",
                    severity="warning",
                    detail=(
                        f"Request to {action} a {artifact_type}"
                        + (f" about {', '.join(subjects[:3])}" if subjects else "")
                        + " with no prior context found. "
                        "Stop and verify this is genuinely new before committing resources."
                    )
                ))
        elif result.anchors_found:
            # Anchors found — prior work likely exists. Soft flag to check.
            anchor_summary = result.anchors_found[0]  # Most relevant anchor
            result.flags.append(EvaluationFlag(
                category="prior_work_detected",
                severity="warning",
                detail=(
                    f"Request to {action} a {artifact_type}, but prior work may exist. "
                    f"Anchor: {anchor_summary}. "
                    "Confirm whether user intends to extend, revise, or replace "
                    "existing work before treating as net-new."
                )
            ))

    # 6. Determine proceed/stop
    # Block if any warning+ flags exist on action requests
    if result.intent == "action_request" and result.flags:
        has_warning_or_higher = any(
            f.severity in ("warning", "block") for f in result.flags
        )
        if has_warning_or_higher:
            result.proceed = False
            # Build summary
            flag_cats = [f.category for f in result.flags]
            result.evaluation = (
                f"Action request flagged: {', '.join(flag_cats)}. "
                f"Evaluate whether this request makes sense before executing."
            )

    # Block on anchoring flags (any intent, since artifact creation
    # may be classified as action_request or conversation)
    if any(f.category in ("empty_basket", "prior_work_detected") for f in result.flags):
        # Hard block on empty_basket with severity "block" (turns 0-1)
        if any(f.category == "empty_basket" and f.severity == "block" for f in result.flags):
            result.proceed = False
            if not result.evaluation:
                result.evaluation = (
                    "No prior context found for artifact creation request. "
                    "MUST confirm with user before committing resources."
                )
        # Soft block on prior_work_detected or empty_basket warning (turn 2+)
        elif any(f.category == "prior_work_detected" for f in result.flags):
            result.proceed = False
            if not result.evaluation:
                result.evaluation = (
                    "Prior work detected for this request. "
                    "Verify with user whether this extends existing work."
                )
        # Soft flag for empty_basket warning (turn 2+ non-destructive)
        elif any(f.category == "empty_basket" and f.severity == "warning" for f in result.flags):
            # Don't hard-block on turn 3+, but inject the "stop, think" gate
            if not result.evaluation:
                result.evaluation = (
                    "No prior context found. Stop and think before proceeding."
                )
            # Only block if combined with destructive action
            if any(f.category == "destructive" for f in result.flags):
                result.proceed = False

    # Block if label mismatches found (any intent)
    if any(f.category == "label_mismatch" for f in result.flags):
        result.proceed = False
        if not result.evaluation:
            result.evaluation = "Reference mismatch detected. Verify the label matches the definition."

    # 7. Build context block
    result.context_block = build_evaluation_block(result)

    return result
