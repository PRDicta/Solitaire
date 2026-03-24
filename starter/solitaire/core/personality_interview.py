"""
The Librarian — Personality Interview System
Scenario-based calibration questions that map to trait deltas.

Each question presents a natural-language scenario with 3 options.
Each option maps to specific trait modifiers (deltas) that get applied
on top of a template baseline or DEFAULT_TRAITS for build-your-own.

The interview is designed to cover the key behavioral axes:
  - autonomy → initiative
  - pushback tolerance → conviction, assertiveness
  - communication style → warmth, empathy
  - humor level → humor
  - detail focus → observance, assertiveness
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ─── Constants ──────────────────────────────────────────────────────────────

# Maximum aggregate delta per trait across all interview questions.
# Prevents extreme swings from stacking.
MAX_AGGREGATE_DELTA = 0.30

# Minimum trait value after deltas applied
TRAIT_FLOOR = 0.05

# Maximum trait value after deltas applied
TRAIT_CEILING = 0.95


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class InterviewOption:
    """A single answer choice for an interview question."""
    key: str                            # "A", "B", "C"
    text: str                           # Natural-language answer text
    trait_deltas: Dict[str, float]      # {trait_name: delta_value}
    description: str = ""               # Optional short label for UI

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "text": self.text,
            "trait_effect": ", ".join(
                f"{t}: {round(d * 100):+d}%" for t, d in self.trait_deltas.items()
            ) if self.trait_deltas else "neutral",
            "description": self.description,
        }


@dataclass
class InterviewQuestion:
    """A single calibration question in the personality interview."""
    id: str                              # Unique identifier
    title: str                           # Short display title
    prompt: str                          # The scenario text shown to user
    affects: List[str]                   # Which traits this question covers
    options: List[InterviewOption]       # The 3 answer choices
    explanation: str                     # Why this question matters
    required: bool = True                # Core vs optional
    order: int = 0                       # Display order

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "prompt": self.prompt,
            "affects": self.affects,
            "options": [o.to_dict() for o in self.options],
            "explanation": self.explanation,
            "required": self.required,
            "order": self.order,
        }

    def get_option(self, key: str) -> Optional[InterviewOption]:
        """Look up an option by key (case-insensitive)."""
        key_upper = key.strip().upper()
        for opt in self.options:
            if opt.key.upper() == key_upper:
                return opt
        return None


# ─── Core Interview Questions ───────────────────────────────────────────────

CORE_QUESTIONS: List[InterviewQuestion] = [
    InterviewQuestion(
        id="autonomy_preference",
        title="Initiative",
        prompt=(
            "You're in the middle of something. "
            "When should I jump in?"
        ),
        affects=["initiative"],
        options=[
            InterviewOption(
                key="A",
                text="Wait for me — I'll ask when I need you",
                trait_deltas={"initiative": -0.15},
                description="On call",
            ),
            InterviewOption(
                key="B",
                text="Flag what you see — I'll decide what to act on",
                trait_deltas={},
                description="Spotter",
            ),
            InterviewOption(
                key="C",
                text="Run with it — take action on what you're confident about and loop me in after",
                trait_deltas={"initiative": 0.20},
                description="Co-pilot",
            ),
        ],
        explanation="This calibrates when I act on my own vs. wait for direction.",
        order=1,
    ),

    InterviewQuestion(
        id="pushback_tolerance",
        title="Pushback",
        prompt=(
            "You propose a direction. I notice potential issues. "
            "How should I respond?"
        ),
        affects=["conviction", "assertiveness"],
        options=[
            InterviewOption(
                key="A",
                text="Support your choice unless it's clearly wrong",
                trait_deltas={"conviction": -0.20, "assertiveness": -0.15},
                description="Supportive",
            ),
            InterviewOption(
                key="B",
                text="Surface concerns thoughtfully, but defer to your judgment",
                trait_deltas={"conviction": 0.0, "assertiveness": -0.05},
                description="Diplomatic",
            ),
            InterviewOption(
                key="C",
                text="Push back hard with evidence when I think you're off-track",
                trait_deltas={"conviction": 0.25, "assertiveness": 0.15},
                description="Challenger",
            ),
        ],
        explanation="This tells me how willing you are to hear contrarian views.",
        order=2,
    ),

    InterviewQuestion(
        id="communication_warmth",
        title="Communication Style",
        prompt="How should I frame my responses?",
        affects=["warmth", "empathy"],
        options=[
            InterviewOption(
                key="A",
                text="Professional and clinical — facts first, minimal emotion",
                trait_deltas={"warmth": -0.25, "empathy": -0.10},
                description="Clinical",
            ),
            InterviewOption(
                key="B",
                text="Balanced — clear but conversational",
                trait_deltas={},
                description="Balanced",
            ),
            InterviewOption(
                key="C",
                text=(
                    "Warm and encouraging — celebrate wins, "
                    "acknowledge the effort behind them"
                ),
                trait_deltas={"warmth": 0.25, "empathy": 0.15},
                description="Warm",
            ),
        ],
        explanation="This shapes how I communicate with you day-to-day.",
        order=3,
    ),

    InterviewQuestion(
        id="humor_preference",
        title="Humor",
        prompt="When should I use humor?",
        affects=["humor"],
        options=[
            InterviewOption(
                key="A",
                text="Never — keep it strictly professional",
                trait_deltas={"humor": -0.35},
                description="Serious",
            ),
            InterviewOption(
                key="B",
                text="Rarely — only if the moment naturally calls for it",
                trait_deltas={"humor": -0.10},
                description="Dry",
            ),
            InterviewOption(
                key="C",
                text="Often — use wit and levity to lighten the mood",
                trait_deltas={"humor": 0.25},
                description="Witty",
            ),
        ],
        explanation="This helps me match your communication style.",
        order=4,
    ),

    InterviewQuestion(
        id="detail_obsession",
        title="Detail Catching",
        prompt=(
            "You miss a detail that matters later. "
            "What should I have done?"
        ),
        affects=["observance", "assertiveness"],
        options=[
            InterviewOption(
                key="A",
                text="Just execute what you asked — it's your call to add detail",
                trait_deltas={"observance": -0.20, "assertiveness": -0.15},
                description="Executor",
            ),
            InterviewOption(
                key="B",
                text=(
                    "Surface missed details if I notice them, "
                    "but don't be annoying"
                ),
                trait_deltas={"observance": 0.05, "assertiveness": -0.05},
                description="Observant",
            ),
            InterviewOption(
                key="C",
                text=(
                    "Flag anomalies, contradictions, and details aggressively "
                    "— I should catch what you might miss"
                ),
                trait_deltas={"observance": 0.30, "assertiveness": 0.15},
                description="Vigilant",
            ),
        ],
        explanation="This calibrates how much detail-catching I should do.",
        order=5,
    ),
]


# ─── Optional Questions ─────────────────────────────────────────────────────

OPTIONAL_QUESTIONS: List[InterviewQuestion] = [
    InterviewQuestion(
        id="learning_pace",
        title="Learning Pace",
        prompt=(
            "I notice you repeating a pattern that could be improved. "
            "How should I respond?"
        ),
        affects=["initiative", "conviction"],
        options=[
            InterviewOption(
                key="A",
                text="Once is enough — I'll figure it out",
                trait_deltas={"initiative": -0.10, "conviction": -0.10},
                description="Hands-off",
            ),
            InterviewOption(
                key="B",
                text="Gently suggest an alternative next time it comes up",
                trait_deltas={"initiative": 0.05, "conviction": 0.05},
                description="Nudge",
            ),
            InterviewOption(
                key="C",
                text=(
                    "Proactively draft solutions and patterns "
                    "to break the loop"
                ),
                trait_deltas={"initiative": 0.15, "conviction": 0.10},
                description="Proactive",
            ),
        ],
        explanation="This tells me how aggressively to optimize your workflows.",
        required=False,
        order=6,
    ),

    InterviewQuestion(
        id="disagreement_style",
        title="Disagreement Resolution",
        prompt=(
            "We disagree on something important. "
            "What's your preferred way to resolve it?"
        ),
        affects=["assertiveness", "conviction"],
        options=[
            InterviewOption(
                key="A",
                text="You decide — I'll support your choice",
                trait_deltas={"assertiveness": -0.15, "conviction": -0.15},
                description="Deferential",
            ),
            InterviewOption(
                key="B",
                text="Show me the data and reasoning — let's align",
                trait_deltas={"assertiveness": 0.05, "conviction": 0.05},
                description="Collaborative",
            ),
            InterviewOption(
                key="C",
                text="Convince me — I expect strong evidence for pushback",
                trait_deltas={"assertiveness": 0.15, "conviction": 0.15},
                description="Demanding",
            ),
        ],
        explanation=(
            "This shapes how I handle moments where "
            "we see things differently."
        ),
        required=False,
        order=7,
    ),
]


# ─── All questions by ID ────────────────────────────────────────────────────

ALL_QUESTIONS: Dict[str, InterviewQuestion] = {
    q.id: q for q in CORE_QUESTIONS + OPTIONAL_QUESTIONS
}

CORE_QUESTION_IDS: List[str] = [q.id for q in CORE_QUESTIONS]
OPTIONAL_QUESTION_IDS: List[str] = [q.id for q in OPTIONAL_QUESTIONS]


# ─── Interview Engine ────────────────────────────────────────────────────────

class InterviewEngine:
    """Process interview answers and compute trait deltas."""

    def __init__(self, include_optional: bool = False):
        self.include_optional = include_optional

    @property
    def question_ids(self) -> List[str]:
        """Return ordered list of question IDs for this interview."""
        ids = list(CORE_QUESTION_IDS)
        if self.include_optional:
            ids.extend(OPTIONAL_QUESTION_IDS)
        return ids

    @property
    def questions(self) -> List[InterviewQuestion]:
        """Return ordered list of questions for this interview."""
        return [ALL_QUESTIONS[qid] for qid in self.question_ids]

    def get_question(self, question_id: str) -> Optional[InterviewQuestion]:
        """Look up a question by ID."""
        return ALL_QUESTIONS.get(question_id)

    def validate_answer(
        self, question_id: str, answer_key: str
    ) -> Tuple[bool, str]:
        """Validate that an answer key is valid for a question.

        Returns:
            (is_valid, error_message)
        """
        question = ALL_QUESTIONS.get(question_id)
        if not question:
            return False, f"Unknown question: {question_id}"
        opt = question.get_option(answer_key)
        if not opt:
            valid_keys = [o.key for o in question.options]
            return False, (
                f"Invalid answer '{answer_key}' for {question_id}. "
                f"Valid: {valid_keys}"
            )
        return True, ""

    def process_answer(
        self, question_id: str, answer_key: str
    ) -> Dict[str, float]:
        """Process a single answer and return its trait deltas.

        Args:
            question_id: Which question was answered.
            answer_key: The selected option key (A, B, or C).

        Returns:
            Dict of {trait_name: delta_value} for this answer.

        Raises:
            ValueError: If question_id or answer_key is invalid.
        """
        valid, error = self.validate_answer(question_id, answer_key)
        if not valid:
            raise ValueError(error)

        question = ALL_QUESTIONS[question_id]
        option = question.get_option(answer_key)
        return dict(option.trait_deltas)

    def compute_trait_deltas(
        self, answers: Dict[str, str]
    ) -> Dict[str, float]:
        """Aggregate trait deltas from all interview answers.

        Sums deltas per trait across all answered questions,
        then clamps each trait's aggregate to ±MAX_AGGREGATE_DELTA.

        Args:
            answers: {question_id: answer_key} for each answered question.

        Returns:
            Dict of {trait_name: clamped_aggregate_delta}.
        """
        aggregated: Dict[str, float] = {}

        for question_id, answer_key in answers.items():
            try:
                deltas = self.process_answer(question_id, answer_key)
            except ValueError:
                continue  # Skip invalid answers gracefully

            for trait, delta in deltas.items():
                aggregated[trait] = aggregated.get(trait, 0.0) + delta

        # Clamp each trait's aggregate
        clamped = {}
        for trait, total in aggregated.items():
            clamped[trait] = max(
                -MAX_AGGREGATE_DELTA,
                min(MAX_AGGREGATE_DELTA, total),
            )

        return clamped

    def explain_modifiers(
        self,
        answers: Dict[str, str],
        baseline: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, str]]:
        """Generate human-readable explanations for each trait modifier.

        Returns a list of explanations showing how each answer
        affected the trait profile, suitable for the preview step.

        Args:
            answers: {question_id: answer_key} for each answered question.
            baseline: Optional baseline traits for context.

        Returns:
            List of dicts: [{trait, delta, explanation, from_question}]
        """
        explanations = []

        for question_id, answer_key in answers.items():
            question = ALL_QUESTIONS.get(question_id)
            if not question:
                continue
            option = question.get_option(answer_key)
            if not option:
                continue

            for trait, delta in option.trait_deltas.items():
                if abs(delta) < 0.001:
                    continue  # Skip zero deltas

                direction = "increased" if delta > 0 else "decreased"
                pct = round(abs(delta) * 100)
                explanations.append({
                    "trait": trait,
                    "delta": delta,
                    "direction": direction,
                    "from_question": question.title,
                    "explanation": (
                        f"{trait.capitalize()} {direction} by {pct}% "
                        f"based on your {question.title.lower()} preference "
                        f"(\"{option.description}\")"
                    ),
                })

        return explanations

    def get_coverage_report(
        self, answers: Dict[str, str]
    ) -> Dict[str, bool]:
        """Check which traits have been calibrated by the interview.

        Returns:
            Dict of {trait_name: was_covered}.
        """
        from .persona import VALID_TRAIT_NAMES
        covered = set()
        for question_id, answer_key in answers.items():
            question = ALL_QUESTIONS.get(question_id)
            if question:
                covered.update(question.affects)
        return {t: t in covered for t in VALID_TRAIT_NAMES}


def apply_deltas_to_baseline(
    baseline: Dict[str, float],
    deltas: Dict[str, float],
) -> Dict[str, float]:
    """Apply interview deltas to a baseline trait profile.

    Clamps final values to [TRAIT_FLOOR, TRAIT_CEILING].

    Args:
        baseline: Starting trait values (from template or defaults).
        deltas: Aggregate deltas from compute_trait_deltas().

    Returns:
        New trait dict with deltas applied and clamped.
    """
    result = dict(baseline)
    for trait, delta in deltas.items():
        if trait in result:
            result[trait] = max(
                TRAIT_FLOOR,
                min(TRAIT_CEILING, result[trait] + delta),
            )
    return result
