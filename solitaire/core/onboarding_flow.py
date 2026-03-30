"""
Solitaire — Onboarding Flow Engine (v2: Template-Free)
Structured multi-step pipeline for first-session persona setup.

Produces JSON steps that any agent (Claude, etc.) can present to the user.
Handles: welcome → intent → live research → trait proposal → working style
         → interview → naming → north star → seed questions → preview → confirm → apply.

The flow is stateful: OnboardingContext accumulates answers across steps.
The FlowEngine is stateless: given a context, it knows what step comes next.

v2 changes: Templates removed. Persona construction uses live research against
the user's intent with heuristic fallback. Working style preferences replace
template selection. Progressive revelation gates features by session count.
"""
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .personality_interview import (
    InterviewEngine,
    InterviewQuestion,
    CORE_QUESTION_IDS,
    OPTIONAL_QUESTION_IDS,
    ALL_QUESTIONS,
    apply_deltas_to_baseline,
    TRAIT_FLOOR,
    TRAIT_CEILING,
)
from .persona import (
    DEFAULT_TRAITS,
    VALID_TRAIT_NAMES,
    DriftConfig,
    SharingConfig,
    PersonaProfile,
    PersonaIdentity,
    DomainEnvelope,
    BehavioralTriggers,
    TraitProfile,
)


# ─── Constants ──────────────────────────────────────────────────────────────

FLOW_VERSION = "2.1"

STEP_TYPES = {"info", "question", "multiple_choice", "preview", "confirm"}

# ─── First-Message Classification ────────────────────────────────────────────
# Patterns that indicate the user's first message is a real task, not a greeting
# or willingness to set up. These users should get help first, onboarding later.

GREETING_PATTERNS = [
    r'^h(i|ey|ello|owdy)\b',
    r'^(good\s+)?(morning|afternoon|evening)\b',
    r'^(what\'?s?\s+up|sup)\b',
    r'^yo\b',
]

TASK_SIGNALS = [
    r'\?$',                         # ends with question mark
    r'```',                         # contains code block
    r'\b(fix|debug|help|solve|build|create|write|analyze|review|explain)\b',
    r'\b(error|bug|issue|problem|broken)\b',
    r'\b(how\s+(do|can|should|would))\b',
    r'\b(can\s+you|could\s+you|please)\b',
    r'\b(i\s+need|i\s+want|i\'m\s+trying)\b',
]

# ─── Vague Intent Detection ─────────────────────────────────────────────────
# Phrases that signal the user doesn't have a clear use case yet.

VAGUE_INTENT_PHRASES = [
    r'^(just\s+)?(help|stuff|things|idk|dunno|whatever)',
    r'^(i\s+don\'?t\s+know)',
    r'^(anything|everything|general)',
    r'^(not\s+sure)',
    r'^(just\s+trying\s+(it|this)\s+out)',
]

INTENT_CATEGORY_PICKER = [
    {"key": "writing", "label": "Writing and content creation",
     "intent_seed": "writing and content creation"},
    {"key": "work", "label": "Work and business tasks",
     "intent_seed": "business operations and work tasks"},
    {"key": "creative", "label": "Creative projects (design, art, music)",
     "intent_seed": "creative projects and design"},
    {"key": "coding", "label": "Coding and engineering",
     "intent_seed": "software engineering and coding"},
    {"key": "learning", "label": "Learning and research",
     "intent_seed": "learning and research"},
    {"key": "personal", "label": "Personal organization and life admin",
     "intent_seed": "personal organization and productivity"},
    {"key": "other", "label": "Something else (I'll describe it)",
     "intent_seed": ""},
]

# Maximum aggregate delta per trait from working style preferences
WORKING_STYLE_MAX_DELTA = 0.15

# Domain suggestions for manual domain config (fallback)
DOMAIN_SUGGESTIONS = {
    "primary": [
        "general",
        "business-operations",
        "software-development",
        "creative-production",
        "financial-analysis",
        "content-production",
        "research",
        "education",
    ],
    "secondary": [
        "finance",
        "marketing",
        "product",
        "design",
        "data-analysis",
        "project-management",
        "client-management",
        "writing",
    ],
}

# ─── Intent-to-Traits Heuristic Table ───────────────────────────────────────
# Used when web search is unavailable. Keywords are regex alternations.
# Multiple matches are averaged, weighted by keyword hit count.

INTENT_TRAIT_HEURISTICS = {
    "finance|trading|accounting|tax|audit|bookkeeping": {
        "observance": 0.85, "conviction": 0.75, "assertiveness": 0.70,
        "warmth": 0.40, "initiative": 0.65, "empathy": 0.45, "humor": 0.30,
    },
    "creative|writing|design|art|music|storytelling": {
        "observance": 0.60, "conviction": 0.55, "assertiveness": 0.50,
        "warmth": 0.70, "initiative": 0.65, "empathy": 0.70, "humor": 0.60,
    },
    "operations|management|startup|business|strategy": {
        "observance": 0.75, "conviction": 0.70, "assertiveness": 0.70,
        "warmth": 0.50, "initiative": 0.80, "empathy": 0.55, "humor": 0.40,
    },
    "engineering|coding|development|software|programming|devops": {
        "observance": 0.80, "conviction": 0.70, "assertiveness": 0.65,
        "warmth": 0.45, "initiative": 0.70, "empathy": 0.45, "humor": 0.35,
    },
    "coaching|therapy|wellness|support|counseling|mental health": {
        "observance": 0.70, "conviction": 0.45, "assertiveness": 0.40,
        "warmth": 0.85, "initiative": 0.55, "empathy": 0.90, "humor": 0.50,
    },
    "research|academic|science|analysis|data": {
        "observance": 0.85, "conviction": 0.65, "assertiveness": 0.55,
        "warmth": 0.45, "initiative": 0.60, "empathy": 0.45, "humor": 0.30,
    },
    "gaming|esports|competitive|warhammer|tabletop": {
        "observance": 0.80, "conviction": 0.80, "assertiveness": 0.75,
        "warmth": 0.45, "initiative": 0.70, "empathy": 0.40, "humor": 0.50,
    },
    "legal|compliance|regulatory|contract|policy": {
        "observance": 0.90, "conviction": 0.80, "assertiveness": 0.75,
        "warmth": 0.35, "initiative": 0.60, "empathy": 0.40, "humor": 0.20,
    },
    "sales|marketing|growth|content|social media": {
        "observance": 0.65, "conviction": 0.65, "assertiveness": 0.70,
        "warmth": 0.65, "initiative": 0.80, "empathy": 0.60, "humor": 0.55,
    },
    "education|teaching|tutoring|curriculum|training": {
        "observance": 0.75, "conviction": 0.55, "assertiveness": 0.50,
        "warmth": 0.80, "initiative": 0.65, "empathy": 0.80, "humor": 0.55,
    },
}

# ─── Working Style Definitions ──────────────────────────────────────────────
# Each category has options that map to trait deltas.

WORKING_STYLE_CATEGORIES = [
    {
        "id": "communication",
        "title": "Communication",
        "prompt": "How should I communicate with you?",
        "options": [
            {
                "key": "direct",
                "label": "Short and direct. Don't over-explain.",
                "trait_deltas": {"warmth": -0.05, "assertiveness": 0.05},
            },
            {
                "key": "thorough",
                "label": "Thorough. I'd rather have too much context than too little.",
                "trait_deltas": {"observance": 0.05, "warmth": 0.05},
            },
            {
                "key": "mirror",
                "label": "Match my energy. Short question, short answer.",
                "trait_deltas": {},
                "flags": ["mirror_input_length"],
            },
        ],
    },
    {
        "id": "feedback",
        "title": "Feedback",
        "prompt": "How should I handle disagreements?",
        "options": [
            {
                "key": "tell_me",
                "label": "Tell me when I'm wrong. I'd rather be corrected than comfortable.",
                "trait_deltas": {"conviction": 0.10, "assertiveness": 0.05},
            },
            {
                "key": "suggest",
                "label": "Suggest alternatives, but let me decide.",
                "trait_deltas": {"conviction": 0.05},
            },
            {
                "key": "support_first",
                "label": "Support first, critique only when I ask.",
                "trait_deltas": {"conviction": -0.10, "warmth": 0.05},
            },
        ],
    },
    {
        "id": "initiative",
        "title": "Initiative",
        "prompt": "How proactive should I be?",
        "options": [
            {
                "key": "see_do",
                "label": "If you see something that needs doing, do it.",
                "trait_deltas": {"initiative": 0.10},
            },
            {
                "key": "suggest_wait",
                "label": "Suggest things, but wait for my go-ahead.",
                "trait_deltas": {"initiative": 0.05},
            },
            {
                "key": "only_asked",
                "label": "Only do exactly what I ask.",
                "trait_deltas": {"initiative": -0.10},
            },
        ],
    },
    {
        "id": "pacing",
        "title": "Pacing",
        "prompt": "How do you work?",
        "options": [
            {
                "key": "focused",
                "label": "Long focused sessions. Don't interrupt unless something's wrong.",
                "trait_deltas": {"observance": 0.05},
                "flags": ["save_observations"],
            },
            {
                "key": "jumpy",
                "label": "I jump between things. Help me keep track.",
                "trait_deltas": {"observance": 0.10, "initiative": 0.05},
            },
            {
                "key": "unsure",
                "label": "Not sure yet. Let's figure it out as we go.",
                "trait_deltas": {},
            },
        ],
    },
]

# ─── Seed Questions ─────────────────────────────────────────────────────────

SEED_QUESTIONS = [
    {
        "id": "user_name",
        "prompt": "What's your name?",
        "placeholder": "e.g., Philip",
        "required": False,
        "knowledge_template": "{name} is the user's name.",
    },
    {
        "id": "user_role",
        "prompt": "What's your role or what do you do?",
        "placeholder": "e.g., CEO of a SaaS startup, freelance designer",
        "required": False,
        "knowledge_template": "The user works as {role}.",
    },
    {
        "id": "current_focus",
        "prompt": "What are you working on right now?",
        "placeholder": "e.g., launching a new product, writing a novel",
        "required": False,
        "knowledge_template": "The user is currently working on: {focus}.",
        "high_significance": True,
    },
    {
        "id": "tools",
        "prompt": "What tools or platforms do you use regularly?",
        "placeholder": "e.g., VS Code, Figma, Slack, Excel",
        "required": False,
        "knowledge_template": "The user regularly uses: {tools}.",
    },
    {
        "id": "freeform",
        "prompt": "Anything else I should know about you or how you work?",
        "placeholder": "",
        "required": False,
        "knowledge_template": "{freeform}",
    },
]


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class ResearchResult:
    """Result of live research or heuristic inference from user intent."""
    inferred_traits: Dict[str, float]
    primary_domain: str
    secondary_domains: List[str]
    excluded_domains: List[str]
    conviction_seeds: List[str]
    research_summary: str
    research_source: str          # "web" or "heuristic"
    confidence: float             # 0-1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class OnboardingStep:
    """A single step in the onboarding journey.

    Serialized to JSON for agent consumption. The agent renders
    the step, collects user input, and passes it back to process_input().
    """
    step_id: str
    step_type: str                   # info, question, multiple_choice, preview, confirm
    title: str
    description: str
    content: Any                     # Step-specific payload (dict or str)
    expected_input_type: str = ""    # text, choice_key, multi_select, json, none
    next_steps: Any = None           # str, list, or dict (conditional routing)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "expected_input_type": self.expected_input_type,
            "next_steps": self.next_steps,
            "metadata": self.metadata,
        }


@dataclass
class OnboardingContext:
    """Stateful context accumulated across the onboarding flow.

    Persisted between steps so the agent can resume after each user input.
    """
    # Flow state
    current_step: str = "welcome"
    completed_steps: List[str] = field(default_factory=list)
    deferred: bool = False  # True when onboarding was deferred for a task-first user

    # Smart Capture state
    scan_result: Optional[Dict[str, Any]] = None
    smart_capture_consent: Optional[str] = None  # "yes", "selective", "skip", "all"
    smart_capture_sources: List[str] = field(default_factory=list)  # Selected source IDs
    ingestion_plan: Optional[Dict[str, Any]] = None
    smart_capture_completed: bool = False

    # User inputs
    user_intent: str = ""
    persona_name: str = ""
    persona_key: str = ""

    # Research results
    research_result: Optional[Dict[str, Any]] = None
    baseline_traits: Dict[str, float] = field(default_factory=dict)

    # Working style
    working_style_answers: Dict[str, str] = field(default_factory=dict)
    working_style_deltas: Dict[str, float] = field(default_factory=dict)
    working_style_flags: List[str] = field(default_factory=list)

    # Interview
    interview_answers: Dict[str, str] = field(default_factory=dict)
    trait_deltas: Dict[str, float] = field(default_factory=dict)

    # Domain config
    domain_config: Dict[str, Any] = field(default_factory=dict)

    # North star
    north_star: str = ""

    # Seed questions
    seed_answers: Dict[str, str] = field(default_factory=dict)

    # Generated output
    generated_persona: Dict[str, Any] = field(default_factory=dict)
    final_traits: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OnboardingContext":
        return cls(**{
            k: v for k, v in d.items()
            if k in cls.__dataclass_fields__
        })

    def mark_step_completed(self, step_id: str):
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)


# ─── Flow Engine ─────────────────────────────────────────────────────────────

class FlowEngine:
    """Orchestrate the onboarding pipeline.

    Stateless: given a context, produces the next step.
    The agent drives the loop: get_next_step -> render -> collect input -> process_input -> repeat.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize the flow engine.

        Args:
            templates_dir: Legacy parameter, kept for backward compatibility.
                          No longer used for template loading.
        """
        self.interview = InterviewEngine(include_optional=False)

    # ─── First-Message Classification (Critical 1) ───────────────────

    @staticmethod
    def classify_first_message(message: str) -> str:
        """Classify a user's first message to determine onboarding strategy.

        Returns:
            "task"        - User has a real task. Help first, onboard later.
            "greeting"    - Casual greeting. Proceed with onboarding.
            "setup_ready" - Substantive but not task-urgent. Proceed with onboarding.
        """
        if not message or not message.strip():
            return "setup_ready"

        msg = message.strip()
        msg_lower = msg.lower()

        # Check greetings first (short messages that are just hellos)
        if len(msg.split()) <= 5:
            for pattern in GREETING_PATTERNS:
                if re.search(pattern, msg_lower):
                    return "greeting"

        # Check task signals
        task_hits = 0
        for pattern in TASK_SIGNALS:
            if re.search(pattern, msg_lower if '?' not in pattern else msg):
                task_hits += 1

        # Strong task signal: 2+ indicators, or the message is long (>30 words)
        # and contains at least one task indicator
        word_count = len(msg.split())
        if task_hits >= 2:
            return "task"
        if task_hits >= 1 and word_count > 30:
            return "task"

        # Long messages without task signals are still probably substantive
        if word_count > 50:
            return "task"

        return "setup_ready"

    @staticmethod
    def _is_vague_intent(intent: str) -> bool:
        """Check if a user's stated intent is too vague for useful research.

        Returns True if the intent matches vague patterns or has fewer
        than 3 meaningful words after stripping filler.
        """
        if not intent:
            return True

        intent_lower = intent.strip().lower()

        # Check against known vague phrases
        for pattern in VAGUE_INTENT_PHRASES:
            if re.search(pattern, intent_lower):
                return True

        # Strip common filler words and check remaining word count
        filler = {
            "i", "me", "my", "just", "like", "with", "some", "a", "an",
            "the", "to", "for", "and", "or", "do", "can", "want", "need",
            "stuff", "things", "help", "use", "it", "be",
        }
        words = [w for w in re.findall(r'\w+', intent_lower) if w not in filler]
        if len(words) < 2:
            return True

        return False

    # ─── Step Builders ───────────────────────────────────────────────

    def _build_welcome(self, ctx: OnboardingContext) -> OnboardingStep:
        return OnboardingStep(
            step_id="welcome",
            step_type="info",
            title="Welcome to Solitaire",
            description="Let's set up your AI partner.",
            content=(
                "I'm an AI partner that learns how you work and gets better "
                "over time. Everything you tell me is stored locally on your "
                "machine.\n\n"
                "We'll build a profile based on what you tell me. "
                "This takes 2-3 minutes, and you can skip any step. "
                'Or say "just get me started" to jump straight in.'
            ),
            expected_input_type="none",
            next_steps="intent_capture",
            metadata={
                "auto_advance": True,
                "quickstart_available": True,
            },
        )

    def _build_smart_capture(self, ctx: OnboardingContext) -> OnboardingStep:
        """Build the smart capture step when sources are detected."""
        scan = ctx.scan_result or {}
        sources = scan.get("sources", [])

        if not sources:
            # No sources found, present manual fallback
            return self._build_smart_capture_manual(ctx)

        # Build source list for display
        source_summaries = []
        for s in sources:
            summary = {
                "name": s.get("display_name", s.get("source_id", "Unknown")),
                "entry_count": s.get("entry_count_estimate", 0),
                "size": s.get("size_description", "unknown"),
            }
            if s.get("age_days"):
                days = s["age_days"]
                if days < 30:
                    summary["age"] = f"{days} days"
                elif days < 365:
                    summary["age"] = f"{days // 30} months"
                else:
                    summary["age"] = f"over {days // 365} year{'s' if days // 365 != 1 else ''}"
            source_summaries.append(summary)

        total_entries = scan.get("total_entry_estimate", 0)
        age_desc = scan.get("combined_age_description", "some time")
        size_desc = scan.get("total_size_description", "")
        plan = ctx.ingestion_plan or {}
        strategy = plan.get("strategy", "immediate")

        # Build message based on strategy
        if strategy == "immediate":
            message = (
                f"I can see you've been working with Claude for a while. "
                f"There's about {age_desc} of context here "
                f"({total_entries} entries, {size_desc}). "
                f"Want me to get up to speed before we start?"
            )
            options = [
                {"key": "yes", "label": "Yes, absorb my existing context"},
                {"key": "selective", "label": "Let me choose which sources to include"},
                {"key": "skip", "label": "Start fresh, I'll build context from scratch"},
            ]
        else:
            message = (
                f"I can see you've been working with Claude for {age_desc}. "
                f"There's a lot here ({total_entries} entries, {size_desc}). "
                f"I can absorb the highlights now and work through the rest "
                f"in the background. Sound good?"
            )
            options = [
                {"key": "yes", "label": "Yes, start with the highlights"},
                {"key": "selective", "label": "Let me choose which sources to include"},
                {"key": "all", "label": "Absorb everything now (this may take a few minutes)"},
                {"key": "skip", "label": "Start fresh"},
            ]

        return OnboardingStep(
            step_id="smart_capture",
            step_type="confirm",
            title="Existing Context Detected",
            description="Found existing memory data on this machine.",
            content={
                "message": message,
                "sources": source_summaries,
                "options": options,
                "default": "yes",
                "ingestion_plan": plan,
            },
            expected_input_type="choice_key",
            next_steps="intent_capture",
            metadata={
                "strategy": strategy,
                "source_count": len(sources),
                "total_entries": total_entries,
            },
        )

    def _build_smart_capture_manual(self, ctx: OnboardingContext) -> OnboardingStep:
        """Build the manual fallback when no sources are auto-detected."""
        return OnboardingStep(
            step_id="smart_capture_manual",
            step_type="question",
            title="Existing Context",
            description="Check for existing memory data.",
            content={
                "message": (
                    "Do you have an existing memory system you'd like me "
                    "to connect to? If so, point me to the folder or file "
                    "and I'll get up to speed."
                ),
                "options": [
                    {"key": "path", "label": "Yes, here's where it lives"},
                    {"key": "skip", "label": "No, let's start fresh"},
                ],
                "default": "skip",
                "accepts_path": True,
            },
            expected_input_type="text",
            next_steps="intent_capture",
        )

    def _build_smart_capture_selective(self, ctx: OnboardingContext) -> OnboardingStep:
        """Build the selective source picker."""
        scan = ctx.scan_result or {}
        sources = scan.get("sources", [])

        options = []
        for s in sources:
            name = s.get("display_name", s.get("source_id", "Unknown"))
            count = s.get("entry_count_estimate", 0)
            options.append({
                "key": s.get("source_id", "unknown"),
                "label": f"{name} ({count} entries)",
                "default": True,
            })

        return OnboardingStep(
            step_id="smart_capture_selective",
            step_type="multiple_choice",
            title="Choose Sources",
            description="Select which sources to absorb.",
            content={
                "message": "Which sources should I absorb?",
                "options": options,
            },
            expected_input_type="multi_select",
            next_steps="intent_capture",
        )

    def _build_intent_capture(self, ctx: OnboardingContext) -> OnboardingStep:
        return OnboardingStep(
            step_id="intent_capture",
            step_type="question",
            title="What's this for?",
            description="This shapes everything: traits, domain, and defaults.",
            content={
                "prompt": "What do you primarily use AI for?",
                "placeholder": (
                    "e.g., 'manage my startup's operations', "
                    "'creative writing partner', 'competitive gaming coach'"
                ),
                "hint": (
                    "Be specific. 'Help me run my SaaS business' gives "
                    "better results than 'business stuff'."
                ),
            },
            expected_input_type="text",
            next_steps="live_research",
        )

    def _build_intent_followup(self, ctx: OnboardingContext) -> OnboardingStep:
        """When intent is vague, offer a category picker instead of useless research."""
        return OnboardingStep(
            step_id="intent_followup",
            step_type="multiple_choice",
            title="Can you narrow it down?",
            description=(
                "That's a bit broad for me to build a useful profile. "
                "Pick the closest category and I'll tune from there."
            ),
            content={
                "prompt": "What's the closest match?",
                "options": [
                    {"key": cat["key"], "label": cat["label"]}
                    for cat in INTENT_CATEGORY_PICKER
                ],
            },
            expected_input_type="choice_key",
            next_steps="live_research",
            metadata={"skippable": True},
        )

    def _build_deferred_prompt(self, ctx: OnboardingContext) -> OnboardingStep:
        """Prompt for deferred onboarding, shown after a few sessions.

        The agent triggers this when it detects a returning user who
        was deferred on first run and has now completed 2+ sessions.
        """
        return OnboardingStep(
            step_id="deferred_prompt",
            step_type="multiple_choice",
            title="Set Up Your Profile",
            description=(
                "I've been learning from our conversations. "
                "Want to set up a profile so I can tune my style to you?"
            ),
            content={
                "prompt": "This takes 2-3 minutes and makes me noticeably better.",
                "options": [
                    {"key": "yes", "label": "Sure, let's do it"},
                    {"key": "later", "label": "Not now"},
                ],
            },
            expected_input_type="choice_key",
            next_steps={
                "yes": "intent_capture",
                "later": "cancelled",
            },
            metadata={"deferred_onboarding": True},
        )

    def _build_live_research(self, ctx: OnboardingContext) -> OnboardingStep:
        """Build the live research step.

        The actual research is triggered by the caller (CLI or agent) before
        this step renders. This step shows the results. If research hasn't
        run yet, the step signals the caller to run it.
        """
        if ctx.research_result:
            return OnboardingStep(
                step_id="live_research",
                step_type="info",
                title="Research Complete",
                description="Here's what I found.",
                content={
                    "status": "complete",
                    "research": ctx.research_result,
                },
                expected_input_type="none",
                next_steps="trait_proposal",
                metadata={"auto_advance": True},
            )
        else:
            return OnboardingStep(
                step_id="live_research",
                step_type="info",
                title="Researching...",
                description="Let me look into that. One moment...",
                content={
                    "status": "pending",
                    "action": "run_research",
                    "intent": ctx.user_intent,
                },
                expected_input_type="none",
                next_steps="trait_proposal",
                metadata={
                    "auto_advance": False,
                    "requires_research": True,
                },
            )

    def _build_trait_proposal(self, ctx: OnboardingContext) -> OnboardingStep:
        """Show the researched trait profile for acceptance or tweaking.

        If research confidence is below 0.3 (all-moderate defaults from vague
        intent), skip the trait card entirely. An all-moderate card communicates
        nothing and wastes the user's time. Go straight to working_style and
        let the system calibrate from real use.
        """
        research = ctx.research_result or {}
        traits = research.get("inferred_traits", dict(DEFAULT_TRAITS))
        source = research.get("research_source", "default")
        summary = research.get("research_summary", "")
        confidence = research.get("confidence", 0.0)
        conviction_seeds = research.get("conviction_seeds", [])

        # Low-confidence skip: all-moderate card is meaningless
        if confidence < 0.3:
            ctx.baseline_traits = traits
            return OnboardingStep(
                step_id="trait_proposal",
                step_type="info",
                title="Profile Calibration",
                description=(
                    "I don't have enough signal to build a useful profile yet. "
                    "I'll start with balanced defaults and calibrate as we work together."
                ),
                content={
                    "status": "low_confidence_skip",
                    "confidence": confidence,
                    "message": (
                        "Your working style preferences (next step) will give me "
                        "the first useful signal. After a few sessions, I'll tune "
                        "further from how you actually work."
                    ),
                },
                expected_input_type="none",
                next_steps="working_style",
                metadata={"auto_advance": True, "low_confidence": True},
            )

        trait_display = {}
        for t in VALID_TRAIT_NAMES:
            val = traits.get(t, 0.5)
            trait_display[t] = {
                "value": round(val * 100),
                "label": self._trait_label(t, val),
                "description": self._trait_description(t, val),
            }

        return OnboardingStep(
            step_id="trait_proposal",
            step_type="preview",
            title="Proposed Profile",
            description=(
                f"Based on your intent, here's what I'd build. "
                f"{'Built from web research.' if source == 'web' else 'Built from domain heuristics.'}"
            ),
            content={
                "traits": trait_display,
                "domain": {
                    "primary": research.get("primary_domain", "general"),
                    "secondary": research.get("secondary_domains", []),
                    "excluded": research.get("excluded_domains", []),
                },
                "conviction_seeds": conviction_seeds,
                "research_summary": summary,
                "confidence": confidence,
            },
            expected_input_type="choice_key",
            next_steps={
                "accept": "working_style",
                "tweak": "trait_tweak",
                "skip": "working_style",
            },
            metadata={
                "cancel_available": True,
                "options": [
                    {"key": "accept", "label": "Looks good"},
                    {"key": "tweak", "label": "Let me adjust"},
                    {"key": "skip", "label": "Skip — use defaults"},
                ],
            },
        )

    def _build_trait_tweak(self, ctx: OnboardingContext) -> OnboardingStep:
        """Allow manual adjustment of proposed traits."""
        traits = ctx.baseline_traits or dict(DEFAULT_TRAITS)

        return OnboardingStep(
            step_id="trait_tweak",
            step_type="question",
            title="Adjust Traits",
            description=(
                "Adjust any trait. Use natural language ('make it warmer') "
                "or specific values ('warmth: 70')."
            ),
            content={
                "current_traits": {
                    t: round(v * 100) for t, v in traits.items()
                },
                "valid_traits": list(VALID_TRAIT_NAMES),
                "hint": (
                    "Examples: 'higher conviction', 'warmth: 80', "
                    "'more assertive, less warm'"
                ),
            },
            expected_input_type="text",
            next_steps="working_style",
        )

    def _build_working_style(self, ctx: OnboardingContext) -> OnboardingStep:
        """Build the working style preferences step.

        Shows all 4 categories at once for the agent to present
        sequentially or as a single form.
        """
        categories = []
        for cat in WORKING_STYLE_CATEGORIES:
            if cat["id"] in ctx.working_style_answers:
                continue
            categories.append({
                "id": cat["id"],
                "title": cat["title"],
                "prompt": cat["prompt"],
                "options": [
                    {"key": opt["key"], "label": opt["label"]}
                    for opt in cat["options"]
                ],
            })

        return OnboardingStep(
            step_id="working_style",
            step_type="multiple_choice",
            title="Working Style",
            description=(
                "A few quick preferences. Skip any you'd rather figure out later."
            ),
            content={
                "categories": categories,
                "skip_label": "Skip — figure it out as we go",
            },
            expected_input_type="json",
            next_steps="interview_offer",
            metadata={
                "cancel_available": False,
                "skippable": True,
            },
        )

    def _build_interview_offer(self, ctx: OnboardingContext) -> OnboardingStep:
        """Offer the optional personality interview."""
        return OnboardingStep(
            step_id="interview_offer",
            step_type="multiple_choice",
            title="Personality Interview",
            description=(
                "Want to fine-tune with 5 quick scenario questions? "
                "Takes about 2 minutes."
            ),
            content={
                "prompt": "These scenarios help calibrate specific behaviors.",
                "options": [
                    {"key": "yes", "label": "Yes, let's do it"},
                    {"key": "skip", "label": "Skip — I'm good"},
                ],
            },
            expected_input_type="choice_key",
            next_steps={
                "yes": self._first_interview_step_id(),
                "skip": "naming",
            },
        )

    def _build_interview_step(
        self, question: InterviewQuestion, ctx: OnboardingContext
    ) -> OnboardingStep:
        """Build an interview question step."""
        question_ids = self.interview.question_ids
        idx = question_ids.index(question.id) if question.id in question_ids else 0
        total = len(question_ids)

        next_q_id = self._next_interview_question_id(question.id)
        next_step = f"interview_{next_q_id}" if next_q_id else "naming"

        return OnboardingStep(
            step_id=f"interview_{question.id}",
            step_type="question",
            title=f"Interview: {question.title}",
            description=question.explanation,
            content={
                "prompt": question.prompt,
                "options": [opt.to_dict() for opt in question.options],
            },
            expected_input_type="choice_key",
            next_steps=next_step,
            metadata={
                "interview_question": True,
                "question_id": question.id,
                "question_number": idx + 1,
                "total_questions": total,
                "affects_traits": question.affects,
                "can_skip": not question.required,
                "skip_options": [
                    {"key": "skip", "label": "Skip this question"},
                    {"key": "skip_rest", "label": "Skip remaining questions"},
                ],
            },
        )

    def _build_naming(self, ctx: OnboardingContext) -> OnboardingStep:
        """Ask user to name their Solitaire instance."""
        suggested = self._suggest_name(ctx.user_intent)

        return OnboardingStep(
            step_id="naming",
            step_type="question",
            title="Name Your Instance",
            description="Give your Solitaire instance a name.",
            content={
                "prompt": "What should we call it?",
                "placeholder": suggested,
                "hint": (
                    "This appears as a prefix on every response "
                    f"(e.g., [{suggested}]). Keep it short."
                ),
            },
            expected_input_type="text",
            next_steps="north_star",
        )

    def _build_north_star(self, ctx: OnboardingContext) -> OnboardingStep:
        """Ask for an optional guiding principle."""
        return OnboardingStep(
            step_id="north_star",
            step_type="multiple_choice",
            title="North Star",
            description=(
                "A North Star is a guiding principle that shapes priorities "
                "and decision-making. Optional, but powerful."
            ),
            content={
                "prompt": "Want to set a guiding principle for this instance?",
                "options": [
                    {
                        "key": "set",
                        "label": "Set one now",
                        "description": "Write a guiding principle in your own words.",
                    },
                    {
                        "key": "later",
                        "label": "Let it emerge",
                        "description": (
                            "Skip for now. Set later with "
                            "'solitaire identity north-star \"...\"'"
                        ),
                    },
                ],
                "hint": (
                    "Examples: 'Always prioritize accuracy over speed.' "
                    "'Challenge my assumptions before validating them.'"
                ),
            },
            expected_input_type="choice_key",
            next_steps={
                "set": "north_star_input",
                "later": "seed_questions",
            },
        )

    def _build_north_star_input(self, ctx: OnboardingContext) -> OnboardingStep:
        """Free text input for north star."""
        return OnboardingStep(
            step_id="north_star_input",
            step_type="question",
            title="Your North Star",
            description="Write your guiding principle.",
            content={
                "prompt": "What should guide this instance's priorities?",
                "placeholder": "e.g., 'Always prioritize accuracy over speed.'",
            },
            expected_input_type="text",
            next_steps="seed_questions",
        )

    def _build_seed_questions(self, ctx: OnboardingContext) -> OnboardingStep:
        """Build the seed questions step to populate the rolodex."""
        questions = []
        for sq in SEED_QUESTIONS:
            if sq["id"] in ctx.seed_answers:
                continue
            questions.append({
                "id": sq["id"],
                "prompt": sq["prompt"],
                "placeholder": sq["placeholder"],
                "required": sq["required"],
            })

        return OnboardingStep(
            step_id="seed_questions",
            step_type="question",
            title="About You",
            description=(
                "A few quick questions to get started. "
                "Skip any you'd rather tell me later."
            ),
            content={
                "questions": questions,
                "skip_label": "Skip — you'll learn as we go",
            },
            expected_input_type="json",
            next_steps="preview",
            metadata={"skippable": True},
        )

    def _build_preview(self, ctx: OnboardingContext) -> OnboardingStep:
        """Show the final persona for review before applying."""
        if not ctx.generated_persona:
            ctx = self._generate_persona(ctx)

        persona = ctx.generated_persona
        traits = persona.get("traits", {})

        baseline = ctx.baseline_traits or dict(DEFAULT_TRAITS)
        trait_breakdown = {}
        for t in VALID_TRAIT_NAMES:
            final_val = traits.get(t, 0.5)
            base_val = baseline.get(t, 0.5)
            ws_delta = ctx.working_style_deltas.get(t, 0.0)
            iv_delta = ctx.trait_deltas.get(t, 0.0)
            trait_breakdown[t] = {
                "value": f"{round(final_val * 100)}%",
                "baseline": f"{round(base_val * 100)}%",
                "working_style_delta": f"{round(ws_delta * 100):+d}%",
                "interview_delta": f"{round(iv_delta * 100):+d}%",
                "label": self._trait_label(t, final_val),
            }

        seed_count = len([v for v in ctx.seed_answers.values() if v.strip()])

        return OnboardingStep(
            step_id="preview",
            step_type="preview",
            title="Review Your Profile",
            description="Here's what we've built. Review and confirm, or go back.",
            content={
                "identity": persona.get("identity", {}),
                "traits": trait_breakdown,
                "domain": persona.get("domain", {}),
                "north_star": ctx.north_star or "(not set)",
                "seed_knowledge_count": seed_count,
                "conviction_seeds": persona.get("triggers", {}).get(
                    "conviction_overrides", []
                ),
                "source": (
                    f"Research ({ctx.research_result.get('research_source', 'heuristic')})"
                    if ctx.research_result
                    else "Defaults"
                ),
            },
            expected_input_type="choice_key",
            next_steps={
                "confirm": "confirm",
                "redo_interview": "interview_offer",
                "redo_style": "working_style",
                "start_over": "intent_capture",
            },
            metadata={
                "allow_back": True,
                "options": [
                    {"key": "confirm", "label": "Confirm"},
                    {"key": "redo_style", "label": "Redo working style"},
                    {"key": "redo_interview", "label": "Redo interview"},
                    {"key": "start_over", "label": "Start over"},
                ],
            },
        )

    def _build_confirm(self, ctx: OnboardingContext) -> OnboardingStep:
        """Final confirmation before applying."""
        name = ctx.generated_persona.get("identity", {}).get("name", "Solitaire")
        seed_count = len([v for v in ctx.seed_answers.values() if v.strip()])

        actions = [
            f'Create persona "{name}" (key: "{ctx.persona_key}")',
            "Register in persona directory",
        ]
        if seed_count:
            actions.append(f"Ingest {seed_count} seed knowledge entries")
        if ctx.north_star:
            actions.append("Set North Star")
        actions.append("Boot into your new instance")

        return OnboardingStep(
            step_id="confirm",
            step_type="confirm",
            title="Create Your Instance",
            description="Ready to go. Confirming will set up your profile.",
            content={
                "summary": f'Your instance "{name}" is configured. Confirming will:',
                "actions": actions,
            },
            expected_input_type="choice_key",
            next_steps={
                "confirm": "locale_capture",
                "start_over": "intent_capture",
            },
            metadata={"confirm_step": True},
        )

    def _build_locale_capture(self, ctx: OnboardingContext) -> OnboardingStep:
        """Ask user for location."""
        return OnboardingStep(
            step_id="locale_capture",
            step_type="multiple_choice",
            title="Where Are You Based?",
            description=(
                "This helps with location-specific guidance "
                "(tax codes, regulations, currency). You can skip "
                "and set it later with 'solitaire locale set <country>'."
            ),
            content={
                "options": [
                    {"key": "US", "label": "United States", "description": "GAAP, IRS, USD"},
                    {"key": "CA", "label": "Canada", "description": "IFRS, CRA, CAD"},
                    {"key": "GB", "label": "United Kingdom", "description": "IFRS, HMRC, GBP"},
                    {"key": "AU", "label": "Australia", "description": "IFRS, ATO, AUD"},
                    {"key": "DE", "label": "Germany", "description": "IFRS, BZSt, EUR"},
                    {"key": "BR", "label": "Brazil", "description": "IFRS, Receita Federal, BRL"},
                    {"key": "GT", "label": "Guatemala", "description": "IFRS, SAT, GTQ"},
                    {"key": "skip", "label": "Skip for now", "description": "Set locale later via CLI"},
                ],
            },
            expected_input_type="choice_key",
            next_steps="apply",
            metadata={"cancel_available": False},
        )

    def _build_cancelled(self, ctx: OnboardingContext) -> OnboardingStep:
        """Terminal step when user cancels."""
        return OnboardingStep(
            step_id="cancelled",
            step_type="info",
            title="Setup Cancelled",
            description="No changes were made.",
            content={
                "status": "cancelled",
                "message": (
                    "Persona creation cancelled. No changes were made. "
                    "You can start again anytime."
                ),
            },
            metadata={"terminal": True},
        )

    # ─── Flow Navigation ─────────────────────────────────────────────

    def get_next_step(self, ctx: OnboardingContext) -> OnboardingStep:
        """Given current context, return the next step to present."""
        step_id = ctx.current_step

        if step_id == "welcome":
            return self._build_welcome(ctx)
        elif step_id == "smart_capture":
            return self._build_smart_capture(ctx)
        elif step_id == "smart_capture_manual":
            return self._build_smart_capture_manual(ctx)
        elif step_id == "smart_capture_selective":
            return self._build_smart_capture_selective(ctx)
        elif step_id == "intent_capture":
            return self._build_intent_capture(ctx)
        elif step_id == "intent_followup":
            return self._build_intent_followup(ctx)
        elif step_id == "deferred_prompt":
            return self._build_deferred_prompt(ctx)
        elif step_id == "live_research":
            return self._build_live_research(ctx)
        elif step_id == "trait_proposal":
            return self._build_trait_proposal(ctx)
        elif step_id == "trait_tweak":
            return self._build_trait_tweak(ctx)
        elif step_id == "working_style":
            return self._build_working_style(ctx)
        elif step_id == "interview_offer":
            return self._build_interview_offer(ctx)
        elif step_id.startswith("interview_"):
            q_id = step_id.replace("interview_", "")
            question = ALL_QUESTIONS.get(q_id)
            if question:
                return self._build_interview_step(question, ctx)
            ctx.current_step = "naming"
            return self._build_naming(ctx)
        elif step_id == "naming":
            return self._build_naming(ctx)
        elif step_id == "north_star":
            return self._build_north_star(ctx)
        elif step_id == "north_star_input":
            return self._build_north_star_input(ctx)
        elif step_id == "seed_questions":
            return self._build_seed_questions(ctx)
        elif step_id == "preview":
            return self._build_preview(ctx)
        elif step_id == "confirm":
            return self._build_confirm(ctx)
        elif step_id == "locale_capture":
            return self._build_locale_capture(ctx)
        elif step_id == "cancelled":
            return self._build_cancelled(ctx)
        elif step_id == "apply":
            return OnboardingStep(
                step_id="apply",
                step_type="info",
                title="Applying...",
                description="Your instance is being created.",
                content={"status": "ready_to_apply"},
                metadata={"terminal": True},
            )
        else:
            return self._build_welcome(ctx)

    def process_input(
        self,
        ctx: OnboardingContext,
        step_id: str,
        input_data: Any,
    ) -> OnboardingContext:
        """Process user input for a step and advance context."""
        ctx.mark_step_completed(step_id)

        input_str = str(input_data).strip()

        # Cancel from any step
        if input_str == "cancel":
            ctx.current_step = "cancelled"
            return ctx

        # ─── Critical 3: Global quickstart from any step ─────────────
        # "quickstart" jumps straight to apply with sensible defaults.
        # Auto-names from intent if available, otherwise "Partner".
        if input_str == "quickstart" and step_id not in ("apply", "cancelled", "confirm"):
            if not ctx.persona_name:
                ctx.persona_name = self._suggest_name(ctx.user_intent) if ctx.user_intent else "Partner"
                ctx.persona_key = self._slugify(ctx.persona_name)
            if not ctx.baseline_traits:
                if ctx.user_intent:
                    result = self.run_heuristic_research(ctx.user_intent)
                    ctx.research_result = result.to_dict()
                    ctx.baseline_traits = result.inferred_traits
                    ctx.domain_config = {
                        "primary": result.primary_domain,
                        "secondary": result.secondary_domains,
                        "excluded": result.excluded_domains,
                    }
                else:
                    ctx.baseline_traits = dict(DEFAULT_TRAITS)
            ctx = self._generate_persona(ctx)
            ctx.current_step = "apply"
            return ctx

        if step_id == "welcome":
            # After welcome, run environment scan and route to smart capture
            # The scan_result should be set on ctx by the caller (CLI or agent)
            # before calling process_input, or we route to smart_capture which
            # handles both detected and manual-fallback paths.
            ctx.current_step = "smart_capture"

        elif step_id == "smart_capture":
            choice = input_str
            ctx.smart_capture_consent = choice
            if choice == "yes" or choice == "all":
                # Consent given. The caller (CLI/agent) should run ingestion
                # using the scan_result and ingestion_plan on ctx.
                # Mark all detected sources as selected.
                scan = ctx.scan_result or {}
                ctx.smart_capture_sources = [
                    s.get("source_id", "") for s in scan.get("sources", [])
                ]
                ctx.smart_capture_completed = True
                ctx.current_step = "intent_capture"
            elif choice == "selective":
                ctx.current_step = "smart_capture_selective"
            elif choice == "skip":
                ctx.smart_capture_completed = True
                ctx.current_step = "intent_capture"
            else:
                ctx.current_step = "intent_capture"

        elif step_id == "smart_capture_manual":
            choice = input_str
            if choice == "skip" or choice == "no":
                ctx.smart_capture_completed = True
                ctx.current_step = "intent_capture"
            else:
                # User provided a path. Store it for the caller to scan + ingest.
                ctx.smart_capture_sources = [choice]
                ctx.smart_capture_consent = "path"
                ctx.smart_capture_completed = True
                ctx.current_step = "intent_capture"

        elif step_id == "smart_capture_selective":
            # input_data should be a list of selected source IDs or a
            # comma-separated string
            if isinstance(input_data, list):
                ctx.smart_capture_sources = input_data
            elif isinstance(input_data, str):
                ctx.smart_capture_sources = [
                    s.strip() for s in input_data.split(",") if s.strip()
                ]
            ctx.smart_capture_consent = "selective"
            ctx.smart_capture_completed = True
            ctx.current_step = "intent_capture"

        elif step_id == "intent_capture":
            ctx.user_intent = str(input_data).strip()
            # ─── Critical 2: Vague intent detection ──────────────────
            if self._is_vague_intent(ctx.user_intent):
                ctx.current_step = "intent_followup"
            else:
                ctx.current_step = "live_research"

        elif step_id == "intent_followup":
            # Category picker response
            choice = input_str
            if choice == "skip":
                ctx.current_step = "live_research"
            elif choice == "other":
                # Send back to intent_capture for a second try
                ctx.completed_steps = [
                    s for s in ctx.completed_steps if s != "intent_capture"
                ]
                ctx.current_step = "intent_capture"
            else:
                # Map category key to intent seed
                for cat in INTENT_CATEGORY_PICKER:
                    if cat["key"] == choice:
                        ctx.user_intent = cat["intent_seed"]
                        break
                ctx.current_step = "live_research"

        elif step_id == "deferred_prompt":
            choice = input_str
            if choice == "yes":
                ctx.current_step = "intent_capture"
            else:
                ctx.current_step = "cancelled"

        elif step_id == "live_research":
            # Research result should be set on ctx before calling process_input.
            # If it's passed as input_data (dict), store it.
            if isinstance(input_data, dict) and "inferred_traits" in input_data:
                ctx.research_result = input_data
                ctx.baseline_traits = input_data.get(
                    "inferred_traits", dict(DEFAULT_TRAITS)
                )
                ctx.domain_config = {
                    "primary": input_data.get("primary_domain", "general"),
                    "secondary": input_data.get("secondary_domains", []),
                    "excluded": input_data.get("excluded_domains", []),
                }
            elif not ctx.research_result:
                # No research provided, run heuristic
                result = self.run_heuristic_research(ctx.user_intent)
                ctx.research_result = result.to_dict()
                ctx.baseline_traits = result.inferred_traits
                ctx.domain_config = {
                    "primary": result.primary_domain,
                    "secondary": result.secondary_domains,
                    "excluded": result.excluded_domains,
                }
            ctx.current_step = "trait_proposal"

        elif step_id == "trait_proposal":
            choice = str(input_data).strip()
            if choice == "accept" or choice == "auto_advance":
                # auto_advance covers the low-confidence skip path
                ctx.current_step = "working_style"
            elif choice == "tweak":
                ctx.current_step = "trait_tweak"
            elif choice == "skip":
                ctx.baseline_traits = dict(DEFAULT_TRAITS)
                ctx.current_step = "working_style"
            else:
                ctx.current_step = "working_style"

        elif step_id == "trait_tweak":
            tweaks = self._parse_trait_tweaks(str(input_data), ctx.baseline_traits)
            ctx.baseline_traits.update(tweaks)
            for t in ctx.baseline_traits:
                ctx.baseline_traits[t] = max(
                    TRAIT_FLOOR, min(TRAIT_CEILING, ctx.baseline_traits[t])
                )
            ctx.current_step = "working_style"

        elif step_id == "working_style":
            if isinstance(input_data, dict):
                for cat_id, choice_key in input_data.items():
                    ctx.working_style_answers[cat_id] = choice_key
                ctx.working_style_deltas, ctx.working_style_flags = (
                    self._compute_working_style_deltas(ctx.working_style_answers)
                )
            ctx.current_step = "interview_offer"

        elif step_id == "interview_offer":
            choice = str(input_data).strip()
            if choice == "yes":
                ctx.current_step = self._first_interview_step_id()
            else:
                ctx.current_step = "naming"

        elif step_id.startswith("interview_"):
            q_id = step_id.replace("interview_", "")
            answer_key = str(input_data).strip()

            if answer_key.lower() == "skip_rest":
                # Skip all remaining interview questions
                ctx.current_step = "naming"
            elif answer_key.lower() == "skip":
                # Skip this question only, advance to next
                next_q = self._next_interview_question_id(q_id)
                if next_q:
                    ctx.current_step = f"interview_{next_q}"
                else:
                    ctx.current_step = "naming"
            else:
                ctx.interview_answers[q_id] = answer_key.upper()
                ctx.trait_deltas = self.interview.compute_trait_deltas(
                    ctx.interview_answers
                )
                next_q = self._next_interview_question_id(q_id)
                if next_q:
                    ctx.current_step = f"interview_{next_q}"
                else:
                    ctx.current_step = "naming"

        elif step_id == "naming":
            name = str(input_data).strip()
            if name:
                ctx.persona_name = name
                ctx.persona_key = self._slugify(name)
            else:
                suggested = self._suggest_name(ctx.user_intent)
                ctx.persona_name = suggested
                ctx.persona_key = self._slugify(suggested)
            ctx.current_step = "north_star"

        elif step_id == "north_star":
            choice = str(input_data).strip()
            if choice == "set":
                ctx.current_step = "north_star_input"
            else:
                ctx.north_star = ""
                ctx.current_step = "seed_questions"

        elif step_id == "north_star_input":
            ctx.north_star = str(input_data).strip()
            ctx.current_step = "seed_questions"

        elif step_id == "seed_questions":
            if isinstance(input_data, dict):
                for q_id, answer in input_data.items():
                    if answer and str(answer).strip():
                        ctx.seed_answers[q_id] = str(answer).strip()
            ctx = self._generate_persona(ctx)
            ctx.current_step = "preview"

        elif step_id == "preview":
            choice = str(input_data).strip()
            if choice == "confirm":
                ctx.current_step = "confirm"
            elif choice == "redo_interview":
                ctx.interview_answers = {}
                ctx.trait_deltas = {}
                ctx.current_step = "interview_offer"
            elif choice == "redo_style":
                ctx.working_style_answers = {}
                ctx.working_style_deltas = {}
                ctx.working_style_flags = []
                ctx.current_step = "working_style"
            elif choice == "start_over":
                ctx.research_result = None
                ctx.baseline_traits = {}
                ctx.working_style_answers = {}
                ctx.working_style_deltas = {}
                ctx.working_style_flags = []
                ctx.interview_answers = {}
                ctx.trait_deltas = {}
                ctx.persona_name = ""
                ctx.persona_key = ""
                ctx.domain_config = {}
                ctx.north_star = ""
                ctx.seed_answers = {}
                ctx.generated_persona = {}
                ctx.final_traits = {}
                ctx.current_step = "intent_capture"
            else:
                ctx.current_step = "confirm"

        elif step_id == "confirm":
            choice = str(input_data).strip()
            if choice == "start_over":
                ctx.current_step = "intent_capture"
            else:
                ctx.current_step = "locale_capture"

        elif step_id == "locale_capture":
            choice = str(input_data).strip()
            if choice and choice != "skip":
                ctx.domain_config["locale_country"] = choice.upper()
            ctx.current_step = "apply"

        return ctx

    # ─── Research ─────────────────────────────────────────────────────

    def run_heuristic_research(self, user_intent: str) -> ResearchResult:
        """Run heuristic trait inference from user intent.

        Uses the INTENT_TRAIT_HEURISTICS table to infer traits
        from keyword matching. Multiple matches are averaged,
        weighted by keyword hit count.
        """
        if not user_intent:
            return ResearchResult(
                inferred_traits=dict(DEFAULT_TRAITS),
                primary_domain="general",
                secondary_domains=[],
                excluded_domains=[],
                conviction_seeds=[],
                research_summary="No intent provided. Using balanced defaults.",
                research_source="heuristic",
                confidence=0.0,
            )

        intent_lower = user_intent.lower()
        matches = []

        for pattern, traits in INTENT_TRAIT_HEURISTICS.items():
            keywords = pattern.split("|")
            hit_count = 0
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', intent_lower):
                    hit_count += 1
            if hit_count > 0:
                matches.append((traits, hit_count))

        if not matches:
            return ResearchResult(
                inferred_traits=dict(DEFAULT_TRAITS),
                primary_domain=self._infer_primary_domain(intent_lower),
                secondary_domains=[],
                excluded_domains=[],
                conviction_seeds=[],
                research_summary=(
                    f"No strong keyword matches for '{user_intent}'. "
                    "Using balanced defaults. Your working style preferences "
                    "and interview will fine-tune from here."
                ),
                research_source="heuristic",
                confidence=0.2,
            )

        # Weighted average of matching trait profiles
        total_weight = sum(w for _, w in matches)
        averaged = {}
        for t in VALID_TRAIT_NAMES:
            weighted_sum = sum(
                traits.get(t, 0.5) * weight for traits, weight in matches
            )
            averaged[t] = round(weighted_sum / total_weight, 2)

        primary = self._infer_primary_domain(intent_lower)
        secondary = self._infer_secondary_domains(intent_lower, primary)
        conviction_seeds = self._generate_conviction_seeds(intent_lower, matches)

        matched_domains = []
        for pattern, _ in INTENT_TRAIT_HEURISTICS.items():
            keywords = pattern.split("|")
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', intent_lower):
                    matched_domains.append(kw)
                    break

        confidence = min(0.85, 0.3 + (len(matches) * 0.15))

        return ResearchResult(
            inferred_traits=averaged,
            primary_domain=primary,
            secondary_domains=secondary,
            excluded_domains=[],
            conviction_seeds=conviction_seeds,
            research_summary=(
                f"Matched domain signals: {', '.join(matched_domains)}. "
                f"Profile tuned for {primary} work."
            ),
            research_source="heuristic",
            confidence=confidence,
        )

    # ─── Persona Generation ──────────────────────────────────────────

    def _generate_persona(self, ctx: OnboardingContext) -> OnboardingContext:
        """Generate the full persona dict from accumulated context.

        Trait application order:
        1. Research baseline (or defaults)
        2. Working style deltas (capped at WORKING_STYLE_MAX_DELTA per trait)
        3. Interview deltas (capped at MAX_AGGREGATE_DELTA per trait)
        """
        baseline = deepcopy(ctx.baseline_traits) if ctx.baseline_traits else dict(DEFAULT_TRAITS)

        # Layer 2: Working style deltas
        ws_deltas = ctx.working_style_deltas or {}
        after_ws = {}
        for t in VALID_TRAIT_NAMES:
            base_val = baseline.get(t, 0.5)
            ws_d = ws_deltas.get(t, 0.0)
            ws_d = max(-WORKING_STYLE_MAX_DELTA, min(WORKING_STYLE_MAX_DELTA, ws_d))
            after_ws[t] = base_val + ws_d

        # Layer 3: Interview deltas
        interview_deltas = ctx.trait_deltas or {}
        final_traits = {}
        for t in VALID_TRAIT_NAMES:
            ws_val = after_ws.get(t, 0.5)
            iv_d = interview_deltas.get(t, 0.0)
            final_val = ws_val + iv_d
            final_traits[t] = max(TRAIT_FLOOR, min(TRAIT_CEILING, round(final_val, 2)))

        ctx.final_traits = final_traits

        domain = ctx.domain_config if ctx.domain_config else {
            "primary": "general",
            "secondary": [],
            "excluded": [],
            "skill_orientation": {"active": [], "watching": [], "ignored": []},
        }

        name = ctx.persona_name or "Solitaire"
        role = self._infer_role(ctx.user_intent)
        description = self._generate_description(ctx, final_traits)

        conviction_seeds = []
        if ctx.research_result:
            conviction_seeds = ctx.research_result.get("conviction_seeds", [])

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ctx.generated_persona = {
            "schema_version": "1.0",
            "identity": {
                "name": name,
                "role": role,
                "description": description,
            },
            "traits": final_traits,
            "domain": domain,
            "triggers": {
                "conviction_overrides": conviction_seeds,
                "initiative_triggers": [],
            },
            "drift": DriftConfig().to_dict(),
            "sharing": SharingConfig(consent_given_at=now).to_dict(),
            "custom_signals": [],
            "meta": {
                "created_at": now,
                "created_by": "onboarding_v2",
                "template_source": None,
                "user_intent": ctx.user_intent,
                "research_source": (
                    ctx.research_result.get("research_source", "heuristic")
                    if ctx.research_result else "default"
                ),
                "research_confidence": (
                    ctx.research_result.get("confidence", 0.0)
                    if ctx.research_result else 0.0
                ),
                "working_style": ctx.working_style_answers,
                "revelation_stage": 1,
                "model_affinity": None,
                "minimum_solitaire_version": "1.0.0",
            },
        }

        if ctx.north_star:
            ctx.generated_persona["north_star"] = ctx.north_star

        return ctx

    def _generate_description(
        self, ctx: OnboardingContext, traits: Dict[str, float]
    ) -> str:
        """Generate a natural-language persona description."""
        parts = []

        role_desc = self._intent_role_descriptor(ctx.user_intent or "")
        if ctx.user_intent:
            parts.append(f"{role_desc}, custom-built from your stated intent.")
        else:
            parts.append("Custom-built through the onboarding flow.")

        highlights = []
        # All traits rendered at appropriate intensity. Moderate traits are
        # meaningful personality signals, not absence of personality.
        _onboard_bands = {
            "initiative": {
                "high": "proactive in building ahead",
                "moderate": "takes initiative on familiar ground",
                "low": "waits for direction before acting",
            },
            "conviction": {
                "high": "pushes back with evidence when warranted",
                "moderate": "holds positions but stays open",
                "low": "supportive, defers to your judgment",
            },
            "assertiveness": {
                "high": "surfaces concerns directly",
                "moderate": "clear but measured in delivery",
                "low": "gentle in delivery",
            },
            "warmth": {
                "high": "warm and encouraging in tone",
                "moderate": "professional warmth, reads the room",
                "low": "clinical and direct",
            },
            "observance": {
                "high": "high vigilance for missed details",
                "moderate": "attentive to patterns",
                "low": "focused on what's directly relevant",
            },
            "humor": {
                "high": "uses wit freely",
                "moderate": "dry humor when it fits",
                "low": "keeps things serious",
            },
            "empathy": {
                "high": "tracks emotional undercurrents",
                "moderate": "emotionally aware",
                "low": "task-focused",
            },
        }
        for trait_name, bands in _onboard_bands.items():
            val = traits.get(trait_name, 0.5)
            if val >= 0.7:
                highlights.append(bands["high"])
            elif val >= 0.4:
                highlights.append(bands["moderate"])
            else:
                highlights.append(bands["low"])

        if highlights:
            parts.append(
                ", ".join(
                    h.capitalize() if i == 0 else h
                    for i, h in enumerate(highlights)
                ) + "."
            )

        return " ".join(parts)

    # ─── Working Style Helpers ────────────────────────────────────────

    def _compute_working_style_deltas(
        self, answers: Dict[str, str]
    ) -> Tuple[Dict[str, float], List[str]]:
        """Compute trait deltas and behavioral flags from working style answers."""
        deltas: Dict[str, float] = {}
        flags: List[str] = []

        for cat in WORKING_STYLE_CATEGORIES:
            cat_id = cat["id"]
            if cat_id not in answers:
                continue
            choice_key = answers[cat_id]
            for opt in cat["options"]:
                if opt["key"] == choice_key:
                    for trait, delta in opt.get("trait_deltas", {}).items():
                        deltas[trait] = deltas.get(trait, 0.0) + delta
                    for flag in opt.get("flags", []):
                        if flag not in flags:
                            flags.append(flag)
                    break

        for t in deltas:
            deltas[t] = max(
                -WORKING_STYLE_MAX_DELTA,
                min(WORKING_STYLE_MAX_DELTA, deltas[t])
            )

        return deltas, flags

    # ─── Research Helpers ─────────────────────────────────────────────

    def _infer_primary_domain(self, intent_lower: str) -> str:
        """Infer primary domain from intent keywords."""
        domain_keywords = {
            "business-operations": ["operations", "startup", "business", "management", "strategy"],
            "software-development": ["engineering", "coding", "development", "software", "programming", "devops"],
            "creative-production": ["creative", "writing", "design", "art", "music", "storytelling"],
            "financial-analysis": ["finance", "trading", "accounting", "tax", "audit", "bookkeeping"],
            "content-production": ["content", "social media", "marketing", "copywriting"],
            "research": ["research", "academic", "science", "analysis"],
            "education": ["education", "teaching", "tutoring", "curriculum", "training"],
            "legal": ["legal", "compliance", "regulatory", "contract", "policy"],
            "gaming": ["gaming", "esports", "competitive", "warhammer", "tabletop"],
            "wellness": ["coaching", "therapy", "wellness", "support", "counseling", "mental health"],
        }

        best_domain = "general"
        best_score = 0

        for domain, keywords in domain_keywords.items():
            score = sum(
                1 for kw in keywords
                if re.search(r'\b' + re.escape(kw) + r'\b', intent_lower)
            )
            if score > best_score:
                best_score = score
                best_domain = domain

        return best_domain

    def _infer_secondary_domains(
        self, intent_lower: str, primary: str
    ) -> List[str]:
        """Infer secondary domains from intent, excluding the primary."""
        all_domains = self._infer_all_matching_domains(intent_lower)
        return [d for d in all_domains if d != primary][:3]

    def _infer_all_matching_domains(self, intent_lower: str) -> List[str]:
        """Return all domains that have keyword matches, sorted by score."""
        domain_keywords = {
            "finance": ["finance", "financial", "money", "budget", "pricing"],
            "marketing": ["marketing", "growth", "brand", "social media"],
            "product": ["product", "roadmap", "feature", "user"],
            "design": ["design", "ui", "ux", "visual", "layout"],
            "data-analysis": ["data", "analysis", "metrics", "analytics"],
            "project-management": ["project", "timeline", "milestone", "sprint"],
            "client-management": ["client", "customer", "relationship"],
            "writing": ["writing", "content", "blog", "article"],
        }

        scored = []
        for domain, keywords in domain_keywords.items():
            score = sum(
                1 for kw in keywords
                if re.search(r'\b' + re.escape(kw) + r'\b', intent_lower)
            )
            if score > 0:
                scored.append((domain, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored]

    def _generate_conviction_seeds(
        self, intent_lower: str, matches: List[Tuple[Dict, int]]
    ) -> List[str]:
        """Generate domain-specific conviction override seeds."""
        seeds = []

        conviction_map = {
            "finance|trading|accounting": [
                "Flag transactions or commitments above established thresholds without approval",
                "Surface discrepancies in financial calculations immediately",
            ],
            "operations|management|startup|business": [
                "Flag timeline commitments that conflict with current capacity",
                "Surface pricing decisions below established floors",
                "Flag contradictions with previously stated priorities",
            ],
            "engineering|coding|development|software": [
                "Flag missing error handling in critical paths",
                "Surface security concerns before they reach production",
            ],
            "legal|compliance|regulatory": [
                "Flag compliance risks before they become violations",
                "Surface contract terms that conflict with stated positions",
            ],
            "coaching|therapy|wellness": [
                "Flag patterns that suggest the user is avoiding a topic",
            ],
        }

        for pattern, seed_list in conviction_map.items():
            keywords = pattern.split("|")
            if any(re.search(r'\b' + re.escape(kw) + r'\b', intent_lower)
                   for kw in keywords):
                seeds.extend(seed_list)

        return seeds[:3]

    # ─── Trait Helpers ────────────────────────────────────────────────

    def _parse_trait_tweaks(
        self, input_text: str, current_traits: Dict[str, float]
    ) -> Dict[str, float]:
        """Parse natural language or specific trait adjustments."""
        tweaks = dict(current_traits)
        text = input_text.lower()

        specific = re.findall(r'(\w+)\s*:\s*(\d+)', text)
        for trait_name, value in specific:
            for valid_name in VALID_TRAIT_NAMES:
                if valid_name.startswith(trait_name) or trait_name in valid_name:
                    tweaks[valid_name] = int(value) / 100.0
                    break

        higher = re.findall(r'(?:higher|more|increase)\s+(\w+)', text)
        lower = re.findall(r'(?:lower|less|decrease|reduce)\s+(\w+)', text)

        for word in higher:
            for valid_name in VALID_TRAIT_NAMES:
                if valid_name.startswith(word) or word in valid_name:
                    tweaks[valid_name] = min(TRAIT_CEILING, tweaks.get(valid_name, 0.5) + 0.10)
                    break

        for word in lower:
            for valid_name in VALID_TRAIT_NAMES:
                if valid_name.startswith(word) or word in valid_name:
                    tweaks[valid_name] = max(TRAIT_FLOOR, tweaks.get(valid_name, 0.5) - 0.10)
                    break

        return tweaks

    @staticmethod
    def _trait_description(trait: str, value: float) -> str:
        """Generate a human-readable description for a trait value."""
        descriptions = {
            "observance": {"high": "Catches details you might miss", "low": "Focuses on the big picture"},
            "assertiveness": {"high": "Surfaces concerns directly", "low": "Gentle in delivery"},
            "conviction": {"high": "Pushes back with evidence when warranted", "low": "Defers to your judgment"},
            "warmth": {"high": "Warm and encouraging", "low": "Professional, clinical"},
            "humor": {"high": "Uses wit freely", "low": "Keeps things focused"},
            "initiative": {"high": "Builds ahead without being asked", "low": "Waits for direction"},
            "empathy": {"high": "Reads the room, acknowledges context", "low": "Stays task-focused"},
        }
        level = "high" if value >= 0.6 else "low"
        return descriptions.get(trait, {}).get(level, "")

    @staticmethod
    def _trait_label(trait: str, value: float) -> str:
        """Map a trait value to a human-readable label."""
        if value >= 0.85:
            return "very high"
        elif value >= 0.70:
            return "high"
        elif value >= 0.55:
            return "moderate-high"
        elif value >= 0.45:
            return "moderate"
        elif value >= 0.30:
            return "moderate-low"
        elif value >= 0.15:
            return "low"
        else:
            return "very low"

    def _intent_role_descriptor(self, user_intent: str) -> str:
        """Extract a concise role descriptor from the user's stated intent."""
        if not user_intent:
            return "A custom assistant"

        role_patterns = [
            r'\b(sommelier|chef|bartender|barista)\b',
            r'\b(analyst|advisor|strategist|planner|manager)\b',
            r'\b(coach|mentor|tutor|trainer|instructor)\b',
            r'\b(editor|writer|copywriter|designer)\b',
            r'\b(engineer|developer|architect)\b',
            r'\b(recruiter|consultant|specialist)\b',
            r'\b(accountant|auditor|controller)\b',
            r'\b(researcher|scientist)\b',
            r'\b(assistant|organizer|coordinator)\b',
        ]

        intent_lower = user_intent.lower()
        for pattern in role_patterns:
            match = re.search(pattern, intent_lower)
            if match:
                found_role = match.group(1)
                article = "An" if found_role[0] in "aeiou" else "A"
                return f"{article} {found_role}"

        who_match = re.search(
            r'\b(a|an)\s+(\w+(?:\s+\w+)?)\s+(?:who|that|for)\b', intent_lower
        )
        if who_match:
            role = who_match.group(2)
            article = who_match.group(1).capitalize()
            return f"{article} {role}"

        return "A custom assistant"

    def _infer_role(self, user_intent: str) -> str:
        """Infer a role key from user intent."""
        if not user_intent:
            return "custom"
        descriptor = self._intent_role_descriptor(user_intent)
        role = re.sub(r'^(A|An)\s+', '', descriptor)
        return self._slugify(role) if role != "custom assistant" else "custom"

    def _suggest_name(self, user_intent: str) -> str:
        """Suggest a persona name from user intent."""
        if not user_intent:
            return "Solitaire"

        name_map = {
            "operations|management|startup|business": "Ops",
            "finance|trading|accounting": "Ledger",
            "creative|writing|design|art": "Muse",
            "engineering|coding|development|software": "Forge",
            "research|academic|science": "Scout",
            "coaching|therapy|wellness": "Guide",
            "gaming|esports|competitive": "Tactician",
            "legal|compliance|regulatory": "Sentinel",
            "sales|marketing|growth": "Signal",
            "education|teaching|tutoring": "Sage",
        }

        intent_lower = user_intent.lower()
        for pattern, name in name_map.items():
            keywords = pattern.split("|")
            if any(re.search(r'\b' + re.escape(kw) + r'\b', intent_lower)
                   for kw in keywords):
                return name

        return "Solitaire"

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a persona name to a valid key slug."""
        slug = name.lower().strip()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'[\s]+', '-', slug)
        slug = re.sub(r'-+', '-', slug)
        slug = slug.strip('-')
        return slug or "custom"

    @staticmethod
    def _summarize_traits(traits: Dict[str, float]) -> str:
        """Create a human-readable trait summary."""
        if not traits:
            return "Default profile"
        highlights = []
        for t in ["observance", "assertiveness", "conviction",
                   "warmth", "humor", "initiative", "empathy"]:
            val = traits.get(t, 0.5)
            if val >= 0.75:
                highlights.append(f"High {t}")
            elif val <= 0.35:
                highlights.append(f"Low {t}")
        return ", ".join(highlights) if highlights else "Balanced profile"

    # ─── Interview Helpers ────────────────────────────────────────────

    def _first_interview_step_id(self) -> str:
        ids = self.interview.question_ids
        return f"interview_{ids[0]}" if ids else "naming"

    def _next_interview_question_id(self, current_id: str) -> Optional[str]:
        ids = self.interview.question_ids
        if current_id not in ids:
            return None
        idx = ids.index(current_id)
        if idx + 1 < len(ids):
            return ids[idx + 1]
        return None


# ─── Context Persistence ─────────────────────────────────────────────────────

def save_onboarding_context(
    ctx: OnboardingContext,
    session_id: str,
    base_dir: str,
) -> str:
    """Persist onboarding context to a JSON file. Returns the file path."""
    path = os.path.join(base_dir, f".onboarding_{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ctx.to_dict(), f, indent=2)
    return path


def load_onboarding_context(
    session_id: str,
    base_dir: str,
) -> OnboardingContext:
    """Load onboarding context from a JSON file. Returns empty context if not found."""
    path = os.path.join(base_dir, f".onboarding_{session_id}.json")
    if not os.path.exists(path):
        return OnboardingContext()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return OnboardingContext.from_dict(data)
    except (json.JSONDecodeError, IOError):
        return OnboardingContext()


def cleanup_onboarding_context(session_id: str, base_dir: str):
    """Remove the onboarding context file after completion."""
    path = os.path.join(base_dir, f".onboarding_{session_id}.json")
    if os.path.exists(path):
        os.remove(path)


# ─── Boot Integration Helper ─────────────────────────────────────────────────

def build_onboarding_payload(templates_dir: str = None) -> Dict[str, Any]:
    """Build the structured onboarding payload for boot JSON.

    Called from cmd_boot() when first boot is detected.
    Returns the full flow structure for agent consumption.
    """
    engine = FlowEngine()
    ctx = OnboardingContext()
    first_step = engine.get_next_step(ctx)

    return {
        "onboarding_required": True,
        "flow_version": FLOW_VERSION,
        "first_step": first_step.to_dict(),
        "flow_metadata": {
            "total_steps_estimated": 13,
            "required_steps": [
                "welcome", "intent_capture", "live_research",
                "trait_proposal", "naming", "preview", "confirm",
            ],
            "optional_steps": [
                "working_style", "interview", "north_star",
                "seed_questions", "intent_followup",
            ],
            "interview_questions": 5,
            "template_free": True,
            "quickstart_available": True,
            "deferred_onboarding_available": True,
            "classify_first_message": True,
        },
    }
