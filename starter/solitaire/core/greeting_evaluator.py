"""
The Librarian — Greeting Evaluator
Evaluates the greeting protocol at boot time to determine how the persona
should open a session. Uses persona traits, auto-recall context, and
relationship history to select the right greeting style and example.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GreetingContext:
    """Contextual inputs for greeting evaluation.

    Built from boot-time data: persona traits, auto-recall results,
    and session history.
    """
    # Effective warmth trait value (post-drift)
    warmth: float = 0.5
    # Effective initiative trait value
    initiative: float = 0.5
    # Effective humor trait value
    humor: float = 0.5
    # Open threads from auto-recall (short descriptions)
    open_threads: List[str] = field(default_factory=list)
    # Time-sensitive items (Phase 2 — empty for now)
    time_sensitive_items: List[str] = field(default_factory=list)
    # Total sessions this persona has run (relationship depth)
    total_sessions: int = 0
    # Whether the current session was resumed (vs. fresh)
    resumed: bool = False
    # Data Poetry: experiential context from most recent encoding
    experiential_arc: Optional[str] = None      # e.g. "from discovery to quiet satisfaction"
    experiential_themes: List[str] = field(default_factory=list)  # e.g. ["building", "recognition"]


def evaluate_greeting(persona, context: GreetingContext) -> Dict[str, Any]:
    """Evaluate the greeting protocol for a persona at boot time.

    Args:
        persona: PersonaProfile instance (must have .greeting and .traits)
        context: GreetingContext with auto-recall and session data

    Returns:
        Dict with:
            style: str — the greeting style used
            warmth_category: str — "high" | "mid" | "low"
            memory_weave: bool — whether to reference a past thread
            memory_thread: Optional[str] — the thread to reference
            humor_active: bool — whether humor is appropriate
            selected_example: str — the greeting example text
            time_sensitive_override: bool — whether a deadline forced contextual
    """
    protocol = persona.greeting

    # 1. Determine warmth category
    if context.warmth >= protocol.warmth_threshold + 0.15:
        warmth_category = "high"
    elif context.warmth >= protocol.warmth_threshold - 0.10:
        warmth_category = "mid"
    else:
        warmth_category = "low"

    # 2. Check for time-sensitive override (Phase 2 — initiative-driven)
    time_sensitive_override = False
    if (context.initiative > 0.7
            and context.time_sensitive_items
            and not context.resumed):
        time_sensitive_override = True

    # 3. Determine memory weave
    memory_weave = False
    memory_thread = None
    if (protocol.memory_reference
            and context.open_threads
            and not context.resumed):
        memory_weave = True
        # Pick the most recent thread (first in list)
        memory_thread = context.open_threads[0]

    # 4. Select greeting example
    if time_sensitive_override:
        # Override: lead with the deadline/time-sensitive item
        example_key = "contextual"
    elif memory_weave:
        example_key = "contextual"
    else:
        example_key = f"{warmth_category}_warmth"

    examples = protocol.examples or {}
    selected_example = examples.get(example_key, examples.get("mid_warmth", ""))

    # Substitute [THREAD] placeholder if present
    if memory_thread and "[THREAD]" in selected_example:
        selected_example = selected_example.replace("[THREAD]", memory_thread)

    # 5. Humor gate
    humor_active = context.humor > 0.5

    # 6. Data Poetry: experiential greeting path
    # When an experiential encoding exists, offer the arc as an alternative
    # greeting context. The operations template decides whether to use it.
    experiential_context = None
    if (context.experiential_arc
            and not context.resumed
            and not time_sensitive_override):
        experiential_context = {
            "arc": context.experiential_arc,
            "themes": context.experiential_themes,
        }
        # If we have experiential context but no memory thread,
        # this becomes the primary greeting texture
        if not memory_weave:
            memory_weave = True

    return {
        "style": protocol.style,
        "warmth_category": warmth_category,
        "memory_weave": memory_weave,
        "memory_thread": memory_thread,
        "humor_active": humor_active,
        "selected_example": selected_example,
        "time_sensitive_override": time_sensitive_override,
        "experiential_context": experiential_context,
    }
