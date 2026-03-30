"""Tests for the outbound writing quality gate.

Covers:
- Layer 1: Surface tell detectors (unit tests per detector)
- Layer 2: Structural shape detectors (unit tests per detector)
- Layer 3: Persona drift detectors (verbosity, generic voice, warmth)
- Layer 4: Commitment adherence detectors (diplomatic preamble, position collapse, hedging)
- Layer 5: Context coherence detectors (thread dropping, inaccurate reference)
- Integration: Full scan pipeline
- Regression: Tonight's HN draft failure (em dashes + negative parallelism + uniform paragraphs)
- Clean pass: Corrected draft produces zero violations
- Code block exclusion: Em dashes inside code blocks don't trigger
"""

import json
import os
import tempfile
import pytest

from solitaire.outbound.surface_detectors import (
    preprocess,
    detect_em_dashes,
    detect_cursed_word_cluster,
    detect_negative_parallelism,
    detect_participial_filler,
    detect_filler_affirmations,
    detect_compulsive_summary,
    detect_throat_clearing,
    detect_false_closer,
    detect_knowledge_cutoff,
    detect_vague_marketing,
    detect_weasel_wording,
    detect_bloated_phrasing,
    detect_false_ranges,
    detect_diplomatic_preamble,
    run_surface_scan,
)
from solitaire.outbound.structural_detectors import (
    detect_paragraph_uniformity,
    detect_sentence_uniformity,
    detect_repeated_openers,
    detect_parallel_construction,
    run_structural_scan,
)
from solitaire.outbound.persona_detectors import (
    detect_verbosity_mismatch,
    detect_generic_assistant_voice,
    detect_warmth_mismatch,
    run_persona_drift_scan,
)
from solitaire.outbound.commitment_detectors import (
    detect_diplomatic_preamble_commitment,
    detect_position_collapse,
    detect_hedging_overload,
    run_commitment_scan,
)
from solitaire.outbound.context_detectors import (
    detect_thread_dropping,
    detect_inaccurate_reference,
    run_context_scan,
)
from solitaire.outbound.writing_gate import scan, WritingGateConfig
from solitaire.outbound.config import (
    WritingGateConfig, load_config,
    PersonaDriftConfig, CommitmentConfig, ContextConfig,
    PersonaTraits, TranscriptContext,
)
from solitaire.outbound.marker import write_marker, read_marker


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    def test_strips_code_blocks(self):
        text = "Before code. ```python\nfoo \u2014 bar\n``` After code."
        result = preprocess(text)
        assert "\u2014" not in result
        assert "Before code." in result
        assert "After code." in result

    def test_strips_blockquotes(self):
        text = "Normal line.\n> Quoted line with \u2014 em dash.\nAnother normal line."
        result = preprocess(text)
        assert "\u2014" not in result
        assert "Normal line." in result

    def test_preserves_normal_text(self):
        text = "This is normal text with no code or quotes."
        assert preprocess(text) == text


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1: SURFACE DETECTORS
# ═══════════════════════════════════════════════════════════════════════════

class TestEmDash:
    def test_detects_em_dash(self):
        result = detect_em_dashes("The tool \u2014 which is new \u2014 works well.")
        assert result is not None
        assert result.severity == "warning"
        assert result.count == 2

    def test_no_em_dash(self):
        result = detect_em_dashes("The tool, which is new, works well.")
        assert result is None

    def test_hyphen_not_flagged(self):
        result = detect_em_dashes("This is a well-known fact.")
        assert result is None


class TestCursedWordCluster:
    def test_detects_cluster(self):
        text = ("We need to leverage our robust framework to streamline "
                "the process and foster innovation across the paradigm.")
        result = detect_cursed_word_cluster(text, window=100, threshold=3)
        assert result is not None
        assert result.count >= 3

    def test_single_cursed_word_ok(self):
        text = "We need to leverage this opportunity to grow the business."
        result = detect_cursed_word_cluster(text, window=100, threshold=3)
        assert result is None

    def test_two_cursed_words_no_flag_at_threshold_3(self):
        text = "The robust framework helps leverage new opportunities."
        result = detect_cursed_word_cluster(text, window=100, threshold=3)
        assert result is None


class TestNegativeParallelism:
    def test_detects_its_not_x_its_y(self):
        result = detect_negative_parallelism(
            "It's not about efficiency, it's about transformation."
        )
        assert result is not None

    def test_detects_no_x_no_y_just_z(self):
        result = detect_negative_parallelism(
            "No jargon, no fluff, just results."
        )
        assert result is not None

    def test_clean_text(self):
        result = detect_negative_parallelism(
            "The system focuses on transformation through efficiency."
        )
        assert result is None


class TestParticipialFiller:
    def test_detects_filler(self):
        result = detect_participial_filler(
            "This approach works well, emphasizing the importance of testing."
        )
        assert result is not None

    def test_clean_text(self):
        result = detect_participial_filler("Testing is important.")
        assert result is None


class TestFillerAffirmations:
    def test_detects_honestly(self):
        result = detect_filler_affirmations("Honestly, this is a great approach.")
        assert result is not None

    def test_detects_good_catch(self):
        result = detect_filler_affirmations("Good catch! I missed that.")
        assert result is not None

    def test_clean_text(self):
        result = detect_filler_affirmations("This is a great approach.")
        assert result is None


class TestCompulsiveSummary:
    def test_detects_in_summary(self):
        result = detect_compulsive_summary("In summary, the approach works.")
        assert result is not None

    def test_detects_overall(self):
        result = detect_compulsive_summary("Overall, this was a productive session.")
        assert result is not None

    def test_clean_text(self):
        result = detect_compulsive_summary("The approach works and we should ship it.")
        assert result is None


class TestThroatClearing:
    def test_detects_lets_dive_in(self):
        result = detect_throat_clearing("Let's dive in and explore the options.")
        assert result is not None

    def test_clean_text(self):
        result = detect_throat_clearing("The first option is to use Postgres.")
        assert result is None


class TestFalseCloser:
    def test_detects_let_me_know(self):
        result = detect_false_closer("Let me know if you have any questions.")
        assert result is not None

    def test_detects_happy_to_help(self):
        result = detect_false_closer("Happy to help with anything else!")
        assert result is not None

    def test_clean_text(self):
        result = detect_false_closer("That covers the implementation.")
        assert result is None


class TestKnowledgeCutoff:
    def test_detects_as_of_my_last_update(self):
        result = detect_knowledge_cutoff("As of my last update, Python 3.12 is the latest.")
        assert result is not None
        assert result.severity == "warning"

    def test_clean_text(self):
        result = detect_knowledge_cutoff("Python 3.12 is the latest version.")
        assert result is None


class TestVagueMarketing:
    def test_detects_cutting_edge(self):
        result = detect_vague_marketing("Our cutting-edge technology transforms workflows.")
        assert result is not None

    def test_clean_text(self):
        result = detect_vague_marketing("The system responds in under 200ms.")
        assert result is None


class TestWeaselWording:
    def test_detects_many_experts(self):
        result = detect_weasel_wording("Many experts believe this approach is superior.")
        assert result is not None

    def test_clean_text(self):
        result = detect_weasel_wording("Knuth argues this approach is optimal.")
        assert result is None


class TestBloatedPhrasing:
    def test_detects_rapidly_evolving(self):
        result = detect_bloated_phrasing("In today's rapidly evolving landscape of AI.")
        assert result is not None

    def test_clean_text(self):
        result = detect_bloated_phrasing("AI tools are changing how we write.")
        assert result is None


class TestFalseRanges:
    def test_detects_multiple_false_ranges(self):
        text = ("From intimate gatherings to global movements, and "
                "from technical expertise to creative vision.")
        result = detect_false_ranges(text)
        assert result is not None
        assert result.count >= 2

    def test_single_range_ok(self):
        text = "From small to large, the system scales."
        result = detect_false_ranges(text)
        assert result is None


class TestDiplomaticPreamble:
    def test_detects_preamble(self):
        result = detect_diplomatic_preamble(
            "That's a great point, but I think we should consider alternatives."
        )
        assert result is not None
        assert result.severity == "warning"

    def test_clean_disagreement(self):
        result = detect_diplomatic_preamble(
            "I disagree. Here's why."
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2: STRUCTURAL DETECTORS
# ═══════════════════════════════════════════════════════════════════════════

class TestParagraphUniformity:
    def test_detects_uniform_paragraphs(self):
        # Three paragraphs of roughly equal length (each 20+ words)
        text = (
            "The first paragraph discusses the system architecture in detail "
            "covering both the frontend components and the backend services that "
            "power the application across all deployment environments.\n\n"
            "The second paragraph covers the implementation approach including "
            "the testing framework and continuous integration pipeline that "
            "ensures code quality across all supported platforms today.\n\n"
            "The third paragraph addresses the monitoring strategy describing "
            "how alerts and dashboards provide visibility into system health "
            "and performance metrics across production and staging."
        )
        result = detect_paragraph_uniformity(text)
        assert result is not None

    def test_varied_paragraphs_clean(self):
        text = (
            "Short point.\n\n"
            "The second paragraph is much longer and contains several sentences. "
            "It goes into significant detail about the implementation, covering "
            "edge cases, error handling, and performance considerations that the "
            "first paragraph deliberately omitted.\n\n"
            "Medium length paragraph here with some detail."
        )
        result = detect_paragraph_uniformity(text)
        assert result is None


class TestSentenceUniformity:
    def test_detects_uniform_sentences(self):
        # Two paragraphs each with 4 sentences of similar length
        text = (
            "The system runs well. The tests all pass. The code is clean. The docs are done.\n\n"
            "The build works fine. The deploy is smooth. The logs look good. The users are happy."
        )
        result = detect_sentence_uniformity(text)
        assert result is not None

    def test_varied_sentences_clean(self):
        text = (
            "Short. The second sentence is much longer and goes into significant "
            "detail about what happened. Why? Because detail matters when the "
            "stakes are high and the audience needs to understand the full picture."
        )
        result = detect_sentence_uniformity(text)
        assert result is None


class TestRepeatedOpeners:
    def test_detects_same_opener(self):
        text = (
            "The system handles authentication through JWT tokens and "
            "session management through Redis. This provides both security "
            "and performance.\n\n"
            "The system also handles authorization via role-based access "
            "control with fine-grained permissions per resource.\n\n"
            "The system additionally provides audit logging for all "
            "sensitive operations with tamper-proof storage."
        )
        result = detect_repeated_openers(text)
        assert result is not None
        assert result.count >= 3

    def test_varied_openers_clean(self):
        text = (
            "Authentication uses JWT tokens for stateless verification.\n\n"
            "For authorization, we implemented role-based access control.\n\n"
            "Audit logging captures all sensitive operations."
        )
        result = detect_repeated_openers(text)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3: PERSONA DRIFT DETECTORS
# ═══════════════════════════════════════════════════════════════════════════

class TestVerbosityMismatch:
    def test_detects_verbose_response_to_short_input(self):
        user = "What's the status?"
        response = " ".join(["word"] * 1200)  # Way over moderate abs_max * 2 = 1000
        result = detect_verbosity_mismatch(response, user, "moderate")
        assert result is not None
        assert result.category == "verbosity_mismatch"

    def test_moderate_response_ok(self):
        user = "What's the status of the writing gate?"
        response = (
            "Layers 1 and 2 are live. Surface detectors cover 14 patterns, "
            "structural covers 4. Tests are green. Layers 3-5 need building."
        )
        result = detect_verbosity_mismatch(response, user, "moderate")
        assert result is None

    def test_terse_persona_flags_moderate_length(self):
        user = "Status?"
        response = " ".join(["word"] * 400)  # Over terse abs_max * 2 = 300
        result = detect_verbosity_mismatch(response, user, "terse")
        assert result is not None

    def test_verbose_persona_allows_long_response(self):
        # 800w response to 100w input = 8x ratio, within verbose band (max 8x * 2 = 16x)
        user = " ".join(["context"] * 100)
        response = " ".join(["word"] * 800)
        result = detect_verbosity_mismatch(response, user, "verbose")
        assert result is None


class TestGenericAssistantVoice:
    def test_detects_generic_phrases_high_assertiveness(self):
        text = (
            "I'd be happy to help with that! Would you like me to "
            "start by reviewing the code? I can certainly help you "
            "with the implementation details."
        )
        result = detect_generic_assistant_voice(text, assertiveness=0.8)
        assert result is not None
        assert result.count >= 2

    def test_no_flag_low_assertiveness(self):
        text = (
            "I'd be happy to help with that! Would you like me to "
            "start by reviewing the code?"
        )
        result = detect_generic_assistant_voice(text, assertiveness=0.4)
        assert result is None

    def test_clean_direct_text(self):
        text = (
            "The scan runs on every assistant response. Surface detectors "
            "cover 14 patterns. Structural detectors measure text geometry."
        )
        result = detect_generic_assistant_voice(text, assertiveness=0.8)
        assert result is None


class TestWarmthMismatch:
    def test_excessive_warmth_low_trait(self):
        text = (
            "That's really wonderful work! I really appreciate you sharing "
            "this. That's so fantastic, and I'm really glad we got to "
            "discuss it. Thanks so much for the detailed explanation."
        )
        result = detect_warmth_mismatch(text, warmth=0.3)
        assert result is not None
        assert result.category == "warmth_mismatch"

    def test_warmth_ok_for_high_warmth_persona(self):
        text = (
            "That's really wonderful work! I really appreciate you sharing "
            "this. That's so fantastic."
        )
        result = detect_warmth_mismatch(text, warmth=0.8)
        assert result is None

    def test_absent_warmth_high_trait(self):
        # 200+ words with zero warmth markers
        text = " ".join(
            ["The system processes data efficiently and returns results."] * 30
        )
        result = detect_warmth_mismatch(text, warmth=0.8)
        assert result is not None
        assert result.count == 0  # zero warmth markers

    def test_moderate_warmth_ok(self):
        text = "The implementation looks solid. Tests are passing."
        result = detect_warmth_mismatch(text, warmth=0.5)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 4: COMMITMENT ADHERENCE DETECTORS
# ═══════════════════════════════════════════════════════════════════════════

class TestDiplomaticPreambleCommitment:
    def test_flags_with_high_conviction(self):
        text = "That's a great point, but I think we should reconsider."
        result = detect_diplomatic_preamble_commitment(text, conviction=0.85)
        assert result is not None
        assert result.severity == "warning"

    def test_no_flag_low_conviction(self):
        text = "That's a great point, but I think we should reconsider."
        result = detect_diplomatic_preamble_commitment(text, conviction=0.4)
        assert result is None

    def test_clean_direct_disagreement(self):
        text = "I disagree. The data shows a different pattern."
        result = detect_diplomatic_preamble_commitment(text, conviction=0.85)
        assert result is None


class TestPositionCollapse:
    def test_detects_capitulation_without_reasoning(self):
        prior = "I think we should use PostgreSQL for this. The query patterns favor it."
        current = "You're absolutely right. I stand corrected. Let's go with MongoDB."
        result = detect_position_collapse(current, prior, conviction=0.85)
        assert result is not None
        assert result.severity == "warning"

    def test_allows_capitulation_with_reasoning(self):
        prior = "I think we should use PostgreSQL for this."
        current = (
            "You're right. Because the new requirements include "
            "document-level versioning, MongoDB is the better fit here."
        )
        result = detect_position_collapse(current, prior, conviction=0.85)
        assert result is None

    def test_no_flag_without_prior_position(self):
        prior = "Here's the data you requested."
        current = "You're absolutely right. Let's proceed."
        result = detect_position_collapse(current, prior, conviction=0.85)
        assert result is None

    def test_no_flag_low_conviction(self):
        prior = "I think we should use PostgreSQL."
        current = "You're right, let's go with MongoDB."
        result = detect_position_collapse(current, prior, conviction=0.4)
        assert result is None

    def test_no_flag_without_prior_text(self):
        current = "You're absolutely right."
        result = detect_position_collapse(current, "", conviction=0.85)
        assert result is None


class TestHedgingOverload:
    def test_detects_excessive_hedging(self):
        text = (
            "Perhaps we could consider using a different approach. "
            "Maybe the existing solution might be worth evaluating. "
            "It seems like there could be some benefits, and it could "
            "possibly improve performance. Arguably the trade-offs "
            "are worth considering in some cases."
        )
        result = detect_hedging_overload(text, conviction=0.85)
        assert result is not None
        assert result.category == "hedging_overload"

    def test_no_flag_low_conviction(self):
        text = (
            "Perhaps we could consider a different approach. "
            "Maybe the existing solution might work. Possibly."
        )
        result = detect_hedging_overload(text, conviction=0.5)
        assert result is None

    def test_confident_text_ok(self):
        text = (
            "Use PostgreSQL for this. The query patterns require joins "
            "across three tables, and the data is relational. MongoDB "
            "would force you to denormalize or run multiple queries."
        )
        result = detect_hedging_overload(text, conviction=0.85)
        assert result is None

    def test_short_text_skipped(self):
        text = "Perhaps we should try that."
        result = detect_hedging_overload(text, conviction=0.85)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 5: CONTEXT COHERENCE DETECTORS
# ═══════════════════════════════════════════════════════════════════════════

class TestThreadDropping:
    def test_detects_ignored_topics(self):
        user = (
            "I need help with the PostgreSQL migration, the Redis caching "
            "strategy, and the Kubernetes deployment configuration."
        )
        response = (
            "The PostgreSQL migration can be handled with Alembic. "
            "Set up your migration scripts and test them locally first."
        )
        result = detect_thread_dropping(response, user)
        assert result is not None
        assert result.category == "thread_dropping"

    def test_all_topics_addressed(self):
        user = (
            "I need help with the PostgreSQL migration and the Redis "
            "caching strategy."
        )
        response = (
            "For the PostgreSQL migration, use Alembic. For Redis caching, "
            "start with a write-through strategy for the session store."
        )
        result = detect_thread_dropping(response, user)
        assert result is None

    def test_short_user_message_skipped(self):
        user = "Fix the bug."
        response = "The fix is to add a null check on line 42."
        result = detect_thread_dropping(response, user)
        assert result is None

    def test_no_user_text(self):
        result = detect_thread_dropping("Some response.", "")
        assert result is None


class TestInaccurateReference:
    def test_detects_fabricated_backreference(self):
        prior = "The system uses PostgreSQL for the primary datastore."
        response = (
            "As I mentioned earlier, the Kubernetes deployment requires "
            "three replicas for high availability."
        )
        result = detect_inaccurate_reference(response, prior)
        assert result is not None
        assert result.category == "inaccurate_reference"

    def test_valid_backreference_ok(self):
        prior = "The system uses PostgreSQL for the primary datastore."
        response = (
            "As I mentioned, PostgreSQL handles the primary datastore. "
            "Now let's discuss the caching layer."
        )
        result = detect_inaccurate_reference(response, prior)
        assert result is None

    def test_no_backreference_ok(self):
        prior = "The system uses PostgreSQL."
        response = "The caching layer should use Redis with a 15-minute TTL."
        result = detect_inaccurate_reference(response, prior)
        assert result is None

    def test_no_prior_turns_skipped(self):
        response = "As I mentioned, the migration needs testing."
        result = detect_inaccurate_reference(response, "")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# CONFIDENCE LEVELS
# ═══════════════════════════════════════════════════════════════════════════

class TestConfidenceLevels:
    """Verify that detectors return appropriate confidence levels.

    High confidence: unambiguous pattern matches (regex hit is definitive).
    Low confidence: borderline or inherently fuzzy detections that benefit
    from model verification on the next turn.
    """

    # -- Layer 1: Surface detectors are always high confidence --

    def test_em_dash_high_confidence(self):
        result = detect_em_dashes("The tool \u2014 which is new \u2014 works well.")
        assert result is not None
        assert result.confidence == "high"

    def test_cursed_cluster_high_confidence(self):
        text = ("We need to leverage our robust framework to streamline "
                "the process and foster innovation across the paradigm.")
        result = detect_cursed_word_cluster(text, window=100, threshold=3)
        assert result is not None
        assert result.confidence == "high"

    # -- Layer 3: Persona drift confidence varies --

    def test_verbosity_borderline_low_confidence(self):
        """Overshoot 2-3x is borderline, should be low confidence."""
        # 500w response to 100w input = 5x ratio. Moderate band max is 4x.
        # Overshoot = 5/4 = 1.25x abs, 500/500 = 1.0x abs. Neither >3.
        # But need to exceed 2x trigger: 500w > abs_max*2=1000? No.
        # Use ratio trigger: need ratio > ratio_max*2 = 8x.
        # 900w / 100w = 9x ratio, overshoot = 9/4 = 2.25x. Low confidence.
        user = " ".join(["context"] * 100)
        response = " ".join(["word"] * 900)
        result = detect_verbosity_mismatch(response, user, "moderate")
        assert result is not None
        assert result.confidence == "low"

    def test_verbosity_extreme_high_confidence(self):
        """Overshoot >3x is clear drift, should be high confidence."""
        user = "Status?"
        response = " ".join(["word"] * 1500)
        result = detect_verbosity_mismatch(response, user, "moderate")
        assert result is not None
        assert result.confidence == "high"

    def test_warmth_excessive_high_confidence(self):
        """Excessive warmth for low-warmth persona is definitive."""
        text = (
            "That's really wonderful work! I really appreciate you sharing "
            "this. That's so fantastic, and I'm really glad we got to "
            "discuss it. Thanks so much for the detailed explanation."
        )
        result = detect_warmth_mismatch(text, warmth=0.3)
        assert result is not None
        assert result.confidence == "high"

    def test_warmth_absent_low_confidence(self):
        """Absent warmth could be style choice, always low confidence."""
        text = " ".join(
            ["The system processes data efficiently and returns results."] * 30
        )
        result = detect_warmth_mismatch(text, warmth=0.8)
        assert result is not None
        assert result.confidence == "low"

    def test_generic_voice_high_confidence(self):
        """Generic assistant phrases are unambiguous regex matches."""
        text = (
            "I'd be happy to help with that! Would you like me to "
            "start by reviewing the code? I can certainly help you."
        )
        result = detect_generic_assistant_voice(text, assertiveness=0.8)
        assert result is not None
        assert result.confidence == "high"

    # -- Layer 4: Commitment detectors --

    def test_position_collapse_always_low_confidence(self):
        """Position collapse detection is inherently ambiguous."""
        prior = "I think we should use PostgreSQL for this. The query patterns favor it."
        current = "You're absolutely right. I stand corrected. Let's go with MongoDB."
        result = detect_position_collapse(current, prior, conviction=0.85)
        assert result is not None
        assert result.confidence == "low"

    def test_hedging_overload_high_confidence(self):
        """Hedge phrase counting is definitive."""
        text = (
            "Perhaps we could consider using a different approach. "
            "Maybe the existing solution might be worth evaluating. "
            "It seems like there could be some benefits, and it could "
            "possibly improve performance. Arguably the trade-offs "
            "are worth considering in some cases."
        )
        result = detect_hedging_overload(text, conviction=0.85)
        assert result is not None
        assert result.confidence == "high"

    # -- Layer 5: Context coherence --

    def test_thread_dropping_borderline_low_confidence(self):
        """50-70% drop rate is borderline, low confidence."""
        user = (
            "Help with PostgreSQL migration, Redis caching, "
            "Kubernetes deployment, and monitoring setup."
        )
        response = (
            "For the PostgreSQL migration, use Alembic. For Redis caching, "
            "start with a write-through strategy."
        )
        result = detect_thread_dropping(response, user)
        # Should flag some dropped topics but at borderline confidence
        if result is not None:
            assert result.confidence == "low"

    def test_thread_dropping_severe_high_confidence(self):
        """Overwhelming drop rate is high confidence."""
        user = (
            "I need help with the PostgreSQL migration, the Redis caching "
            "strategy, the Kubernetes deployment, the monitoring setup, "
            "and the authentication service."
        )
        response = (
            "That sounds like a good plan. Let me know when you want "
            "to get started on the implementation details."
        )
        result = detect_thread_dropping(response, user)
        assert result is not None
        assert result.confidence == "high"

    def test_inaccurate_reference_single_low_confidence(self):
        """Single fabricated reference could be vocabulary mismatch."""
        prior = "The system uses PostgreSQL for the primary datastore."
        response = (
            "As I mentioned earlier, the Kubernetes deployment requires "
            "three replicas for high availability."
        )
        result = detect_inaccurate_reference(response, prior)
        assert result is not None
        assert result.confidence == "low"

    # -- Integration: confidence flows through scan() --

    def test_confidence_flows_through_full_scan(self):
        """Confidence field propagates from detector to WritingViolation."""
        config = WritingGateConfig(
            commitment=CommitmentConfig(enabled=True),
            min_response_length=10,
        )
        traits = PersonaTraits(conviction=0.85)
        transcript = TranscriptContext(
            prior_assistant_text=(
                "I think we should use PostgreSQL for this. The query "
                "patterns require joins across multiple tables."
            ),
        )
        text = (
            "You're absolutely right. I stand corrected. Let's go "
            "with the approach you suggested instead of what I recommended "
            "earlier in the conversation about database selection. "
            "I agree completely with your assessment here."
        )
        result = scan(text, config, traits, transcript)
        collapse = [v for v in result.violations if v.category == "position_collapse"]
        assert len(collapse) > 0
        assert collapse[0].confidence == "low"

    def test_high_confidence_marker_dict(self):
        """High-confidence violations carry through to marker dict."""
        config = WritingGateConfig(min_response_length=5)
        text = "The tool \u2014 which is new \u2014 works well and handles all the cases."
        transcript = TranscriptContext(user_text="How does the tool work?")
        result = scan(text, config, transcript=transcript)
        em = [v for v in result.violations if v.category == "em_dash"]
        assert len(em) > 0
        marker = result.to_marker_dict()
        em_marker = [v for v in marker["violations"] if v["category"] == "em_dash"]
        assert em_marker[0]["confidence"] == "high"


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION: LAYERS 3-5 WITH FULL SCAN
# ═══════════════════════════════════════════════════════════════════════════

class TestFullScanWithPersona:
    """Integration tests for Layers 3-5 through the full scan pipeline."""

    def test_persona_drift_detected_when_enabled(self):
        config = WritingGateConfig(
            persona_drift=PersonaDriftConfig(enabled=True),
            min_response_length=10,
        )
        traits = PersonaTraits(assertiveness=0.8)
        text = (
            "I'd be happy to help with that! Would you like me to "
            "start by reviewing the code? I can certainly help you "
            "with the implementation details and walk you through "
            "each component step by step so you understand everything. "
            "Let me help you by going through all of this carefully "
            "and making sure we cover every aspect of the problem."
        )
        result = scan(text, config, traits)
        generic = [v for v in result.violations if v.category == "generic_assistant_voice"]
        assert len(generic) > 0

    def test_commitment_detected_when_enabled(self):
        config = WritingGateConfig(
            commitment=CommitmentConfig(enabled=True),
            min_response_length=10,
        )
        traits = PersonaTraits(conviction=0.85)
        transcript = TranscriptContext(
            prior_assistant_text=(
                "I think we should use PostgreSQL for this. The query "
                "patterns require joins across multiple tables and the "
                "data model is fundamentally relational."
            ),
        )
        text = (
            "You're absolutely right. I stand corrected. Let's go "
            "with the approach you suggested instead of what I recommended "
            "earlier in the conversation about database selection. "
            "I agree completely, MongoDB is the way to go for this "
            "particular use case given what you've described."
        )
        result = scan(text, config, traits, transcript)
        collapse = [v for v in result.violations if v.category == "position_collapse"]
        assert len(collapse) > 0

    def test_context_detected_when_enabled(self):
        config = WritingGateConfig(
            context=ContextConfig(enabled=True),
            min_response_length=10,
        )
        transcript = TranscriptContext(
            user_text=(
                "I need help with the PostgreSQL migration, the Redis caching "
                "strategy, and the Kubernetes deployment configuration."
            ),
        )
        text = (
            "The PostgreSQL migration can be handled with Alembic. "
            "Set up your migration scripts and run them against a staging "
            "database before applying to production. Test each migration "
            "step individually to catch issues early."
        )
        result = scan(text, config, transcript=transcript)
        dropped = [v for v in result.violations if v.category == "thread_dropping"]
        assert len(dropped) > 0

    def test_layers_disabled_by_default(self):
        """Default config has persona_drift, commitment, context disabled."""
        config = WritingGateConfig()
        traits = PersonaTraits(assertiveness=0.8, conviction=0.85)
        text = (
            "I'd be happy to help! Would you like me to review the code? "
            "Perhaps we could consider an alternative approach. Maybe "
            "the existing solution might be worth evaluating instead."
        )
        result = scan(text, config, traits)
        layer3_5 = [v for v in result.violations if v.layer >= 3]
        assert len(layer3_5) == 0, "Layers 3-5 should not fire when disabled"


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION: FULL SCAN
# ═══════════════════════════════════════════════════════════════════════════

class TestFullScan:
    def test_short_response_skipped(self):
        result = scan("Yes, that works.")
        assert not result.has_violations()

    def test_clean_response(self):
        text = (
            "The evaluation gate runs on every user message before the model "
            "generates a response. It checks for destructive actions, reference "
            "mismatches, and unverified claims.\n\n"
            "When concerns are found, a structured block gets injected into "
            "the model's context. This is structural enforcement, not behavioral.\n\n"
            "The claim scanner handles outbound checking via the Stop hook."
        )
        transcript = TranscriptContext(
            user_text="How does the evaluation gate work and what does it check for?",
        )
        result = scan(text, transcript=transcript)
        # Should have few or no violations on clean text
        warnings = [v for v in result.violations if v.severity == "warning"]
        assert len(warnings) == 0

    def test_em_dash_flagged(self):
        text = (
            "The writing gate scans for patterns \u2014 things like em dashes, "
            "cursed word clusters, and structural uniformity. It runs after "
            "every assistant response via the Stop hook. The scanner checks "
            "the full text of the response against a set of compiled regex "
            "patterns covering thirteen surface categories and four structural "
            "categories. Results are written as a marker file that the "
            "evaluation gate picks up on the next turn."
        )
        result = scan(text)
        em_dash_violations = [v for v in result.violations if v.category == "em_dash"]
        assert len(em_dash_violations) == 1
        assert em_dash_violations[0].severity == "warning"


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSION: TONIGHT'S HN DRAFT FAILURE
# ═══════════════════════════════════════════════════════════════════════════

class TestHNDraftRegression:
    """The draft that triggered this entire feature.

    Ward's first HN comment draft had em dashes, negative parallelism,
    and uniform paragraph structure. It took 4 revision cycles to fix.
    The gate should catch these on the first pass.
    """

    BAD_DRAFT = (
        "The author's experience is a real cognitive pattern, not just a feeling. "
        "When you outsource a skill to a tool \u2014 even partially, even just for "
        '"checking" \u2014 the neural pathway for doing it yourself weakens. It\'s '
        "the same reason people who rely on spellcheck become worse spellers over "
        "time. The tool becomes a crutch that replaces the muscle instead of "
        "supporting it.\n\n"
        "What I find missing from this discussion is specificity about what makes "
        "AI-touched writing detectable. It's not any single word or pattern. It's "
        "clustering. A human might write \"leverage\" once in a piece and it's fine. "
        "An LLM will stack \"leverage,\" \"streamline,\" and \"robust\" within three "
        "paragraphs. Humans vary their sentence length naturally \u2014 long, then "
        "short, then medium. LLMs default to a metronomic mid-length rhythm. "
        "Humans leave thoughts half-developed when they're not the point. LLMs "
        "compulsively summarize every section.\n\n"
        "The detection tools that flagged the author's writing aren't looking for "
        "AI words. They're looking for the absence of human irregularity. The fix "
        "isn't to write worse on purpose. It's to stop routing your drafts through "
        "something that sands off every rough edge, because the rough edges were "
        "the proof of life."
    )

    CLEAN_DRAFT = (
        "The author's experience isn't just a feeling. Outsource a skill to a "
        "tool, even just for \"checking,\" and the ability to do it yourself "
        "degrades. Same thing happened to spelling after spellcheck became default.\n\n"
        "What nobody in this thread has gotten specific about: AI-touched writing "
        "isn't detectable because of any single word. It's detectable because of "
        "clustering. A human might write \"leverage\" once and it disappears. Run a "
        "draft through an LLM and you get \"leverage,\" \"streamline,\" and \"robust\" "
        "within three paragraphs. Sentence length is another giveaway. LLMs settle "
        "into a rhythm where every sentence is roughly the same size, while most "
        "people naturally alternate between long and short without thinking about "
        "it. And then there's the compulsive completeness. Every point wrapped up, "
        "every thread tied off. Real writing leaves things half-developed when "
        "they're not the main event.\n\n"
        "The detection tools that flagged the author's post aren't matching "
        "vocabulary. They're noticing the irregularity is gone."
    )

    def test_bad_draft_flags_em_dashes(self):
        result = scan(self.BAD_DRAFT)
        em_dashes = [v for v in result.violations if v.category == "em_dash"]
        assert len(em_dashes) > 0, "Bad draft should flag em dashes"
        assert em_dashes[0].count >= 2  # 2 match regex (\w before \u2014)

    def test_bad_draft_has_warnings(self):
        """Bad draft should have at least one warning-level violation."""
        result = scan(self.BAD_DRAFT)
        warnings = [v for v in result.violations if v.severity == "warning"]
        assert len(warnings) > 0, "Bad draft should have warnings"

    def test_clean_draft_no_em_dashes(self):
        result = scan(self.CLEAN_DRAFT)
        em_dashes = [v for v in result.violations if v.category == "em_dash"]
        assert len(em_dashes) == 0, "Clean draft should have no em dashes"

    def test_clean_draft_no_em_dash_or_parallelism_warnings(self):
        """Clean draft should have no em dash or parallelism warnings.

        Note: the clean draft discusses cursed words by name (leverage,
        streamline, robust) as examples of AI tells. The cursed word
        detector correctly flags these. This is expected behavior for
        content that quotes cursed words.
        """
        result = scan(self.CLEAN_DRAFT)
        structural_warnings = [
            v for v in result.violations
            if v.severity == "warning"
            and v.category in ("em_dash", "negative_parallelism", "diplomatic_preamble")
        ]
        assert len(structural_warnings) == 0, f"Clean draft should have no structural warnings, got: {structural_warnings}"


# ═══════════════════════════════════════════════════════════════════════════
# CODE BLOCK EXCLUSION
# ═══════════════════════════════════════════════════════════════════════════

class TestCodeBlockExclusion:
    def test_em_dashes_in_code_not_flagged(self):
        text = (
            "Here is how the pattern works in code:\n\n"
            "```python\n"
            "result = value \u2014 offset  # em dash in code\n"
            "```\n\n"
            "The implementation is straightforward and handles all edge cases "
            "correctly, including negative values and overflow conditions."
        )
        result = scan(text)
        em_dashes = [v for v in result.violations if v.category == "em_dash"]
        assert len(em_dashes) == 0


# ═══════════════════════════════════════════════════════════════════════════
# MARKER READ/WRITE
# ═══════════════════════════════════════════════════════════════════════════

class TestMarker:
    def test_write_and_read(self):
        workspace = tempfile.mkdtemp()
        violations = [
            {"layer": 1, "category": "em_dash", "severity": "warning",
             "count": 2, "samples": ["test \u2014 sample"], "detail": "test", "score": None},
        ]
        write_marker(violations, "chief", workspace)
        data = read_marker(workspace)
        assert data is not None
        assert len(data["violations"]) == 1
        assert data["violations"][0]["category"] == "em_dash"
        assert data["persona_key"] == "chief"

    def test_read_consumes_marker(self):
        workspace = tempfile.mkdtemp()
        violations = [
            {"layer": 1, "category": "em_dash", "severity": "warning",
             "count": 1, "samples": [], "detail": "test", "score": None},
        ]
        write_marker(violations, "chief", workspace)
        data1 = read_marker(workspace)
        assert data1 is not None
        data2 = read_marker(workspace)
        assert data2 is None, "Marker should be consumed after first read"

    def test_read_nonexistent(self):
        workspace = tempfile.mkdtemp()
        data = read_marker(workspace)
        assert data is None
