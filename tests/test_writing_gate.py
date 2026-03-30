"""Tests for the outbound writing quality gate.

Covers:
- Layer 1: Surface tell detectors (unit tests per detector)
- Layer 2: Structural shape detectors (unit tests per detector)
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
from solitaire.outbound.writing_gate import scan, WritingGateConfig
from solitaire.outbound.config import WritingGateConfig, load_config
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
        result = scan(text)
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
