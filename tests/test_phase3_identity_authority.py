"""
Tests for Phase 3: Engine as Identity Authority.

Tests the cognitive profile module, identity context block builder,
commitments block wrapper, and format adapter.
"""
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import pytest

from solitaire.core.cognitive_profile import (
    build_cognitive_profile,
    TRAIT_DESCRIPTIONS,
    _value_to_band,
)
from solitaire.core.format_adapter import FormatAdapter


# ─── Minimal test doubles ────────────────────────────────────────────────

@dataclass
class FakeIdentity:
    name: str = "TestAgent"
    role: str = "test-partner"
    domain_scope: str = "testing"


@dataclass
class FakeTraits:
    observance: float = 0.5
    assertiveness: float = 0.5
    conviction: float = 0.5
    warmth: float = 0.5
    humor: float = 0.5
    initiative: float = 0.5
    empathy: float = 0.5

    def to_dict(self):
        return {
            "observance": self.observance,
            "assertiveness": self.assertiveness,
            "conviction": self.conviction,
            "warmth": self.warmth,
            "humor": self.humor,
            "initiative": self.initiative,
            "empathy": self.empathy,
        }


@dataclass
class FakePersona:
    identity: FakeIdentity = field(default_factory=FakeIdentity)
    traits: FakeTraits = field(default_factory=FakeTraits)
    _baseline_traits: Optional[FakeTraits] = None
    state: Optional[object] = None


# ─── Cognitive Profile Tests ─────────────────────────────────────────────

class TestCognitiveProfile:

    def test_basic_rendering(self):
        """Profile renders all 7 traits with correct delimiters."""
        persona = FakePersona()
        profile = build_cognitive_profile(persona)

        assert "═══ COGNITIVE PROFILE ═══" in profile
        assert "═══ END COGNITIVE PROFILE ═══" in profile
        assert "Identity: TestAgent (test-partner)" in profile
        assert "Primary domain: testing" in profile
        assert "Disposition:" in profile

    def test_all_traits_present(self):
        """Every trait dimension appears in the output."""
        persona = FakePersona()
        profile = build_cognitive_profile(persona)

        # All 7 traits should produce a line (moderate band at 0.5)
        # Count the disposition lines (start with "- ")
        disposition_lines = [
            l for l in profile.split("\n")
            if l.startswith("- ")
        ]
        assert len(disposition_lines) == 7

    def test_very_high_band(self):
        """Traits at 0.85+ render as 'defining trait'."""
        persona = FakePersona(traits=FakeTraits(observance=0.90))
        profile = build_cognitive_profile(persona)
        assert "defining trait" in profile

    def test_very_low_band(self):
        """Traits at <0.20 render as the lowest band."""
        persona = FakePersona(traits=FakeTraits(humor=0.10))
        profile = build_cognitive_profile(persona)
        assert "no humor" in profile

    def test_drift_shown_inline(self):
        """Drift from baseline appears inline with the trait."""
        persona = FakePersona(
            traits=FakeTraits(conviction=0.88),
            _baseline_traits=FakeTraits(conviction=0.85),
        )
        profile = build_cognitive_profile(persona)
        assert "drift:" in profile
        assert "↑" in profile

    def test_negative_drift(self):
        """Negative drift shows downward arrow."""
        persona = FakePersona(
            traits=FakeTraits(warmth=0.40),
            _baseline_traits=FakeTraits(warmth=0.50),
        )
        profile = build_cognitive_profile(persona)
        assert "↓" in profile

    def test_no_drift_when_equal(self):
        """No drift indicator when trait equals baseline."""
        persona = FakePersona(
            traits=FakeTraits(warmth=0.50),
            _baseline_traits=FakeTraits(warmth=0.50),
        )
        profile = build_cognitive_profile(persona)
        assert "drift:" not in profile

    def test_none_persona_returns_empty(self):
        """None persona produces empty string."""
        assert build_cognitive_profile(None) == ""

    def test_no_conn_graceful(self):
        """Without DB connection, texture is skipped but profile renders."""
        persona = FakePersona()
        profile = build_cognitive_profile(persona, conn=None)
        assert "═══ COGNITIVE PROFILE ═══" in profile
        assert "texture:" not in profile

    def test_no_domain_scope(self):
        """Missing domain_scope skips the domain line."""
        persona = FakePersona(identity=FakeIdentity(domain_scope=""))
        profile = build_cognitive_profile(persona)
        assert "Primary domain:" not in profile


class TestValueToBand:

    def test_bands(self):
        assert _value_to_band(0.90) == "very_high"
        assert _value_to_band(0.85) == "very_high"
        assert _value_to_band(0.70) == "high"
        assert _value_to_band(0.50) == "moderate"
        assert _value_to_band(0.30) == "low"
        assert _value_to_band(0.10) == "very_low"
        assert _value_to_band(0.0) == "very_low"


# ─── Format Adapter Tests ────────────────────────────────────────────────

class TestFormatAdapter:

    @pytest.fixture
    def sample_blocks(self):
        return {
            "cognitive_profile": "═══ COGNITIVE PROFILE ═══\n\nIdentity: Ward (partner)\n\n═══ END COGNITIVE PROFILE ═══",
            "identity": "═══ IDENTITY CONTEXT ═══\n\nNorth Star: Be accurate.\n\n═══ END IDENTITY CONTEXT ═══",
            "briefing": "═══ SITUATIONAL BRIEFING ═══\n\nWork streams: testing\n\n═══ END BRIEFING ═══",
        }

    def test_claude_format(self, sample_blocks):
        """Claude format joins blocks with double newlines."""
        adapter = FormatAdapter("claude")
        output = adapter.render(sample_blocks)

        assert "═══ COGNITIVE PROFILE ═══" in output
        assert "═══ IDENTITY CONTEXT ═══" in output
        assert "═══ SITUATIONAL BRIEFING ═══" in output

    def test_claude_section_ordering(self, sample_blocks):
        """Claude format renders sections in the defined order."""
        adapter = FormatAdapter("claude")
        output = adapter.render(sample_blocks)

        profile_pos = output.index("COGNITIVE PROFILE")
        identity_pos = output.index("IDENTITY CONTEXT")
        briefing_pos = output.index("SITUATIONAL BRIEFING")

        assert profile_pos < identity_pos < briefing_pos

    def test_openai_format(self, sample_blocks):
        """OpenAI format uses markdown headers instead of ═══ delimiters."""
        adapter = FormatAdapter("openai")
        output = adapter.render(sample_blocks)

        assert "## Cognitive Profile" in output
        assert "## Identity Context" in output
        assert "## Situational Briefing" in output
        # Delimiters should be stripped
        assert "═══" not in output

    def test_raw_format(self, sample_blocks):
        """Raw format returns valid JSON."""
        adapter = FormatAdapter("raw")
        output = adapter.render(sample_blocks)

        parsed = json.loads(output)
        assert "cognitive_profile" in parsed
        assert "identity" in parsed
        assert "briefing" in parsed

    def test_invalid_format_raises(self):
        """Unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            FormatAdapter("gemini")

    def test_empty_blocks(self):
        """Empty blocks produce empty output."""
        adapter = FormatAdapter("claude")
        output = adapter.render({})
        assert output == ""

    def test_none_blocks_skipped(self, sample_blocks):
        """Blocks with empty/None values are skipped."""
        sample_blocks["identity"] = ""
        adapter = FormatAdapter("claude")
        output = adapter.render(sample_blocks)

        assert "COGNITIVE PROFILE" in output
        assert "IDENTITY CONTEXT" not in output


# ─── Identity Context Block Tests ────────────────────────────────────────

class TestIdentityContextBlock:

    @pytest.fixture
    def db(self):
        """Create an in-memory DB with identity graph schema."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        from solitaire.storage.identity_graph import IdentityGraph
        ig = IdentityGraph(conn)
        from solitaire.storage.identity_graph import ensure_identity_graph_schema
        ensure_identity_graph_schema(conn)
        return conn

    def test_empty_graph_returns_empty(self, db):
        """No identity data produces empty string."""
        from solitaire.storage.identity_graph import IdentityGraph
        ig = IdentityGraph(db)
        block = ig.build_identity_context_block()
        assert block == ""

    def test_north_star_rendered(self, db):
        """North star appears in the identity block."""
        from solitaire.storage.identity_graph import IdentityGraph, IdentityNode
        ig = IdentityGraph(db)
        ig.add_node(IdentityNode(
            id="ns_1",
            node_type="motivation",
            content="Be honest about what I observe.",
            status="active",
            metadata={"role": "north_star"},
        ))
        block = ig.build_identity_context_block()
        assert "North Star:" in block
        assert "Be honest about what I observe." in block

    def test_growth_edges_rendered(self, db):
        """Active growth edges appear in the identity block."""
        from solitaire.storage.identity_graph import IdentityGraph, IdentityNode
        ig = IdentityGraph(db)
        ig.add_node(IdentityNode(
            id="ge_1",
            node_type="growth_edge",
            content="Stay in reflective moments.",
            status="practicing",
            first_seen="2026-03-20T00:00:00",
            last_seen="2026-03-29T00:00:00",
        ))
        block = ig.build_identity_context_block()
        assert "Growth edges (active):" in block
        assert "Stay in reflective moments." in block
        assert "practicing" in block

    def test_patterns_rendered(self, db):
        """Top patterns appear in the identity block."""
        from solitaire.storage.identity_graph import IdentityGraph, IdentityNode
        ig = IdentityGraph(db)
        ig.add_node(IdentityNode(
            id="pat_1",
            node_type="pattern",
            content="Action bias overriding verification.",
            observation_count=11,
            trajectory="stable",
        ))
        block = ig.build_identity_context_block()
        assert "Known patterns:" in block
        assert "Action bias overriding verification." in block
        assert "stable" in block
        assert "observed 11x" in block

    def test_delimiters_present(self, db):
        """Identity block has proper delimiters."""
        from solitaire.storage.identity_graph import IdentityGraph, IdentityNode
        ig = IdentityGraph(db)
        ig.add_node(IdentityNode(
            id="ge_1",
            node_type="growth_edge",
            content="Test edge.",
            status="identified",
        ))
        block = ig.build_identity_context_block()
        assert "═══ IDENTITY CONTEXT ═══" in block
        assert "═══ END IDENTITY CONTEXT ═══" in block


# ─── Commitments Block Wrapper Tests ─────────────────────────────────────

class TestCommitmentsBlockWrapper:

    @pytest.fixture
    def db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        from solitaire.storage.identity_graph import IdentityGraph
        ig = IdentityGraph(conn)
        from solitaire.storage.identity_graph import ensure_identity_graph_schema
        ensure_identity_graph_schema(conn)
        return conn

    def test_wrapper_returns_string(self, db):
        """build_commitments_block returns a string, not a tuple."""
        from solitaire.storage.identity_graph import IdentityGraph
        ig = IdentityGraph(db)
        result = ig.build_commitments_block()
        assert isinstance(result, str)

    def test_wrapper_works_without_session_id(self, db):
        """build_commitments_block works without explicit session_id."""
        from solitaire.storage.identity_graph import IdentityGraph
        ig = IdentityGraph(db)
        # Should not raise
        result = ig.build_commitments_block()
        assert isinstance(result, str)
