"""
Tests for the cognitive profile rendering in SolitaireEngine.

Covers:
- 5-band trait mapping (very_high / high / moderate / low / very_low)
- Texture layer (signal-history-driven trait specificity)
- Growth milestones (recent trait movement)
- Drift display
- Pluralization (session vs sessions)
"""
import sqlite3
import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

from solitaire.core.persona import (
    PersonaProfile,
    TraitProfile,
    PersonaIdentity,
    DomainEnvelope,
    VALID_TRAIT_NAMES,
)
from solitaire.engine import SolitaireEngine


# ─── Helpers ──────────────────────────────────────────────────────────────

def _make_engine_with_persona(traits: Dict[str, float], baseline: Optional[Dict[str, float]] = None) -> SolitaireEngine:
    """Create a SolitaireEngine with a mocked persona, bypassing full boot."""
    engine = object.__new__(SolitaireEngine)
    engine._lib = MagicMock()

    trait_profile = TraitProfile(**{t: traits.get(t, 0.5) for t in VALID_TRAIT_NAMES})
    baseline_profile = TraitProfile(**{t: (baseline or traits).get(t, 0.5) for t in VALID_TRAIT_NAMES})

    persona = PersonaProfile(
        identity=PersonaIdentity(name="TestPersona", role="test-role"),
        traits=trait_profile,
        domain=DomainEnvelope(primary="testing", secondary=["validation"]),
    )
    persona._baseline_traits = baseline_profile

    engine._lib.persona = persona
    # No rolodex connection by default (texture/milestones degrade gracefully)
    engine._lib.rolodex = MagicMock()
    engine._lib.rolodex.conn = MagicMock()
    engine._lib.rolodex.conn.execute = MagicMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))

    return engine


# ─── Band Selection ──────────────────────────────────────────────────────

class TestBandSelection:
    """5-band trait mapping: very_high / high / moderate / low / very_low."""

    def test_very_high_band(self):
        """Traits >= 0.85 render as 'defining trait'."""
        engine = _make_engine_with_persona({"observance": 0.90})
        profile = engine._build_cognitive_profile()
        assert "defining trait" in profile
        assert "flags patterns, anomalies, and edge cases before they surface" in profile

    def test_high_band(self):
        """Traits 0.65-0.84 render as the 'high' description."""
        engine = _make_engine_with_persona({"observance": 0.75})
        profile = engine._build_cognitive_profile()
        assert "highly observant" in profile
        assert "defining trait" not in profile or "highly observant" in profile

    def test_moderate_band(self):
        """Traits 0.40-0.64 render as 'moderate'."""
        engine = _make_engine_with_persona({"warmth": 0.55})
        profile = engine._build_cognitive_profile()
        assert "moderate warmth" in profile

    def test_low_band(self):
        """Traits 0.20-0.39 render as 'low'."""
        engine = _make_engine_with_persona({"humor": 0.25})
        profile = engine._build_cognitive_profile()
        assert "serious-toned" in profile

    def test_very_low_band(self):
        """Traits < 0.20 render as 'very_low'."""
        engine = _make_engine_with_persona({"humor": 0.10})
        profile = engine._build_cognitive_profile()
        assert "no humor" in profile

    def test_boundary_085(self):
        """Exactly 0.85 should be very_high."""
        engine = _make_engine_with_persona({"conviction": 0.85})
        profile = engine._build_cognitive_profile()
        assert "defining trait" in profile
        assert "holds positions firmly" in profile

    def test_boundary_065(self):
        """Exactly 0.65 should be high."""
        engine = _make_engine_with_persona({"empathy": 0.65})
        profile = engine._build_cognitive_profile()
        assert "empathetic" in profile

    def test_boundary_040(self):
        """Exactly 0.40 should be moderate."""
        engine = _make_engine_with_persona({"assertiveness": 0.40})
        profile = engine._build_cognitive_profile()
        assert "measured directness" in profile

    def test_boundary_020(self):
        """Exactly 0.20 should be low."""
        engine = _make_engine_with_persona({"initiative": 0.20})
        profile = engine._build_cognitive_profile()
        assert "responsive" in profile
        assert "passive" not in profile

    def test_all_seven_traits_rendered(self):
        """All 7 traits must appear in the profile regardless of value."""
        engine = _make_engine_with_persona({
            "observance": 0.90,
            "assertiveness": 0.80,
            "conviction": 0.85,
            "warmth": 0.55,
            "humor": 0.50,
            "initiative": 0.85,
            "empathy": 0.65,
        })
        profile = engine._build_cognitive_profile()
        # Count disposition lines (each starts with "- ")
        disp_lines = [l for l in profile.split("\n") if l.strip().startswith("- ")]
        assert len(disp_lines) == 7

    def test_all_bands_have_unique_descriptions(self):
        """Each trait at each band should produce a distinct description."""
        test_values = [0.95, 0.75, 0.50, 0.30, 0.10]
        descriptions = set()
        for val in test_values:
            engine = _make_engine_with_persona({"observance": val})
            profile = engine._build_cognitive_profile()
            # Extract the observance line
            for line in profile.split("\n"):
                if line.strip().startswith("- ") and ("observ" in line.lower() or "defining" in line.lower() or "narrowly" in line.lower()):
                    descriptions.add(line.strip())
                    break
        assert len(descriptions) == 5, f"Expected 5 unique descriptions, got {len(descriptions)}: {descriptions}"


# ─── Profile Structure ───────────────────────────────────────────────────

class TestProfileStructure:
    """Structural elements of the cognitive profile block."""

    def test_header_and_footer(self):
        """Profile has matching header and footer."""
        engine = _make_engine_with_persona({"observance": 0.5})
        profile = engine._build_cognitive_profile()
        assert "═══ COGNITIVE PROFILE ═══" in profile
        assert "═══ END COGNITIVE PROFILE ═══" in profile

    def test_identity_line(self):
        """Identity renders name and role."""
        engine = _make_engine_with_persona({"observance": 0.5})
        profile = engine._build_cognitive_profile()
        assert "Identity: TestPersona (test-role)" in profile

    def test_domain_line(self):
        """Domain renders when domain_scope is set."""
        engine = _make_engine_with_persona({"observance": 0.5})
        # Set domain_scope on identity
        engine._lib.persona.identity.domain_scope = "test-domain"
        profile = engine._build_cognitive_profile()
        assert "Primary domain: test-domain" in profile

    def test_no_persona_returns_empty(self):
        """No persona loaded returns empty string."""
        engine = object.__new__(SolitaireEngine)
        engine._lib = MagicMock()
        engine._lib.persona = None
        profile = engine._build_cognitive_profile()
        assert profile == ""


# ─── Drift Display ───────────────────────────────────────────────────────

class TestDriftDisplay:
    """Drift direction indicator on trait lines."""

    def test_positive_drift_shown(self):
        """When effective > baseline, drift arrow appears."""
        engine = _make_engine_with_persona(
            traits={"conviction": 0.88},
            baseline={"conviction": 0.85},
        )
        profile = engine._build_cognitive_profile()
        assert "drift:" in profile
        assert "↑" in profile

    def test_negative_drift_shown(self):
        """When effective < baseline, downward drift shown."""
        engine = _make_engine_with_persona(
            traits={"initiative": 0.80},
            baseline={"initiative": 0.85},
        )
        profile = engine._build_cognitive_profile()
        assert "↓" in profile

    def test_no_drift_no_indicator(self):
        """When effective == baseline, no drift indicator."""
        engine = _make_engine_with_persona(
            traits={"warmth": 0.55},
            baseline={"warmth": 0.55},
        )
        profile = engine._build_cognitive_profile()
        assert "drift:" not in profile


# ─── Texture Layer ────────────────────────────────────────────────────────

class TestTextureLayer:
    """Signal-history-driven trait texture."""

    def test_texture_renders_when_drift_data_exists(self):
        """Texture lines appear when drift entries provide signal stats."""
        engine = _make_engine_with_persona({"observance": 0.90})

        # Build drift entries using an in-memory SQLite DB for realistic Row objects
        import json
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE drift (content TEXT, created_at TEXT, session_id TEXT)")
        for i in range(5):
            conn.execute(
                "INSERT INTO drift VALUES (?, ?, ?)",
                (
                    json.dumps({
                        "signal": "positive_acknowledgment",
                        "traits_affected": {"observance": 0.01},
                        "active_profile_snapshot": {"observance": 0.90 + i * 0.001},
                        "confidence": 0.8,
                        "reinforcement_count": i + 1,
                        "session_id": f"sess-{i}",
                    }),
                    f"2026-03-{20 + i:02d}T12:00:00",
                    f"sess-{i}",
                ),
            )
        rows = conn.execute("SELECT content, created_at, session_id FROM drift ORDER BY created_at").fetchall()

        engine._lib.rolodex.conn.execute.return_value = MagicMock(
            fetchall=MagicMock(return_value=rows)
        )

        profile = engine._build_cognitive_profile()
        assert "texture:" in profile
        assert "positive acknowledgment" in profile

    def test_texture_graceful_degradation(self):
        """Texture fails gracefully when DB query fails."""
        engine = _make_engine_with_persona({"observance": 0.90})
        engine._lib.rolodex.conn.execute.side_effect = Exception("DB error")

        profile = engine._build_cognitive_profile()
        # Should still render the profile, just without texture
        assert "═══ COGNITIVE PROFILE ═══" in profile
        assert "defining trait" in profile
        assert "texture:" not in profile

    def test_texture_pluralization_single_session(self):
        """'1 session' not '1 sessions'."""
        engine = _make_engine_with_persona({"humor": 0.50})

        import json
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE drift (content TEXT, created_at TEXT, session_id TEXT)")
        conn.execute(
            "INSERT INTO drift VALUES (?, ?, ?)",
            (
                json.dumps({
                    "signal": "humor_landed",
                    "traits_affected": {"humor": 0.01},
                    "active_profile_snapshot": {"humor": 0.51},
                    "confidence": 0.7,
                    "reinforcement_count": 1,
                    "session_id": "single-sess",
                }),
                "2026-03-25T12:00:00",
                "single-sess",
            ),
        )
        rows = conn.execute("SELECT content, created_at, session_id FROM drift").fetchall()

        engine._lib.rolodex.conn.execute.return_value = MagicMock(
            fetchall=MagicMock(return_value=rows)
        )

        profile = engine._build_cognitive_profile()
        if "texture:" in profile:
            assert "1 session," in profile
            assert "1 sessions," not in profile

    def test_texture_pluralization_multiple_sessions(self):
        """'3 sessions' not '3 session'."""
        engine = _make_engine_with_persona({"conviction": 0.85})

        import json
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE drift (content TEXT, created_at TEXT, session_id TEXT)")
        for i in range(5):
            conn.execute(
                "INSERT INTO drift VALUES (?, ?, ?)",
                (
                    json.dumps({
                        "signal": "pushback_accepted",
                        "traits_affected": {"conviction": 0.02},
                        "active_profile_snapshot": {"conviction": 0.85 + i * 0.002},
                        "confidence": 0.9,
                        "reinforcement_count": i + 1,
                        "session_id": f"sess-{i}",
                    }),
                    f"2026-03-{20 + i:02d}T12:00:00",
                    f"sess-{i}",
                ),
            )
        rows = conn.execute("SELECT content, created_at, session_id FROM drift ORDER BY created_at").fetchall()

        engine._lib.rolodex.conn.execute.return_value = MagicMock(
            fetchall=MagicMock(return_value=rows)
        )

        profile = engine._build_cognitive_profile()
        if "texture:" in profile:
            assert "5 sessions" in profile


# ─── Growth Milestones ───────────────────────────────────────────────────

class TestGrowthMilestones:
    """Growth milestone rendering from trait history."""

    def test_milestones_render_when_significant_movement(self):
        """Growth line appears when a trait moves >= 0.005."""
        engine = _make_engine_with_persona(
            traits={"conviction": 0.87},
            baseline={"conviction": 0.85},
        )

        # Mock persona state with trait history
        mock_state = MagicMock()
        mock_state.trait_history = {
            "conviction": [
                {"session_id": "s1", "effective_value": 0.850},
                {"session_id": "s2", "effective_value": 0.855},
                {"session_id": "s3", "effective_value": 0.860},
                {"session_id": "s4", "effective_value": 0.870},
            ]
        }
        engine._lib.persona._state = mock_state

        profile = engine._build_cognitive_profile()
        assert "Recent growth:" in profile
        assert "conviction" in profile.split("Recent growth:")[1]
        assert "reinforced" in profile

    def test_milestones_not_shown_when_movement_too_small(self):
        """No growth line when trait movement < 0.005."""
        engine = _make_engine_with_persona(
            traits={"conviction": 0.8503},
            baseline={"conviction": 0.85},
        )

        mock_state = MagicMock()
        mock_state.trait_history = {
            "conviction": [
                {"session_id": "s1", "effective_value": 0.8500},
                {"session_id": "s2", "effective_value": 0.8501},
                {"session_id": "s3", "effective_value": 0.8503},
            ]
        }
        engine._lib.persona._state = mock_state

        profile = engine._build_cognitive_profile()
        assert "Recent growth:" not in profile

    def test_milestones_graceful_without_state(self):
        """No crash when persona.state is None."""
        engine = _make_engine_with_persona({"observance": 0.90})
        engine._lib.persona._state = None

        profile = engine._build_cognitive_profile()
        assert "═══ COGNITIVE PROFILE ═══" in profile
        assert "Recent growth:" not in profile

    def test_milestones_softened_direction(self):
        """Downward movement shows 'softened'."""
        engine = _make_engine_with_persona(
            traits={"initiative": 0.82},
            baseline={"initiative": 0.85},
        )

        mock_state = MagicMock()
        mock_state.trait_history = {
            "initiative": [
                {"session_id": "s1", "effective_value": 0.850},
                {"session_id": "s2", "effective_value": 0.840},
                {"session_id": "s3", "effective_value": 0.830},
                {"session_id": "s4", "effective_value": 0.820},
            ]
        }
        engine._lib.persona._state = mock_state

        profile = engine._build_cognitive_profile()
        assert "softened" in profile

    def test_milestones_session_pluralization(self):
        """Growth milestones use correct session/sessions."""
        engine = _make_engine_with_persona(
            traits={"warmth": 0.62},
            baseline={"warmth": 0.55},
        )

        mock_state = MagicMock()
        mock_state.trait_history = {
            "warmth": [
                {"session_id": "s1", "effective_value": 0.550},
                {"session_id": "s1", "effective_value": 0.570},
                {"session_id": "s1", "effective_value": 0.620},
            ]
        }
        engine._lib.persona._state = mock_state

        profile = engine._build_cognitive_profile()
        if "Recent growth:" in profile:
            growth_part = profile.split("Recent growth:")[1]
            assert "1 session" in growth_part
            assert "1 sessions" not in growth_part


# ─── Integration: Ward-Like Profile ─────────────────────────────────────

class TestWardProfile:
    """Full integration test matching Ward's actual trait values."""

    def test_ward_trait_mapping(self):
        """Ward's values map to the expected bands."""
        engine = _make_engine_with_persona({
            "observance": 0.90,    # very_high
            "assertiveness": 0.80, # high
            "conviction": 0.85,    # very_high
            "warmth": 0.55,        # moderate
            "humor": 0.50,         # moderate
            "initiative": 0.85,    # very_high
            "empathy": 0.65,       # high
        })
        profile = engine._build_cognitive_profile()

        # Three traits should be "defining trait"
        assert profile.count("defining trait") == 3

        # Assertiveness at 0.80 = high
        assert "direct and concise" in profile

        # Warmth at 0.55 = moderate
        assert "moderate warmth" in profile

        # Humor at 0.50 = moderate
        assert "dry humor" in profile

        # Empathy at 0.65 = high
        assert "empathetic" in profile
