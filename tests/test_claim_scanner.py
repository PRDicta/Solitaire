"""
Tests for the claim scanner.

Part 1: Evaluation gate enhancement (preflight patterns, marker file I/O).
Part 2: Stop hook (transcript scanning, marker file writing).
"""
import json
import os
import tempfile
from pathlib import Path

import pytest

from solitaire.retrieval.evaluation_gate import (
    EvaluationFlag,
    _check_unverified_claims,
    _has_remote_context,
    _has_verification,
    _read_claim_marker,
    _CLAIM_MARKER_DIR,
    evaluate_message,
)


# ---------------------------------------------------------------------------
# Unit tests: pattern detection helpers
# ---------------------------------------------------------------------------


class TestRemoteContext:
    """Tests for _has_remote_context pattern matching."""

    def test_possessive_machine_reference(self):
        assert _has_remote_context("Check her machine for the update")

    def test_named_person_machine(self):
        assert _has_remote_context("Bernie's laptop has the old version")

    def test_brenna_system(self):
        assert _has_remote_context("Brenna's installation is in a weird state")

    def test_their_server(self):
        assert _has_remote_context("The config on their server needs updating")

    def test_another_machine(self):
        assert _has_remote_context("We need to check another machine")

    def test_remote_ssh(self):
        assert _has_remote_context("I'll ssh in and check")

    def test_local_reference_no_match(self):
        assert not _has_remote_context("Delete the old config files")

    def test_generic_discussion_no_match(self):
        assert not _has_remote_context("How do we handle the update?")

    def test_presentation_no_match(self):
        """'her presentation' should not match 'her machine/system'."""
        assert not _has_remote_context("Her presentation isn't working well as a narrative")


class TestVerificationSignals:
    """Tests for _has_verification pattern matching."""

    def test_i_checked(self):
        assert _has_verification("I checked and the service is down")

    def test_i_verified(self):
        assert _has_verification("I verified the port is open")

    def test_output_shows(self):
        assert _has_verification("The output shows three errors")

    def test_log_says(self):
        assert _has_verification("The log says connection refused")

    def test_cant_verify(self):
        assert _has_verification("I can't verify whether it's running")

    def test_no_verification(self):
        assert not _has_verification("The database isn't running on their server")


# ---------------------------------------------------------------------------
# Unit tests: _check_unverified_claims
# ---------------------------------------------------------------------------


class TestCheckUnverifiedClaims:
    """Tests for the main claim scanner function."""

    def test_remote_file_op_triggers_block(self):
        """File ops on remote targets produce a block-severity flag."""
        flags = _check_unverified_claims(
            "Delete the corrupted files from Brenna's laptop"
        )
        cats = [f.category for f in flags]
        assert "unverified_file_op" in cats
        block_flag = next(f for f in flags if f.category == "unverified_file_op")
        assert block_flag.severity == "block"

    def test_remote_state_claim_triggers_warning(self):
        """State assertions about remote systems produce a warning."""
        flags = _check_unverified_claims(
            "The database isn't running on their server"
        )
        cats = [f.category for f in flags]
        assert "unverified_state_claim" in cats
        flag = next(f for f in flags if f.category == "unverified_state_claim")
        assert flag.severity == "warning"

    def test_remote_reference_without_claim_triggers_info(self):
        """Remote reference without state claim or file op gets info flag."""
        flags = _check_unverified_claims(
            "We should look at her machine next"
        )
        cats = [f.category for f in flags]
        assert "unverified_remote" in cats
        flag = next(f for f in flags if f.category == "unverified_remote")
        assert flag.severity == "info"

    def test_screenshot_reasoning_triggers_info(self):
        """Screenshot-based reasoning produces an info flag."""
        flags = _check_unverified_claims(
            "Based on the screenshot, the config is wrong"
        )
        cats = [f.category for f in flags]
        assert "partial_evidence" in cats

    def test_verified_claim_no_flag(self):
        """Remote state claim with verification signal does NOT trigger."""
        flags = _check_unverified_claims(
            "I checked and the database isn't running on their server"
        )
        # Should not have unverified_state_claim or unverified_file_op
        cats = [f.category for f in flags]
        assert "unverified_state_claim" not in cats
        assert "unverified_file_op" not in cats

    def test_local_only_no_flag(self):
        """Local-only references produce no claim flags."""
        flags = _check_unverified_claims("Delete the old config files")
        assert len(flags) == 0

    def test_generic_conversation_no_flag(self):
        """Normal conversation without remote/state patterns is clean."""
        flags = _check_unverified_claims("How should we approach the refactor?")
        assert len(flags) == 0

    def test_it_looks_like_triggers_partial_evidence(self):
        """'It looks like' triggers partial evidence flag."""
        flags = _check_unverified_claims(
            "It looks like their system is misconfigured"
        )
        cats = [f.category for f in flags]
        assert "partial_evidence" in cats

    def test_move_files_on_remote_triggers_block(self):
        """Move operation on remote target triggers block."""
        flags = _check_unverified_claims(
            "Move the old backup files from his machine to the archive"
        )
        cats = [f.category for f in flags]
        assert "unverified_file_op" in cats

    def test_never_applied_on_remote_triggers_warning(self):
        """'never applied' about a remote system triggers warning."""
        flags = _check_unverified_claims(
            "The update never applied on their system"
        )
        cats = [f.category for f in flags]
        assert "unverified_state_claim" in cats


# ---------------------------------------------------------------------------
# Marker file tests
# ---------------------------------------------------------------------------


class TestClaimMarker:
    """Tests for marker file reading/consumption."""

    def test_reads_marker_file(self, tmp_path):
        """Marker file is read and returned as dict."""
        marker_dir = Path(_CLAIM_MARKER_DIR)
        marker_dir.mkdir(parents=True, exist_ok=True)

        import hashlib
        ws = str(tmp_path)
        ws_hash = hashlib.md5(ws.encode()).hexdigest()[:12]
        marker_path = marker_dir / ws_hash

        marker_data = {
            "timestamp": "2026-03-27T14:22:00Z",
            "claims_detected": [{"category": "state_claim", "text": "test"}],
            "summary": "1 unverified claim detected.",
        }
        marker_path.write_text(json.dumps(marker_data), encoding="utf-8")

        result = _read_claim_marker(ws)
        assert result is not None
        assert result["summary"] == "1 unverified claim detected."

    def test_marker_consumed_after_reading(self, tmp_path):
        """Marker file is deleted after being read."""
        marker_dir = Path(_CLAIM_MARKER_DIR)
        marker_dir.mkdir(parents=True, exist_ok=True)

        import hashlib
        ws = str(tmp_path)
        ws_hash = hashlib.md5(ws.encode()).hexdigest()[:12]
        marker_path = marker_dir / ws_hash

        marker_path.write_text(json.dumps({"claims_detected": [{"text": "x"}], "summary": "test"}))
        _read_claim_marker(ws)
        assert not marker_path.exists()

    def test_no_marker_returns_none(self, tmp_path):
        """Missing marker file returns None without error."""
        result = _read_claim_marker(str(tmp_path))
        assert result is None

    def test_no_marker_dir_returns_none(self):
        """Missing marker directory returns None without error."""
        result = _read_claim_marker("/nonexistent/path/workspace")
        assert result is None

    def test_no_workspace_returns_none(self):
        """None workspace returns None."""
        result = _read_claim_marker(None)
        assert result is None

    def test_marker_produces_prior_claims_flag(self, tmp_path):
        """Marker file results in prior_unverified_claims flag."""
        marker_dir = Path(_CLAIM_MARKER_DIR)
        marker_dir.mkdir(parents=True, exist_ok=True)

        import hashlib
        ws = str(tmp_path)
        ws_hash = hashlib.md5(ws.encode()).hexdigest()[:12]
        marker_path = marker_dir / ws_hash

        marker_data = {
            "claims_detected": [{"category": "state_claim"}],
            "summary": "Asserted database state without checking.",
        }
        marker_path.write_text(json.dumps(marker_data), encoding="utf-8")

        flags = _check_unverified_claims("What should we do next?", workspace_dir=ws)
        cats = [f.category for f in flags]
        assert "prior_unverified_claims" in cats
        flag = next(f for f in flags if f.category == "prior_unverified_claims")
        assert flag.severity == "warning"
        assert "database state" in flag.detail


# ---------------------------------------------------------------------------
# Integration: evaluate_message with claim scanner
# ---------------------------------------------------------------------------


class TestEvaluateMessageClaims:
    """Tests for claim scanner integration in evaluate_message."""

    def test_unverified_file_op_blocks_proceed(self):
        """File op on unverified remote system sets proceed=False."""
        result = evaluate_message(
            "Delete the old files from Brenna's laptop"
        )
        assert result.proceed is False
        assert any(f.category == "unverified_file_op" for f in result.flags)

    def test_context_block_includes_claim_check(self):
        """Claim flags produce a CLAIM CHECK section in the context block."""
        result = evaluate_message(
            "The update never applied on their system"
        )
        assert "CLAIM CHECK" in result.context_block
        assert "Stop. Think. Check. Be Sure." in result.context_block

    def test_clean_message_no_claim_block(self):
        """Normal message produces no claim-related context block content."""
        result = evaluate_message("How should we approach the refactor?")
        assert "CLAIM CHECK" not in result.context_block

    def test_workspace_dir_passed_through(self, tmp_path):
        """workspace_dir parameter is accepted without error."""
        result = evaluate_message(
            "Check what's happening",
            workspace_dir=str(tmp_path),
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Part 2: Stop hook tests
# ---------------------------------------------------------------------------

# Import the hook as a module
import importlib.util
_hook_path = str(Path(__file__).resolve().parent.parent / "skill" / "hooks" / "claude-code-claim-scanner.py")
_spec = importlib.util.spec_from_file_location("claim_scanner_hook", _hook_path)
_hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hook)


def _make_transcript(tmp_path, assistant_text, user_text="How is it going?"):
    """Helper: write a minimal JSONL transcript."""
    transcript = tmp_path / "transcript.jsonl"
    lines = []
    if user_text:
        lines.append(json.dumps({
            "type": "user",
            "message": {"content": user_text},
        }))
    lines.append(json.dumps({
        "type": "assistant",
        "message": {"content": assistant_text},
    }))
    transcript.write_text("\n".join(lines), encoding="utf-8")
    return str(transcript)


class TestStopHookScanner:
    """Tests for the Stop hook's scan_for_claims function."""

    def test_state_claim_detected(self):
        """Definitive state claim about remote system is caught."""
        text = "The database isn't running on their server. We should restart it."
        claims = _hook.scan_for_claims(text)
        assert len(claims) > 0
        assert any(c["category"] == "state_claim" for c in claims)

    def test_file_op_recommendation_detected(self):
        """File op recommendation targeting remote system is caught."""
        text = "You should delete the old config files from Brenna's machine."
        claims = _hook.scan_for_claims(text)
        assert len(claims) > 0
        assert any(c["category"] == "file_op_recommendation" for c in claims)

    def test_narrative_construction_detected(self):
        """Narrative construction about remote state is caught."""
        text = "What happened is their server crashed and the data was lost."
        claims = _hook.scan_for_claims(text)
        assert len(claims) > 0
        assert any(c["category"] == "narrative_construction" for c in claims)

    def test_verified_response_passes(self):
        """Response with verification language is not flagged."""
        text = "I checked and the database isn't running on their server."
        claims = _hook.scan_for_claims(text)
        assert len(claims) == 0

    def test_cant_verify_passes(self):
        """Response with explicit uncertainty is not flagged."""
        text = "I can't verify whether their server is running. We should check."
        claims = _hook.scan_for_claims(text)
        assert len(claims) == 0

    def test_local_only_passes(self):
        """Response about local system is not flagged."""
        text = "The database isn't running. Let me restart it for you."
        claims = _hook.scan_for_claims(text)
        assert len(claims) == 0

    def test_short_response_skipped(self):
        """Response under 20 chars is skipped."""
        claims = _hook.scan_for_claims("ok, done")
        assert len(claims) == 0

    def test_none_response_skipped(self):
        """None response is handled gracefully."""
        claims = _hook.scan_for_claims(None)
        assert len(claims) == 0


class TestStopHookTranscript:
    """Tests for transcript reading."""

    def test_extracts_last_assistant(self, tmp_path):
        """Extracts the last assistant message from transcript."""
        path = _make_transcript(tmp_path, "Their server is down.")
        text = _hook.extract_last_assistant(path)
        assert text == "Their server is down."

    def test_missing_transcript(self):
        """Missing transcript returns None."""
        assert _hook.extract_last_assistant("/nonexistent/path") is None

    def test_empty_transcript(self, tmp_path):
        """Empty transcript returns None."""
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert _hook.extract_last_assistant(str(p)) is None


class TestStopHookMarkerWrite:
    """Tests for marker file writing."""

    def test_writes_marker(self, tmp_path):
        """Claims produce a marker file."""
        claims = [{"category": "state_claim", "text": "server is down"}]
        _hook.write_marker(claims, workspace=str(tmp_path))

        import hashlib
        ws_hash = hashlib.md5(str(tmp_path).encode()).hexdigest()[:12]
        marker_path = Path(_hook.MARKER_DIR) / ws_hash
        assert marker_path.exists()
        data = json.loads(marker_path.read_text())
        assert len(data["claims_detected"]) == 1
        assert "state_claim" in data["categories"]
        # Cleanup
        marker_path.unlink()

    def test_marker_has_summary(self, tmp_path):
        """Marker file includes a human-readable summary."""
        claims = [
            {"category": "state_claim", "text": "database isn't running"},
            {"category": "narrative_construction", "text": "what happened is"},
        ]
        _hook.write_marker(claims, workspace=str(tmp_path))

        import hashlib
        ws_hash = hashlib.md5(str(tmp_path).encode()).hexdigest()[:12]
        marker_path = Path(_hook.MARKER_DIR) / ws_hash
        data = json.loads(marker_path.read_text())
        assert "2 unverified claim" in data["summary"]
        # Cleanup
        marker_path.unlink()
