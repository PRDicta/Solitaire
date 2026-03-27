"""
Tests for the Solitaire universal updater.

Covers: target state detection, circuit breaker, data-first backup,
migration execution, and update checker.
"""
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add updater/ to path so we can import update.py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "updater"))

import update as updater
from solitaire.core.update_checker import UpdateChecker, parse_semver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with a seeded rolodex.db and personas/."""
    db_path = tmp_path / "rolodex.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE rolodex_entries "
        "(id INTEGER PRIMARY KEY, content TEXT, topic TEXT)"
    )
    conn.execute(
        "CREATE TABLE conversations "
        "(id TEXT PRIMARY KEY, started_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE identity_nodes "
        "(id TEXT PRIMARY KEY, content TEXT, node_type TEXT)"
    )
    for i in range(10):
        conn.execute(
            "INSERT INTO rolodex_entries VALUES (?, ?, ?)",
            (i, f"entry {i}", "test"),
        )
    conn.commit()
    conn.close()

    # Create personas/
    personas_dir = tmp_path / "personas" / "chief"
    personas_dir.mkdir(parents=True)
    (personas_dir / "config.yaml").write_text("name: Chief\n")
    (personas_dir / "identity.json").write_text('{"nodes": []}\n')

    # Create sessions/
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "active.json").write_text('{"id": "123"}\n')

    # Create session marker
    (tmp_path / ".solitaire_session").write_text(
        '{"session_id": "abc", "persona_key": "chief"}\n'
    )

    return tmp_path


@pytest.fixture
def v1_dual_workspace(tmp_path):
    """Create a v1.0.0-style workspace with src/ and starter/."""
    db_path = tmp_path / "rolodex.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE rolodex_entries (id INTEGER PRIMARY KEY, content TEXT)"
    )
    conn.execute(
        "CREATE TABLE conversations (id TEXT PRIMARY KEY)"
    )
    conn.execute(
        "CREATE TABLE identity_nodes (id TEXT PRIMARY KEY, content TEXT)"
    )
    conn.execute("INSERT INTO rolodex_entries VALUES (1, 'test')")
    conn.commit()
    conn.close()

    # v1.0.0 layout
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "__version__.py").write_text('__version__ = "1.0.0"\n')
    (src_dir / "engine.py").write_text("# old engine\n")

    starter_dir = tmp_path / "starter" / "solitaire"
    starter_dir.mkdir(parents=True)
    (starter_dir / "__version__.py").write_text('__version__ = "1.0.0"\n')

    # Legacy files
    (tmp_path / "ai_writing_tells.md").write_text("# tells\n")
    (tmp_path / "FIRST_INTERACTION.md").write_text("# first\n")

    return tmp_path


@pytest.fixture
def update_package(tmp_path):
    """Create a mock update package with solitaire/ code."""
    pkg_dir = tmp_path / "update_pkg"
    sol_dir = pkg_dir / "solitaire"
    sol_dir.mkdir(parents=True)
    (sol_dir / "__version__.py").write_text('__version__ = "1.3.0"\n')
    (sol_dir / "__init__.py").write_text("")
    (sol_dir / "engine.py").write_text("# new engine\n")
    (pkg_dir / "pyproject.toml").write_text('[project]\nname = "solitaire"\n')
    (pkg_dir / "README.md").write_text("# Solitaire\n")
    return pkg_dir


# ---------------------------------------------------------------------------
# detect_target_state tests
# ---------------------------------------------------------------------------

class TestDetectTargetState:
    def test_unified_layout(self, workspace):
        """v1.1.0+ workspace with solitaire/ directory."""
        sol_dir = workspace / "solitaire"
        sol_dir.mkdir()
        (sol_dir / "__version__.py").write_text('__version__ = "1.2.0"\n')

        state = updater.detect_target_state(workspace)
        assert state["layout"] == "v1_unified"
        assert state["version"] == "1.2.0"
        assert "solitaire/" in state["code_locations"]
        assert state["has_rolodex"] is True
        assert state["rolodex_entry_count"] == 10
        assert state["has_personas"] is True
        assert state["persona_count"] > 0

    def test_dual_tree_layout(self, v1_dual_workspace):
        """v1.0.0 workspace with src/ and starter/."""
        state = updater.detect_target_state(v1_dual_workspace)
        assert state["layout"] == "v1_dual"
        assert state["version"] == "1.0.0"
        assert "src/" in state["code_locations"]
        assert state["has_rolodex"] is True
        assert "ai_writing_tells.md" in state["legacy_files"]

    def test_unknown_layout(self, tmp_path):
        """Workspace with rolodex.db but no recognizable code layout."""
        db_path = tmp_path / "rolodex.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

        state = updater.detect_target_state(tmp_path)
        assert state["layout"] == "unknown"
        assert state["version"] is None
        assert state["has_rolodex"] is True

    def test_empty_workspace(self, tmp_path):
        """Completely empty directory."""
        state = updater.detect_target_state(tmp_path)
        assert state["layout"] == "unknown"
        assert state["version"] is None
        assert state["has_rolodex"] is False
        assert state["has_personas"] is False


# ---------------------------------------------------------------------------
# CircuitBreaker tests
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_single_failure_does_not_trip(self):
        cb = updater.CircuitBreaker(max_consecutive=2)
        cb.record_failure("step1", "error1")
        assert cb.should_abort() is False

    def test_two_consecutive_failures_trips(self):
        cb = updater.CircuitBreaker(max_consecutive=2)
        cb.record_failure("step1", "error1")
        cb.record_failure("step2", "error2")
        assert cb.should_abort() is True

    def test_success_resets_counter(self):
        cb = updater.CircuitBreaker(max_consecutive=2)
        cb.record_failure("step1", "error1")
        cb.record_success("step2")
        cb.record_failure("step3", "error3")
        assert cb.should_abort() is False

    def test_summary_shows_all_failures(self):
        cb = updater.CircuitBreaker(max_consecutive=5)
        cb.record_failure("a", "err_a")
        cb.record_success("b")
        cb.record_failure("c", "err_c")
        summary = cb.summary()
        assert "a" in summary
        assert "c" in summary
        assert "err_a" in summary


# ---------------------------------------------------------------------------
# backup_user_data tests
# ---------------------------------------------------------------------------

class TestBackupUserData:
    def test_creates_atomic_sqlite_backup(self, workspace):
        """Backup produces a valid SQLite file with all tables."""
        backup_dir, report = updater.backup_user_data(workspace)

        assert backup_dir.exists()
        assert isinstance(report["rolodex"], dict)

        # Verify the backup is a valid SQLite DB
        backup_db = backup_dir / "rolodex.db"
        assert backup_db.exists()
        conn = sqlite3.connect(str(backup_db))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "rolodex_entries" in tables
        assert "identity_nodes" in tables
        count = conn.execute("SELECT COUNT(*) FROM rolodex_entries").fetchone()[0]
        assert count == 10
        conn.close()

    def test_backs_up_personas(self, workspace):
        """Personas directory is copied to backup."""
        backup_dir, report = updater.backup_user_data(workspace)
        assert isinstance(report["personas"], dict)
        assert report["personas"]["files"] > 0
        assert (backup_dir / "personas" / "chief" / "config.yaml").exists()

    def test_backs_up_sessions(self, workspace):
        """Sessions directory is copied to backup."""
        backup_dir, report = updater.backup_user_data(workspace)
        assert isinstance(report["sessions"], dict)
        assert (backup_dir / "sessions" / "active.json").exists()

    def test_backs_up_session_marker(self, workspace):
        """Session marker file is copied to backup."""
        backup_dir, report = updater.backup_user_data(workspace)
        assert report["session_marker"] is not None
        assert report["session_marker"] != "not_present"
        assert (backup_dir / ".solitaire_session").exists()

    def test_handles_missing_rolodex(self, tmp_path):
        """No crash when rolodex.db doesn't exist."""
        backup_dir, report = updater.backup_user_data(tmp_path)
        assert report["rolodex"] == "not_present"

    def test_handles_missing_personas(self, tmp_path):
        """No crash when personas/ doesn't exist."""
        # Create a minimal DB so backup_user_data has something
        db = tmp_path / "rolodex.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE rolodex_entries (id INTEGER)")
        conn.execute("CREATE TABLE conversations (id TEXT)")
        conn.execute("CREATE TABLE identity_nodes (id TEXT)")
        conn.commit()
        conn.close()

        backup_dir, report = updater.backup_user_data(tmp_path)
        assert report["personas"] == "not_present"

    def test_shared_timestamp(self, workspace):
        """All backup items share the same timestamp directory."""
        backup_dir, _ = updater.backup_user_data(workspace)
        # The backup dir name contains a single timestamp
        assert "pre-update-data-" in backup_dir.name


# ---------------------------------------------------------------------------
# execute_migration tests
# ---------------------------------------------------------------------------

class TestExecuteMigration:
    def test_all_steps_succeed(self, workspace, update_package):
        """All migration steps complete successfully."""
        # Create solitaire/ so v1_unified steps can remove it
        sol_dir = workspace / "solitaire"
        sol_dir.mkdir()
        (sol_dir / "__version__.py").write_text('__version__ = "1.2.0"\n')

        pre_state = updater.detect_target_state(workspace)
        steps = updater.MIGRATIONS["v1_unified"]
        breaker = updater.CircuitBreaker(max_consecutive=2)

        completed = updater.execute_migration(
            steps, workspace, update_package, breaker, pre_state=pre_state
        )
        assert len(completed) == len(steps)
        step_names = [name for name, _ in completed]
        assert "copy_new_code" in step_names
        assert "verify_data_intact" in step_names

    def test_circuit_breaker_aborts(self, workspace, update_package):
        """Two consecutive failures trigger UpdateAborted."""
        def always_fail(ws, ud, ctx):
            raise RuntimeError("intentional failure")

        steps = [
            ("fail1", always_fail),
            ("fail2", always_fail),
            ("never_reached", always_fail),
        ]
        breaker = updater.CircuitBreaker(max_consecutive=2)

        with pytest.raises(updater.UpdateAborted) as exc_info:
            updater.execute_migration(
                steps, workspace, update_package, breaker
            )
        assert "Circuit breaker" in str(exc_info.value)

    def test_single_failure_continues(self, workspace, update_package):
        """A single failure does not abort if followed by success."""
        results = []

        def fail_once(ws, ud, ctx):
            raise RuntimeError("one-time fail")

        def succeed(ws, ud, ctx):
            results.append("ok")
            return "success"

        steps = [
            ("will_fail", fail_once),
            ("will_succeed", succeed),
        ]
        breaker = updater.CircuitBreaker(max_consecutive=2)

        completed = updater.execute_migration(
            steps, workspace, update_package, breaker
        )
        # Only the success step is in completed
        assert len(completed) == 1
        assert completed[0][0] == "will_succeed"


# ---------------------------------------------------------------------------
# verify_data_intact tests
# ---------------------------------------------------------------------------

class TestVerifyDataIntact:
    def test_data_intact(self, workspace):
        """No issues when all data is present."""
        pre_state = updater.detect_target_state(workspace)
        # Add solitaire/ for layout detection
        sol_dir = workspace / "solitaire"
        sol_dir.mkdir(exist_ok=True)
        (sol_dir / "__version__.py").write_text('__version__ = "1.2.0"\n')
        pre_state = updater.detect_target_state(workspace)

        ctx = {"pre_state": pre_state}
        result = updater.verify_data_intact(workspace, workspace, ctx)
        assert "intact" in result.lower()

    def test_missing_rolodex_raises(self, workspace):
        """Raises if rolodex.db disappears during migration."""
        pre_state = updater.detect_target_state(workspace)
        # Simulate rolodex.db being deleted
        (workspace / "rolodex.db").unlink()

        ctx = {"pre_state": pre_state}
        with pytest.raises(RuntimeError, match="MISSING"):
            updater.verify_data_intact(workspace, workspace, ctx)

    def test_missing_personas_raises(self, workspace):
        """Raises if personas/ disappears during migration."""
        pre_state = updater.detect_target_state(workspace)
        # Remove personas
        import shutil
        shutil.rmtree(workspace / "personas")

        ctx = {"pre_state": pre_state}
        with pytest.raises(RuntimeError, match="MISSING"):
            updater.verify_data_intact(workspace, workspace, ctx)


# ---------------------------------------------------------------------------
# UpdateChecker tests
# ---------------------------------------------------------------------------

class TestParseVersion:
    def test_valid_versions(self):
        assert parse_semver("1.2.3") == (1, 2, 3)
        assert parse_semver("0.0.1") == (0, 0, 1)
        assert parse_semver("10.20.30") == (10, 20, 30)

    def test_invalid_version(self):
        assert parse_semver("") == (0, 0, 0)
        assert parse_semver(None) == (0, 0, 0)
        assert parse_semver("abc") == (0, 0, 0)

    def test_comparison(self):
        assert parse_semver("1.3.0") > parse_semver("1.2.0")
        assert parse_semver("2.0.0") > parse_semver("1.9.9")
        assert parse_semver("1.2.0") == parse_semver("1.2.0")


class TestUpdateChecker:
    def test_snooze_prevents_check(self, tmp_path):
        """Snoozed checker returns update_available=False."""
        checker = UpdateChecker(tmp_path, "1.2.0")
        checker.snooze(days=7)

        result = checker.check()
        assert result is not None
        assert result["update_available"] is False
        assert result.get("reason") == "snoozed"

    def test_clear_snooze(self, tmp_path):
        """Clearing snooze allows checks to proceed."""
        checker = UpdateChecker(tmp_path, "1.2.0")
        checker.snooze(days=7)
        checker.clear_snooze()

        assert not checker._is_snoozed()

    def test_expired_snooze_clears(self, tmp_path):
        """Expired snooze is automatically cleared."""
        checker = UpdateChecker(tmp_path, "1.2.0")
        # Write an already-expired snooze
        snooze_data = {
            "snooze_until": "2020-01-01T00:00:00+00:00",
            "snooze_version": None,
            "created_at": "2020-01-01T00:00:00+00:00",
        }
        (tmp_path / ".update_snooze.json").write_text(
            json.dumps(snooze_data), encoding="utf-8"
        )
        assert not checker._is_snoozed()

    def test_cache_is_used(self, tmp_path):
        """Cached results are returned without network call."""
        checker = UpdateChecker(tmp_path, "1.2.0")
        # Write a cache entry
        from datetime import datetime, timezone
        cache_data = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "result": {
                "update_available": True,
                "current_version": "1.2.0",
                "latest_version": "1.3.0",
                "download_url": "https://example.com/update.zip",
                "release_notes": "New features",
                "html_url": "https://github.com/example",
            },
        }
        (tmp_path / ".update_cache.json").write_text(
            json.dumps(cache_data), encoding="utf-8"
        )

        result = checker.check()
        assert result["update_available"] is True
        assert result["latest_version"] == "1.3.0"

    @patch("solitaire.core.update_checker.urlopen")
    def test_network_error_returns_none(self, mock_urlopen, tmp_path):
        """Network failure returns None (non-fatal)."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("network error")

        checker = UpdateChecker(tmp_path, "1.2.0")
        result = checker.check()
        assert result is None


# ---------------------------------------------------------------------------
# Full v1.0.0 -> current migration test
# ---------------------------------------------------------------------------

class TestV1DualMigration:
    def test_full_migration(self, v1_dual_workspace, update_package):
        """v1.0.0 workspace migrates cleanly to current."""
        pre_state = updater.detect_target_state(v1_dual_workspace)
        assert pre_state["layout"] == "v1_dual"

        # Run backup gate first
        backup_dir, report = updater.backup_user_data(v1_dual_workspace)
        assert backup_dir.exists()

        # Run migration
        steps = updater.MIGRATIONS["v1_dual"]
        breaker = updater.CircuitBreaker(max_consecutive=2)
        completed = updater.execute_migration(
            steps, v1_dual_workspace, update_package, breaker,
            pre_state=pre_state,
        )

        # Verify: src/ and starter/ removed
        assert not (v1_dual_workspace / "src").exists()
        assert not (v1_dual_workspace / "starter").exists()

        # Verify: legacy files removed
        assert not (v1_dual_workspace / "ai_writing_tells.md").exists()
        assert not (v1_dual_workspace / "FIRST_INTERACTION.md").exists()

        # Verify: new solitaire/ installed
        assert (v1_dual_workspace / "solitaire" / "__version__.py").exists()
        new_version = updater.read_version(v1_dual_workspace / "solitaire")
        assert new_version == "1.3.0"

        # Verify: rolodex.db untouched
        assert (v1_dual_workspace / "rolodex.db").exists()
        conn = sqlite3.connect(str(v1_dual_workspace / "rolodex.db"))
        count = conn.execute(
            "SELECT COUNT(*) FROM rolodex_entries"
        ).fetchone()[0]
        conn.close()
        assert count == 1  # original data preserved
