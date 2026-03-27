"""
Tests for the Solitaire rolling backup system.

Covers: BackupManager staleness checks, backup creation,
rotation, listing, and config integration.
"""
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from solitaire.storage.backup import BackupManager
from solitaire.utils.config import LibrarianConfig


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace with a seeded rolodex.db."""
    db_path = tmp_path / "rolodex.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test_data (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO test_data VALUES (1, 'hello')")
    conn.execute("INSERT INTO test_data VALUES (2, 'world')")
    conn.commit()
    conn.close()
    return tmp_path


@pytest.fixture
def manager(workspace):
    """BackupManager with default settings."""
    return BackupManager(
        workspace_dir=workspace,
        db_path=workspace / "rolodex.db",
        retention_count=3,
        max_age_hours=24.0,
    )


class TestNeedsBackup:
    def test_needs_backup_no_backups_exist(self, manager):
        assert manager.needs_backup() is True

    def test_needs_backup_recent_backup_exists(self, manager):
        manager.create_backup()
        assert manager.needs_backup() is False

    def test_needs_backup_stale_backup(self, workspace):
        bm = BackupManager(
            workspace_dir=workspace,
            db_path=workspace / "rolodex.db",
            max_age_hours=0.0,  # Any backup is immediately stale
        )
        bm.create_backup()
        # With max_age_hours=0, even a fresh backup is "stale"
        assert bm.needs_backup() is True

    def test_needs_backup_disabled(self, workspace):
        bm = BackupManager(
            workspace_dir=workspace,
            db_path=workspace / "rolodex.db",
            enabled=False,
        )
        assert bm.needs_backup() is False

    def test_needs_backup_no_database(self, tmp_path):
        bm = BackupManager(
            workspace_dir=tmp_path,
            db_path=tmp_path / "nonexistent.db",
        )
        assert bm.needs_backup() is False


class TestCreateBackup:
    def test_creates_valid_sqlite_file(self, manager, workspace):
        result = manager.create_backup()
        assert result["status"] == "ok"
        assert result["size_bytes"] > 0

        # Verify the backup is a valid SQLite DB with our test data
        backup_path = result["path"]
        conn = sqlite3.connect(backup_path)
        rows = conn.execute("SELECT value FROM test_data ORDER BY id").fetchall()
        conn.close()
        assert rows == [("hello",), ("world",)]

    def test_backup_naming_convention(self, manager):
        result = manager.create_backup()
        assert result["filename"].startswith("rolodex_")
        assert result["filename"].endswith(".db")

    def test_backup_dir_created_automatically(self, manager):
        assert not manager.backup_dir.exists()
        manager.create_backup()
        assert manager.backup_dir.exists()

    def test_no_database_returns_skip(self, tmp_path):
        bm = BackupManager(
            workspace_dir=tmp_path,
            db_path=tmp_path / "nonexistent.db",
        )
        result = bm.create_backup()
        assert result["status"] == "skip"

    def test_multiple_backups_have_unique_names(self, manager):
        r1 = manager.create_backup()
        # Ensure different timestamp
        time.sleep(1.1)
        r2 = manager.create_backup()
        assert r1["filename"] != r2["filename"]


class TestRotation:
    def test_rotate_keeps_retention_count(self, manager):
        for _ in range(5):
            manager.create_backup()
            time.sleep(0.1)  # Ensure distinct timestamps

        files_before = manager._backup_files()
        assert len(files_before) == 5

        deleted = manager.rotate()
        assert len(deleted) == 2

        files_after = manager._backup_files()
        assert len(files_after) == 3

    def test_rotate_preserves_newest(self, manager):
        names = []
        for _ in range(4):
            r = manager.create_backup()
            names.append(r["filename"])
            time.sleep(0.1)

        manager.rotate()
        remaining = [f.name for f in manager._backup_files()]
        # The 3 newest should survive
        assert names[-1] in remaining
        assert names[-2] in remaining
        assert names[-3] in remaining
        assert names[0] not in remaining

    def test_rotate_noop_under_limit(self, manager):
        manager.create_backup()
        deleted = manager.rotate()
        assert deleted == []

    def test_retention_count_minimum_is_one(self, workspace):
        bm = BackupManager(
            workspace_dir=workspace,
            db_path=workspace / "rolodex.db",
            retention_count=0,  # Should be clamped to 1
        )
        assert bm.retention_count == 1


class TestCheckAndBackup:
    def test_creates_backup_when_none_exist(self, manager):
        result = manager.check_and_backup()
        assert result["status"] == "ok"
        assert len(manager._backup_files()) == 1

    def test_skips_when_current(self, manager):
        manager.create_backup()
        result = manager.check_and_backup()
        assert result["status"] == "current"

    def test_returns_disabled_when_off(self, workspace):
        bm = BackupManager(
            workspace_dir=workspace,
            db_path=workspace / "rolodex.db",
            enabled=False,
        )
        result = bm.check_and_backup()
        assert result["status"] == "disabled"

    def test_rotates_after_backup(self, workspace):
        bm = BackupManager(
            workspace_dir=workspace,
            db_path=workspace / "rolodex.db",
            retention_count=2,
            max_age_hours=0.0,
        )
        # Create 3 backups (all immediately stale with max_age_hours=0)
        for _ in range(3):
            bm.create_backup()
            time.sleep(0.1)

        result = bm.check_and_backup()
        assert result["status"] == "ok"
        assert len(bm._backup_files()) == 2


class TestListBackups:
    def test_empty_when_no_backups(self, manager):
        assert manager.list_backups() == []

    def test_returns_metadata(self, manager):
        manager.create_backup()
        backups = manager.list_backups()
        assert len(backups) == 1
        assert "filename" in backups[0]
        assert "size_bytes" in backups[0]
        assert "modified" in backups[0]

    def test_newest_first_ordering(self, manager):
        manager.create_backup()
        time.sleep(0.1)
        manager.create_backup()
        backups = manager.list_backups()
        assert len(backups) == 2
        # First entry should be newer
        assert backups[0]["modified"] >= backups[1]["modified"]


class TestConfigIntegration:
    def test_from_config(self, workspace):
        config = LibrarianConfig(
            backup_enabled=True,
            backup_retention_count=5,
            backup_max_age_hours=12.0,
        )
        bm = BackupManager.from_config(workspace, config)
        assert bm.retention_count == 5
        assert bm.max_age_hours == 12.0
        assert bm.enabled is True
        assert bm.db_path == workspace / "rolodex.db"

    def test_config_defaults(self):
        config = LibrarianConfig()
        assert config.backup_enabled is True
        assert config.backup_retention_count == 3
        assert config.backup_max_age_hours == 24.0
