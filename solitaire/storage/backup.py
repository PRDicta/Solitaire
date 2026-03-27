"""
Solitaire — Rolling Backup Manager

Atomic SQLite backups with configurable retention and staleness checks.
Triggered on boot when the latest backup exceeds max_age_hours.
"""
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BackupManager:
    """Manages rolling SQLite backups for the rolodex database."""

    BACKUP_PREFIX = "rolodex_"
    BACKUP_SUFFIX = ".db"

    def __init__(
        self,
        workspace_dir: Path,
        db_path: Path,
        retention_count: int = 3,
        max_age_hours: float = 24.0,
        enabled: bool = True,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.backup_dir = self.workspace_dir / "backups"
        self.db_path = Path(db_path)
        self.retention_count = max(1, retention_count)
        self.max_age_hours = max(0.0, max_age_hours)
        self.enabled = enabled

    @classmethod
    def from_config(cls, workspace_dir: Path, config) -> "BackupManager":
        """Create from a LibrarianConfig instance."""
        return cls(
            workspace_dir=workspace_dir,
            db_path=workspace_dir / config.db_path,
            retention_count=config.backup_retention_count,
            max_age_hours=config.backup_max_age_hours,
            enabled=config.backup_enabled,
        )

    def _backup_files(self) -> List[Path]:
        """Return existing backup files sorted newest-first."""
        if not self.backup_dir.exists():
            return []
        files = [
            f for f in self.backup_dir.iterdir()
            if f.name.startswith(self.BACKUP_PREFIX)
            and f.name.endswith(self.BACKUP_SUFFIX)
            and f.is_file()
        ]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return files

    def latest_backup(self) -> Optional[Path]:
        """Return the most recent backup file, or None."""
        files = self._backup_files()
        return files[0] if files else None

    def needs_backup(self) -> bool:
        """True if no backups exist or the latest is older than max_age_hours."""
        if not self.enabled:
            return False
        if not self.db_path.exists():
            return False
        latest = self.latest_backup()
        if latest is None:
            return True
        age_hours = (
            datetime.now(timezone.utc).timestamp() - latest.stat().st_mtime
        ) / 3600
        return age_hours >= self.max_age_hours

    def create_backup(self) -> Dict[str, Any]:
        """Create an atomic SQLite backup using conn.backup().

        Returns:
            Dict with path, size_bytes, timestamp, or error.
        """
        if not self.db_path.exists():
            return {"status": "skip", "reason": "no_database"}

        self.backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_name = f"{self.BACKUP_PREFIX}{timestamp}{self.BACKUP_SUFFIX}"
        backup_path = self.backup_dir / backup_name

        try:
            source = sqlite3.connect(str(self.db_path))
            dest = sqlite3.connect(str(backup_path))
            try:
                source.backup(dest)
            finally:
                dest.close()
                source.close()

            size = backup_path.stat().st_size
            logger.info("Backup created: %s (%d bytes)", backup_path.name, size)
            return {
                "status": "ok",
                "path": str(backup_path),
                "filename": backup_name,
                "size_bytes": size,
                "timestamp": timestamp,
            }
        except Exception as e:
            logger.error("Backup failed: %s", e)
            # Clean up partial backup
            if backup_path.exists():
                try:
                    backup_path.unlink()
                except OSError:
                    pass
            return {"status": "error", "reason": str(e)}

    def rotate(self) -> List[str]:
        """Delete oldest backups exceeding retention_count.

        Returns:
            List of deleted filenames.
        """
        files = self._backup_files()
        deleted = []
        while len(files) > self.retention_count:
            oldest = files.pop()
            try:
                oldest.unlink()
                deleted.append(oldest.name)
                logger.info("Rotated out backup: %s", oldest.name)
            except OSError as e:
                logger.warning("Could not delete %s: %s", oldest.name, e)
        return deleted

    def check_and_backup(self) -> Dict[str, Any]:
        """Staleness check + conditional backup + rotation.

        This is the main entry point, called on boot.

        Returns:
            Dict with action taken and details.
        """
        if not self.enabled:
            return {"status": "disabled"}

        if not self.needs_backup():
            latest = self.latest_backup()
            return {
                "status": "current",
                "latest": latest.name if latest else None,
            }

        result = self.create_backup()
        if result["status"] == "ok":
            deleted = self.rotate()
            result["rotated"] = deleted
        return result

    def list_backups(self) -> List[Dict[str, Any]]:
        """Return available backups with metadata, newest-first."""
        files = self._backup_files()
        backups = []
        for f in files:
            stat = f.stat()
            backups.append({
                "filename": f.name,
                "path": str(f),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            })
        return backups
