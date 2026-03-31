"""
Solitaire — Rolling Backup Manager

Atomic SQLite backups and persona config snapshots with configurable
retention and staleness checks. Triggered on boot when the latest
backup exceeds max_age_hours.

Safety invariant: user data (rolodex.db AND personas/) is always
backed up together. Code can be re-downloaded; user data cannot.
"""
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BackupManager:
    """Manages rolling backups for rolodex database AND persona configs."""

    BACKUP_PREFIX = "rolodex_"
    BACKUP_SUFFIX = ".db"
    PERSONA_PREFIX = "personas_"

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

    def create_backup(self, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """Create an atomic SQLite backup using conn.backup().

        Args:
            timestamp: Optional shared timestamp string. Generated if not provided.

        Returns:
            Dict with path, size_bytes, timestamp, or error.
        """
        if not self.db_path.exists():
            return {"status": "skip", "reason": "no_database"}

        self.backup_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
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

    # --- Persona backup ---

    def _persona_dir(self) -> Path:
        """Return the personas directory."""
        return self.workspace_dir / "personas"

    def _persona_backup_dirs(self) -> List[Path]:
        """Return existing persona backup directories sorted newest-first."""
        if not self.backup_dir.exists():
            return []
        dirs = [
            d for d in self.backup_dir.iterdir()
            if d.name.startswith(self.PERSONA_PREFIX)
            and d.is_dir()
        ]
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return dirs

    def create_persona_backup(self, timestamp: str) -> Dict[str, Any]:
        """Copy the entire personas/ directory to backups/.

        Returns:
            Dict with path, file_count, or error.
        """
        persona_dir = self._persona_dir()
        if not persona_dir.exists() or not any(persona_dir.iterdir()):
            return {"status": "skip", "reason": "no_personas"}

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup_name = f"{self.PERSONA_PREFIX}{timestamp}"
        backup_path = self.backup_dir / backup_name

        try:
            shutil.copytree(persona_dir, backup_path)
            file_count = sum(1 for _ in backup_path.rglob("*") if _.is_file())
            logger.info(
                "Persona backup created: %s (%d files)",
                backup_path.name, file_count,
            )
            return {
                "status": "ok",
                "path": str(backup_path),
                "dirname": backup_name,
                "file_count": file_count,
            }
        except Exception as e:
            logger.error("Persona backup failed: %s", e)
            if backup_path.exists():
                try:
                    shutil.rmtree(backup_path)
                except OSError:
                    pass
            return {"status": "error", "reason": str(e)}

    def rotate_personas(self) -> List[str]:
        """Delete oldest persona backups exceeding retention_count."""
        dirs = self._persona_backup_dirs()
        deleted = []
        while len(dirs) > self.retention_count:
            oldest = dirs.pop()
            try:
                shutil.rmtree(oldest)
                deleted.append(oldest.name)
                logger.info("Rotated out persona backup: %s", oldest.name)
            except OSError as e:
                logger.warning("Could not delete %s: %s", oldest.name, e)
        return deleted

    # --- Rotation ---

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

    # --- Main entry point ---

    def check_and_backup(self) -> Dict[str, Any]:
        """Staleness check + conditional backup + rotation for ALL user data.

        Backs up both rolodex.db and personas/ together using the same
        timestamp. This is the main entry point, called on boot.

        Safety invariant: user data is always backed up as a unit.
        rolodex.db and personas/ snapshots share a timestamp so they
        can be restored together if needed.

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

        # Shared timestamp so DB and persona backups are paired
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

        result = self.create_backup(timestamp=timestamp)
        persona_result = self.create_persona_backup(timestamp=timestamp)

        if result["status"] == "ok":
            deleted = self.rotate()
            result["rotated"] = deleted
        if persona_result["status"] == "ok":
            deleted = self.rotate_personas()
            persona_result["rotated"] = deleted

        result["personas"] = persona_result
        return result

    def get_matching_persona_backup(self, db_backup_path: Path) -> Optional[Path]:
        """Find the persona backup with the same timestamp as a DB backup.

        Extracts the timestamp from the DB backup filename and looks for
        a matching personas_ directory.
        """
        # Extract timestamp: rolodex_YYYYMMDD_HHMMSS_FFFFFF.db
        name = db_backup_path.name
        ts = name.replace(self.BACKUP_PREFIX, "").replace(self.BACKUP_SUFFIX, "")
        persona_dir = self.backup_dir / f"{self.PERSONA_PREFIX}{ts}"
        if persona_dir.exists() and persona_dir.is_dir():
            return persona_dir
        return None

    def restore_from_backup(self, backup_path: Path) -> Dict[str, Any]:
        """Restore rolodex.db and personas/ from a backup snapshot.

        Creates a safety snapshot of the current state before restoring.
        Rebuilds FTS indexes after restore.

        Args:
            backup_path: Path to the .db backup file.

        Returns:
            Dict with status, safety_backup, restored files, and FTS rebuild result.
        """
        result: Dict[str, Any] = {
            "status": "error",
            "message": "",
            "safety_backup": None,
            "restored_db": False,
            "restored_personas": False,
            "fts_rebuild": None,
        }

        if not backup_path.exists():
            result["message"] = f"Backup file not found: {backup_path}"
            return result

        # Step 1: Safety snapshot of current state
        safety_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safety = self.create_backup(timestamp=f"pre_restore_{safety_ts}")
        self.create_persona_backup(timestamp=f"pre_restore_{safety_ts}")
        result["safety_backup"] = safety.get("path")

        # Step 2: Restore database
        try:
            source = sqlite3.connect(str(backup_path))
            dest = sqlite3.connect(str(self.db_path))
            try:
                source.backup(dest)
            finally:
                dest.close()
                source.close()
            result["restored_db"] = True
            logger.info("Restored database from %s", backup_path.name)
        except Exception as e:
            result["message"] = f"Database restore failed: {e}"
            return result

        # Step 3: Restore personas if matching backup exists
        persona_backup = self.get_matching_persona_backup(backup_path)
        if persona_backup:
            persona_dir = self._persona_dir()
            try:
                if persona_dir.exists():
                    shutil.rmtree(persona_dir)
                shutil.copytree(persona_backup, persona_dir)
                result["restored_personas"] = True
                logger.info("Restored personas from %s", persona_backup.name)
            except Exception as e:
                result["message"] = f"Persona restore failed (DB was restored): {e}"
                return result

        # Step 4: Rebuild FTS indexes
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            from .fts_rebuild import rebuild_all_fts
            fts_result = rebuild_all_fts(conn)
            conn.close()
            result["fts_rebuild"] = fts_result
        except Exception as e:
            result["fts_rebuild"] = {"status": "error", "reason": str(e)}

        result["status"] = "ok"
        result["message"] = "Restore complete"
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
