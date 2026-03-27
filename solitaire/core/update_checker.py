"""
Auto-update checker for Solitaire.

Checks GitHub releases API on boot, compares versions,
and can download + apply updates via the bundled update.py script.

All network calls use stdlib (urllib) with short timeouts.
Failure at any point is non-fatal: the engine boots normally.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile


# GitHub API endpoint for latest release
GITHUB_REPO = "PRDicta/Solitaire-for-Agents"
RELEASES_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# Cache settings
CACHE_FILE = ".update_cache.json"
CACHE_MAX_AGE_HOURS = 24

# Network timeout (seconds)
REQUEST_TIMEOUT = 5
DOWNLOAD_TIMEOUT = 120


def parse_semver(version_str: str) -> Tuple[int, ...]:
    """Parse a version string like '1.2.3' into a comparable tuple."""
    parts = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str or "")
    if not parts:
        return (0, 0, 0)
    return (int(parts.group(1)), int(parts.group(2)), int(parts.group(3)))


class UpdateChecker:
    """
    Manages update checking, snooze state, and update application.

    Usage:
        checker = UpdateChecker(workspace, current_version)
        result = checker.check()
        if result and result["update_available"]:
            checker.apply_update(result["download_url"])
    """

    def __init__(self, workspace: Path, current_version: str):
        self.workspace = Path(workspace)
        self.current_version = current_version
        self._cache_path = self.workspace / CACHE_FILE
        self._snooze_path = self.workspace / ".update_snooze.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> Optional[Dict[str, Any]]:
        """
        Check if an update is available. Returns None on any error.

        Returns dict with:
            update_available: bool
            current_version: str
            latest_version: str
            download_url: str (zip URL for the release)
            release_notes: str (first 500 chars of release body)
            html_url: str (GitHub release page)
        """
        # Check snooze
        if self._is_snoozed():
            return {"update_available": False, "reason": "snoozed"}

        # Check cache
        cached = self._read_cache()
        if cached:
            return cached

        # Query GitHub
        try:
            release = self._fetch_latest_release()
        except Exception:
            return None

        if not release:
            return None

        latest_version = release.get("tag_name", "").lstrip("v")
        current_tuple = parse_semver(self.current_version)
        latest_tuple = parse_semver(latest_version)

        # Find the source zip asset or fall back to zipball_url
        download_url = release.get("zipball_url", "")
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if name.startswith("solitaire-update") and name.endswith(".zip"):
                download_url = asset.get("browser_download_url", download_url)
                break

        result = {
            "update_available": latest_tuple > current_tuple,
            "current_version": self.current_version,
            "latest_version": latest_version,
            "download_url": download_url,
            "release_notes": (release.get("body") or "")[:500],
            "html_url": release.get("html_url", ""),
        }

        # Cache the result
        self._write_cache(result)
        return result

    def snooze(self, days: int, version: Optional[str] = None):
        """
        Snooze update notifications for N days.
        If version is specified, only snooze that specific version.
        """
        now = datetime.now(timezone.utc)
        expiry = now.isoformat()  # placeholder
        # Calculate expiry
        from datetime import timedelta
        expiry = (now + timedelta(days=days)).isoformat()

        data = {
            "snooze_until": expiry,
            "snooze_version": version,
            "created_at": now.isoformat(),
        }
        self._snooze_path.write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )

    def clear_snooze(self):
        """Remove snooze state."""
        if self._snooze_path.exists():
            self._snooze_path.unlink()

    def apply_update(self, download_url: str) -> Dict[str, Any]:
        """
        Download and apply an update.

        Downloads the release zip, extracts it, and runs update.py
        from the extracted package as a subprocess.

        Returns dict with status and details.
        """
        result = {"status": "error", "message": ""}

        # Download to temp directory
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="solitaire-update-"))
            zip_path = tmp_dir / "update.zip"

            req = Request(download_url)
            req.add_header("User-Agent", f"Solitaire/{self.current_version}")
            with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
                zip_path.write_bytes(resp.read())

        except Exception as e:
            result["message"] = f"Download failed: {e}"
            return result

        # Extract
        try:
            extract_dir = tmp_dir / "extracted"
            with ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            # Find the update package directory (may be nested one level)
            update_dir = extract_dir
            subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1 and (subdirs[0] / "solitaire").is_dir():
                update_dir = subdirs[0]
            elif not (update_dir / "solitaire").is_dir():
                result["message"] = "Invalid update package: no solitaire/ directory"
                return result

        except Exception as e:
            result["message"] = f"Extraction failed: {e}"
            return result

        # Run update.py from the extracted package
        update_script = update_dir / "update.py"
        if not update_script.exists():
            result["message"] = "update.py not found in package"
            return result

        try:
            proc = subprocess.run(
                [sys.executable, str(update_script), str(update_dir)],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.workspace),
            )

            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr
            result["returncode"] = proc.returncode

            if proc.returncode == 0:
                result["status"] = "ok"
                result["message"] = "Update applied successfully"
                # Clear cache so next boot sees the new version
                self._clear_cache()
                self.clear_snooze()
            else:
                result["message"] = f"Update script failed (exit {proc.returncode})"

        except subprocess.TimeoutExpired:
            result["message"] = "Update script timed out after 5 minutes"
        except Exception as e:
            result["message"] = f"Failed to run update script: {e}"
        finally:
            # Clean up temp dir (best-effort)
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _read_cache(self) -> Optional[Dict[str, Any]]:
        """Read cached check result if fresh enough."""
        if not self._cache_path.exists():
            return None
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(data.get("cached_at", ""))
            age_hours = (
                datetime.now(timezone.utc) - cached_at
            ).total_seconds() / 3600
            if age_hours < CACHE_MAX_AGE_HOURS:
                return data.get("result")
        except Exception:
            pass
        return None

    def _write_cache(self, result: Dict[str, Any]):
        """Cache a check result."""
        try:
            data = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "result": result,
            }
            self._cache_path.write_text(
                json.dumps(data, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

    def _clear_cache(self):
        """Remove cached check result."""
        try:
            if self._cache_path.exists():
                self._cache_path.unlink()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Snooze management
    # ------------------------------------------------------------------

    def _is_snoozed(self) -> bool:
        """Check if updates are currently snoozed."""
        if not self._snooze_path.exists():
            return False
        try:
            data = json.loads(self._snooze_path.read_text(encoding="utf-8"))
            expiry = datetime.fromisoformat(data["snooze_until"])
            now = datetime.now(timezone.utc)
            if now >= expiry:
                # Snooze expired, clean up
                self._snooze_path.unlink()
                return False

            # If snooze is version-specific, check if a newer version exists
            snooze_version = data.get("snooze_version")
            if snooze_version:
                # Only snoozed for this specific version
                # If we somehow know a newer version is out, don't suppress
                # (but we don't have that info without checking GitHub)
                pass

            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # GitHub API
    # ------------------------------------------------------------------

    def _fetch_latest_release(self) -> Optional[Dict[str, Any]]:
        """Fetch latest release info from GitHub API."""
        req = Request(RELEASES_URL)
        req.add_header("Accept", "application/vnd.github.v3+json")
        req.add_header("User-Agent", f"Solitaire/{self.current_version}")

        try:
            with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError):
            return None


def check_for_updates(
    workspace: Path,
    current_version: str,
) -> Optional[Dict[str, Any]]:
    """
    Convenience function for boot integration.
    Returns update info dict or None.
    Non-fatal: any exception returns None.
    """
    try:
        checker = UpdateChecker(workspace, current_version)
        return checker.check()
    except Exception:
        return None
