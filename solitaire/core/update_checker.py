"""
Auto-update checker for Solitaire.

Checks GitHub releases API on boot, compares versions,
and applies updates via git operations.

Update mechanism:
  - Git clone users (.git/ exists): git fetch + selective checkout
  - Zip/installer users (no .git/): bootstrap git, then same checkout
  - Both converge to the same update path after first update

All network calls use stdlib (urllib) with short timeouts.
Failure at any point is non-fatal: the engine boots normally.
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen


# GitHub repo
GITHUB_REPO = "PRDicta/Solitaire-for-Agents"
GITHUB_URL = f"https://github.com/{GITHUB_REPO}.git"
RELEASES_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# Cache settings
CACHE_FILE = ".update_cache.json"
CACHE_MAX_AGE_HOURS = 24

# Network timeout (seconds)
REQUEST_TIMEOUT = 5

# Code paths to checkout from git (everything except user data)
CODE_PATHS = [
    ".gitignore",
    "solitaire/",
    "pyproject.toml",
    "CLAUDE.md",
    "README.md",
    "skill/",
    "mcp-server/",
    "LICENSE",
    "COMMERCIAL_LICENSE.md",
    "updater/",
]


def parse_semver(version_str: str) -> Tuple[int, ...]:
    """Parse a version string like '1.2.3' into a comparable tuple."""
    parts = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str or "")
    if not parts:
        return (0, 0, 0)
    return (int(parts.group(1)), int(parts.group(2)), int(parts.group(3)))


def _run_git(args: List[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a git command and return the result.

    Default timeout is 30s for local operations. Network-bound calls
    (fetch, clone) should pass timeout=15 explicitly.
    """
    return subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd),
    )


class UpdateChecker:
    """
    Manages update checking, snooze state, and git-based update application.

    Usage:
        checker = UpdateChecker(workspace, current_version)
        result = checker.check()
        if result and result["update_available"]:
            checker.apply_update(result["latest_version"])
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
            tag: str (git tag, e.g. "v1.3.0")
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

        tag = release.get("tag_name", "")
        latest_version = tag.lstrip("v")
        current_tuple = parse_semver(self.current_version)
        latest_tuple = parse_semver(latest_version)

        result = {
            "update_available": latest_tuple > current_tuple,
            "current_version": self.current_version,
            "latest_version": latest_version,
            "tag": tag,
            "release_notes": (release.get("body") or "")[:500],
            "html_url": release.get("html_url", ""),
        }

        self._write_cache(result)
        return result

    def snooze(self, days: int, version: Optional[str] = None):
        """
        Snooze update notifications for N days.
        If version is specified, only snooze that specific version.
        """
        now = datetime.now(timezone.utc)
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

    def has_git(self) -> bool:
        """Check if the workspace is a git repository."""
        return (self.workspace / ".git").exists()

    def apply_update(self, target_version: str) -> Dict[str, Any]:
        """
        Apply an update using git operations.

        For git workspaces: fetch + selective checkout of code paths.
        For non-git workspaces: bootstrap git first, then same checkout.

        The selective checkout only touches CODE_PATHS. User data
        (rolodex.db, personas/, sessions/, etc.) is gitignored and
        structurally untouched.

        Args:
            target_version: Version to update to (e.g. "1.3.0").
                           Will be resolved to git tag "v1.3.0".

        Returns:
            Dict with status, steps completed, and any errors.
        """
        tag = f"v{target_version}" if not target_version.startswith("v") else target_version
        result = {
            "status": "error",
            "message": "",
            "steps": [],
            "tag": tag,
        }

        try:
            # Step 1: Ensure git is available
            git_check = subprocess.run(
                ["git", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if git_check.returncode != 0:
                result["message"] = "git is not installed or not on PATH"
                return result
            result["steps"].append("git_available")

            # Step 2: Bootstrap git if needed
            if not self.has_git():
                bootstrap_result = self._bootstrap_git()
                if not bootstrap_result["ok"]:
                    result["message"] = f"Git bootstrap failed: {bootstrap_result['error']}"
                    return result
                result["steps"].append("git_bootstrapped")
            else:
                result["steps"].append("git_exists")

            # Step 3: Fetch latest from remote
            fetch = _run_git(
                ["fetch", "origin", "--tags", "--force"],
                cwd=self.workspace,
                timeout=15,
            )
            if fetch.returncode != 0:
                result["message"] = f"git fetch failed: {fetch.stderr.strip()}"
                return result
            result["steps"].append("fetched")

            # Step 4: Verify tag exists
            tag_check = _run_git(
                ["rev-parse", "--verify", f"refs/tags/{tag}"],
                cwd=self.workspace,
            )
            if tag_check.returncode != 0:
                result["message"] = f"Tag {tag} not found in remote"
                return result
            result["steps"].append("tag_verified")

            # Step 5: Selective checkout of code paths only
            checkout_args = ["checkout", tag, "--"] + CODE_PATHS
            checkout = _run_git(checkout_args, cwd=self.workspace)
            if checkout.returncode != 0:
                # Some paths may not exist in older tags; try one by one
                failed_paths = []
                for path in CODE_PATHS:
                    single = _run_git(
                        ["checkout", tag, "--", path],
                        cwd=self.workspace,
                    )
                    if single.returncode != 0:
                        failed_paths.append(path)
                if failed_paths:
                    result["steps"].append(
                        f"checkout_partial (failed: {', '.join(failed_paths)})"
                    )
                else:
                    result["steps"].append("checkout_individual")
            else:
                result["steps"].append("checkout_complete")

            # Step 6: Reinstall package
            pip_result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e",
                 str(self.workspace), "--break-system-packages", "-q"],
                capture_output=True, text=True, timeout=60,
            )
            if pip_result.returncode == 0:
                result["steps"].append("pip_installed")
            else:
                # Non-fatal: pip install may fail but code is already in place
                result["steps"].append(
                    f"pip_warning: {pip_result.stderr.strip()[:200]}"
                )

            # Step 7: Verify version matches
            version_file = self.workspace / "solitaire" / "__version__.py"
            if version_file.exists():
                text = version_file.read_text(encoding="utf-8")
                match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
                if match:
                    installed = match.group(1)
                    expected = tag.lstrip("v")
                    if installed == expected:
                        result["steps"].append(f"version_verified ({installed})")
                    else:
                        result["steps"].append(
                            f"version_mismatch (expected {expected}, got {installed})"
                        )

            # Success
            result["status"] = "ok"
            result["message"] = f"Updated to {tag}"
            self._clear_cache()
            self.clear_snooze()

        except subprocess.TimeoutExpired:
            result["message"] = "Git operation timed out"
        except FileNotFoundError:
            result["message"] = "git not found. Install git to enable updates."
        except Exception as e:
            result["message"] = f"Update failed: {e}"

        return result

    # ------------------------------------------------------------------
    # Git bootstrap (for zip/installer users)
    # ------------------------------------------------------------------

    def _bootstrap_git(self) -> Dict[str, Any]:
        """
        Initialize git in a non-git workspace and add the remote.
        After this, the workspace can use git fetch/checkout for updates.
        """
        result = {"ok": False, "error": ""}

        # git init
        init = _run_git(["init"], cwd=self.workspace)
        if init.returncode != 0:
            result["error"] = f"git init failed: {init.stderr.strip()}"
            return result

        # Add remote
        remote = _run_git(
            ["remote", "add", "origin", GITHUB_URL],
            cwd=self.workspace,
        )
        if remote.returncode != 0:
            # Remote may already exist
            if "already exists" not in remote.stderr:
                result["error"] = f"git remote add failed: {remote.stderr.strip()}"
                return result

        result["ok"] = True
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
                self._snooze_path.unlink()
                return False
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
