"""
Solitaire Universal Updater

Single-file, stdlib-only update script. Works for any Solitaire user
upgrading from any GitHub release version (v1.0.0+).

Safety gates:
  1. Target state discovery before any file operations
  2. Data-first backup (atomic SQLite + personas/ + sessions/) as prerequisite gate
  3. Circuit breaker: abort after 2 consecutive operation failures

Never touches user data (rolodex.db, personas/, backups/, sessions/).
"""

import os
import re
import sys
import sqlite3
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Files/dirs that are USER DATA and must never be overwritten or deleted
PROTECTED = {
    "rolodex.db",
    "rolodex.db-wal",
    "rolodex.db-shm",
    "personas",
    "backups",
    "sessions",
    ".solitaire_session",
}

# Files/dirs to copy from update package to workspace
DISTRIBUTABLE = [
    "solitaire",
    "pyproject.toml",
    "CLAUDE.md",
    "README.md",
    "skill",
    "mcp-server",
    "LICENSE",
    "COMMERCIAL_LICENSE.md",
]

# v1.0.0 remnants to clean up
LEGACY_CLEANUP = [
    "ai_writing_tells.md",
    "FIRST_INTERACTION.md",
    "starter",
]

# Skip patterns during copy
SKIP_PATTERNS = {"__pycache__", "*.pyc", "*.pyo", ".git", ".pytest_cache"}

# Key tables that must exist in a valid rolodex.db
REQUIRED_TABLES = {"rolodex_entries", "conversations", "identity_nodes"}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_banner(text, char="="):
    width = max(len(text) + 4, 44)
    print(char * width)
    print(f"  {text}")
    print(char * width)


def print_ok(msg):
    print(f"  [OK] {msg}")


def print_err(msg):
    print(f"  [!!] {msg}")


def print_info(msg):
    print(f"  {msg}")


# ---------------------------------------------------------------------------
# Version reading
# ---------------------------------------------------------------------------

def read_version(solitaire_dir):
    """Read __version__ from a solitaire/src package directory."""
    version_file = Path(solitaire_dir) / "__version__.py"
    if not version_file.exists():
        return None
    text = version_file.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Target State Discovery (Safety Gate #1)
# ---------------------------------------------------------------------------

def detect_target_state(workspace):
    """
    Examine the actual machine state before any operations.
    Returns a dict describing what exists on disk right now.
    """
    workspace = Path(workspace)
    state = {
        "version": None,
        "layout": "unknown",
        "has_rolodex": False,
        "rolodex_size": 0,
        "rolodex_tables": [],
        "rolodex_entry_count": 0,
        "has_personas": False,
        "persona_count": 0,
        "has_sessions": False,
        "has_session_marker": False,
        "code_locations": [],
        "legacy_files": [],
        "egg_info_dirs": [],
    }

    # Check unified layout (v1.1.0+): solitaire/__version__.py at workspace root
    v_file = workspace / "solitaire" / "__version__.py"
    if v_file.exists():
        state["version"] = read_version(workspace / "solitaire")
        state["layout"] = "v1_unified"
        state["code_locations"].append("solitaire/")

    # Check dual-tree layout (v1.0.0): src/ and/or starter/solitaire/
    for old_path in ["src", "starter/solitaire"]:
        v_file = workspace / old_path / "__version__.py"
        if v_file.exists():
            if state["version"] is None:
                state["version"] = read_version(workspace / old_path)
            state["layout"] = "v1_dual"
            state["code_locations"].append(old_path + "/")

    # Data inventory: rolodex.db
    rdb = workspace / "rolodex.db"
    if rdb.exists():
        state["has_rolodex"] = True
        state["rolodex_size"] = rdb.stat().st_size
        try:
            conn = sqlite3.connect(str(rdb))
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            state["rolodex_tables"] = [row[0] for row in cursor.fetchall()]
            if "rolodex_entries" in state["rolodex_tables"]:
                count = conn.execute(
                    "SELECT COUNT(*) FROM rolodex_entries"
                ).fetchone()[0]
                state["rolodex_entry_count"] = count
            conn.close()
        except sqlite3.Error:
            pass  # DB exists but can't read it; still note it exists

    # Data inventory: personas/
    personas = workspace / "personas"
    if personas.is_dir():
        items = list(personas.iterdir())
        state["has_personas"] = len(items) > 0
        state["persona_count"] = len(items)

    # Data inventory: sessions/
    sessions = workspace / "sessions"
    state["has_sessions"] = sessions.is_dir() and any(sessions.iterdir())

    # Data inventory: .solitaire_session
    state["has_session_marker"] = (workspace / ".solitaire_session").exists()

    # Legacy file scan
    for name in LEGACY_CLEANUP + ["src"]:
        if (workspace / name).exists():
            state["legacy_files"].append(name)

    # Egg-info dirs
    if workspace.is_dir():
        for item in workspace.iterdir():
            if item.name.endswith(".egg-info") and item.is_dir():
                state["egg_info_dirs"].append(item.name)

    return state


# ---------------------------------------------------------------------------
# Circuit Breaker (Safety Gate #3)
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Abort after max_consecutive consecutive failures."""

    def __init__(self, max_consecutive=2):
        self.max = max_consecutive
        self.consecutive = 0
        self.failures = []

    def record_success(self, step_name):
        self.consecutive = 0

    def record_failure(self, step_name, error):
        self.consecutive += 1
        self.failures.append((step_name, str(error)))

    def should_abort(self):
        return self.consecutive >= self.max

    def summary(self):
        if not self.failures:
            return "No failures recorded."
        lines = [f"  - {name}: {err}" for name, err in self.failures]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data-First Backup (Safety Gate #2 / Prerequisite Gate)
# ---------------------------------------------------------------------------

def backup_user_data(workspace, timestamp=None):
    """
    Atomic backup of ALL user data before any code operations.
    This is the prerequisite gate: if it fails, nothing else runs.

    Backs up:
      - rolodex.db (atomic SQLite conn.backup, NOT file copy)
      - personas/ directory tree
      - sessions/ directory tree
      - .solitaire_session file

    Returns (backup_dir, report_dict). Raises on any failure.
    """
    workspace = Path(workspace)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    backup_dir = workspace / "backups" / f"pre-update-data-{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "backup_dir": str(backup_dir),
        "rolodex": None,
        "personas": None,
        "sessions": None,
        "session_marker": None,
    }

    # 1. Atomic SQLite backup of rolodex.db
    rdb = workspace / "rolodex.db"
    if rdb.exists():
        backup_db = backup_dir / "rolodex.db"
        try:
            src_conn = sqlite3.connect(str(rdb))
            dst_conn = sqlite3.connect(str(backup_db))
            src_conn.backup(dst_conn)
            dst_conn.close()
            src_conn.close()
        except Exception as e:
            # Clean up partial backup
            if backup_db.exists():
                backup_db.unlink()
            raise RuntimeError(f"Atomic SQLite backup failed: {e}") from e

        # Verify backup integrity
        try:
            verify_conn = sqlite3.connect(str(backup_db))
            cursor = verify_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            missing = REQUIRED_TABLES - tables
            if missing:
                verify_conn.close()
                raise RuntimeError(
                    f"Backup verification failed: missing tables {missing}"
                )
            entry_count = verify_conn.execute(
                "SELECT COUNT(*) FROM rolodex_entries"
            ).fetchone()[0]
            verify_conn.close()
            report["rolodex"] = {
                "path": str(backup_db),
                "size": backup_db.stat().st_size,
                "tables": len(tables),
                "entries": entry_count,
            }
        except sqlite3.Error as e:
            raise RuntimeError(
                f"Backup verification failed: cannot read backup DB: {e}"
            ) from e
    else:
        report["rolodex"] = "not_present"

    # 2. Copy personas/ directory
    personas_src = workspace / "personas"
    if personas_src.is_dir() and any(personas_src.iterdir()):
        personas_dst = backup_dir / "personas"
        try:
            shutil.copytree(personas_src, personas_dst)
            count = sum(1 for _ in personas_dst.rglob("*") if _.is_file())
            report["personas"] = {"path": str(personas_dst), "files": count}
        except Exception as e:
            raise RuntimeError(f"Persona backup failed: {e}") from e
    else:
        report["personas"] = "not_present"

    # 3. Copy sessions/ directory
    sessions_src = workspace / "sessions"
    if sessions_src.is_dir() and any(sessions_src.iterdir()):
        sessions_dst = backup_dir / "sessions"
        try:
            shutil.copytree(sessions_src, sessions_dst)
            count = sum(1 for _ in sessions_dst.rglob("*") if _.is_file())
            report["sessions"] = {"path": str(sessions_dst), "files": count}
        except Exception as e:
            raise RuntimeError(f"Sessions backup failed: {e}") from e
    else:
        report["sessions"] = "not_present"

    # 4. Copy .solitaire_session
    session_marker = workspace / ".solitaire_session"
    if session_marker.exists():
        dst = backup_dir / ".solitaire_session"
        try:
            shutil.copy2(session_marker, dst)
            report["session_marker"] = str(dst)
        except Exception as e:
            raise RuntimeError(f"Session marker backup failed: {e}") from e
    else:
        report["session_marker"] = "not_present"

    return backup_dir, report


def backup_code(workspace, current_version):
    """
    Best-effort backup of current code files.
    Failure here does NOT abort the update (code can be re-downloaded).
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version_str = current_version or "unknown"
    backup_name = f"pre-update-code-v{version_str}-{timestamp}"
    backup_dir = workspace / "backups" / backup_name

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backed_up = 0

        # Backup any code directories that exist
        for dirname in ["solitaire", "src", "starter", "skill"]:
            src = workspace / dirname
            if src.is_dir():
                shutil.copytree(
                    src, backup_dir / dirname,
                    ignore=shutil.ignore_patterns(*SKIP_PATTERNS),
                    dirs_exist_ok=True,
                )
                backed_up += 1

        # Backup individual config files
        for fname in ["pyproject.toml", "CLAUDE.md", "README.md",
                       "_boot_context.md", "_boot_ops.md"]:
            src = workspace / fname
            if src.exists():
                shutil.copy2(src, backup_dir / fname)
                backed_up += 1

        return backup_dir, backed_up
    except Exception as e:
        print_info(f"Code backup note: {e} (non-fatal, continuing)")
        return None, 0


# ---------------------------------------------------------------------------
# Migration Steps
# ---------------------------------------------------------------------------

def remove_dir_step(dirname):
    """Factory: returns a step function that removes a directory."""
    def step(workspace, update_dir, ctx):
        target = workspace / dirname
        if target.is_dir():
            shutil.rmtree(target)
            return f"Removed {dirname}/"
        return f"{dirname}/ not present, skipped"
    step.__name__ = f"remove_{dirname}"
    return step


def remove_legacy_files(workspace, update_dir, ctx):
    """Remove v1.0.0 remnant files."""
    removed = []
    for name in LEGACY_CLEANUP:
        target = workspace / name
        if target.is_file():
            target.unlink()
            removed.append(name)
        elif target.is_dir():
            shutil.rmtree(target)
            removed.append(name + "/")
    return f"Legacy cleanup: {', '.join(removed) if removed else 'nothing to remove'}"


def clean_egg_info(workspace, update_dir, ctx):
    """Remove .egg-info directories."""
    removed = []
    for item in workspace.iterdir():
        if item.name.endswith(".egg-info") and item.is_dir():
            shutil.rmtree(item)
            removed.append(item.name)
    return f"Cleaned: {', '.join(removed) if removed else 'no egg-info dirs'}"


def copy_distributable(workspace, update_dir, ctx):
    """Copy new files from update package to workspace."""
    copied = []
    for name in DISTRIBUTABLE:
        src = update_dir / name
        dst = workspace / name
        if not src.exists():
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(
                src, dst,
                ignore=shutil.ignore_patterns(*SKIP_PATTERNS),
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(src, dst)
        copied.append(name)
    ctx["copied"] = copied
    return f"Copied: {', '.join(copied)}"


def verify_data_intact(workspace, update_dir, ctx):
    """
    Post-migration check: confirm user data survived.
    Compares against pre_state stored in ctx.
    """
    pre = ctx.get("pre_state", {})
    issues = []

    # Check rolodex.db
    rdb = workspace / "rolodex.db"
    if pre.get("has_rolodex"):
        if not rdb.exists():
            issues.append("rolodex.db is MISSING after migration")
        else:
            current_size = rdb.stat().st_size
            pre_size = pre.get("rolodex_size", 0)
            # Allow some variance for WAL compaction, but flag if dramatically smaller
            if pre_size > 0 and current_size < pre_size * 0.5:
                issues.append(
                    f"rolodex.db shrank dramatically: {pre_size} -> {current_size} bytes"
                )

    # Check personas/
    if pre.get("has_personas"):
        personas = workspace / "personas"
        if not personas.is_dir():
            issues.append("personas/ directory is MISSING after migration")
        else:
            current_count = len(list(personas.iterdir()))
            pre_count = pre.get("persona_count", 0)
            if current_count < pre_count:
                issues.append(
                    f"personas/ lost items: {pre_count} -> {current_count}"
                )

    # Check sessions/
    if pre.get("has_sessions"):
        if not (workspace / "sessions").is_dir():
            issues.append("sessions/ directory is MISSING after migration")

    # Check session marker
    if pre.get("has_session_marker"):
        if not (workspace / ".solitaire_session").exists():
            issues.append(".solitaire_session is MISSING after migration")

    if issues:
        raise RuntimeError(
            "Data integrity check failed:\n  " + "\n  ".join(issues)
        )
    return "All user data intact"


# ---------------------------------------------------------------------------
# Migration Registry
# ---------------------------------------------------------------------------

MIGRATIONS = {
    "v1_dual": [
        ("remove_src", remove_dir_step("src")),
        ("remove_starter", remove_dir_step("starter")),
        ("remove_legacy_files", remove_legacy_files),
        ("clean_egg_info", clean_egg_info),
        ("copy_new_code", copy_distributable),
        ("verify_data_intact", verify_data_intact),
    ],
    "v1_unified": [
        ("remove_solitaire", remove_dir_step("solitaire")),
        ("remove_legacy_files", remove_legacy_files),
        ("clean_egg_info", clean_egg_info),
        ("copy_new_code", copy_distributable),
        ("verify_data_intact", verify_data_intact),
    ],
    "unknown": [
        ("remove_legacy_files", remove_legacy_files),
        ("clean_egg_info", clean_egg_info),
        ("copy_new_code", copy_distributable),
        ("verify_data_intact", verify_data_intact),
    ],
}


# ---------------------------------------------------------------------------
# Migration Executor
# ---------------------------------------------------------------------------

class UpdateAborted(Exception):
    """Raised when the circuit breaker trips."""
    def __init__(self, message, completed_steps=None):
        super().__init__(message)
        self.completed_steps = completed_steps or []


def execute_migration(steps, workspace, update_dir, breaker, pre_state=None):
    """
    Run migration steps in order. Each step gets (workspace, update_dir, ctx).
    The circuit breaker aborts after max consecutive failures.
    Returns list of (step_name, result_msg) for completed steps.
    """
    workspace = Path(workspace)
    update_dir = Path(update_dir)
    ctx = {"pre_state": pre_state or {}}
    completed = []

    for step_name, step_fn in steps:
        try:
            result = step_fn(workspace, update_dir, ctx)
            breaker.record_success(step_name)
            completed.append((step_name, result))
            print_ok(f"{step_name}: {result}")
        except Exception as e:
            breaker.record_failure(step_name, e)
            print_err(f"{step_name}: {e}")

            if breaker.should_abort():
                raise UpdateAborted(
                    f"Circuit breaker tripped after {breaker.max} consecutive failures.\n"
                    f"Failure log:\n{breaker.summary()}",
                    completed_steps=completed,
                )

    return completed


# ---------------------------------------------------------------------------
# Workspace Detection
# ---------------------------------------------------------------------------

def find_workspace(update_dir):
    """
    Find the Solitaire workspace. Strategy:
    1. Check for solitaire_workspace.txt in the update package
    2. Walk upward from update dir looking for rolodex.db or pyproject.toml
    3. Ask the user
    """
    update_dir = Path(update_dir)

    # Strategy 1: explicit workspace file
    ws_file = update_dir / "solitaire_workspace.txt"
    if ws_file.exists():
        ws_path = Path(ws_file.read_text(encoding="utf-8").strip())
        if ws_path.exists() and (ws_path / "rolodex.db").exists():
            return ws_path

    # Strategy 2: walk upward (user extracted zip inside workspace)
    current = update_dir.parent
    for _ in range(5):
        if (current / "rolodex.db").exists():
            return current
        if (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Strategy 3: ask the user
    print()
    print_info("Could not auto-detect your Solitaire workspace.")
    print_info("Please enter the full path to your Solitaire folder")
    print_info("(the folder that contains rolodex.db):")
    print()
    while True:
        try:
            user_path = input("  Path: ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            return None
        if not user_path:
            return None
        ws = Path(user_path)
        if ws.exists() and (ws / "rolodex.db").exists():
            return ws
        print_err(f"No rolodex.db found at {ws}")
        print_info("Please try again, or press Enter to cancel.")


# ---------------------------------------------------------------------------
# Reinstall
# ---------------------------------------------------------------------------

def reinstall(workspace):
    """Run pip install -e to register the updated package."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(workspace),
             "--break-system-packages", "-q"],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0, result.stderr.strip()
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(workspace, expected_version, pre_state=None):
    """Verify the update succeeded, comparing against pre-update state."""
    workspace = Path(workspace)
    checks = {}

    # Check rolodex.db
    rdb = workspace / "rolodex.db"
    if rdb.exists():
        size_kb = rdb.stat().st_size / 1024
        checks["rolodex.db"] = f"OK ({size_kb:.0f} KB)"
        if pre_state and pre_state.get("has_rolodex"):
            pre_kb = pre_state["rolodex_size"] / 1024
            checks["rolodex.db"] += f" [was {pre_kb:.0f} KB]"
    elif pre_state and pre_state.get("has_rolodex"):
        checks["rolodex.db"] = "MISSING (was present before update!)"
    else:
        checks["rolodex.db"] = "not present (OK if new install)"

    # Check personas/
    personas = workspace / "personas"
    if personas.is_dir():
        count = sum(1 for _ in personas.iterdir())
        checks["personas/"] = f"OK ({count} items)"
        if pre_state and pre_state.get("has_personas"):
            checks["personas/"] += f" [was {pre_state['persona_count']}]"
    elif pre_state and pre_state.get("has_personas"):
        checks["personas/"] = "MISSING (was present before update!)"
    else:
        checks["personas/"] = "not present (OK if new install)"

    # Check sessions/
    if pre_state and pre_state.get("has_sessions"):
        if (workspace / "sessions").is_dir():
            checks["sessions/"] = "OK"
        else:
            checks["sessions/"] = "MISSING (was present before update!)"

    # Check version
    installed_version = read_version(workspace / "solitaire")
    if installed_version == expected_version:
        checks["version"] = f"OK (v{installed_version})"
    else:
        checks["version"] = (
            f"MISMATCH (expected {expected_version}, got {installed_version})"
        )

    return checks


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def rollback(workspace, data_backup_dir, code_backup_dir=None):
    """
    Attempt to restore from backup. Data backup takes priority.
    """
    workspace = Path(workspace)
    restored = []

    # Restore data if needed (check if it's still there first)
    if data_backup_dir:
        data_backup_dir = Path(data_backup_dir)

        # Restore rolodex.db from atomic backup if the original is damaged
        backup_rdb = data_backup_dir / "rolodex.db"
        live_rdb = workspace / "rolodex.db"
        if backup_rdb.exists() and not live_rdb.exists():
            shutil.copy2(backup_rdb, live_rdb)
            restored.append("rolodex.db")

        # Restore personas/
        backup_personas = data_backup_dir / "personas"
        live_personas = workspace / "personas"
        if backup_personas.is_dir() and not live_personas.is_dir():
            shutil.copytree(backup_personas, live_personas)
            restored.append("personas/")

        # Restore sessions/
        backup_sessions = data_backup_dir / "sessions"
        live_sessions = workspace / "sessions"
        if backup_sessions.is_dir() and not live_sessions.is_dir():
            shutil.copytree(backup_sessions, live_sessions)
            restored.append("sessions/")

        # Restore session marker
        backup_marker = data_backup_dir / ".solitaire_session"
        live_marker = workspace / ".solitaire_session"
        if backup_marker.exists() and not live_marker.exists():
            shutil.copy2(backup_marker, live_marker)
            restored.append(".solitaire_session")

    # Restore code from code backup
    if code_backup_dir:
        code_backup_dir = Path(code_backup_dir)
        try:
            backup_sol = code_backup_dir / "solitaire"
            if backup_sol.is_dir():
                dst = workspace / "solitaire"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(backup_sol, dst)
                restored.append("solitaire/")

            for fname in ["pyproject.toml", "CLAUDE.md", "README.md"]:
                src = code_backup_dir / fname
                if src.exists():
                    shutil.copy2(src, workspace / fname)
                    restored.append(fname)
        except Exception:
            pass  # Code restoration is best-effort

    return restored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print_err("Usage: python update.py <update_package_dir>")
        return 1

    update_dir = Path(sys.argv[1]).resolve()

    # Validate update package has the new solitaire/ code
    if not (update_dir / "solitaire").is_dir():
        print_err(f"No solitaire/ folder found in {update_dir}")
        return 1

    new_version = read_version(update_dir / "solitaire")
    if not new_version:
        print_err("Could not read version from update package")
        return 1

    # Step 1: Find workspace
    print_info("Looking for your Solitaire workspace...")
    workspace = find_workspace(update_dir)
    if not workspace:
        print_err("Could not find workspace. Update cancelled.")
        return 1

    workspace = Path(workspace).resolve()

    # Step 2: Discover target state (Safety Gate #1)
    print()
    print_info("Examining current installation...")
    pre_state = detect_target_state(workspace)

    version_label = pre_state["version"] or "pre-1.0"
    layout_label = pre_state["layout"]

    print()
    print_info(f"Workspace:   {workspace}")
    print_info(f"Current:     v{version_label}")
    print_info(f"Layout:      {layout_label}")
    if pre_state["code_locations"]:
        print_info(f"Code dirs:   {', '.join(pre_state['code_locations'])}")
    print_info(f"Updating to: v{new_version}")
    if pre_state["has_rolodex"]:
        size_kb = pre_state["rolodex_size"] / 1024
        print_info(
            f"Data:        rolodex.db ({size_kb:.0f} KB, "
            f"{pre_state['rolodex_entry_count']} entries)"
        )
    if pre_state["has_personas"]:
        print_info(f"             personas/ ({pre_state['persona_count']} items)")
    if pre_state["legacy_files"]:
        print_info(f"Legacy:      {', '.join(pre_state['legacy_files'])}")
    print()

    # Step 3: Data-first backup (Safety Gate #2 / Prerequisite Gate)
    print_info("Backing up user data (prerequisite gate)...")
    try:
        data_backup_dir, backup_report = backup_user_data(workspace)
        if isinstance(backup_report.get("rolodex"), dict):
            rdb_info = backup_report["rolodex"]
            print_ok(
                f"Rolodex backed up: {rdb_info['entries']} entries, "
                f"{rdb_info['tables']} tables"
            )
        else:
            print_info("  No rolodex.db to back up")
        if isinstance(backup_report.get("personas"), dict):
            print_ok(f"Personas backed up: {backup_report['personas']['files']} files")
        if isinstance(backup_report.get("sessions"), dict):
            print_ok(f"Sessions backed up: {backup_report['sessions']['files']} files")
        print_ok(f"Data backup saved to: {data_backup_dir}")
    except Exception as e:
        print_err(f"DATA BACKUP FAILED: {e}")
        print_err("Update cancelled. Nothing was changed.")
        print_err("Your data is safe. The backup gate prevented any operations.")
        return 1

    # Step 4: Code backup (best-effort)
    print()
    print_info("Backing up current code (best-effort)...")
    code_backup_dir, code_count = backup_code(workspace, pre_state["version"])
    if code_backup_dir:
        print_ok(f"Code backed up ({code_count} items)")

    # Step 5: Execute migration
    print()
    print_info(f"Running migration for layout: {layout_label}")
    steps = MIGRATIONS.get(layout_label, MIGRATIONS["unknown"])
    breaker = CircuitBreaker(max_consecutive=2)

    try:
        completed = execute_migration(
            steps, workspace, update_dir, breaker, pre_state=pre_state
        )
    except UpdateAborted as e:
        print()
        print_err(f"UPDATE ABORTED: {e}")
        print_info("Attempting rollback...")
        restored = rollback(workspace, data_backup_dir, code_backup_dir)
        if restored:
            print_ok(f"Restored: {', '.join(restored)}")
        else:
            print_info("Nothing needed restoring (data was not touched).")
        print_info(f"Data backup: {data_backup_dir}")
        if code_backup_dir:
            print_info(f"Code backup: {code_backup_dir}")
        return 1
    except Exception as e:
        print()
        print_err(f"Unexpected error during migration: {e}")
        print_info("Attempting rollback...")
        restored = rollback(workspace, data_backup_dir, code_backup_dir)
        if restored:
            print_ok(f"Restored: {', '.join(restored)}")
        print_info(f"Data backup: {data_backup_dir}")
        return 1

    # Step 6: Reinstall (non-fatal)
    print()
    print_info("Registering updated package...")
    pip_ok, pip_err = reinstall(workspace)
    if pip_ok:
        print_ok("Package registered")
    else:
        print_info(f"pip install note: {pip_err}")
        print_info("(This is usually fine. Cowork will retry on next boot.)")

    # Step 7: Final verification
    print()
    checks = verify(workspace, new_version, pre_state=pre_state)

    print()
    print_banner(f"UPDATE COMPLETE: v{version_label}  -->  v{new_version}")
    print()
    print_info("Verification:")
    for key, status in checks.items():
        print_info(f"    {key:20s} {status}")
    print()
    print_info("Backups:")
    print_info(f"    Data: {data_backup_dir}")
    if code_backup_dir:
        print_info(f"    Code: {code_backup_dir}")
    print()
    print_info("Open Cowork and start a new session. You're all set.")
    print()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print_info("Update cancelled.")
        sys.exit(1)
    except Exception as e:
        print()
        print_err(f"Unexpected error: {e}")
        print_info("Contact support for help.")
        sys.exit(1)
