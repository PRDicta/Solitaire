"""
Build a Solitaire update package (zip).

Run from the solitaire repo root:
    python updater/build_update.py

Outputs: updater/output/solitaire-update-vX.Y.Z.zip

Optional: pass a workspace path to embed in the zip for auto-detection:
    python updater/build_update.py --workspace "C:\\Users\\Brenna\\Solitaire"
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


# Directories/files to include in the update package
INCLUDE = [
    "solitaire",
    "pyproject.toml",
    "CLAUDE.md",
    "README.md",
    "skill",
    "mcp-server",
    "LICENSE",
    "COMMERCIAL_LICENSE.md",
]

# Updater scripts to include
UPDATER_SCRIPTS = [
    "UPDATE.bat",
    "update.ps1",
    "update.py",
]

# Patterns to exclude from the copy
EXCLUDE_PATTERNS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
    "rolodex.db",
    "rolodex.db-wal",
    "rolodex.db-shm",
    "personas",
    "backups",
    "sessions",
    ".solitaire_session",
}

# Data that must never be included in an update package
PROTECTED_DATA = [
    "rolodex.db",
    "rolodex.db-wal",
    "rolodex.db-shm",
    "personas",
    "backups",
    "sessions",
    ".solitaire_session",
]


def should_exclude(path):
    """Check if a path component matches exclusion patterns."""
    name = path.name
    if name in EXCLUDE_PATTERNS:
        return True
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*") and name.endswith(pattern[1:]):
            return True
    for part in path.parts:
        if part in EXCLUDE_PATTERNS:
            return True
    return False


def read_version(version_file):
    """Read version from a __version__.py file."""
    text = version_file.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        return None
    return match.group(1)


def check_version_consistency(repo_root):
    """Verify __version__.py and pyproject.toml agree on version."""
    version_py = repo_root / "solitaire" / "__version__.py"
    pyproject = repo_root / "pyproject.toml"

    py_version = read_version(version_py)
    if not py_version:
        print("ERROR: Could not read version from solitaire/__version__.py")
        return None

    if pyproject.exists():
        toml_text = pyproject.read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*"([^"]+)"', toml_text)
        if match:
            toml_version = match.group(1)
            if toml_version != py_version:
                print(f"ERROR: Version mismatch!")
                print(f"  __version__.py: {py_version}")
                print(f"  pyproject.toml: {toml_version}")
                print(f"Fix this before building a release package.")
                return None

    return py_version


def build_migration_meta(version):
    """Create migration metadata for the update package."""
    return {
        "target_version": version,
        "min_compatible_source": "1.0.0",
        "supported_layouts": ["v1_dual", "v1_unified", "unknown"],
        "protected_data": PROTECTED_DATA,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "requires_python": ">=3.10",
        "schema_version": 1,
    }


def build_package(repo_root, output_dir, workspace_path=None):
    # Check version consistency first
    version = check_version_consistency(repo_root)
    if not version:
        sys.exit(1)

    package_name = f"solitaire-update-v{version}"
    staging = output_dir / package_name

    # Clean previous build
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
        if staging.exists():
            print(f"WARNING: Could not fully clean {staging}")
            print("Delete it manually and retry.")
            sys.exit(1)
    staging.mkdir(parents=True)

    print(f"Building update package v{version}...")
    print()

    # Copy distributable files
    for name in INCLUDE:
        src = repo_root / name
        if not src.exists():
            print(f"  SKIP  {name} (not found)")
            continue

        dst = staging / name
        if src.is_dir():
            shutil.copytree(
                src, dst,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", "*.pyo", ".git", ".pytest_cache"
                ),
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(src, dst)
        print(f"  COPY  {name}")

    # Copy updater scripts (including update.py)
    updater_dir = repo_root / "updater"
    for script in UPDATER_SCRIPTS:
        src = updater_dir / script
        if src.exists():
            shutil.copy2(src, staging / script)
            print(f"  COPY  {script} (updater)")
        else:
            print(f"  SKIP  {script} (not found in updater/)")

    # Write migration metadata
    meta = build_migration_meta(version)
    meta_path = staging / "migration_meta.json"
    meta_path.write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  WRITE migration_meta.json")

    # Write workspace hint if provided
    if workspace_path:
        ws_file = staging / "solitaire_workspace.txt"
        ws_file.write_text(str(workspace_path), encoding="utf-8")
        print(f"  WRITE solitaire_workspace.txt -> {workspace_path}")

    # Create zip
    zip_path = output_dir / f"{package_name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for file_path in sorted(staging.rglob("*")):
            if file_path.is_file() and not should_exclude(file_path):
                arcname = file_path.relative_to(output_dir)
                zf.write(file_path, arcname)

    # Clean staging
    try:
        shutil.rmtree(staging, ignore_errors=True)
    except Exception:
        pass

    # Report
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print()
    print(f"Package ready: {zip_path}")
    print(f"Size: {size_mb:.1f} MB")
    print()
    print("Contents:")
    print(f"  Code:     {', '.join(INCLUDE)}")
    print(f"  Updater:  {', '.join(UPDATER_SCRIPTS)}")
    print(f"  Meta:     migration_meta.json")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Build Solitaire update package")
    parser.add_argument("--workspace", help="Workspace path to embed for auto-detection")
    args = parser.parse_args()

    # Determine repo root (updater/ is one level down)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if not (repo_root / "solitaire" / "__version__.py").exists():
        print("ERROR: Run this script from the solitaire repo root or from updater/")
        sys.exit(1)

    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)

    workspace = args.workspace if args.workspace else None
    build_package(repo_root, output_dir, workspace)


if __name__ == "__main__":
    main()
