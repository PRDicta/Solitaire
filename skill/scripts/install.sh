#!/usr/bin/env bash
# Solitaire — Install script for agentskills.io skill packaging
# Installs the solitaire package and verifies core dependencies.
#
# Exit codes:
#   0 — install succeeded, import verified
#   1 — install failed (missing Python, pip failure, import failure)
set -euo pipefail

MIN_PYTHON="3.10"

log() { echo "[solitaire-install] $*"; }
err() { echo "[solitaire-install] ERROR: $*" >&2; }

# ── Find Python ──────────────────────────────────────────────────────────

PY=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PY="$candidate"
        break
    fi
done
[ -z "$PY" ] && { err "Python not found. Install Python >= $MIN_PYTHON."; exit 1; }

# ── Check version ────────────────────────────────────────────────────────

py_version=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
py_major=$(echo "$py_version" | cut -d. -f1)
py_minor=$(echo "$py_version" | cut -d. -f2)
req_major=$(echo "$MIN_PYTHON" | cut -d. -f1)
req_minor=$(echo "$MIN_PYTHON" | cut -d. -f2)

if [ "$py_major" -lt "$req_major" ] || { [ "$py_major" -eq "$req_major" ] && [ "$py_minor" -lt "$req_minor" ]; }; then
    err "Python $py_version found, but >= $MIN_PYTHON required."
    exit 1
fi
log "Python $py_version OK ($PY)"

# ── Install ──────────────────────────────────────────────────────────────

log "Installing Solitaire..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."

if [ -f "${REPO_ROOT}/pyproject.toml" ]; then
    log "Found pyproject.toml — installing from source"
    "$PY" -m pip install -e "${REPO_ROOT}" --quiet --break-system-packages 2>/dev/null \
        || "$PY" -m pip install -e "${REPO_ROOT}" --quiet
else
    err "pyproject.toml not found at ${REPO_ROOT}. Clone the repo first:"
    err "  git clone https://github.com/PRDicta/Solitaire.git"
    exit 1
fi

# ── Verify import ────────────────────────────────────────────────────────

version=$("$PY" -c "from solitaire import __version__; print(__version__)" 2>&1) \
    || { err "Install completed but import failed."; exit 1; }
log "Solitaire $version imported OK"

# ── Check CLI entry point ────────────────────────────────────────────────

if command -v solitaire &>/dev/null; then
    log "CLI entry point: $(command -v solitaire)"
else
    log "NOTE: CLI not on PATH. Use '$PY -m solitaire' or add pip's bin dir to PATH."
fi

lo