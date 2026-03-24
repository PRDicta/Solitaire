#!/usr/bin/env bash
# Solitaire MCP Server launcher
# Dependencies are vendored in ./vendor/ — no pip install needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/server.py"
