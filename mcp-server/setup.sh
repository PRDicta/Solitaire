#!/usr/bin/env bash
# Install vendored dependencies for the Solitaire MCP server.
# Run once after cloning, or after adding new dependencies.
# The vendor/ directory is gitignored — each environment installs its own.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENDOR_DIR="$SCRIPT_DIR/vendor"

echo "Installing MCP server dependencies into $VENDOR_DIR ..."
pip install --target="$VENDOR_DIR" --upgrade \
    mcp \
    httpx \
    pydantic \
    2>&1

echo "Done. Vendored dependencies installed to $VENDOR_DIR"
