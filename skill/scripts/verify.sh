#!/usr/bin/env bash
# Solitaire — Verify script for agentskills.io skill packaging
# Runs a smoke test: boot -> ingest -> recall -> end against a temp workspace.
#
# Exit codes:
#   0 — all checks passed
#   1 — one or more checks failed
set -euo pipefail

PASS=0
FAIL=0
WORKSPACE=""

log()  { echo "[solitaire-verify] $*"; }
err()  { echo "[solitaire-verify] FAIL: $*" >&2; }
pass() { log "PASS: $1"; PASS=$((PASS + 1)); }
fail() { err "$1"; FAIL=$((FAIL + 1)); }

cleanup() {
    [ -n "$WORKSPACE" ] && [ -d "$WORKSPACE" ] && rm -rf "$WORKSPACE"
}
trap cleanup EXIT

# ── Resolve CLI ──────────────────────────────────────────────────────────

resolve_cli() {
    if command -v solitaire &>/dev/null; then
        echo "solitaire"
        return
    fi
    local py=""
    for candidate in python3 python; do
        if command -v "$candidate" &>/dev/null; then py="$candidate"; break; fi
    done
    if [ -n "$py" ] && "$py" -c "import solitaire" &>/dev/null; then
        echo "$py -m solitaire"
        return
    fi
    echo ""
}

run_cmd() {
    local desc="$1"; shift
    local output
    if output=$(SOLITAIRE_WORKSPACE="$WORKSPACE" eval "$@" 2>/dev/null); then
        echo "$output"
        return 0
    else
        fail "$desc"
        echo ""
        return 1
    fi
}

json_field() {
    python3 -c "import sys,json; d=json.load(sys.stdin); print(d$1)" 2>/dev/null
}

# ── Main ─────────────────────────────────────────────────────────────────

main() {
    log "Running Solitaire smoke test..."

    WORKSPACE=$(mktemp -d -t solitaire-verify-XXXXXXXX)
    log "Temp workspace: $WORKSPACE"

    local cli
    cli=$(resolve_cli)
    if [ -z "$cli" ]; then
        fail "Could not find solitaire CLI or Python module"
        log "Results: $PASS passed, $FAIL failed"
        exit 1
    fi
    log "Using CLI: $cli"

    # ── 0. Import check ─────────────────────────────────────────────────

    local py=""
    for candidate in python3 python; do
        command -v "$candidate" &>/dev/null && { py="$candidate"; break; }
    done
    if [ -n "$py" ] && "$py" -c "from solitaire import SolitaireEngine, __version__" &>/dev/null; then
        pass "Python import"
    else
        fail "Python import"
    fi

    # ── 1. Boot ──────────────────────────────────────────────────────────

    local output status

    output=$(run_cmd "boot --pre-persona" "$cli boot --pre-persona") || true
    status=$(echo "$output" | json_field "['status']") || true
    [ "$status" = "ok" ] && pass "boot --pre-persona" || fail "boot --pre-persona"

    # Create minimal default persona
    mkdir -p "$WORKSPACE/personas/default"
    cat > "$WORKSPACE/personas/default/persona.yaml" << 'YAML'
identity:
  name: Default
  role: assistant
  description: Default verification persona
traits:
  observance: 0.5
  assertiveness: 0.5
  conviction: 0.5
  warmth: 0.5
  humor: 0.5
  initiative: 0.5
  empathy: 0.5
YAML

    output=$(run_cmd "boot --persona default" "$cli boot --persona default --cold") || true
    status=$(echo "$output" | json_field "['status']") || true
    [ "$status" = "ok" ] && pass "boot --persona default" || fail "boot --persona default"

    # ── 2. Remember ──────────────────────────────────────────────────────

    output=$(run_cmd "remember" "$cli remember 'Test user prefers dark mode'") || true
    if [ -n "$output" ]; then
        pass "remember"
    else
        fail "remember"
    fi

    # ── 3. Ingest ────────────────────────────────────────────────────────

    # Use positional args (not stdin) and messages long enough to pass ingestion filters
    output=$(run_cmd "ingest-turn" \
        "$cli ingest-turn 'I need to review the quarterly revenue report for Q1 2026 and compare it against our projections from last December' 'Based on the Q1 2026 data, revenue came in at 2.3M against a projected 2.1M, representing a 9.5 percent beat on the December forecast'") || true
    if echo "$output" | grep -q '"user"'; then
        pass "ingest-turn"
    else
        fail "ingest-turn"
    fi

    # ── 4. Recall ────────────────────────────────────────────────────────

    output=$(run_cmd "recall" "$cli recall 'color preferences' --no-preflight") || true
    if [ -n "$output" ]; then
        local found
        found=$(echo "$output" | json_field ".get('entries_found', 0)") || true
        pass "recall ($found entries)"
    else
        fail "recall"
    fi

    # ── 5. Pulse ─────────────────────────────────────────────────────────

    output=$(run_cmd "pulse" "$cli pulse") || true
    if echo "$output" | grep -q '"alive"'; then
        pass "pulse"
    else
        fail "pulse"
    fi

    # ── 6. End ───────────────────────────────────────────────────────────

    output=$(run_cmd "end" "$cli end 'Verification complete'") || true
    status=$(echo "$output" | json_field "['status']") || true
    [ "$status" = "ok" ] && pass "end" || fail "end"

    # ── Summary ──────────────────────────────────────────────────────────

    echo ""
    log "Results: $PASS passed, $FAIL failed"
    [ "$FAIL" -gt 0 ] && exit 1
    log "All checks passed."
    exit 0
}

main "$@"
