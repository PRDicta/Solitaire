# Changelog

All notable changes to Solitaire are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.4.0] - 2026-03-29

### Added
- Identity resonance, self-healing knowledge graph, cognitive consolidation
- Identity scaffolding wired into onboarding
- Over-hedging signature detection
- Smart capture onboarding (environment scanner, 7-reader registry)
- Outbound writing gate (5-layer quality enforcement)
- pytest-cov coverage threshold (80% floor) in CI
- SECURITY.md with vulnerability disclosure process
- `solitaire restore` command for backup recovery
- FTS rebuild capability via `solitaire rebuild-index`
- Queue write-ahead journal for crash recovery
- Pre-operation safety backups before destructive operations
- Hook error logging to `.solitaire/hook-errors.log`

### Changed
- Subprocess timeouts: network calls reduced from 60s to 15s
- pip install timeout reduced from 120s to 60s

### Fixed
- Silent exception handlers in identity measurement now log to stderr
- `correct()` cross-session contamination
- Session resume JSONL/pressure rebinding
- Identity graph core classification
- Shared knowledge FTS sanitization
- Datetime naive/aware mismatch in supersession collapse

## [1.3.3] - 2026-03-28

### Fixed
- `correct()` cross-session contamination and resume JSONL/pressure rebinding

## [1.3.2] - 2026-03-28

### Fixed
- Identity graph core classification and shared knowledge FTS sanitization

## [1.3.1] - 2026-03-27

### Fixed
- Codex review fixes (session handoff, test cleanup)
- Session tail (red-hot context) and residue history
- Codex launch blockers and README updates

## [1.3.0] - 2026-03-27

### Added
- Universal auto-update system (git-based)
- Claim scanner: preflight gate and Stop hook for unverified state assertions
- Persona backup added to rolling backup system

### Changed
- Update mechanism switched from zip-download to git-based

## [1.2.0] - 2026-03-27

### Added
- Cognitive profile: 5-band traits, texture layer, growth milestones
- Behavioral genome wired into boot context

### Fixed
- Persona trait rendering: surface all 7 dimensions

## [1.1.0] - 2026-03-26

### Added
- Rolling backup system with atomic SQLite backups
- Behavioral scorer promoted to primary
- Claude Code auto-ingest hook
- Writing standards enforcement in engine ops block
- Session handoff reliability improvements

## [1.0.0] - 2026-03-24

### Added
- Initial release
- Persistent memory with full-text search (FTS5)
- Knowledge graph with entity extraction
- Persona system with adaptive drift
- Token compression engine
- Claude Code hook integration (boot, recall, ingest)
- AGPL-3.0 + commercial dual license
