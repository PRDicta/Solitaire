"""
Sync engine for the symbiosis adapter.

Three sync tiers, all shipping in v1:

1. ONE-SHOT: Manual import. User triggers it, runs once, reports results.
   Already built in Phase 1 (ImportOrchestrator).

2. PERIODIC: Scheduled re-import. Checks for new/changed entries at a
   configured interval. Uses the dedup system to skip already-imported
   entries. Stateless between runs: the dedup_key in the rolodex IS the
   sync state.

3. LIVE WATCH: File-system watcher for directories (auto-memory, text).
   Detects new/modified files and imports them automatically. Uses
   polling (not inotify) for cross-platform compatibility and FUSE
   reliability.

The sync engine ties together:
- ReaderRegistry (which reader handles which source)
- ImportOrchestrator (the actual import pipeline)
- Source configs (what's connected and how)
- Sync state (when was the last sync, what happened)

Design: adapter is read-only. No write-back to source systems.
"""

import os
import json
import time
import logging
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum

from .reader_registry import ReaderRegistry
from .import_orchestrator import ImportOrchestrator
from ..core.types import ImportResult

logger = logging.getLogger(__name__)


class SyncTier(Enum):
    ONE_SHOT = "one-shot"
    PERIODIC = "periodic"
    LIVE_WATCH = "live-watch"


class SyncStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class SourceConfig:
    """Configuration for a connected source.

    Stored in the sync state file. Contains everything needed to
    reconnect and re-sync a source.
    """
    source_id: str                       # Reader type (e.g., "auto-memory")
    name: str                            # Human-readable name
    config: Dict[str, Any]               # Reader-specific config (path, etc.)
    sync_tier: SyncTier = SyncTier.ONE_SHOT
    enabled: bool = True
    connected_at: str = ""               # ISO timestamp
    last_sync_at: Optional[str] = None   # ISO timestamp
    last_sync_result: Optional[Dict[str, Any]] = None
    periodic_interval_seconds: int = 3600  # Default: 1 hour
    watch_poll_seconds: int = 30           # Default: 30s poll interval

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "source_id": self.source_id,
            "name": self.name,
            "config": self.config,
            "sync_tier": self.sync_tier.value,
            "enabled": self.enabled,
            "connected_at": self.connected_at,
            "last_sync_at": self.last_sync_at,
            "last_sync_result": self.last_sync_result,
            "periodic_interval_seconds": self.periodic_interval_seconds,
            "watch_poll_seconds": self.watch_poll_seconds,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceConfig":
        tier_str = d.get("sync_tier", "one-shot")
        try:
            tier = SyncTier(tier_str)
        except ValueError:
            tier = SyncTier.ONE_SHOT

        return cls(
            source_id=d["source_id"],
            name=d.get("name", d["source_id"]),
            config=d.get("config", {}),
            sync_tier=tier,
            enabled=d.get("enabled", True),
            connected_at=d.get("connected_at", ""),
            last_sync_at=d.get("last_sync_at"),
            last_sync_result=d.get("last_sync_result"),
            periodic_interval_seconds=d.get("periodic_interval_seconds", 3600),
            watch_poll_seconds=d.get("watch_poll_seconds", 30),
        )


@dataclass
class SyncResult:
    """Outcome of a sync operation."""
    source_name: str
    sync_tier: str
    status: str  # "ok", "error", "skipped"
    import_result: Optional[ImportResult] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    synced_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "source_name": self.source_name,
            "sync_tier": self.sync_tier,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "synced_at": self.synced_at,
        }
        if self.import_result:
            d["imported"] = self.import_result.imported
            d["skipped_duplicate"] = self.import_result.skipped_duplicate
            d["skipped_error"] = self.import_result.skipped_error
            d["total_candidates"] = self.import_result.total_candidates
        if self.error:
            d["error"] = self.error
        return d


class SyncEngine:
    """Manages source connections, sync scheduling, and live watching.

    Usage:
        engine = SyncEngine(
            registry=registry,
            orchestrator_factory=lambda: ImportOrchestrator(...),
            state_dir="/path/to/state",
        )
        engine.load_state()

        # Connect a new source
        engine.connect("auto-memory", "Cowork Memory", {"path": "/..."}, SyncTier.LIVE_WATCH)

        # One-shot sync
        result = engine.sync("Cowork Memory")

        # Start live watchers
        engine.start_watchers()

        # Periodic sync (call from a scheduler)
        results = engine.sync_periodic()
    """

    def __init__(
        self,
        registry: ReaderRegistry,
        orchestrator_factory,  # Callable[[], ImportOrchestrator]
        state_dir: str = "",
    ):
        self.registry = registry
        self.orchestrator_factory = orchestrator_factory
        self.state_dir = state_dir
        self.sources: Dict[str, SourceConfig] = {}  # Keyed by name
        self._watchers: Dict[str, threading.Thread] = {}
        self._watcher_stop: Dict[str, threading.Event] = {}
        self._state_file = os.path.join(state_dir, "sync_state.json") if state_dir else ""

    # ── Connection management ────────────────────────────────────────────

    def connect(
        self,
        source_id: str,
        name: str,
        config: Dict[str, Any],
        sync_tier: SyncTier = SyncTier.ONE_SHOT,
        periodic_interval: int = 3600,
        watch_poll: int = 30,
    ) -> Dict[str, Any]:
        """Connect a new source. Validates config before connecting.

        Returns dict with 'connected' bool and optional 'error' str.
        """
        reader = self.registry.get(source_id)
        if not reader:
            return {"connected": False, "error": f"Unknown source type: {source_id}"}

        # Validate config
        validation = reader.validate(config)
        if not validation.get("valid"):
            return {"connected": False, "error": validation.get("error", "Validation failed")}

        # Check for name collision
        if name in self.sources:
            return {"connected": False, "error": f"Source name already in use: {name}"}

        source = SourceConfig(
            source_id=source_id,
            name=name,
            config=config,
            sync_tier=sync_tier,
            enabled=True,
            connected_at=datetime.now(timezone.utc).isoformat(),
            periodic_interval_seconds=periodic_interval,
            watch_poll_seconds=watch_poll,
        )
        self.sources[name] = source
        self._save_state()

        return {"connected": True, "source": source.to_dict(), **validation}

    def disconnect(self, name: str) -> Dict[str, Any]:
        """Disconnect a source. Stops any active watcher."""
        if name not in self.sources:
            return {"disconnected": False, "error": f"Source not found: {name}"}

        # Stop watcher if running
        self._stop_watcher(name)

        del self.sources[name]
        self._save_state()
        return {"disconnected": True, "name": name}

    def list_sources(self) -> List[Dict[str, Any]]:
        """List all connected sources with their sync status."""
        result = []
        for name, source in sorted(self.sources.items()):
            info = source.to_dict()
            info["watcher_active"] = name in self._watchers
            result.append(info)
        return result

    def get_source(self, name: str) -> Optional[SourceConfig]:
        """Get a source config by name."""
        return self.sources.get(name)

    # ── Sync operations ──────────────────────────────────────────────────

    def sync(self, name: str, dry_run: bool = False) -> SyncResult:
        """Run a one-shot sync for a named source.

        This is the primary sync method. Works for any sync tier:
        one-shot sources get synced on demand, periodic/watch sources
        can also be manually triggered.
        """
        start = time.time()
        synced_at = datetime.now(timezone.utc).isoformat()

        source = self.sources.get(name)
        if not source:
            return SyncResult(
                source_name=name,
                sync_tier="unknown",
                status="error",
                error=f"Source not found: {name}",
                synced_at=synced_at,
            )

        if not source.enabled:
            return SyncResult(
                source_name=name,
                sync_tier=source.sync_tier.value,
                status="skipped",
                error="Source is disabled",
                synced_at=synced_at,
            )

        reader = self.registry.get(source.source_id)
        if not reader:
            return SyncResult(
                source_name=name,
                sync_tier=source.sync_tier.value,
                status="error",
                error=f"Reader not found: {source.source_id}",
                synced_at=synced_at,
            )

        try:
            # Re-validate config (source may have moved/changed)
            validation = reader.validate(source.config)
            if not validation.get("valid"):
                return SyncResult(
                    source_name=name,
                    sync_tier=source.sync_tier.value,
                    status="error",
                    error=f"Validation failed: {validation.get('error')}",
                    synced_at=synced_at,
                )

            # Read candidates
            candidates = reader.read(source.config)

            # Import
            orchestrator = self.orchestrator_factory()
            import_result = orchestrator.run(
                candidates,
                source_id=source.source_id,
                dry_run=dry_run,
            )

            # Update source state
            source.last_sync_at = synced_at
            source.last_sync_result = {
                "imported": import_result.imported,
                "skipped_duplicate": import_result.skipped_duplicate,
                "skipped_error": import_result.skipped_error,
                "total": import_result.total_candidates,
            }
            self._save_state()

            return SyncResult(
                source_name=name,
                sync_tier=source.sync_tier.value,
                status="ok",
                import_result=import_result,
                duration_seconds=time.time() - start,
                synced_at=synced_at,
            )

        except Exception as e:
            logger.error(f"Sync failed for {name}: {e}")
            return SyncResult(
                source_name=name,
                sync_tier=source.sync_tier.value,
                status="error",
                error=str(e),
                duration_seconds=time.time() - start,
                synced_at=synced_at,
            )

    def sync_periodic(self) -> List[SyncResult]:
        """Sync all periodic sources that are due.

        Call this from a scheduler (e.g., every minute). Only syncs
        sources whose interval has elapsed since their last sync.
        """
        results = []
        now = time.time()

        for name, source in self.sources.items():
            if source.sync_tier != SyncTier.PERIODIC:
                continue
            if not source.enabled:
                continue

            # Check if due
            if source.last_sync_at:
                try:
                    last = datetime.fromisoformat(source.last_sync_at)
                    elapsed = now - last.timestamp()
                    if elapsed < source.periodic_interval_seconds:
                        continue
                except (ValueError, OSError):
                    pass  # Can't parse last sync time, sync anyway

            result = self.sync(name)
            results.append(result)

        return results

    # ── Live Watch (Tier 3) ──────────────────────────────────────────────

    def start_watchers(self) -> int:
        """Start file-system watchers for all live-watch sources.

        Returns the number of watchers started. Uses polling, not
        inotify, for FUSE compatibility and cross-platform reliability.
        """
        started = 0
        for name, source in self.sources.items():
            if source.sync_tier != SyncTier.LIVE_WATCH:
                continue
            if not source.enabled:
                continue
            if name in self._watchers:
                continue  # Already watching

            if self._start_watcher(name, source):
                started += 1

        return started

    def stop_watchers(self) -> int:
        """Stop all active watchers."""
        stopped = 0
        for name in list(self._watchers.keys()):
            self._stop_watcher(name)
            stopped += 1
        return stopped

    def _start_watcher(self, name: str, source: SourceConfig) -> bool:
        """Start a polling watcher for a single source."""
        stop_event = threading.Event()
        self._watcher_stop[name] = stop_event

        thread = threading.Thread(
            target=self._watch_loop,
            args=(name, source, stop_event),
            daemon=True,
            name=f"symbiosis-watcher-{name}",
        )
        thread.start()
        self._watchers[name] = thread
        return True

    def _stop_watcher(self, name: str) -> None:
        """Stop a specific watcher."""
        stop_event = self._watcher_stop.pop(name, None)
        if stop_event:
            stop_event.set()

        thread = self._watchers.pop(name, None)
        if thread:
            thread.join(timeout=5.0)

    def _watch_loop(self, name: str, source: SourceConfig, stop: threading.Event) -> None:
        """Polling loop for live-watch sources.

        Tracks file modification times. When a file changes or appears,
        triggers a sync. Uses content hashing for change detection:
        the dedup system handles the rest.
        """
        poll_interval = source.watch_poll_seconds
        last_snapshot: Dict[str, float] = {}  # path -> mtime

        # Build initial snapshot
        last_snapshot = self._snapshot_source(source)

        while not stop.is_set():
            stop.wait(poll_interval)
            if stop.is_set():
                break

            try:
                current = self._snapshot_source(source)

                # Detect changes: new files or modified files
                changed = False
                for path, mtime in current.items():
                    if path not in last_snapshot or mtime > last_snapshot[path]:
                        changed = True
                        break

                if changed:
                    logger.info(f"Change detected in {name}, syncing...")
                    self.sync(name)

                last_snapshot = current

            except Exception as e:
                logger.warning(f"Watch loop error for {name}: {e}")

    def _snapshot_source(self, source: SourceConfig) -> Dict[str, float]:
        """Take a snapshot of file mtimes for a source's path."""
        snapshot = {}
        path = source.config.get("path", "")
        if not path or not os.path.exists(path):
            return snapshot

        if os.path.isfile(path):
            try:
                snapshot[path] = os.path.getmtime(path)
            except OSError:
                pass
        elif os.path.isdir(path):
            try:
                for entry in os.scandir(path):
                    if entry.is_file():
                        try:
                            snapshot[entry.path] = entry.stat().st_mtime
                        except OSError:
                            pass
            except OSError:
                pass

        return snapshot

    # ── Reconciliation Safety Net ────────────────────────────────────────

    def reconcile(self, name: str) -> SyncResult:
        """Full reconciliation sync for a source.

        Same as sync() but with explicit logging and status tracking.
        The dedup system IS the reconciliation: re-importing a source
        that's already been imported will skip all existing entries
        via dedup_key matching. Only genuinely new or modified entries
        get imported.

        This is the "safety net" described in the plan. If periodic
        sync misses something or a live watcher glitches, reconcile
        catches it.
        """
        result = self.sync(name)
        if result.status == "ok":
            logger.info(
                f"Reconciliation for {name}: "
                f"{result.import_result.imported if result.import_result else 0} new, "
                f"{result.import_result.skipped_duplicate if result.import_result else 0} already synced"
            )
        return result

    # ── State persistence ────────────────────────────────────────────────

    def _save_state(self) -> None:
        """Persist source configs to disk."""
        if not self._state_file:
            return
        try:
            state = {
                "sources": {name: src.to_dict() for name, src in self.sources.items()},
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save sync state: {e}")

    def load_state(self) -> int:
        """Load source configs from disk. Returns number of sources loaded."""
        if not self._state_file or not os.path.isfile(self._state_file):
            return 0
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            sources_dict = state.get("sources", {})
            for name, src_dict in sources_dict.items():
                self.sources[name] = SourceConfig.from_dict(src_dict)
            return len(self.sources)
        except Exception as e:
            logger.warning(f"Failed to load sync state: {e}")
            return 0

    def status(self) -> Dict[str, Any]:
        """Get overall sync engine status."""
        return {
            "total_sources": len(self.sources),
            "enabled_sources": sum(1 for s in self.sources.values() if s.enabled),
            "active_watchers": len(self._watchers),
            "sources_by_tier": {
                tier.value: sum(1 for s in self.sources.values() if s.sync_tier == tier)
                for tier in SyncTier
            },
        }
