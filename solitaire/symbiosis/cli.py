"""
CLI interface for the symbiosis adapter.

Commands:
    adapter sources          List available source types (from registry)
    adapter connect          Connect a new source
    adapter disconnect       Disconnect a source
    adapter import           One-shot import from a connected source
    adapter sync             Sync a source (respects sync tier)
    adapter sync-all         Sync all due periodic sources
    adapter status           Show connected sources and sync status
    adapter reconcile        Full reconciliation for a source
    adapter watch-start      Start live watchers
    adapter watch-stop       Stop live watchers

All commands return JSON for programmatic consumption.
The CLI is designed to be called from the Librarian CLI or from
a standalone Solitaire CLI.
"""

import json
import sys
import os
from typing import Dict, Any, Optional, List

from .reader_registry import ReaderRegistry
from .sync_engine import SyncEngine, SyncTier
from .import_orchestrator import ImportOrchestrator


def _json_out(data: Any) -> str:
    """Serialize to compact JSON for CLI output."""
    return json.dumps(data, indent=2, default=str)


class AdapterCLI:
    """CLI handler for symbiosis adapter commands.

    Wraps the SyncEngine and ReaderRegistry with CLI-friendly
    input/output. Each method corresponds to a CLI subcommand and
    returns a dict suitable for JSON serialization.
    """

    def __init__(self, sync_engine: SyncEngine, registry: ReaderRegistry):
        self.engine = sync_engine
        self.registry = registry

    # ── Discovery ────────────────────────────────────────────────────────

    def cmd_sources(self) -> Dict[str, Any]:
        """List available source types from the reader registry.

        Shows what kinds of sources can be connected (auto-memory,
        jsonl, chatgpt-export, text).
        """
        sources = self.registry.list_sources()
        return {
            "status": "ok",
            "available_sources": sources,
            "count": len(sources),
        }

    # ── Connection management ────────────────────────────────────────────

    def cmd_connect(
        self,
        source_id: str,
        name: str,
        config: Dict[str, Any],
        sync_tier: str = "one-shot",
        periodic_interval: int = 3600,
        watch_poll: int = 30,
    ) -> Dict[str, Any]:
        """Connect a new source.

        Args:
            source_id: Reader type (e.g., "auto-memory", "jsonl")
            name: Human-readable name for this connection
            config: Reader-specific config (e.g., {"path": "/some/dir"})
            sync_tier: "one-shot", "periodic", or "live-watch"
            periodic_interval: Seconds between periodic syncs (default 3600)
            watch_poll: Seconds between watch polls (default 30)
        """
        try:
            tier = SyncTier(sync_tier)
        except ValueError:
            return {"status": "error", "error": f"Invalid sync tier: {sync_tier}. Use: one-shot, periodic, live-watch"}

        result = self.engine.connect(
            source_id=source_id,
            name=name,
            config=config,
            sync_tier=tier,
            periodic_interval=periodic_interval,
            watch_poll=watch_poll,
        )

        if result.get("connected"):
            return {"status": "ok", **result}
        else:
            return {"status": "error", **result}

    def cmd_disconnect(self, name: str) -> Dict[str, Any]:
        """Disconnect a source by name."""
        result = self.engine.disconnect(name)
        if result.get("disconnected"):
            return {"status": "ok", **result}
        else:
            return {"status": "error", **result}

    # ── Import / Sync ────────────────────────────────────────────────────

    def cmd_import(self, name: str, dry_run: bool = False) -> Dict[str, Any]:
        """Run a one-shot import for a connected source.

        This is the primary way to import data. The dedup system
        ensures re-running is safe (duplicates are skipped).
        """
        result = self.engine.sync(name, dry_run=dry_run)
        output = result.to_dict()
        output["dry_run"] = dry_run
        return output

    def cmd_sync(self, name: str) -> Dict[str, Any]:
        """Sync a specific source (same as import but uses sync naming)."""
        result = self.engine.sync(name)
        return result.to_dict()

    def cmd_sync_all(self) -> Dict[str, Any]:
        """Sync all periodic sources that are due."""
        results = self.engine.sync_periodic()
        return {
            "status": "ok",
            "synced": len(results),
            "results": [r.to_dict() for r in results],
        }

    def cmd_reconcile(self, name: str) -> Dict[str, Any]:
        """Full reconciliation sync for a source."""
        result = self.engine.reconcile(name)
        return result.to_dict()

    # ── Status ───────────────────────────────────────────────────────────

    def cmd_status(self) -> Dict[str, Any]:
        """Show overall sync engine status and connected sources."""
        engine_status = self.engine.status()
        sources = self.engine.list_sources()
        return {
            "status": "ok",
            **engine_status,
            "sources": sources,
        }

    # ── Live Watch ───────────────────────────────────────────────────────

    def cmd_watch_start(self) -> Dict[str, Any]:
        """Start file-system watchers for all live-watch sources."""
        started = self.engine.start_watchers()
        return {
            "status": "ok",
            "watchers_started": started,
            "active_watchers": len(self.engine._watchers),
        }

    def cmd_watch_stop(self) -> Dict[str, Any]:
        """Stop all active watchers."""
        stopped = self.engine.stop_watchers()
        return {
            "status": "ok",
            "watchers_stopped": stopped,
        }

    # ── Dispatch ─────────────────────────────────────────────────────────

    def dispatch(self, args: List[str]) -> Dict[str, Any]:
        """Dispatch a CLI command from argument list.

        Expected format: ["adapter", "<subcommand>", ...args]
        or just ["<subcommand>", ...args] if 'adapter' prefix is stripped.

        Returns a dict suitable for JSON serialization.
        """
        if not args:
            return self._help()

        cmd = args[0].lower().replace("-", "_")

        if cmd == "sources":
            return self.cmd_sources()

        elif cmd == "connect":
            if len(args) < 4:
                return {"status": "error", "error": "Usage: adapter connect <source_id> <name> <path> [sync_tier]"}
            source_id = args[1]
            name = args[2]
            path = args[3]
            sync_tier = args[4] if len(args) > 4 else "one-shot"
            return self.cmd_connect(source_id, name, {"path": path}, sync_tier)

        elif cmd == "disconnect":
            if len(args) < 2:
                return {"status": "error", "error": "Usage: adapter disconnect <name>"}
            return self.cmd_disconnect(args[1])

        elif cmd == "import":
            if len(args) < 2:
                return {"status": "error", "error": "Usage: adapter import <name> [--dry-run]"}
            dry_run = "--dry-run" in args
            return self.cmd_import(args[1], dry_run=dry_run)

        elif cmd == "sync":
            if len(args) < 2:
                return {"status": "error", "error": "Usage: adapter sync <name>"}
            return self.cmd_sync(args[1])

        elif cmd == "sync_all":
            return self.cmd_sync_all()

        elif cmd == "reconcile":
            if len(args) < 2:
                return {"status": "error", "error": "Usage: adapter reconcile <name>"}
            return self.cmd_reconcile(args[1])

        elif cmd == "status":
            return self.cmd_status()

        elif cmd == "watch_start":
            return self.cmd_watch_start()

        elif cmd == "watch_stop":
            return self.cmd_watch_stop()

        elif cmd in ("help", "h"):
            return self._help()

        else:
            return {"status": "error", "error": f"Unknown command: {cmd}", "help": self._help()["commands"]}

    def _help(self) -> Dict[str, Any]:
        """Return help text as structured data."""
        return {
            "status": "ok",
            "commands": {
                "sources": "List available source types",
                "connect <source_id> <name> <path> [tier]": "Connect a new source",
                "disconnect <name>": "Disconnect a source",
                "import <name> [--dry-run]": "One-shot import",
                "sync <name>": "Sync a source",
                "sync-all": "Sync all due periodic sources",
                "reconcile <name>": "Full reconciliation",
                "status": "Show connected sources",
                "watch-start": "Start live watchers",
                "watch-stop": "Stop live watchers",
            },
            "sync_tiers": {
                "one-shot": "Manual import only",
                "periodic": "Auto-sync at configured interval",
                "live-watch": "File-system polling for changes",
            },
        }
