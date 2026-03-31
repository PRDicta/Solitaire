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
from .environment_scanner import scan_environment, ScanResult
from .priority_ranker import classify_corpus, IngestionPlan


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

    # ── Smart Capture ────────────────────────────────────────────────────

    def cmd_scan(
        self,
        workspace: str = "",
        extra_paths: list = None,
        own_db: str = "",
    ) -> Dict[str, Any]:
        """Scan the environment for existing memory sources.

        Returns detected sources with size estimates and an ingestion plan.
        No file contents are read, only filesystem metadata.
        """
        result = scan_environment(
            workspace=workspace or None,
            extra_paths=extra_paths,
            own_db=own_db or None,
        )

        output = result.to_dict()

        # Add ingestion plan if sources were found
        if result.has_sources:
            plan = classify_corpus(
                entry_count=result.total_entry_estimate,
                size_bytes=result.total_size_bytes,
            )
            output["ingestion_plan"] = plan.to_dict()
            output["combined_age_description"] = result.combined_age_description
            output["total_size_description"] = result.total_size_description

        output["status"] = "ok"
        return output

    def cmd_capture(
        self,
        workspace: str = "",
        source_ids: list = None,
        auto: bool = False,
        chunk_mb: float = 10.0,
    ) -> Dict[str, Any]:
        """Run Smart Capture: scan + connect + ingest detected sources.

        This is the high-level command that ties together scanning,
        source connection, and one-shot import. For use during onboarding
        or as a standalone command for existing users.

        Args:
            workspace: Solitaire workspace directory.
            source_ids: Specific source IDs to capture (None = all detected).
            auto: Skip consent prompts (for power users / automation).
            chunk_mb: First-chunk budget in MB for large corpora.
        """
        # Step 1: Scan
        scan = scan_environment(workspace=workspace or None)
        if not scan.has_sources:
            return {
                "status": "ok",
                "message": "No memory sources detected.",
                "sources_found": 0,
            }

        # Filter to requested sources
        sources = scan.sources
        if source_ids:
            sources = [s for s in sources if s.source_id in source_ids]
            if not sources:
                return {
                    "status": "error",
                    "error": f"None of the requested sources found: {source_ids}",
                    "available": [s.source_id for s in scan.sources],
                }

        # Step 2: Connect and import each source
        results = []
        for source in sources:
            # Connect via sync engine
            connect_result = self.engine.connect(
                source_id=source.source_id,
                name=source.display_name,
                config={"path": source.path},
                sync_tier=SyncTier.ONE_SHOT,
            )

            if not connect_result.get("connected"):
                results.append({
                    "source": source.source_id,
                    "status": "error",
                    "error": connect_result.get("error", "Connection failed"),
                })
                continue

            # Import
            sync_result = self.engine.sync(source.display_name)
            results.append({
                "source": source.source_id,
                "name": source.display_name,
                "status": sync_result.status,
                "imported": sync_result.import_result.imported if sync_result.import_result else 0,
                "skipped": sync_result.import_result.skipped_duplicate if sync_result.import_result else 0,
                "duration": round(sync_result.duration_seconds, 2),
            })

        total_imported = sum(r.get("imported", 0) for r in results)
        return {
            "status": "ok",
            "sources_processed": len(results),
            "total_imported": total_imported,
            "results": results,
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

        elif cmd == "scan":
            workspace = args[1] if len(args) > 1 else ""
            extra = args[2:] if len(args) > 2 else None
            return self.cmd_scan(workspace=workspace, extra_paths=extra)

        elif cmd == "capture":
            auto = "--auto" in args
            workspace = ""
            chunk_mb = 10.0
            clean_args = []
            i = 1
            while i < len(args):
                if args[i] == "--workspace" and i + 1 < len(args):
                    workspace = args[i + 1]
                    i += 2
                elif args[i] == "--chunk-mb" and i + 1 < len(args):
                    try:
                        chunk_mb = float(args[i + 1])
                    except ValueError:
                        pass
                    i += 2
                elif args[i].startswith("--"):
                    i += 1
                else:
                    clean_args.append(args[i])
                    i += 1
            source_ids = clean_args if clean_args else None
            return self.cmd_capture(
                workspace=workspace,
                source_ids=source_ids,
                auto=auto,
                chunk_mb=chunk_mb,
            )

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
                "scan [workspace] [extra_paths...]": "Detect memory sources in environment",
                "capture [source_ids...] [--auto]": "Smart Capture: scan + connect + ingest",
            },
            "sync_tiers": {
                "one-shot": "Manual import only",
                "periodic": "Auto-sync at configured interval",
                "live-watch": "File-system polling for changes",
            },
        }
