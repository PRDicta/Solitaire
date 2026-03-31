"""
Symbiosis commands -- external memory import and Smart Capture.

Commands:
    solitaire symbiosis scan [--workspace PATH]
    solitaire symbiosis capture [--workspace PATH] [--auto] [--chunk-mb N] [SOURCE_IDS...]
    solitaire symbiosis sources
    solitaire symbiosis status
"""
import click

from ._engine import get_engine, output_json


def _get_adapter_cli(engine):
    """Build an AdapterCLI from the engine's workspace."""
    from ..symbiosis.reader_registry import default_registry
    from ..symbiosis.sync_engine import SyncEngine
    from ..symbiosis.import_orchestrator import ImportOrchestrator
    from ..symbiosis.cli import AdapterCLI

    registry = default_registry
    registry.auto_discover()

    workspace_dir = engine.workspace_dir

    def _make_orchestrator():
        # The engine's internal Librarian holds the rolodex and DB connection
        lib = getattr(engine, "_lib", None)
        if lib and hasattr(lib, "rolodex"):
            return ImportOrchestrator(
                rolodex=lib.rolodex,
                conn=lib.rolodex.conn,
                session_id=getattr(engine, "_session_id", "") or "",
            )
        # Fallback: open a fresh connection (read-only scan/status still work)
        import sqlite3
        db_path = str(workspace_dir / "rolodex.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        from ..storage.rolodex import Rolodex
        rolodex = Rolodex(conn)
        return ImportOrchestrator(rolodex=rolodex, conn=conn)

    sync = SyncEngine(
        registry=registry,
        orchestrator_factory=_make_orchestrator,
        state_dir=str(workspace_dir),
    )
    return AdapterCLI(sync_engine=sync, registry=registry)


@click.group()
@click.pass_context
def symbiosis(ctx):
    """External memory import and Smart Capture."""
    pass


@symbiosis.command("scan")
@click.option("--workspace", default="", help="Workspace directory to scan")
@click.pass_context
def symbiosis_scan(ctx, workspace):
    """Scan environment for existing memory sources."""
    engine = get_engine(ctx)
    adapter = _get_adapter_cli(engine)
    ws = workspace or str(engine.workspace_dir)
    result = adapter.cmd_scan(workspace=ws, own_db=str(engine.workspace_dir / "rolodex.db"))
    output_json(result)


@symbiosis.command("capture")
@click.option("--workspace", default="", help="Workspace directory")
@click.option("--auto", "auto_mode", is_flag=True, help="Skip consent prompts")
@click.option("--chunk-mb", default=10.0, type=float, help="First-chunk budget in MB")
@click.argument("source_ids", nargs=-1, required=False)
@click.pass_context
def symbiosis_capture(ctx, workspace, auto_mode, chunk_mb, source_ids):
    """Smart Capture: scan, connect, and ingest detected sources."""
    engine = get_engine(ctx)
    adapter = _get_adapter_cli(engine)
    ws = workspace or str(engine.workspace_dir)
    result = adapter.cmd_capture(
        workspace=ws,
        source_ids=list(source_ids) if source_ids else None,
        auto=auto_mode,
        chunk_mb=chunk_mb,
    )
    output_json(result)


@symbiosis.command("sources")
@click.pass_context
def symbiosis_sources(ctx):
    """List available source types from the reader registry."""
    engine = get_engine(ctx)
    adapter = _get_adapter_cli(engine)
    result = adapter.cmd_sources()
    output_json(result)


@symbiosis.command("status")
@click.pass_context
def symbiosis_status(ctx):
    """Show connected sources and sync status."""
    engine = get_engine(ctx)
    adapter = _get_adapter_cli(engine)
    result = adapter.cmd_status()
    output_json(result)
