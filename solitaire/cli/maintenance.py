"""
Maintenance commands — system upkeep and health checks.

Commands:
    solitaire maintenance run
"""
import os
from pathlib import Path

import click

from ._engine import get_engine, output_json, output_error


@click.group(name="maintain")
@click.pass_context
def maintenance(ctx):
    """System maintenance operations."""
    pass


@maintenance.command("run")
@click.option("--full", is_flag=True, help="Run full maintenance cycle")
@click.pass_context
def maintain_run(ctx, full):
    """Run maintenance cycle (weight adjustment, pattern detection, cleanup)."""
    engine = get_engine(ctx)
    results = {}

    # Adjust retrieval weights
    try:
        from ..core.retrieval_feedback import adjust_weights
        weight_result = adjust_weights(
            conn=engine._lib.rolodex.conn,
            session_id=engine._session_id,
        )
        results["weight_adjustment"] = weight_result
    except Exception as e:
        results["weight_adjustment"] = {"error": str(e)}

    # Pattern report
    try:
        results["patterns"] = engine.get_patterns()
    except Exception as e:
        results["patterns"] = {"error": str(e)}

    # Tool proposals check
    try:
        proposals = engine.get_tool_proposals()
        results["pending_proposals"] = len(proposals)
    except Exception:
        results["pending_proposals"] = 0

    # Run MaintenanceEngine consolidation passes (Phase 5)
    try:
        from ..core.maintenance import MaintenanceEngine
        conn = engine._lib.rolodex.conn
        me = MaintenanceEngine(
            conn=conn,
            session_id=engine._session_id,
            token_budget=15000,
            workspace=Path(ctx.obj.get("workspace", os.getcwd())),
        )
        consolidation_result = me.run_all()
        results["consolidation"] = consolidation_result.get("summary", {})

        # Generate and output the human-readable report
        report = me.generate_consolidation_report()
        results["consolidation_report"] = report

        # Ingest the report as a system entry for future recall
        if consolidation_result.get("summary", {}).get("total_actions", 0) > 0:
            _ingest_consolidation_report(conn, report, consolidation_result.get("maintenance_id", ""))
    except Exception as e:
        results["consolidation"] = {"error": str(e)}

    output_json({"status": "ok", "maintenance": results})


def _ingest_consolidation_report(conn, report: str, maintenance_id: str) -> None:
    """Store the consolidation report as a searchable rolodex entry."""
    import uuid
    import json
    from datetime import datetime, timezone

    entry_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        conn.execute(
            """INSERT INTO rolodex_entries
               (id, conversation_id, content, content_type, category, tags,
                source_range, created_at, tier, metadata, provenance)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry_id,
                "maintenance",
                report,
                "structured",
                "note",
                json.dumps(["maintenance", "consolidation-report", date_tag]),
                json.dumps({}),
                now,
                "cold",
                json.dumps({"consolidation_run_id": maintenance_id}),
                "system",
            ),
        )
        # Update FTS index
        conn.execute(
            """INSERT INTO rolodex_fts (entry_id, content, tags, category)
               VALUES (?, ?, ?, ?)""",
            (entry_id, report, f"maintenance consolidation-report {date_tag}", "note"),
        )
        conn.commit()
    except Exception:
        pass  # Non-critical
