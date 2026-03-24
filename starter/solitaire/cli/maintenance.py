"""
Maintenance commands — system upkeep and health checks.

Commands:
    solitaire maintenance run
"""
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

    output_json({"status": "ok", "maintenance": results})
