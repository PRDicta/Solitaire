"""
Onboarding commands -- persona creation flow.

Commands:
    solitaire onboard create [--intent "..."]
    solitaire onboard flow-step <step_id> <input>
"""
import click

from ._engine import get_engine, output_json, output_error


@click.group()
@click.pass_context
def onboard(ctx):
    """Persona creation onboarding flow."""
    pass


@onboard.command("create")
@click.option("--intent", default=None, help="User intent to pre-fill")
@click.pass_context
def onboard_create(ctx, intent):
    """Start the persona creation onboarding flow."""
    engine = get_engine(ctx)
    result = engine.onboard_start(intent=intent)
    output_json(result)


@onboard.command("flow-step")
@click.argument("step_id")
@click.argument("user_input")
@click.pass_context
def onboard_flow_step(ctx, step_id, user_input):
    """Process a single step in the onboarding flow."""
    engine = get_engine(ctx)
    result = engine.onboard_flow_step(step_id, user_input)
    output_json(result)
