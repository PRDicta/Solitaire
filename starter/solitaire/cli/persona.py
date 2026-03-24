"""
Persona commands — list and create personas.

Commands:
    solitaire persona list
    solitaire persona create
"""
import click

from ._engine import get_engine, output_json, output_error


@click.group()
@click.pass_context
def persona(ctx):
    """Persona management."""
    pass


@persona.command("list")
@click.pass_context
def persona_list(ctx):
    """List available personas."""
    engine = get_engine(ctx)
    result = engine.boot_pre_persona()
    # Extract just the persona list
    output_json({
        "personas": result.get("available_personas", []),
        "template_creation_enabled": result.get("template_creation_enabled", False),
    })


@persona.command("create")
@click.pass_context
def persona_create(ctx):
    """Start the persona creation onboarding flow.

    This returns the first step of the onboarding pipeline. The host
    agent walks the user through each step, calling 'onboard flow-step'
    for each response.

    In v1, persona creation uses live research against the user's
    stated intent -- no template library.
    """
    engine = get_engine(ctx)

    # Delegate to the onboarding system
    try:
        from ..core.onboarding_flow import OnboardingFlow
        flow = OnboardingFlow(
            conn=engine._lib.rolodex.conn if engine._lib else None,
            persona_dir=str(engine.persona_dir),
        )
        first_step = flow.start()
        output_json(first_step)
    except ImportError:
        output_error("Onboarding module not available")
    except Exception as e:
        output_error(f"Failed to start onboarding: {e}")
