"""
Boot commands — engine initialization and persona loading.

Commands:
    solitaire boot --pre-persona          Stage 1: list available personas
    solitaire boot --persona KEY          Stage 2: boot with persona
    solitaire boot --resume               Post-compaction: resume last persona
"""
import click

from ._engine import get_engine, output_json, output_error


@click.group(invoke_without_command=True)
@click.option("--pre-persona", is_flag=True, help="Stage 1: return persona list without booting")
@click.option("--persona", "persona_key", default=None, help="Persona key to boot")
@click.option("--resume", is_flag=True, help="Resume last active persona (post-compaction)")
@click.option("--intent", default="", help="Context about what the user is working on")
@click.option("--cold", is_flag=True, help="Skip experiential memory and residue (testing)")
@click.pass_context
def boot(ctx, pre_persona, persona_key, resume, intent, cold):
    """Initialize the Solitaire engine."""
    engine = get_engine(ctx, auto_resume=False)

    if pre_persona:
        result = engine.boot_pre_persona()
        output_json(result)
        return

    if resume:
        result = engine.boot(resume=True, intent=intent, cold=cold)
        output_json(result)
        return

    if not persona_key:
        output_error(
            "Must specify --pre-persona, --persona KEY, or --resume",
            exit_code=2,
        )
        return

    result = engine.boot(persona_key=persona_key, intent=intent, cold=cold)
    output_json(result)
