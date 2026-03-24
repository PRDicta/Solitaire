"""
Solitaire — Persistent memory and evolving identity for AI agents.

The core engine: ingestion, retrieval, persona management, session continuity.
Model-agnostic by design. Returns structured data; the host agent decides
how to inject it into whatever model it uses.

Quick start:

    from solitaire import SolitaireEngine

    engine = SolitaireEngine(workspace_dir="/path/to/data")
    engine.boot(persona_key="default", intent="working on financials")
    engine.ingest(user_msg="...", assistant_msg="...")
    context = engine.recall(query="pricing history")
    engine.remember(fact="Client X prefers email")
    engine.end(summary="Reviewed Q1 pricing")
"""
from .__version__ import __version__
from .engine import SolitaireEngine

__all__ = ["SolitaireEngine", "__version__"]
