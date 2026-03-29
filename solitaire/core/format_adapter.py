"""
The Librarian / Solitaire -- Format Adapter

Phase 3: Engine as Identity Authority.

Adapter that takes structured identity/briefing data and formats it for
the target model. This completes the model-agnostic claim: the engine
produces identity data in a model-neutral structure, and this adapter
renders it for the specific host.

Supported formats:
- "claude": Current output (system prompt blocks with ═══ delimiters)
- "openai": Structured system message (markdown sections)
- "raw": JSON dict (for programmatic consumers)

Default: "claude". Configurable via engine config or direct call.
"""
import json
from typing import Dict, Optional


class FormatAdapter:
    """
    Renders structured boot context blocks for different model targets.

    The engine produces blocks as a dict of section_name -> text content.
    This adapter formats them into the target model's preferred structure.
    """

    SUPPORTED_FORMATS = ("claude", "openai", "raw")

    def __init__(self, format: str = "claude"):
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        self.format = format

    def render(self, blocks: Dict[str, str]) -> str:
        """
        Render boot context blocks into the target format.

        Args:
            blocks: Dict of section_name -> text content. Expected keys
                    include: cognitive_profile, identity, commitments,
                    briefing, residue, user_knowledge, etc.

        Returns:
            Formatted string ready for injection into the model's context.
        """
        if self.format == "claude":
            return self._render_claude(blocks)
        elif self.format == "openai":
            return self._render_openai(blocks)
        elif self.format == "raw":
            return self._render_raw(blocks)
        return self._render_claude(blocks)  # fallback

    def _render_claude(self, blocks: Dict[str, str]) -> str:
        """
        Claude format: sections wrapped in ═══ delimiters.

        This is the current production format. Blocks are already formatted
        with their own delimiters (e.g., ═══ COGNITIVE PROFILE ═══), so
        we join them with double newlines.
        """
        parts = []
        # Ordered rendering: profile first, then identity, then episodic
        section_order = [
            "cognitive_profile",
            "experiential",
            "user_knowledge",
            "resident_knowledge",
            "identity",
            "commitments",
            "briefing",
            "residue",
            "session_tail",
            "intent",
            "tool_proposals",
        ]

        for key in section_order:
            if key in blocks and blocks[key]:
                parts.append(blocks[key].strip())

        # Append any blocks not in the ordered list
        for key, val in blocks.items():
            if key not in section_order and val:
                parts.append(val.strip())

        return "\n\n".join(parts)

    def _render_openai(self, blocks: Dict[str, str]) -> str:
        """
        OpenAI format: markdown-structured system message.

        Strips the ═══ delimiter lines and replaces them with markdown
        headers. Produces a single system message suitable for OpenAI's
        chat completions API.
        """
        parts = []

        section_titles = {
            "cognitive_profile": "Cognitive Profile",
            "identity": "Identity Context",
            "commitments": "Active Commitments",
            "briefing": "Situational Briefing",
            "residue": "Session Context",
            "session_tail": "Recent Conversation",
            "user_knowledge": "User Knowledge",
            "resident_knowledge": "Domain Knowledge",
            "experiential": "Experiential Memory",
            "intent": "Intent Context",
            "tool_proposals": "Available Tools",
        }

        section_order = list(section_titles.keys())

        for key in section_order:
            if key not in blocks or not blocks[key]:
                continue
            title = section_titles.get(key, key.replace("_", " ").title())
            content = self._strip_delimiters(blocks[key])
            if content.strip():
                parts.append(f"## {title}\n\n{content.strip()}")

        # Append unlisted blocks
        for key, val in blocks.items():
            if key not in section_titles and val:
                title = key.replace("_", " ").title()
                content = self._strip_delimiters(val)
                if content.strip():
                    parts.append(f"## {title}\n\n{content.strip()}")

        return "\n\n".join(parts)

    def _render_raw(self, blocks: Dict[str, str]) -> str:
        """
        Raw format: JSON dict of section_name -> content.

        For programmatic consumers that want to parse the data themselves.
        """
        return json.dumps(blocks, indent=2, ensure_ascii=False)

    @staticmethod
    def _strip_delimiters(text: str) -> str:
        """Remove ═══ delimiter lines from a block."""
        lines = text.split("\n")
        stripped = [
            line for line in lines
            if not (line.startswith("═══") and line.endswith("═══"))
        ]
        return "\n".join(stripped)
