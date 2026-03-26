"""
Reader registry for the symbiosis adapter.

Central registry where readers register themselves by source_id.
The registry resolves a source_id to the right reader class, validates
config, and provides discovery for CLI commands ("what sources can I
connect to?").

Usage:
    from solitaire.symbiosis.reader_registry import ReaderRegistry

    registry = ReaderRegistry()
    registry.auto_discover()  # Registers all built-in readers

    reader = registry.get("auto-memory")
    validation = reader.validate({"path": "/some/dir"})
    if validation["valid"]:
        for candidate in reader.read({"path": "/some/dir"}):
            ...
"""

from typing import Dict, Optional, List, Type

from .reader_base import ReaderBase


class ReaderRegistry:
    """Singleton-style registry for symbiosis readers.

    Readers are keyed by source_id (e.g., "auto-memory", "jsonl",
    "chatgpt-export"). Each source_id maps to exactly one reader class.
    Registering a duplicate source_id replaces the previous reader.
    """

    def __init__(self):
        self._readers: Dict[str, ReaderBase] = {}

    def register(self, reader: ReaderBase) -> None:
        """Register a reader instance. Keyed by reader.source_id."""
        self._readers[reader.source_id] = reader

    def get(self, source_id: str) -> Optional[ReaderBase]:
        """Look up a reader by source_id. Returns None if not found."""
        return self._readers.get(source_id)

    def list_sources(self) -> List[Dict[str, str]]:
        """List all registered source types with their descriptions.

        Returns list of dicts with 'source_id' and 'description' keys.
        Useful for CLI discovery ("what sources are available?").
        """
        result = []
        for sid, reader in sorted(self._readers.items()):
            result.append({
                "source_id": sid,
                "description": reader.__class__.__doc__.strip().split("\n")[0] if reader.__class__.__doc__ else sid,
                "reader_class": reader.__class__.__name__,
            })
        return result

    def has(self, source_id: str) -> bool:
        """Check if a reader is registered for this source_id."""
        return source_id in self._readers

    @property
    def count(self) -> int:
        """Number of registered readers."""
        return len(self._readers)

    def auto_discover(self) -> int:
        """Register all built-in readers.

        Returns the number of readers registered. This is the standard
        way to populate the registry. Call once at startup.
        """
        registered = 0

        # Auto-memory reader (Cowork .auto-memory format)
        try:
            from .auto_memory_reader import AutoMemoryReader
            self.register(AutoMemoryReader())
            registered += 1
        except ImportError:
            pass

        # JSONL reader (Librarian rolodex export format)
        try:
            from .jsonl_reader import JSONLReader
            self.register(JSONLReader())
            registered += 1
        except ImportError:
            pass

        # ChatGPT export reader
        try:
            from .chatgpt_reader import ChatGPTExportReader
            self.register(ChatGPTExportReader())
            registered += 1
        except ImportError:
            pass

        # Raw text reader
        try:
            from .text_reader import TextReader
            self.register(TextReader())
            registered += 1
        except ImportError:
            pass

        return registered


# Module-level convenience: a default registry instance.
# Import and call auto_discover() to populate.
default_registry = ReaderRegistry()
