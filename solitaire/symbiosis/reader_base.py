"""
Base class for symbiosis readers.

Every reader takes a source config and yields IngestCandidates.
The contract: regardless of source format, the output is always
the same intermediate type. The enrichment pipeline doesn't care
where the data came from.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any

from ..core.types import IngestCandidate


class ReaderBase(ABC):
    """Abstract base for all symbiosis readers.

    Subclasses implement:
        source_id:  A stable identifier for this source type (e.g., "auto-memory").
        read():     Yields IngestCandidates from the source.
        validate():  Checks whether the source config is valid before reading.
    """

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Stable identifier for this source type."""
        ...

    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate source config. Returns dict with 'valid' bool and optional 'error' str."""
        ...

    @abstractmethod
    def read(self, config: Dict[str, Any]) -> Iterator[IngestCandidate]:
        """Yield IngestCandidates from the source.

        Must not raise on individual bad entries. Instead, yield what you can
        and skip what you can't, logging errors to the candidate's metadata
        with key 'reader_error'.

        Args:
            config: Source-specific configuration. For file-based sources,
                    typically contains 'path'. For API sources, credentials
                    and endpoint info.
        """
        ...
