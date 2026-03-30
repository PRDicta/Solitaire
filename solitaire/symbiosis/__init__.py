"""
Symbiosis Adapter -- External memory ingestion for Solitaire.

Reads from external memory systems (Cowork .auto-memory, JSONL stores,
ChatGPT exports, text files, CLAUDE.md, markdown KBs, other Solitaire
instances) and feeds them through the enrichment pipeline.

Phase 1: .auto-memory reader + one-shot import.      [SHIPPED]
Phase 2: Reader registry + JSONL/ChatGPT readers.    [SHIPPED]
Phase 3: Sync engine (periodic + live watch).         [SHIPPED]
Phase 4: CLI commands + documentation.                [SHIPPED]
Phase 5: Smart Capture (environment scanner, priority ranker,
         CLAUDE.md reader, markdown KB reader, Solitaire reader).  [SHIPPED]

Quick start:
    from solitaire.symbiosis import ReaderRegistry, SyncEngine, AdapterCLI
    from solitaire.symbiosis import AutoMemoryReader, JSONLReader, ChatGPTExportReader, TextReader

    registry = ReaderRegistry()
    registry.auto_discover()  # Registers all built-in readers
"""

# Phase 1
from .reader_base import ReaderBase
from .auto_memory_reader import AutoMemoryReader
from .import_orchestrator import ImportOrchestrator

# Phase 2
from .reader_registry import ReaderRegistry, default_registry
from .jsonl_reader import JSONLReader
from .chatgpt_reader import ChatGPTExportReader
from .text_reader import TextReader

# Phase 3
from .sync_engine import SyncEngine, SyncTier, SyncStatus, SourceConfig, SyncResult

# Phase 4
from .cli import AdapterCLI

# Phase 5: Smart Capture
from .claude_md_reader import ClaudeMdReader
from .markdown_kb_reader import MarkdownKBReader
from .solitaire_reader import SolitaireReader
from .environment_scanner import DetectedSource, ScanResult, scan_environment
from .priority_ranker import (
    IngestionPlan, IngestionStrategy, classify_corpus,
    heuristic_priority_score, rank_candidates_heuristic,
    build_classification_prompt, parse_classification_response,
    rank_candidates_llm,
)

__all__ = [
    # Phase 1
    "ReaderBase",
    "AutoMemoryReader",
    "ImportOrchestrator",
    # Phase 2
    "ReaderRegistry",
    "default_registry",
    "JSONLReader",
    "ChatGPTExportReader",
    "TextReader",
    # Phase 3
    "SyncEngine",
    "SyncTier",
    "SyncStatus",
    "SourceConfig",
    "SyncResult",
    # Phase 4
    "AdapterCLI",
    # Phase 5: Smart Capture
    "ClaudeMdReader",
    "MarkdownKBReader",
    "SolitaireReader",
    "DetectedSource",
    "ScanResult",
    "scan_environment",
    "IngestionPlan",
    "IngestionStrategy",
    "classify_corpus",
    "heuristic_priority_score",
    "rank_candidates_heuristic",
    "build_classification_prompt",
    "parse_classification_response",
    "rank_candidates_llm",
]
