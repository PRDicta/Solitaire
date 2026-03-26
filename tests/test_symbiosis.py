"""
Comprehensive test suite for the Solitaire symbiosis adapter.

Covers all four phases:
- Phase 1: IngestCandidate types, AutoMemoryReader, ImportOrchestrator
- Phase 2: ReaderRegistry, JSONLReader, ChatGPTExportReader, TextReader
- Phase 3: SyncEngine (connect, sync, periodic, watch, reconcile, state)
- Phase 4: AdapterCLI (dispatch, all subcommands)
"""

import os
import json
import time
import tempfile
import shutil
import sqlite3
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

import pytest

# Direct imports to bypass broken engine.py
from solitaire.core.types import (
    IngestCandidate,
    IngestContentType,
    EnrichmentHint,
    ImportResult,
    RolodexEntry,
    ContentModality,
    EntryCategory,
    Tier,
)
from solitaire.symbiosis.reader_base import ReaderBase
from solitaire.symbiosis.auto_memory_reader import (
    AutoMemoryReader,
    _parse_frontmatter,
    _compute_dedup_key,
)
from solitaire.symbiosis.import_orchestrator import (
    ImportOrchestrator,
    candidate_to_entry,
    _get_existing_dedup_keys,
)
from solitaire.symbiosis.reader_registry import ReaderRegistry
from solitaire.symbiosis.jsonl_reader import JSONLReader, _parse_timestamp
from solitaire.symbiosis.chatgpt_reader import (
    ChatGPTExportReader,
    _extract_messages,
    _format_conversation,
)
from solitaire.symbiosis.text_reader import TextReader, _chunk_text
from solitaire.symbiosis.sync_engine import (
    SyncEngine,
    SyncTier,
    SourceConfig,
    SyncResult,
)
from solitaire.symbiosis.cli import AdapterCLI


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_dir():
    """Create a temporary directory, cleaned up after test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def auto_memory_dir(tmp_dir):
    """Create a populated .auto-memory directory."""
    # Index file (should be skipped)
    with open(os.path.join(tmp_dir, "MEMORY.md"), "w") as f:
        f.write("# Memory Index\nThis is the index file.\n")

    # User preference
    with open(os.path.join(tmp_dir, "prefs.md"), "w") as f:
        f.write("---\nname: communication-style\ndescription: User prefers direct communication\ntype: user\n---\n\nAlways be direct and concise. No fluff.\n")

    # Project fact
    with open(os.path.join(tmp_dir, "project.md"), "w") as f:
        f.write("---\nname: solitaire-arch\ndescription: Architecture decisions\ntype: project\n---\n\nSolitaire uses SQLite for local storage.\nThe JSONL canonical store is the source of truth.\n")

    # Reference
    with open(os.path.join(tmp_dir, "ref.md"), "w") as f:
        f.write("---\nname: api-docs\ntype: reference\n---\n\nSee https://docs.example.com/api for details.\n")

    # No frontmatter file
    with open(os.path.join(tmp_dir, "plain.md"), "w") as f:
        f.write("Just some plain text without any frontmatter.\n")

    # Empty body file (should be skipped)
    with open(os.path.join(tmp_dir, "empty.md"), "w") as f:
        f.write("---\nname: empty\ntype: user\n---\n")

    return tmp_dir


@pytest.fixture
def jsonl_file(tmp_dir):
    """Create a sample entries.jsonl file."""
    path = os.path.join(tmp_dir, "entries.jsonl")
    entries = [
        {
            "_seq": 1, "_type": "entry", "_op": "upsert", "_ts": "2026-03-01T10:00:00",
            "id": "entry-001", "content": "Philip prefers direct communication",
            "content_type": "prose", "category": "user_knowledge",
            "tags": ["user_knowledge", "communication"],
            "created_at": "2026-03-01T10:00:00", "provenance": "user-stated",
            "source_type": "conversation", "access_count": 5, "tier": "hot",
            "metadata": {"session": "abc123"},
        },
        {
            "_seq": 2, "_type": "entry", "_op": "upsert", "_ts": "2026-03-01T11:00:00",
            "id": "entry-002", "content": "Solitaire uses SQLite for storage",
            "content_type": "prose", "category": "fact",
            "tags": ["architecture"], "created_at": "2026-03-01T11:00:00",
            "provenance": "assistant-inferred", "source_type": "conversation",
        },
        # Archived entry (should be skipped)
        {
            "_seq": 3, "_type": "entry", "_op": "upsert", "_ts": "2026-03-01T12:00:00",
            "id": "entry-003", "content": "Old archived content",
            "category": "note", "archived_at": "2026-03-02T00:00:00",
        },
        # Non-entry record (should be skipped)
        {
            "_seq": 4, "_type": "chain", "_op": "upsert",
            "id": "chain-001", "summary": "A reasoning chain",
        },
        # Delete operation (should be skipped)
        {
            "_seq": 5, "_type": "entry", "_op": "delete",
            "id": "entry-002",
        },
        # Empty content (should be skipped)
        {
            "_seq": 6, "_type": "entry", "_op": "upsert",
            "id": "entry-004", "content": "",
            "category": "note",
        },
    ]
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


@pytest.fixture
def chatgpt_file(tmp_dir):
    """Create a sample ChatGPT conversations.json file."""
    path = os.path.join(tmp_dir, "conversations.json")
    conversations = [
        {
            "title": "Python Best Practices",
            "id": "conv-001",
            "create_time": 1709000000.0,
            "mapping": {
                "root": {
                    "id": "root",
                    "message": None,
                    "parent": None,
                    "children": ["msg1"],
                },
                "msg1": {
                    "id": "msg1",
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["What are Python best practices?"]},
                        "create_time": 1709000001.0,
                    },
                    "parent": "root",
                    "children": ["msg2"],
                },
                "msg2": {
                    "id": "msg2",
                    "message": {
                        "id": "m2",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Here are some key best practices for Python development..."]},
                        "create_time": 1709000002.0,
                    },
                    "parent": "msg1",
                    "children": ["msg3"],
                },
                "msg3": {
                    "id": "msg3",
                    "message": {
                        "id": "m3",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Tell me more about type hints."]},
                        "create_time": 1709000003.0,
                    },
                    "parent": "msg2",
                    "children": [],
                },
            },
        },
        # Short conversation (should be skipped with min_turns=2)
        {
            "title": "Quick Question",
            "id": "conv-002",
            "create_time": 1709100000.0,
            "mapping": {
                "root": {
                    "id": "root",
                    "message": None,
                    "parent": None,
                    "children": ["msg1"],
                },
                "msg1": {
                    "id": "msg1",
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hi"]},
                        "create_time": 1709100001.0,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        },
    ]
    with open(path, "w") as f:
        json.dump(conversations, f)
    return path


@pytest.fixture
def text_dir(tmp_dir):
    """Create a directory with text files."""
    with open(os.path.join(tmp_dir, "notes.md"), "w") as f:
        f.write("# Project Notes\n\nSome important notes here.\n\nAnother paragraph.\n")
    with open(os.path.join(tmp_dir, "readme.txt"), "w") as f:
        f.write("This is a plain text readme file.\n")
    with open(os.path.join(tmp_dir, "data.csv"), "w") as f:
        f.write("col1,col2\na,b\n")  # Should be skipped (wrong extension)
    return tmp_dir


@pytest.fixture
def mock_rolodex():
    """Create a mock Rolodex for ImportOrchestrator tests."""
    rolodex = MagicMock()
    rolodex.batch_create_entries.return_value = ["id-1", "id-2", "id-3"]
    rolodex.create_entry.return_value = "id-single"
    rolodex.get_entry.return_value = None
    return rolodex


@pytest.fixture
def mock_db():
    """Create an in-memory SQLite database with rolodex schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE rolodex_entries (
            id TEXT PRIMARY KEY,
            content TEXT,
            metadata TEXT
        )
    """)
    return conn


@pytest.fixture
def registry():
    """Create a populated reader registry."""
    r = ReaderRegistry()
    r.auto_discover()
    return r


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 TESTS: Types, AutoMemoryReader, ImportOrchestrator
# ═══════════════════════════════════════════════════════════════════════

class TestIngestCandidate:
    def test_defaults(self):
        c = IngestCandidate(source_ref="test://1", raw_content="hello")
        assert c.content_type == IngestContentType.OTHER
        assert c.enrichment_hint == EnrichmentHint.FULL
        assert c.confidence == 0.5
        assert c.dedup_key is None

    def test_all_fields(self):
        c = IngestCandidate(
            source_ref="file:///test.md",
            raw_content="content",
            content_type=IngestContentType.FACT,
            enrichment_hint=EnrichmentHint.PARTIAL,
            confidence=0.9,
            source_id="test",
            tags=["tag1", "tag2"],
            dedup_key="test:key",
        )
        assert c.content_type == IngestContentType.FACT
        assert c.enrichment_hint == EnrichmentHint.PARTIAL
        assert len(c.tags) == 2


class TestImportResult:
    def test_success_rate_empty(self):
        r = ImportResult(source_id="test")
        assert r.success_rate == 0.0

    def test_success_rate_partial(self):
        r = ImportResult(source_id="test", total_candidates=10, imported=7)
        assert r.success_rate == 0.7


class TestFrontmatterParsing:
    def test_basic_frontmatter(self):
        text = "---\nname: test\ntype: user\n---\nBody content here."
        fm, body = _parse_frontmatter(text)
        assert fm["name"] == "test"
        assert fm["type"] == "user"
        assert body == "Body content here."

    def test_no_frontmatter(self):
        text = "Just plain text."
        fm, body = _parse_frontmatter(text)
        assert fm == {}
        assert body == "Just plain text."

    def test_quoted_values(self):
        text = '---\nname: "quoted value"\n---\nBody.'
        fm, body = _parse_frontmatter(text)
        assert fm["name"] == "quoted value"


class TestAutoMemoryReader:
    def test_source_id(self):
        reader = AutoMemoryReader()
        assert reader.source_id == "auto-memory"

    def test_validate_valid(self, auto_memory_dir):
        reader = AutoMemoryReader()
        result = reader.validate({"path": auto_memory_dir})
        assert result["valid"]
        assert result["file_count"] >= 3  # prefs, project, ref, plain (not MEMORY.md or empty)

    def test_validate_missing_path(self):
        reader = AutoMemoryReader()
        result = reader.validate({"path": "/nonexistent/path"})
        assert not result["valid"]

    def test_validate_no_path(self):
        reader = AutoMemoryReader()
        result = reader.validate({})
        assert not result["valid"]

    def test_read_yields_candidates(self, auto_memory_dir):
        reader = AutoMemoryReader()
        candidates = list(reader.read({"path": auto_memory_dir}))
        # Should get: prefs, project, ref, plain (skip MEMORY.md and empty body)
        assert len(candidates) == 4

    def test_read_type_mapping(self, auto_memory_dir):
        reader = AutoMemoryReader()
        candidates = {c.metadata.get("original_name", ""): c for c in reader.read({"path": auto_memory_dir})}

        # User type -> PREFERENCE
        assert candidates["communication-style"].content_type == IngestContentType.PREFERENCE
        # Project type -> FACT
        assert candidates["solitaire-arch"].content_type == IngestContentType.FACT
        # Reference type -> DOCUMENT
        assert candidates["api-docs"].content_type == IngestContentType.DOCUMENT

    def test_read_skips_memory_md(self, auto_memory_dir):
        reader = AutoMemoryReader()
        candidates = list(reader.read({"path": auto_memory_dir}))
        files = [c.metadata.get("file", "") for c in candidates]
        assert not any("MEMORY.md" in f for f in files)

    def test_read_dedup_keys_stable(self, auto_memory_dir):
        reader = AutoMemoryReader()
        run1 = {c.dedup_key for c in reader.read({"path": auto_memory_dir})}
        run2 = {c.dedup_key for c in reader.read({"path": auto_memory_dir})}
        assert run1 == run2

    def test_read_content_composition(self, auto_memory_dir):
        reader = AutoMemoryReader()
        candidates = list(reader.read({"path": auto_memory_dir}))
        prefs = [c for c in candidates if c.metadata.get("original_name") == "communication-style"]
        assert len(prefs) == 1
        # Content should include name + description + body
        assert "communication-style" in prefs[0].raw_content
        assert "direct and concise" in prefs[0].raw_content


class TestImportOrchestrator:
    def test_candidate_to_entry_mapping(self):
        candidate = IngestCandidate(
            source_ref="test://1",
            raw_content="Test content",
            content_type=IngestContentType.FACT,
            source_id="test",
            dedup_key="test:1",
        )
        entry = candidate_to_entry(candidate, session_id="sess-1")
        assert entry.content == "Test content"
        assert entry.category == EntryCategory.USER_KNOWLEDGE
        assert entry.content_type == ContentModality.PROSE
        assert entry.source_type == "external-import"
        assert "source:external-import" in entry.tags
        assert entry.metadata["dedup_key"] == "test:1"

    def test_dedup_skips_existing(self, mock_rolodex, mock_db):
        # Pre-populate a dedup key
        mock_db.execute(
            "INSERT INTO rolodex_entries (id, content, metadata) VALUES (?, ?, ?)",
            ("existing", "old", json.dumps({"dedup_key": "test:existing"})),
        )

        candidates = [
            IngestCandidate(source_ref="test://1", raw_content="New", source_id="test", dedup_key="test:new"),
            IngestCandidate(source_ref="test://2", raw_content="Existing", source_id="test", dedup_key="test:existing"),
        ]

        orchestrator = ImportOrchestrator(rolodex=mock_rolodex, conn=mock_db, session_id="s1")
        mock_rolodex.batch_create_entries.return_value = ["id-new"]
        result = orchestrator.run(iter(candidates), source_id="test")

        assert result.imported == 1
        assert result.skipped_duplicate == 1
        assert result.total_candidates == 2

    def test_dry_run(self, mock_rolodex, mock_db):
        candidates = [
            IngestCandidate(source_ref="test://1", raw_content="Content", source_id="test", dedup_key="test:1"),
        ]
        orchestrator = ImportOrchestrator(rolodex=mock_rolodex, conn=mock_db, session_id="s1")
        result = orchestrator.run(iter(candidates), source_id="test", dry_run=True)

        assert result.imported == 1  # Counted but not stored
        mock_rolodex.batch_create_entries.assert_not_called()

    def test_error_entries_tracked(self, mock_rolodex, mock_db):
        candidates = [
            IngestCandidate(
                source_ref="test://bad", raw_content="",
                source_id="test", metadata={"reader_error": "File not found"},
            ),
            IngestCandidate(source_ref="test://empty", raw_content="   ", source_id="test"),
        ]
        orchestrator = ImportOrchestrator(rolodex=mock_rolodex, conn=mock_db, session_id="s1")
        result = orchestrator.run(iter(candidates), source_id="test")

        assert result.skipped_error == 2
        assert len(result.errors) == 2

    def test_batch_fallback(self, mock_rolodex, mock_db):
        """If batch insert fails, fall back to one-by-one."""
        mock_rolodex.batch_create_entries.side_effect = Exception("Batch failed")
        mock_rolodex.create_entry.return_value = "id-fallback"

        candidates = [
            IngestCandidate(source_ref="test://1", raw_content="Content", source_id="test", dedup_key="test:1"),
        ]
        orchestrator = ImportOrchestrator(rolodex=mock_rolodex, conn=mock_db, session_id="s1")
        result = orchestrator.run(iter(candidates), source_id="test")

        assert result.imported == 1
        mock_rolodex.create_entry.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 TESTS: ReaderRegistry, JSONLReader, ChatGPTReader, TextReader
# ═══════════════════════════════════════════════════════════════════════

class TestReaderRegistry:
    def test_auto_discover(self, registry):
        assert registry.count == 4
        assert registry.has("auto-memory")
        assert registry.has("jsonl")
        assert registry.has("chatgpt-export")
        assert registry.has("text")

    def test_list_sources(self, registry):
        sources = registry.list_sources()
        ids = {s["source_id"] for s in sources}
        assert ids == {"auto-memory", "jsonl", "chatgpt-export", "text"}

    def test_get_returns_none_for_unknown(self, registry):
        assert registry.get("nonexistent") is None

    def test_register_replaces(self, registry):
        mock_reader = MagicMock(spec=ReaderBase)
        mock_reader.source_id = "auto-memory"
        registry.register(mock_reader)
        assert registry.get("auto-memory") is mock_reader


class TestJSONLReader:
    def test_source_id(self):
        assert JSONLReader().source_id == "jsonl"

    def test_validate_valid(self, jsonl_file):
        reader = JSONLReader()
        result = reader.validate({"path": jsonl_file})
        assert result["valid"]

    def test_validate_missing_file(self):
        reader = JSONLReader()
        result = reader.validate({"path": "/nonexistent.jsonl"})
        assert not result["valid"]

    def test_read_yields_entries(self, jsonl_file):
        reader = JSONLReader()
        candidates = list(reader.read({"path": jsonl_file}))
        # Should get 2: entry-001 and entry-002
        # Skips: archived (entry-003), chain record, delete op, empty content
        assert len(candidates) == 2

    def test_read_category_mapping(self, jsonl_file):
        reader = JSONLReader()
        candidates = {c.metadata["original_id"]: c for c in reader.read({"path": jsonl_file})}
        assert candidates["entry-001"].content_type == IngestContentType.PREFERENCE
        assert candidates["entry-002"].content_type == IngestContentType.FACT

    def test_read_preserves_metadata(self, jsonl_file):
        reader = JSONLReader()
        candidates = list(reader.read({"path": jsonl_file}))
        first = [c for c in candidates if c.metadata.get("original_id") == "entry-001"][0]
        assert first.metadata["original_provenance"] == "user-stated"
        assert first.metadata["original_access_count"] == 5

    def test_read_dedup_by_id(self, jsonl_file):
        reader = JSONLReader()
        candidates = list(reader.read({"path": jsonl_file}))
        dedup_keys = [c.dedup_key for c in candidates]
        assert "jsonl:entry-001" in dedup_keys
        assert "jsonl:entry-002" in dedup_keys

    def test_read_enrichment_hint_partial(self, jsonl_file):
        reader = JSONLReader()
        candidates = list(reader.read({"path": jsonl_file}))
        for c in candidates:
            assert c.enrichment_hint == EnrichmentHint.PARTIAL


class TestParseTimestamp:
    def test_iso_format(self):
        ts = _parse_timestamp("2026-03-01T10:00:00")
        assert ts is not None
        assert ts.year == 2026

    def test_with_microseconds(self):
        ts = _parse_timestamp("2026-03-01T10:00:00.123456")
        assert ts is not None

    def test_date_only(self):
        ts = _parse_timestamp("2026-03-01")
        assert ts is not None

    def test_none_input(self):
        assert _parse_timestamp(None) is None

    def test_invalid(self):
        assert _parse_timestamp("not a date") is None


class TestChatGPTExportReader:
    def test_source_id(self):
        assert ChatGPTExportReader().source_id == "chatgpt-export"

    def test_validate_valid(self, chatgpt_file):
        reader = ChatGPTExportReader()
        result = reader.validate({"path": chatgpt_file})
        assert result["valid"]
        assert result["conversation_count"] == 2

    def test_validate_missing(self):
        reader = ChatGPTExportReader()
        result = reader.validate({"path": "/nonexistent.json"})
        assert not result["valid"]

    def test_read_yields_conversations(self, chatgpt_file):
        reader = ChatGPTExportReader()
        candidates = list(reader.read({"path": chatgpt_file}))
        # conv-001 has 3 messages (2 user, 1 assistant) -> 1 candidate
        # conv-002 has 1 message -> skipped (min_turns=2)
        assert len(candidates) == 1

    def test_read_content_format(self, chatgpt_file):
        reader = ChatGPTExportReader()
        candidates = list(reader.read({"path": chatgpt_file}))
        content = candidates[0].raw_content
        assert "Python Best Practices" in content
        assert "[User]" in content
        assert "[Assistant]" in content

    def test_read_conversation_type(self, chatgpt_file):
        reader = ChatGPTExportReader()
        candidates = list(reader.read({"path": chatgpt_file}))
        assert candidates[0].content_type == IngestContentType.CONVERSATION

    def test_chunking(self, chatgpt_file):
        reader = ChatGPTExportReader()
        # With max_turns=2, the 3-message conversation should split
        candidates = list(reader.read({"path": chatgpt_file, "max_turns_per_chunk": 2}))
        assert len(candidates) == 2  # 3 messages split into chunks of 2

    def test_extract_messages_empty(self):
        assert _extract_messages({}) == []


class TestTextReader:
    def test_source_id(self):
        assert TextReader().source_id == "text"

    def test_validate_dir(self, text_dir):
        reader = TextReader()
        result = reader.validate({"path": text_dir})
        assert result["valid"]
        assert result["file_count"] == 2  # .md and .txt only

    def test_validate_single_file(self, text_dir):
        reader = TextReader()
        result = reader.validate({"path": os.path.join(text_dir, "notes.md")})
        assert result["valid"]
        assert result["file_count"] == 1

    def test_validate_wrong_extension(self, text_dir):
        reader = TextReader()
        result = reader.validate({"path": os.path.join(text_dir, "data.csv")})
        assert not result["valid"]

    def test_read_dir(self, text_dir):
        reader = TextReader()
        candidates = list(reader.read({"path": text_dir}))
        assert len(candidates) == 2

    def test_read_single_file(self, text_dir):
        reader = TextReader()
        candidates = list(reader.read({"path": os.path.join(text_dir, "notes.md")}))
        assert len(candidates) == 1
        assert "Project Notes" in candidates[0].raw_content

    def test_read_content_type(self, text_dir):
        reader = TextReader()
        candidates = list(reader.read({"path": text_dir}))
        types = {c.metadata["extension"]: c.content_type for c in candidates}
        assert types[".md"] == IngestContentType.DOCUMENT
        assert types[".txt"] == IngestContentType.OTHER


class TestChunkText:
    def test_small_text_no_split(self):
        chunks = _chunk_text("Short text.", max_chars=100)
        assert len(chunks) == 1

    def test_large_text_splits(self):
        text = "\n\n".join([f"Paragraph {i}. " * 20 for i in range(10)])
        chunks = _chunk_text(text, max_chars=500)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = _chunk_text("")
        assert len(chunks) == 1


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 TESTS: SyncEngine
# ═══════════════════════════════════════════════════════════════════════

class TestSourceConfig:
    def test_round_trip(self):
        sc = SourceConfig(
            source_id="jsonl", name="test-source",
            config={"path": "/tmp/test.jsonl"},
            sync_tier=SyncTier.PERIODIC,
            periodic_interval_seconds=1800,
        )
        d = sc.to_dict()
        sc2 = SourceConfig.from_dict(d)
        assert sc2.source_id == "jsonl"
        assert sc2.name == "test-source"
        assert sc2.sync_tier == SyncTier.PERIODIC
        assert sc2.periodic_interval_seconds == 1800

    def test_default_tier(self):
        sc = SourceConfig.from_dict({"source_id": "test", "config": {}})
        assert sc.sync_tier == SyncTier.ONE_SHOT


class TestSyncEngine:
    def _make_engine(self, registry, tmp_dir):
        mock_orchestrator = MagicMock(spec=ImportOrchestrator)
        mock_result = ImportResult(source_id="test")
        mock_result.imported = 3
        mock_result.total_candidates = 5
        mock_result.skipped_duplicate = 2
        mock_orchestrator.run.return_value = mock_result

        return SyncEngine(
            registry=registry,
            orchestrator_factory=lambda: mock_orchestrator,
            state_dir=tmp_dir,
        )

    def test_connect_valid_source(self, registry, auto_memory_dir, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        result = engine.connect("auto-memory", "Cowork", {"path": auto_memory_dir})
        assert result["connected"]
        assert len(engine.sources) == 1

    def test_connect_unknown_source(self, registry, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        result = engine.connect("nonexistent", "Bad", {})
        assert not result["connected"]

    def test_connect_invalid_config(self, registry, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        result = engine.connect("auto-memory", "Bad", {"path": "/nonexistent"})
        assert not result["connected"]

    def test_connect_duplicate_name(self, registry, auto_memory_dir, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        engine.connect("auto-memory", "Cowork", {"path": auto_memory_dir})
        result = engine.connect("auto-memory", "Cowork", {"path": auto_memory_dir})
        assert not result["connected"]

    def test_disconnect(self, registry, auto_memory_dir, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        engine.connect("auto-memory", "Cowork", {"path": auto_memory_dir})
        result = engine.disconnect("Cowork")
        assert result["disconnected"]
        assert len(engine.sources) == 0

    def test_disconnect_nonexistent(self, registry, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        result = engine.disconnect("Nonexistent")
        assert not result["disconnected"]

    def test_sync(self, registry, auto_memory_dir, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        engine.connect("auto-memory", "Cowork", {"path": auto_memory_dir})
        result = engine.sync("Cowork")
        assert result.status == "ok"
        assert result.import_result is not None
        assert result.import_result.imported == 3

    def test_sync_disabled_source(self, registry, auto_memory_dir, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        engine.connect("auto-memory", "Cowork", {"path": auto_memory_dir})
        engine.sources["Cowork"].enabled = False
        result = engine.sync("Cowork")
        assert result.status == "skipped"

    def test_sync_nonexistent(self, registry, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        result = engine.sync("Nonexistent")
        assert result.status == "error"

    def test_state_persistence(self, registry, auto_memory_dir, tmp_dir):
        engine1 = self._make_engine(registry, tmp_dir)
        engine1.connect("auto-memory", "Cowork", {"path": auto_memory_dir})
        engine1.sync("Cowork")

        # Load state in a new engine
        engine2 = self._make_engine(registry, tmp_dir)
        loaded = engine2.load_state()
        assert loaded == 1
        assert "Cowork" in engine2.sources
        assert engine2.sources["Cowork"].last_sync_at is not None

    def test_list_sources(self, registry, auto_memory_dir, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        engine.connect("auto-memory", "Cowork", {"path": auto_memory_dir})
        sources = engine.list_sources()
        assert len(sources) == 1
        assert sources[0]["name"] == "Cowork"

    def test_status(self, registry, auto_memory_dir, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        engine.connect("auto-memory", "Cowork", {"path": auto_memory_dir}, sync_tier=SyncTier.PERIODIC)
        status = engine.status()
        assert status["total_sources"] == 1
        assert status["sources_by_tier"]["periodic"] == 1

    def test_sync_periodic_due(self, registry, auto_memory_dir, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        engine.connect("auto-memory", "Cowork", {"path": auto_memory_dir}, sync_tier=SyncTier.PERIODIC, periodic_interval=0)
        # Never synced, so should be due
        results = engine.sync_periodic()
        assert len(results) == 1
        assert results[0].status == "ok"

    def test_sync_periodic_not_due(self, registry, auto_memory_dir, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        engine.connect("auto-memory", "Cowork", {"path": auto_memory_dir}, sync_tier=SyncTier.PERIODIC, periodic_interval=9999)
        # Sync once so last_sync_at is set
        engine.sync("Cowork")
        # Should not be due yet (9999 second interval)
        results = engine.sync_periodic()
        assert len(results) == 0

    def test_reconcile(self, registry, auto_memory_dir, tmp_dir):
        engine = self._make_engine(registry, tmp_dir)
        engine.connect("auto-memory", "Cowork", {"path": auto_memory_dir})
        result = engine.reconcile("Cowork")
        assert result.status == "ok"


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 TESTS: AdapterCLI
# ═══════════════════════════════════════════════════════════════════════

class TestAdapterCLI:
    def _make_cli(self, registry, tmp_dir):
        mock_orchestrator = MagicMock(spec=ImportOrchestrator)
        mock_result = ImportResult(source_id="test")
        mock_result.imported = 2
        mock_result.total_candidates = 3
        mock_result.skipped_duplicate = 1
        mock_orchestrator.run.return_value = mock_result

        engine = SyncEngine(
            registry=registry,
            orchestrator_factory=lambda: mock_orchestrator,
            state_dir=tmp_dir,
        )
        return AdapterCLI(sync_engine=engine, registry=registry)

    def test_sources(self, registry, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        result = cli.dispatch(["sources"])
        assert result["status"] == "ok"
        assert result["count"] == 4

    def test_connect_and_status(self, registry, auto_memory_dir, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        result = cli.dispatch(["connect", "auto-memory", "Cowork", auto_memory_dir])
        assert result["status"] == "ok"

        result = cli.dispatch(["status"])
        assert result["total_sources"] == 1

    def test_import(self, registry, auto_memory_dir, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        cli.dispatch(["connect", "auto-memory", "Cowork", auto_memory_dir])
        result = cli.dispatch(["import", "Cowork"])
        assert result["status"] == "ok"

    def test_import_dry_run(self, registry, auto_memory_dir, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        cli.dispatch(["connect", "auto-memory", "Cowork", auto_memory_dir])
        result = cli.dispatch(["import", "Cowork", "--dry-run"])
        assert result["dry_run"]

    def test_disconnect(self, registry, auto_memory_dir, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        cli.dispatch(["connect", "auto-memory", "Cowork", auto_memory_dir])
        result = cli.dispatch(["disconnect", "Cowork"])
        assert result["status"] == "ok"

    def test_sync_all(self, registry, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        result = cli.dispatch(["sync_all"])
        assert result["status"] == "ok"

    def test_help(self, registry, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        result = cli.dispatch(["help"])
        assert "commands" in result

    def test_unknown_command(self, registry, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        result = cli.dispatch(["unknown"])
        assert result["status"] == "error"

    def test_empty_args(self, registry, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        result = cli.dispatch([])
        assert "commands" in result

    def test_connect_missing_args(self, registry, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        result = cli.dispatch(["connect"])
        assert result["status"] == "error"

    def test_connect_with_tier(self, registry, auto_memory_dir, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        result = cli.dispatch(["connect", "auto-memory", "Cowork", auto_memory_dir, "periodic"])
        assert result["status"] == "ok"

    def test_watch_start_stop(self, registry, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        result = cli.dispatch(["watch_start"])
        assert result["status"] == "ok"
        result = cli.dispatch(["watch_stop"])
        assert result["status"] == "ok"

    def test_reconcile(self, registry, auto_memory_dir, tmp_dir):
        cli = self._make_cli(registry, tmp_dir)
        cli.dispatch(["connect", "auto-memory", "Cowork", auto_memory_dir])
        result = cli.dispatch(["reconcile", "Cowork"])
        assert result["status"] == "ok"
