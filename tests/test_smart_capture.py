"""
Tests for Smart Capture onboarding modules.

Covers:
- EnvironmentScanner (detection probes, scan aggregation)
- PriorityRanker (heuristic scoring, corpus classification, LLM prompt/parse)
- ClaudeMdReader (section splitting, type classification)
- MarkdownKBReader (directory reading, large file splitting)
- SolitaireReader (cross-instance import, dedup, category mapping)
- Onboarding integration (smart_capture step in flow)
"""

import os
import json
import tempfile
import shutil
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from solitaire.core.types import (
    IngestCandidate,
    IngestContentType,
    EnrichmentHint,
)
from solitaire.symbiosis.environment_scanner import (
    scan_environment,
    DetectedSource,
    ScanResult,
    _probe_claude_md,
    _probe_auto_memory,
    _probe_chatgpt_export,
    _probe_markdown_kb,
    _probe_solitaire_instance,
    _mtime_range,
    _dir_size_and_mtimes,
)
from solitaire.symbiosis.priority_ranker import (
    heuristic_priority_score,
    classify_corpus,
    rank_candidates_heuristic,
    build_classification_prompt,
    parse_classification_response,
    rank_candidates_llm,
    IngestionStrategy,
    IngestionPlan,
)
from solitaire.symbiosis.claude_md_reader import (
    ClaudeMdReader,
    _split_sections,
    _classify_section,
)
from solitaire.symbiosis.markdown_kb_reader import MarkdownKBReader
from solitaire.symbiosis.solitaire_reader import SolitaireReader
from solitaire.core.onboarding_flow import OnboardingContext, FlowEngine


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def claude_md_file(tmp_dir):
    content = """# Writing Standards

Always use plain language. No jargon.

## Formatting Rules

- No em dashes
- Short paragraphs

## Project Setup

This is a Python 3.12 project using FastAPI.
Database is PostgreSQL with SQLAlchemy ORM.
"""
    path = os.path.join(tmp_dir, "CLAUDE.md")
    with open(path, "w") as f:
        f.write(content)
    return path


@pytest.fixture
def markdown_kb_dir(tmp_dir):
    kb_dir = os.path.join(tmp_dir, "notes")
    os.makedirs(kb_dir)
    for i in range(15):
        with open(os.path.join(kb_dir, f"note_{i}.md"), "w") as f:
            f.write(f"# Note {i}\n\nContent for note {i}.\n")
    return kb_dir


@pytest.fixture
def solitaire_db(tmp_dir):
    db_path = os.path.join(tmp_dir, "rolodex.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE rolodex_entries (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            content TEXT NOT NULL,
            content_type TEXT NOT NULL,
            category TEXT NOT NULL,
            tags TEXT DEFAULT '[]',
            source_range TEXT DEFAULT '{}',
            access_count INTEGER DEFAULT 0,
            last_accessed DATETIME,
            created_at DATETIME NOT NULL,
            tier TEXT DEFAULT 'cold',
            embedding BLOB,
            linked_ids TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            archived_at DATETIME DEFAULT NULL
        )
    """)
    now = datetime.now(timezone.utc).isoformat()
    entries = [
        ("e1", "s1", "User prefers dark mode", "prose", "user_knowledge", "[]", now, "{}"),
        ("e2", "s1", "Project uses FastAPI", "prose", "fact", "[]", now, "{}"),
        ("e3", "s1", "Yesterday we debugged the auth flow", "conversational", "note", "[]", now, "{}"),
        ("e4", "s1", "External import entry", "prose", "note", "[]", now,
         json.dumps({"import_source_id": "other-instance"})),
    ]
    for eid, cid, content, ctype, cat, tags, created, meta in entries:
        conn.execute(
            "INSERT INTO rolodex_entries (id, conversation_id, content, content_type, category, tags, created_at, metadata) VALUES (?,?,?,?,?,?,?,?)",
            (eid, cid, content, ctype, cat, tags, created, meta)
        )
    conn.commit()
    conn.close()
    return db_path


# ─── EnvironmentScanner Tests ───────────────────────────────────────────────

class TestEnvironmentScanner:

    def test_scan_empty_dir(self, tmp_dir):
        result = scan_environment(workspace=tmp_dir)
        assert isinstance(result, ScanResult)
        assert len(result.sources) >= 0

    def test_probe_claude_md(self, claude_md_file, tmp_dir):
        source = _probe_claude_md([Path(tmp_dir)])
        assert source is not None
        assert source.source_id == "claude-md"
        assert source.confidence == 0.95
        assert source.entry_count_estimate == 1

    def test_probe_auto_memory(self, tmp_dir):
        mem_dir = os.path.join(tmp_dir, "memory")
        os.makedirs(mem_dir)
        for i in range(3):
            with open(os.path.join(mem_dir, f"mem_{i}.md"), "w") as f:
                f.write(f"---\nname: mem{i}\ntype: user\n---\nContent {i}\n")

        source = _probe_auto_memory([Path(tmp_dir)])
        assert source is not None
        assert source.source_id == "auto-memory"
        assert source.entry_count_estimate == 3
        # Should detect YAML frontmatter
        assert source.confidence == 0.9

    def test_probe_chatgpt_export(self, tmp_dir):
        data = [
            {"title": "Test", "mapping": {"n1": {"id": "n1", "message": None, "children": []}}}
        ]
        path = os.path.join(tmp_dir, "conversations.json")
        with open(path, "w") as f:
            json.dump(data, f)

        source = _probe_chatgpt_export([Path(tmp_dir)])
        assert source is not None
        assert source.source_id == "chatgpt-export"
        assert source.entry_count_estimate == 1

    def test_probe_solitaire_instance(self, solitaire_db, tmp_dir):
        source = _probe_solitaire_instance([Path(tmp_dir)])
        assert source is not None
        assert source.source_id == "solitaire-instance"
        assert source.entry_count_estimate == 4

    def test_probe_solitaire_skips_own_db(self, solitaire_db, tmp_dir):
        source = _probe_solitaire_instance([Path(tmp_dir)], own_db=solitaire_db)
        assert source is None

    def test_probe_markdown_kb(self, markdown_kb_dir):
        source = _probe_markdown_kb([Path(markdown_kb_dir)])
        assert source is not None
        assert source.source_id == "markdown-kb"
        assert source.entry_count_estimate == 15

    def test_scan_result_aggregation(self, tmp_dir, claude_md_file):
        result = scan_environment(workspace=tmp_dir)
        assert result.total_size_bytes > 0
        assert result.total_entry_estimate >= 1

    def test_detected_source_descriptions(self):
        s = DetectedSource(
            source_id="test",
            display_name="Test",
            path="/tmp/test",
            entry_count_estimate=100,
            size_bytes=1024 * 500,
        )
        assert s.size_description == "500 KB"

        s2 = DetectedSource(
            source_id="test",
            display_name="Test",
            path="/tmp/test",
            entry_count_estimate=100,
            size_bytes=5 * 1024 * 1024,
        )
        assert s2.size_description == "5.0 MB"

    def test_mtime_range(self):
        now = datetime.now(timezone.utc).timestamp()
        old = now - 86400 * 90  # 90 days ago
        result = _mtime_range([old, now])
        assert result is not None
        assert result[0] < result[1]

    def test_mtime_range_empty(self):
        assert _mtime_range([]) is None


# ─── PriorityRanker Tests ───────────────────────────────────────────────────

class TestPriorityRanker:

    def _make_candidate(self, content_type, tags=None, confidence=0.5, days_ago=0):
        ts = datetime.now(timezone.utc) - timedelta(days=days_ago) if days_ago >= 0 else None
        return IngestCandidate(
            source_ref="test://1",
            raw_content="test content",
            content_type=content_type,
            enrichment_hint=EnrichmentHint.FULL,
            confidence=confidence,
            source_id="test",
            timestamp=ts,
            tags=tags or [],
        )

    def test_preference_scores_higher_than_conversation(self):
        pref = self._make_candidate(IngestContentType.PREFERENCE)
        conv = self._make_candidate(IngestContentType.CONVERSATION)
        assert heuristic_priority_score(pref) > heuristic_priority_score(conv)

    def test_fact_scores_higher_than_other(self):
        fact = self._make_candidate(IngestContentType.FACT)
        other = self._make_candidate(IngestContentType.OTHER)
        assert heuristic_priority_score(fact) > heuristic_priority_score(other)

    def test_recency_increases_score(self):
        recent = self._make_candidate(IngestContentType.FACT, days_ago=1)
        old = self._make_candidate(IngestContentType.FACT, days_ago=300)
        assert heuristic_priority_score(recent) > heuristic_priority_score(old)

    def test_feedback_tags_boost_score(self):
        with_tag = self._make_candidate(IngestContentType.OTHER, tags=["feedback:user"])
        without = self._make_candidate(IngestContentType.OTHER, tags=[])
        assert heuristic_priority_score(with_tag) > heuristic_priority_score(without)

    def test_classify_corpus_immediate(self):
        plan = classify_corpus(entry_count=100, size_bytes=500_000)
        assert plan.strategy == IngestionStrategy.IMMEDIATE
        assert plan.background_remaining == 0
        assert plan.first_chunk_entries == 100

    def test_classify_corpus_chunked(self):
        plan = classify_corpus(entry_count=5000, size_bytes=10_000_000)
        assert plan.strategy == IngestionStrategy.CHUNKED
        assert plan.background_remaining > 0
        assert plan.first_chunk_entries <= 2000

    def test_classify_corpus_large(self):
        plan = classify_corpus(entry_count=50000, size_bytes=100_000_000)
        assert plan.strategy == IngestionStrategy.LARGE
        assert plan.background_remaining > 0

    def test_rank_candidates_respects_budget(self):
        candidates = [
            self._make_candidate(IngestContentType.FACT) for _ in range(20)
        ]
        ranked = rank_candidates_heuristic(candidates, budget=5)
        assert len(ranked) == 5

    def test_build_classification_prompt(self):
        candidates = [
            self._make_candidate(IngestContentType.PREFERENCE),
            self._make_candidate(IngestContentType.CONVERSATION),
        ]
        prompt = build_classification_prompt(candidates)
        assert "Tier 1" in prompt
        assert "[0]" in prompt
        assert "[1]" in prompt

    def test_parse_classification_response(self):
        response = '[{"index": 0, "tier": 1}, {"index": 1, "tier": 3}]'
        tiers = parse_classification_response(response, 2)
        assert tiers == [1, 3]

    def test_parse_classification_response_with_noise(self):
        response = 'Here are the results:\n[{"index": 0, "tier": 2}]\nDone.'
        tiers = parse_classification_response(response, 2)
        assert tiers[0] == 2
        assert tiers[1] == 3  # Default

    def test_parse_classification_response_invalid(self):
        tiers = parse_classification_response("not json", 3)
        assert tiers == [3, 3, 3]

    def test_rank_candidates_llm(self):
        candidates = [
            self._make_candidate(IngestContentType.OTHER, days_ago=1),
            self._make_candidate(IngestContentType.OTHER, days_ago=100),
            self._make_candidate(IngestContentType.OTHER, days_ago=50),
        ]
        tiers = [3, 1, 2]
        ranked = rank_candidates_llm(candidates, tiers, budget=3)
        # Tier 1 should be first
        assert ranked[0] is candidates[1]


# ─── ClaudeMdReader Tests ───────────────────────────────────────────────────

class TestClaudeMdReader:

    def test_validate_valid_file(self, claude_md_file):
        reader = ClaudeMdReader()
        result = reader.validate({"path": claude_md_file})
        assert result["valid"]

    def test_validate_missing_file(self):
        reader = ClaudeMdReader()
        result = reader.validate({"path": "/nonexistent/CLAUDE.md"})
        assert not result["valid"]

    def test_validate_wrong_name(self, tmp_dir):
        path = os.path.join(tmp_dir, "README.md")
        with open(path, "w") as f:
            f.write("# README")
        reader = ClaudeMdReader()
        result = reader.validate({"path": path})
        assert not result["valid"]

    def test_read_produces_candidates(self, claude_md_file):
        reader = ClaudeMdReader()
        candidates = list(reader.read({"path": claude_md_file}))
        assert len(candidates) >= 3  # At least 3 sections
        for c in candidates:
            assert c.source_id == "claude-md"
            assert c.raw_content
            assert c.dedup_key

    def test_section_splitting(self):
        text = "# Header 1\nBody 1\n\n## Header 2\nBody 2\n"
        sections = _split_sections(text)
        assert len(sections) == 2
        assert sections[0][0] == "Header 1"
        assert sections[1][0] == "Header 2"

    def test_classify_preference_section(self):
        ctype = _classify_section("Writing Standards", "Always use plain language. Never use jargon.")
        assert ctype == IngestContentType.PREFERENCE

    def test_classify_fact_section(self):
        ctype = _classify_section("Project Setup", "This is a Python project using FastAPI.")
        assert ctype == IngestContentType.FACT

    def test_dedup_keys_unique(self, claude_md_file):
        reader = ClaudeMdReader()
        candidates = list(reader.read({"path": claude_md_file}))
        keys = [c.dedup_key for c in candidates]
        assert len(keys) == len(set(keys))


# ─── MarkdownKBReader Tests ─────────────────────────────────────────────────

class TestMarkdownKBReader:

    def test_validate_valid_dir(self, markdown_kb_dir):
        reader = MarkdownKBReader()
        result = reader.validate({"path": markdown_kb_dir})
        assert result["valid"]
        assert result["file_count"] == 15

    def test_validate_empty_dir(self, tmp_dir):
        reader = MarkdownKBReader()
        result = reader.validate({"path": tmp_dir, "min_files": 5})
        assert not result["valid"]

    def test_read_produces_candidates(self, markdown_kb_dir):
        reader = MarkdownKBReader()
        candidates = list(reader.read({"path": markdown_kb_dir}))
        assert len(candidates) == 15
        for c in candidates:
            assert c.source_id == "markdown-kb"
            assert c.raw_content

    def test_skips_readme(self, tmp_dir):
        for name in ["README.md", "notes.md"]:
            with open(os.path.join(tmp_dir, name), "w") as f:
                f.write(f"# {name}\nContent.")
        reader = MarkdownKBReader()
        candidates = list(reader.read({"path": tmp_dir}))
        names = [c.metadata.get("filename", "") for c in candidates]
        assert "README.md" not in names
        assert "notes.md" in names

    def test_large_file_splitting(self, tmp_dir):
        # Create a file larger than 4KB with multiple sections
        content = ""
        for i in range(10):
            content += f"# Section {i}\n\n{'A' * 500}\n\n"
        with open(os.path.join(tmp_dir, "big.md"), "w") as f:
            f.write(content)
        reader = MarkdownKBReader()
        candidates = list(reader.read({"path": tmp_dir}))
        assert len(candidates) == 10  # One per section


# ─── SolitaireReader Tests ──────────────────────────────────────────────────

class TestSolitaireReader:

    def test_validate_valid_db(self, solitaire_db):
        reader = SolitaireReader()
        result = reader.validate({"path": solitaire_db})
        assert result["valid"]
        assert result["entry_count"] == 4

    def test_validate_invalid_db(self, tmp_dir):
        bad_db = os.path.join(tmp_dir, "bad.db")
        conn = sqlite3.connect(bad_db)
        conn.execute("CREATE TABLE foo (id INTEGER)")
        conn.close()
        reader = SolitaireReader()
        result = reader.validate({"path": bad_db})
        assert not result["valid"]
        assert "missing rolodex_entries" in result["error"]

    def test_read_produces_candidates(self, solitaire_db):
        reader = SolitaireReader()
        candidates = list(reader.read({"path": solitaire_db}))
        # Should get 3 entries (4 total minus 1 external import skipped)
        assert len(candidates) == 3

    def test_skips_external_imports(self, solitaire_db):
        reader = SolitaireReader()
        candidates = list(reader.read({"path": solitaire_db}))
        source_ids = [c.metadata.get("source_entry_id") for c in candidates]
        assert "e4" not in source_ids  # External import entry

    def test_includes_external_when_requested(self, solitaire_db):
        reader = SolitaireReader()
        candidates = list(reader.read({"path": solitaire_db, "skip_external": False}))
        assert len(candidates) == 4

    def test_category_mapping(self, solitaire_db):
        reader = SolitaireReader()
        candidates = list(reader.read({"path": solitaire_db}))
        type_map = {c.metadata["source_entry_id"]: c.content_type for c in candidates}
        assert type_map["e1"] == IngestContentType.PREFERENCE  # user_knowledge
        assert type_map["e2"] == IngestContentType.FACT         # fact
        assert type_map["e3"] == IngestContentType.CONVERSATION # note

    def test_category_filter(self, solitaire_db):
        reader = SolitaireReader()
        candidates = list(reader.read({
            "path": solitaire_db,
            "categories": ["user_knowledge"],
        }))
        assert len(candidates) == 1
        assert candidates[0].metadata["source_entry_id"] == "e1"

    def test_dedup_keys_stable(self, solitaire_db):
        reader = SolitaireReader()
        run1 = list(reader.read({"path": solitaire_db}))
        run2 = list(reader.read({"path": solitaire_db}))
        keys1 = [c.dedup_key for c in run1]
        keys2 = [c.dedup_key for c in run2]
        assert keys1 == keys2


# ─── Onboarding Integration Tests ──────────────────────────────────────────

class TestOnboardingSmartCapture:

    def test_context_has_smart_capture_fields(self):
        ctx = OnboardingContext()
        assert ctx.scan_result is None
        assert ctx.smart_capture_consent is None
        assert ctx.smart_capture_sources == []
        assert ctx.smart_capture_completed is False

    def test_welcome_routes_to_smart_capture(self):
        engine = FlowEngine()
        ctx = OnboardingContext(current_step="welcome")
        ctx = engine.process_input(ctx, "welcome", "continue")
        assert ctx.current_step == "smart_capture"

    def test_smart_capture_skip_routes_to_intent(self):
        engine = FlowEngine()
        ctx = OnboardingContext(current_step="smart_capture")
        ctx.scan_result = {"sources": []}
        ctx = engine.process_input(ctx, "smart_capture", "skip")
        assert ctx.current_step == "intent_capture"
        assert ctx.smart_capture_completed is True

    def test_smart_capture_yes_sets_sources(self):
        engine = FlowEngine()
        ctx = OnboardingContext(current_step="smart_capture")
        ctx.scan_result = {
            "sources": [
                {"source_id": "auto-memory", "display_name": "Claude Code memory"},
                {"source_id": "claude-md", "display_name": "CLAUDE.md"},
            ]
        }
        ctx = engine.process_input(ctx, "smart_capture", "yes")
        assert ctx.current_step == "intent_capture"
        assert ctx.smart_capture_consent == "yes"
        assert "auto-memory" in ctx.smart_capture_sources
        assert "claude-md" in ctx.smart_capture_sources

    def test_smart_capture_selective_routes_to_picker(self):
        engine = FlowEngine()
        ctx = OnboardingContext(current_step="smart_capture")
        ctx.scan_result = {"sources": [{"source_id": "auto-memory"}]}
        ctx = engine.process_input(ctx, "smart_capture", "selective")
        assert ctx.current_step == "smart_capture_selective"

    def test_selective_stores_chosen_sources(self):
        engine = FlowEngine()
        ctx = OnboardingContext(current_step="smart_capture_selective")
        ctx = engine.process_input(ctx, "smart_capture_selective", ["auto-memory"])
        assert ctx.smart_capture_sources == ["auto-memory"]
        assert ctx.smart_capture_consent == "selective"
        assert ctx.current_step == "intent_capture"

    def test_manual_fallback_skip(self):
        engine = FlowEngine()
        ctx = OnboardingContext(current_step="smart_capture_manual")
        ctx = engine.process_input(ctx, "smart_capture_manual", "skip")
        assert ctx.current_step == "intent_capture"
        assert ctx.smart_capture_completed is True

    def test_manual_fallback_with_path(self):
        engine = FlowEngine()
        ctx = OnboardingContext(current_step="smart_capture_manual")
        ctx = engine.process_input(ctx, "smart_capture_manual", "/home/user/memories")
        assert ctx.smart_capture_sources == ["/home/user/memories"]
        assert ctx.smart_capture_consent == "path"
        assert ctx.current_step == "intent_capture"

    def test_get_next_step_smart_capture(self):
        engine = FlowEngine()
        ctx = OnboardingContext(current_step="smart_capture")
        ctx.scan_result = {
            "sources": [
                {
                    "source_id": "auto-memory",
                    "display_name": "Claude Code memory",
                    "entry_count_estimate": 50,
                    "size_description": "124 KB",
                }
            ],
            "total_entry_estimate": 50,
            "combined_age_description": "3 months",
            "total_size_description": "124 KB",
        }
        ctx.ingestion_plan = {"strategy": "immediate"}
        step = engine.get_next_step(ctx)
        assert step.step_id == "smart_capture"
        assert step.step_type == "confirm"

    def test_get_next_step_manual_fallback(self):
        engine = FlowEngine()
        ctx = OnboardingContext(current_step="smart_capture")
        ctx.scan_result = {"sources": []}
        step = engine.get_next_step(ctx)
        # Should fall through to manual since no sources
        assert step.step_id == "smart_capture_manual"
