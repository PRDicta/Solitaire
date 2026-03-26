"""
One-shot import orchestrator for the symbiosis adapter.

Takes IngestCandidates from any reader, deduplicates against the existing
rolodex, maps to RolodexEntry, and stores via the standard pipeline.

Design decisions:
- Dedup uses a stable key stored in entry metadata (metadata.dedup_key).
  Re-running an import on the same source won't create duplicates.
- Errors on individual entries don't abort the batch. The orchestrator
  tracks per-entry errors and reports them in ImportResult.
- Source tagging: all imported entries get source_type="external-import"
  and provenance="external-import" so the system can distinguish them
  from native conversation entries.
- Entries go through entity extraction and KG integration via the
  existing enrichment helpers when available.
"""

import time
import logging
import sqlite3
from typing import Iterator, Optional, List, Set
from datetime import datetime

from ..core.types import (
    IngestCandidate,
    IngestContentType,
    EnrichmentHint,
    ImportResult,
    RolodexEntry,
    ContentModality,
    EntryCategory,
    Tier,
)

logger = logging.getLogger(__name__)


# IngestContentType -> (ContentModality, EntryCategory) mapping
# This maps the adapter's type system to the rolodex's type system.
_MODALITY_MAP = {
    IngestContentType.FACT: (ContentModality.PROSE, EntryCategory.USER_KNOWLEDGE),
    IngestContentType.CONVERSATION: (ContentModality.CONVERSATIONAL, EntryCategory.NOTE),
    IngestContentType.DOCUMENT: (ContentModality.PROSE, EntryCategory.REFERENCE),
    IngestContentType.PREFERENCE: (ContentModality.PROSE, EntryCategory.USER_KNOWLEDGE),
    IngestContentType.OTHER: (ContentModality.PROSE, EntryCategory.NOTE),
}


def _get_existing_dedup_keys(conn: sqlite3.Connection) -> Set[str]:
    """Load all dedup_keys already in the rolodex.

    Dedup keys are stored in the metadata JSON field under 'dedup_key'.
    This is a full scan, but it only runs once per import. For a typical
    .auto-memory directory (tens to low hundreds of entries), this is
    well under a second.
    """
    keys = set()
    try:
        rows = conn.execute(
            "SELECT metadata FROM rolodex_entries WHERE metadata LIKE '%dedup_key%'"
        ).fetchall()
        import json
        for (meta_str,) in rows:
            try:
                meta = json.loads(meta_str) if meta_str else {}
                dk = meta.get("dedup_key")
                if dk:
                    keys.add(dk)
            except (json.JSONDecodeError, TypeError):
                continue
    except Exception as e:
        logger.warning(f"Could not load dedup keys: {e}")
    return keys


def candidate_to_entry(
    candidate: IngestCandidate,
    session_id: str = "",
) -> RolodexEntry:
    """Convert an IngestCandidate to a RolodexEntry.

    This is the format boundary between the adapter's world and the
    rolodex's world. Everything upstream of this function speaks
    IngestCandidate. Everything downstream speaks RolodexEntry.
    """
    modality, category = _MODALITY_MAP.get(
        candidate.content_type,
        (ContentModality.PROSE, EntryCategory.NOTE),
    )

    # Build metadata preserving adapter provenance
    metadata = dict(candidate.metadata) if candidate.metadata else {}
    metadata["dedup_key"] = candidate.dedup_key
    metadata["import_source_id"] = candidate.source_id
    metadata["import_source_ref"] = candidate.source_ref
    metadata["import_confidence"] = candidate.confidence
    metadata["import_enrichment_hint"] = candidate.enrichment_hint.value

    entry = RolodexEntry(
        content=candidate.raw_content,
        conversation_id=session_id,
        content_type=modality,
        category=category,
        tags=list(candidate.tags) + ["source:external-import", f"source:{candidate.source_id}"],
        tier=Tier.COLD,
        verbatim_source=True,
        source_type="external-import",
        provenance="external-import",
        metadata=metadata,
    )

    # Use the source timestamp as created_at if available
    if candidate.timestamp:
        entry.created_at = candidate.timestamp

    # If the source had a timestamp, set event_time too
    if candidate.timestamp:
        entry.event_time = candidate.timestamp

    return entry


class ImportOrchestrator:
    """Runs a one-shot import from an iterator of IngestCandidates.

    Usage:
        reader = AutoMemoryReader()
        candidates = reader.read({"path": "/path/to/.auto-memory"})
        orchestrator = ImportOrchestrator(rolodex=rolodex, conn=conn)
        result = orchestrator.run(candidates, source_id="auto-memory")
    """

    def __init__(
        self,
        rolodex,  # Rolodex instance
        conn: sqlite3.Connection,
        session_id: str = "",
        entity_extractor=None,  # Optional: EntityExtractorKG instance
        knowledge_graph=None,   # Optional: KnowledgeGraph instance
    ):
        self.rolodex = rolodex
        self.conn = conn
        self.session_id = session_id
        self.entity_extractor = entity_extractor
        self.knowledge_graph = knowledge_graph

    def run(
        self,
        candidates: Iterator[IngestCandidate],
        source_id: str = "",
        dry_run: bool = False,
    ) -> ImportResult:
        """Execute a one-shot import.

        Args:
            candidates: Iterator of IngestCandidates from a reader.
            source_id: Identifier for this import source.
            dry_run: If True, validate and count but don't store anything.

        Returns:
            ImportResult with counts and any errors.
        """
        start_time = time.time()
        result = ImportResult(source_id=source_id)

        # Load existing dedup keys once
        existing_keys = _get_existing_dedup_keys(self.conn)

        entries_to_store: List[RolodexEntry] = []

        for candidate in candidates:
            result.total_candidates += 1

            # Skip error candidates from reader
            if candidate.metadata.get("reader_error"):
                result.skipped_error += 1
                result.errors.append({
                    "source_ref": candidate.source_ref,
                    "error": candidate.metadata["reader_error"],
                    "phase": "reader",
                })
                continue

            # Skip empty content
            if not candidate.raw_content.strip():
                result.skipped_error += 1
                result.errors.append({
                    "source_ref": candidate.source_ref,
                    "error": "Empty content",
                    "phase": "validation",
                })
                continue

            # Deduplication check
            if candidate.dedup_key and candidate.dedup_key in existing_keys:
                result.skipped_duplicate += 1
                continue

            # Convert to RolodexEntry
            try:
                entry = candidate_to_entry(candidate, session_id=self.session_id)
            except Exception as e:
                result.skipped_error += 1
                result.errors.append({
                    "source_ref": candidate.source_ref,
                    "error": str(e),
                    "phase": "conversion",
                })
                continue

            entries_to_store.append(entry)

            # Track the dedup key so we don't import the same entry twice
            # within a single batch (in case the reader yields duplicates)
            if candidate.dedup_key:
                existing_keys.add(candidate.dedup_key)

        if dry_run:
            result.imported = len(entries_to_store)
            result.duration_seconds = time.time() - start_time
            return result

        # Batch store
        if entries_to_store:
            try:
                ids = self.rolodex.batch_create_entries(entries_to_store)
                result.entry_ids = ids
                result.imported = len(ids)
            except Exception as e:
                # If batch fails, fall back to one-by-one
                logger.warning(f"Batch insert failed, falling back to individual: {e}")
                for entry in entries_to_store:
                    try:
                        eid = self.rolodex.create_entry(entry)
                        result.entry_ids.append(eid)
                        result.imported += 1
                    except Exception as e2:
                        result.skipped_error += 1
                        result.errors.append({
                            "source_ref": entry.metadata.get("import_source_ref", ""),
                            "error": str(e2),
                            "phase": "storage",
                        })

        # Post-import enrichment: entity extraction + KG integration
        if result.entry_ids and self.entity_extractor:
            self._enrich_entries(result.entry_ids)

        result.duration_seconds = time.time() - start_time
        return result

    def _enrich_entries(self, entry_ids: List[str]) -> None:
        """Run entity extraction and KG integration on imported entries.

        This is best-effort. Failures here don't affect the import result
        because the entries are already stored and FTS-searchable.
        """
        for eid in entry_ids:
            try:
                entry = self.rolodex.get_entry(eid)
                if not entry:
                    continue

                # Entity extraction
                if self.entity_extractor:
                    try:
                        entities = self.entity_extractor.extract_entities(entry.content)
                        if entities and self.knowledge_graph:
                            for entity in entities:
                                try:
                                    self.knowledge_graph.add_entity(
                                        entity_name=entity.get("name", ""),
                                        entity_type=entity.get("type", "unknown"),
                                        source_entry_id=eid,
                                        metadata=entity.get("metadata", {}),
                                    )
                                except Exception as kg_err:
                                    logger.debug(f"KG insertion failed for entity in {eid}: {kg_err}")
                    except Exception as ext_err:
                        logger.debug(f"Entity extraction failed for {eid}: {ext_err}")

            except Exception as e:
                logger.info(f"Post-import enrichment failed for {eid}: {e}")
