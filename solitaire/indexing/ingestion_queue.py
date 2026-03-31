"""
The Librarian — Ingestion Queue (Phase 8)

Decouples raw message capture (fast, synchronous) from enrichment
(async background workers). Enables 100% ingestion — every message
is stored immediately; extraction, embedding, and categorization
happen in the background.

Architecture:
    ingest() → persist raw stub → enqueue enrichment → return immediately
    [Background workers] → chunk → extract → embed → update DB

Workers pause when a query arrives (retrieval takes priority) and
resume once the search completes.

Crash recovery:
    A write-ahead journal (.solitaire/queue-journal.jsonl) tracks
    in-flight tasks. On clean drain the journal is truncated. On next
    boot, orphaned entries are replayed into the queue.
"""
import asyncio
import logging
import os
import uuid
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Callable, Awaitable, Any
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)

from ..core.types import (
    RolodexEntry, Message, ContentModality, EntryCategory, Tier
)


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionTask:
    """
    A unit of background enrichment work.
    Created when a message is ingested; processed by workers.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message: Optional[Message] = None
    stub_entry_ids: List[str] = field(default_factory=list)
    conversation_id: str = ""
    turn_number: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class QueueJournal:
    """Write-ahead journal for crash recovery.

    Before a task is processed, its ID is written to the journal.
    On completion (success or failure), the entry is removed.
    On clean shutdown, the journal is truncated.
    On next boot, any remaining entries are orphans that need replay.
    """

    def __init__(self, workspace: Optional[Path] = None):
        ws = workspace or Path(os.environ.get("SOLITAIRE_WORKSPACE", os.getcwd()))
        self._dir = ws / ".solitaire"
        self._path = self._dir / "queue-journal.jsonl"

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def record_inflight(self, task: "IngestionTask") -> None:
        """Write a task to the journal before processing."""
        try:
            self._ensure_dir()
            entry = {
                "id": task.id,
                "stub_entry_ids": task.stub_entry_ids,
                "conversation_id": task.conversation_id,
                "turn_number": task.turn_number,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Journal failure must never block ingestion

    def record_complete(self, task_id: str) -> None:
        """Remove a completed task from the journal."""
        try:
            if not self._path.exists():
                return
            lines = self._path.read_text(encoding="utf-8").splitlines()
            remaining = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("id") != task_id:
                        remaining.append(line)
                except json.JSONDecodeError:
                    pass
            self._path.write_text("\n".join(remaining) + "\n" if remaining else "", encoding="utf-8")
        except Exception:
            pass

    def get_orphans(self) -> List[Dict[str, Any]]:
        """Return tasks that were in-flight when the process died."""
        try:
            if not self._path.exists():
                return []
            orphans = []
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    orphans.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            return orphans
        except Exception:
            return []

    def truncate(self) -> None:
        """Clear the journal on clean shutdown."""
        try:
            if self._path.exists():
                self._path.write_text("", encoding="utf-8")
        except Exception:
            pass


class IngestionQueue:
    """
    Async queue with background workers for enrichment tasks.

    Usage:
        queue = IngestionQueue(enrichment_fn=agent.process_enrichment_task)
        await queue.start()

        # Fast path: create stub + enqueue
        stub = queue.create_stub_entry(msg, conversation_id)
        await queue.enqueue(task)

        # On query: pause workers, run search, resume
        await queue.pause()
        results = await searcher.search(...)
        await queue.resume()

        # Shutdown
        await queue.shutdown()
    """

    def __init__(
        self,
        enrichment_fn: Optional[Callable[[IngestionTask], Awaitable[None]]] = None,
        num_workers: int = 2,
        max_queue_size: int = 1000,
        pause_on_query: bool = True,
        workspace: Optional[Path] = None,
    ):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._enrichment_fn = enrichment_fn
        self._num_workers = num_workers
        self._pause_on_query = pause_on_query
        self._journal = QueueJournal(workspace)

        # Worker management
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._paused = asyncio.Event()
        self._paused.set()  # Start unpaused (set = not paused)

        # Stats
        self._pending_count = 0
        self._processing_count = 0
        self._completed_count = 0
        self._failed_count = 0

    # ─── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start background workers. Replays orphaned journal entries."""
        if self._running:
            return
        self._running = True

        # Replay orphaned tasks from previous crash
        orphans = self._journal.get_orphans()
        if orphans:
            logger.info("Replaying %d orphaned ingestion tasks", len(orphans))
            for entry in orphans:
                task = IngestionTask(
                    id=entry.get("id", str(uuid.uuid4())),
                    stub_entry_ids=entry.get("stub_entry_ids", []),
                    conversation_id=entry.get("conversation_id", ""),
                    turn_number=entry.get("turn_number", 0),
                )
                try:
                    self._queue.put_nowait(task)
                    self._pending_count += 1
                except asyncio.QueueFull:
                    logger.warning("Queue full, dropping orphan %s", task.id)

        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self._num_workers)
        ]

    async def shutdown(self) -> None:
        """Graceful shutdown: drain pending work, then stop workers."""
        # Drain pending work before signaling stop (prevents task loss)
        self._paused.set()  # Unpause so workers can process
        await self.wait_for_drain(timeout=15.0)
        self._journal.truncate()
        self._running = False
        # Send poison pills
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        # Wait for workers to finish (with timeout)
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    # ─── Pause / Resume ──────────────────────────────────────────────────

    async def pause(self, reason: str = "query") -> None:
        """
        Pause background workers. Workers finish their current task
        then wait until resumed. Non-blocking for the caller.
        """
        if self._pause_on_query:
            self._paused.clear()  # Clear = paused

    async def resume(self) -> None:
        """Resume paused workers."""
        self._paused.set()  # Set = unpaused

    def is_paused(self) -> bool:
        return not self._paused.is_set()

    # ─── Stub Creation ───────────────────────────────────────────────────

    def create_stub_entry(
        self,
        message: Message,
        conversation_id: str,
    ) -> RolodexEntry:
        """
        Create a minimal RolodexEntry stub for immediate storage.
        The stub has content and FTS-searchable text but no embedding
        or enriched categorization yet. That comes from background workers.
        """
        # Derive provenance from message role
        role_to_provenance = {
            "user": "user-stated",
            "assistant": "assistant-inferred",
            "system": "system",
        }
        provenance = role_to_provenance.get(message.role.value, "unknown")

        return RolodexEntry(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            content=message.content,
            content_type=ContentModality.CONVERSATIONAL,
            category=EntryCategory.NOTE,
            tags=["pending-enrichment"],
            source_range={"turn_number": message.turn_number},
            embedding=None,  # Will be filled by enrichment
            tier=Tier.COLD,
            metadata={"enrichment_status": "pending"},
            provenance=provenance,
        )

    # ─── Enqueue ─────────────────────────────────────────────────────────

    async def enqueue(self, task: IngestionTask) -> None:
        """Add an enrichment task to the queue."""
        self._pending_count += 1
        await self._queue.put(task)

    def enqueue_nowait(self, task: IngestionTask) -> bool:
        """Non-blocking enqueue. Returns False if queue is full."""
        try:
            self._queue.put_nowait(task)
            self._pending_count += 1
            return True
        except asyncio.QueueFull:
            return False

    # ─── Workers ─────────────────────────────────────────────────────────

    async def _worker(self, worker_id: int) -> None:
        """
        Background worker loop.
        Pulls tasks from queue, processes them, respects pause signals.
        """
        while self._running:
            try:
                # Wait if paused
                await self._paused.wait()

                # Get next task (with timeout to check _running flag)
                try:
                    task = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Poison pill for shutdown
                if task is None:
                    break

                # Process the task
                task.status = TaskStatus.PROCESSING
                self._pending_count = max(0, self._pending_count - 1)
                self._processing_count += 1
                self._journal.record_inflight(task)

                try:
                    if self._enrichment_fn:
                        await self._enrichment_fn(task)
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now(timezone.utc)
                    self._completed_count += 1
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    self._failed_count += 1
                finally:
                    self._journal.record_complete(task.id)
                    self._processing_count = max(0, self._processing_count - 1)
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception:
                # Worker must not die
                continue

    # ─── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return queue statistics."""
        return {
            "enabled": True,
            "running": self._running,
            "paused": self.is_paused(),
            "num_workers": self._num_workers,
            "queue_size": self._queue.qsize(),
            "pending": self._pending_count,
            "processing": self._processing_count,
            "completed": self._completed_count,
            "failed": self._failed_count,
        }

    async def wait_for_drain(self, timeout: float = 30.0) -> bool:
        """
        Wait until all queued tasks are processed.
        Returns True if drained, False if timed out.
        Useful for testing and graceful shutdown.
        """
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
