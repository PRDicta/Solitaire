"""
The Librarian — JSONL Storage Layer
Append-only JSONL files as an audit trail and secondary persistence layer.

Design principles:
  1. SQLite (via SQL dumps) is the primary source of truth for queries
     and cross-session persistence. The DB is rebuilt from dumps on boot.
  2. JSONL is an append-only audit trail: every mutation is logged here
     for traceability and potential forensic rebuild, but JSONL append
     failure is non-fatal (the live session continues via SQLite).
  3. No overwrites, no deletes on disk.
  4. Every append flushes the index (no batching, no stale windows).
  5. Concurrent sessions append safely (POSIX atomic append < PIPE_BUF).

Historical note: early versions described JSONL as the single source of
truth with SQLite as a disposable cache. In practice, SQLite + SQL dumps
became the canonical persistence path because JSONL rebuilds are slower
and do not capture schema migrations or index state. The JSONL layer is
retained for auditability.
"""

import json
import os
import struct
import base64
import threading
try:
    import fcntl
except ImportError:
    fcntl = None  # Windows: file locking handled via msvcrt below
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Callable, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field


# Maximum size for atomic POSIX append (bytes). Writes under this are guaranteed
# not to interleave with concurrent appenders on the same file.
PIPE_BUF = 4096


@dataclass
class JsonlRecord:
    """A single record in a JSONL file."""
    seq: int
    record_type: str
    op: str
    ts: str
    session: str
    data: Dict[str, Any]
    offset: int = 0  # byte offset in file (populated on read)

    def to_line(self) -> str:
        """Serialize to a single JSON line (no trailing newline)."""
        obj = {
            "_seq": self.seq,
            "_type": self.record_type,
            "_op": self.op,
            "_ts": self.ts,
            "_session": self.session,
            **self.data
        }
        return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))

    @classmethod
    def from_line(cls, line: str, offset: int = 0) -> Optional['JsonlRecord']:
        """Parse a JSON line into a record. Returns None on parse failure."""
        try:
            obj = json.loads(line)
            return cls(
                seq=obj.pop("_seq", 0),
                record_type=obj.pop("_type", "unknown"),
                op=obj.pop("_op", "create"),
                ts=obj.pop("_ts", ""),
                session=obj.pop("_session", ""),
                data=obj,
                offset=offset,
            )
        except (json.JSONDecodeError, KeyError):
            return None


@dataclass
class JsonlIndex:
    """In-memory index mapping record IDs to byte offsets in a JSONL file."""
    version: int = 1
    last_seq: int = 0
    last_offset: int = 0
    entry_count: int = 0
    # id -> {offset, seq, ts} for create records
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # id -> list of {offset, seq, op} for update/supersede/archive/delete
    updates: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def add_record(self, record_id: str, offset: int, seq: int, ts: str, op: str):
        """Register a record in the index."""
        if op == "create":
            self.entries[record_id] = {"offset": offset, "seq": seq, "ts": ts}
            self.entry_count = len(self.entries)
        else:
            if record_id not in self.updates:
                self.updates[record_id] = []
            self.updates[record_id].append({"offset": offset, "seq": seq, "op": op})
        if seq > self.last_seq:
            self.last_seq = seq
        if offset > self.last_offset:
            self.last_offset = offset

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "last_seq": self.last_seq,
            "last_offset": self.last_offset,
            "entry_count": self.entry_count,
            "entries": self.entries,
            "updates": self.updates,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'JsonlIndex':
        idx = cls()
        idx.version = d.get("version", 1)
        idx.last_seq = d.get("last_seq", 0)
        idx.last_offset = d.get("last_offset", 0)
        idx.entry_count = d.get("entry_count", 0)
        idx.entries = d.get("entries", {})
        idx.updates = d.get("updates", {})
        return idx


class JsonlStore:
    """
    Append-only JSONL storage with byte-offset indexing.

    One instance per JSONL file. Thread-safe for appends within a single process.
    Cross-process safety relies on POSIX atomic append semantics.
    """

    def __init__(self, jsonl_path: str, index_path: Optional[str] = None):
        self._path = Path(jsonl_path)
        self._index_path = Path(index_path) if index_path else self._path.with_suffix('.index.json')
        self._lock = threading.Lock()
        self._seq_counter = 0
        self._index = JsonlIndex()

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Load or rebuild index
        self._load_or_rebuild_index()

    def _load_or_rebuild_index(self):
        """Load index from file, verify against JSONL, rebuild if needed."""
        if self._index_path.exists():
            try:
                with open(self._index_path, 'r') as f:
                    self._index = JsonlIndex.from_dict(json.load(f))
                self._seq_counter = self._index.last_seq
            except (json.JSONDecodeError, KeyError):
                self._index = JsonlIndex()
                self._seq_counter = 0

        # Check if JSONL has data beyond what the index covers
        if self._path.exists():
            file_size = self._path.stat().st_size
            if file_size > 0 and (
                self._index.last_offset == 0 or
                file_size > self._index.last_offset + 1024  # rough: index is stale
            ):
                self._rebuild_index_from(self._index.last_offset if self._index.entry_count > 0 else 0)
        else:
            # No JSONL file yet — fresh start
            self._index = JsonlIndex()
            self._seq_counter = 0

    def _rebuild_index_from(self, start_offset: int = 0):
        """Scan JSONL from offset, updating the index for any new records."""
        if not self._path.exists():
            return

        with open(self._path, 'r', encoding='utf-8') as f:
            if start_offset > 0:
                f.seek(start_offset)
                # Skip to next complete line if we're mid-line
                remainder = f.readline()
                if start_offset > 0 and remainder:
                    start_offset += len(remainder.encode('utf-8'))

            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                record = JsonlRecord.from_line(line, offset)
                if record is None:
                    continue  # skip malformed lines
                record_id = record.data.get("id", "")
                if record_id:
                    self._index.add_record(record_id, offset, record.seq, record.ts, record.op)
                elif record.record_type in ("identity_edge", "identity_reference",
                                             "identity_signal", "topic_assignment"):
                    # These don't have a top-level 'id' — index by composite key or skip
                    composite = f"{record.record_type}:{record.seq}"
                    self._index.add_record(composite, offset, record.seq, record.ts, record.op)
                if record.seq > self._seq_counter:
                    self._seq_counter = record.seq

        self._index.last_offset = self._path.stat().st_size if self._path.exists() else 0
        self._flush_index()

    def _flush_index(self):
        """Write index to disk. Called on every append."""
        try:
            tmp = self._index_path.with_suffix('.tmp')
            with open(tmp, 'w') as f:
                json.dump(self._index.to_dict(), f, separators=(',', ':'))
            os.replace(str(tmp), str(self._index_path))
        except OSError:
            pass  # Index flush failure is non-fatal; index rebuilds on next boot

    def next_seq(self) -> int:
        """Get next monotonic sequence number."""
        self._seq_counter += 1
        return self._seq_counter

    def set_batch_mode(self, enabled: bool):
        """Enable/disable batch mode. In batch mode, index flush is deferred
        until batch_flush() is called. Use for bulk imports only."""
        self._batch_mode = enabled
        if not enabled:
            self._flush_index()

    def batch_flush(self):
        """Flush index after a batch of appends."""
        self._flush_index()

    def append(self, record_type: str, op: str, data: Dict[str, Any],
               session_id: str = "", record_id: Optional[str] = None) -> int:
        """
        Append a record to the JSONL file. Returns the sequence number assigned.

        Thread-safe within a single process. Cross-process safe via POSIX atomic append.
        """
        with self._lock:
            seq = self.next_seq()
            record = JsonlRecord(
                seq=seq,
                record_type=record_type,
                op=op,
                ts=datetime.now(timezone.utc).isoformat(),
                session=session_id,
                data=data,
            )
            line = record.to_line() + "\n"
            line_bytes = line.encode('utf-8')

            # Atomic append
            with open(self._path, 'a', encoding='utf-8') as f:
                offset = f.tell()
                f.write(line)
                f.flush()
                # fsync ensures durability; skip in batch mode for bulk imports
                if not getattr(self, '_batch_mode', False):
                    os.fsync(f.fileno())

            # Update in-memory index
            rid = record_id or data.get("id", "")
            if rid:
                self._index.add_record(rid, offset, seq, record.ts, op)
            self._index.last_offset = offset + len(line_bytes)

            # Flush index to disk on every append (unless batch mode)
            if not getattr(self, '_batch_mode', False):
                self._flush_index()

            return seq

    def read_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Read the current state of a record by ID (applying any updates)."""
        if record_id not in self._index.entries:
            return None

        entry_info = self._index.entries[record_id]
        record = self._read_at_offset(entry_info["offset"])
        if record is None:
            return None

        result = record.data.copy()

        # Apply updates in sequence order
        if record_id in self._index.updates:
            updates = sorted(self._index.updates[record_id], key=lambda u: u["seq"])
            for update_info in updates:
                if update_info["op"] == "delete":
                    return None  # Record was hard-deleted
                update_record = self._read_at_offset(update_info["offset"])
                if update_record:
                    # Merge update fields into result (update records carry only changed fields)
                    for k, v in update_record.data.items():
                        if k != "id":  # Don't overwrite ID
                            result[k] = v

        return result

    def read_by_ids(self, record_ids: List[str]) -> List[Dict[str, Any]]:
        """Batch read multiple records by ID."""
        results = []
        for rid in record_ids:
            record = self.read_by_id(rid)
            if record is not None:
                results.append(record)
        return results

    def read_by_offset(self, offset: int) -> Optional[Dict[str, Any]]:
        """Read a single record at a known byte offset."""
        record = self._read_at_offset(offset)
        return record.data if record else None

    def _read_at_offset(self, offset: int) -> Optional[JsonlRecord]:
        """Low-level: read and parse one line at a byte offset."""
        if not self._path.exists():
            return None
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                f.seek(offset)
                line = f.readline().strip()
                if not line:
                    return None
                return JsonlRecord.from_line(line, offset)
        except (OSError, ValueError):
            return None

    def scan(self, predicate: Optional[Callable[[Dict], bool]] = None,
             record_type: Optional[str] = None,
             limit: int = 0) -> Iterator[Dict[str, Any]]:
        """
        Scan the JSONL file, yielding records that match the predicate.
        Only yields 'create' records (use read_by_id to resolve updates).
        """
        if not self._path.exists():
            return

        count = 0
        with open(self._path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = JsonlRecord.from_line(line)
                if record is None:
                    continue
                if record.op != "create":
                    continue
                if record_type and record.record_type != record_type:
                    continue
                if predicate and not predicate(record.data):
                    continue
                yield record.data
                count += 1
                if limit > 0 and count >= limit:
                    return

    def scan_tail(self, n: int, record_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read the last N create records (by sequence order) from the index."""
        # Filter to creates only
        entries = [
            (rid, info) for rid, info in self._index.entries.items()
        ]
        if record_type:
            # We'd need to read each to check type — use sorted seq instead
            entries.sort(key=lambda x: x[1]["seq"], reverse=True)
            results = []
            for rid, info in entries[:n * 2]:  # over-read to account for type filtering
                record = self._read_at_offset(info["offset"])
                if record and (record.record_type == record_type):
                    results.append(record.data)
                    if len(results) >= n:
                        break
            return list(reversed(results))
        else:
            entries.sort(key=lambda x: x[1]["seq"], reverse=True)
            results = []
            for rid, info in entries[:n]:
                data = self.read_by_id(rid)
                if data:
                    results.append(data)
            return list(reversed(results))

    def keyword_scan(self, terms: List[str], field_name: str = "content",
                     limit: int = 50) -> List[Tuple[Dict[str, Any], float]]:
        """
        Simple keyword search over the JSONL file.
        Returns (record, score) tuples sorted by relevance.
        Used as fallback when session-local FTS doesn't cover the full corpus.
        """
        if not self._path.exists():
            return []

        terms_lower = [t.lower() for t in terms]
        results = []

        with open(self._path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = JsonlRecord.from_line(line)
                if record is None or record.op != "create":
                    continue
                content = record.data.get(field_name, "").lower()
                if not content:
                    continue
                # Score: count of matching terms
                score = sum(1 for t in terms_lower if t in content)
                if score > 0:
                    results.append((record.data, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_all_ids(self) -> List[str]:
        """Return all record IDs in the index (create records only)."""
        return list(self._index.entries.keys())

    def count(self) -> int:
        """Number of live records (creates minus deletes)."""
        deleted = set()
        for rid, updates in self._index.updates.items():
            for u in updates:
                if u["op"] == "delete":
                    deleted.add(rid)
        return len(self._index.entries) - len(deleted)

    @property
    def seq_counter(self) -> int:
        return self._seq_counter

    @property
    def index(self) -> JsonlIndex:
        return self._index


# ─── Embedding Helpers ──────────────────────────────────────────────────────

def encode_embedding(embedding: Optional[list]) -> Optional[str]:
    """Pack float32 list to base64 string for JSONL storage."""
    if embedding is None or len(embedding) == 0:
        return None
    packed = struct.pack(f'{len(embedding)}f', *embedding)
    return base64.b64encode(packed).decode('ascii')


def decode_embedding(b64: Optional[str]) -> Optional[list]:
    """Unpack base64 string to float32 list."""
    if b64 is None:
        return None
    try:
        packed = base64.b64decode(b64)
        count = len(packed) // 4
        return list(struct.unpack(f'{count}f', packed))
    except Exception:
        return None


# ─── Persona JSONL Manager ──────────────────────────────────────────────────

class PersonaJsonlStore:
    """
    Manages all JSONL files for a single persona.

    File layout:
      personas/<key>/entries.jsonl      — rolodex entries
      personas/<key>/identity.jsonl     — identity graph
      personas/<key>/chains.jsonl       — reasoning chains
      personas/<key>/sessions.jsonl     — conversations + messages
      personas/<key>/metadata.jsonl     — topics, assignments, documents, facts
      personas/<key>/index.json         — byte-offset index for entries.jsonl
      personas/<key>/manifest_cache.json — last boot manifest
    """

    def __init__(self, persona_dir: str):
        self.persona_dir = Path(persona_dir)
        self.persona_dir.mkdir(parents=True, exist_ok=True)

        self.entries = JsonlStore(
            str(self.persona_dir / "entries.jsonl"),
            str(self.persona_dir / "entries.index.json"),
        )
        self.identity = JsonlStore(
            str(self.persona_dir / "identity.jsonl"),
            str(self.persona_dir / "identity.index.json"),
        )
        self.chains = JsonlStore(
            str(self.persona_dir / "chains.jsonl"),
            str(self.persona_dir / "chains.index.json"),
        )
        self.sessions = JsonlStore(
            str(self.persona_dir / "sessions.jsonl"),
            str(self.persona_dir / "sessions.index.json"),
        )
        self.metadata = JsonlStore(
            str(self.persona_dir / "metadata.jsonl"),
            str(self.persona_dir / "metadata.index.json"),
        )

    def get_manifest_cache_path(self) -> str:
        return str(self.persona_dir / "manifest_cache.json")

    def stats(self) -> Dict[str, Any]:
        return {
            "entries": self.entries.count(),
            "identity": self.identity.count(),
            "chains": self.chains.count(),
            "sessions": self.sessions.count(),
            "metadata": self.metadata.count(),
            "entries_seq": self.entries.seq_counter,
        }
