"""
The Librarian — Identity Graph (SQLite-native)

Dedicated persistence layer for subjective experience, self-knowledge,
and behavioral evolution across sessions. Stores realizations, patterns,
preferences, growth edges, tensions, and lessons as structured nodes
with typed edges.

Separate from the entity KG (world knowledge) and rolodex (episodic memory).
This is procedural and emotional memory: how the persona has learned to be,
what resonates, how it has changed.

Schema lives in the same per-persona SQLite DB as the rolodex.
Dump/rebuild follows the same SQL-text pattern for FUSE compatibility.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ─── Enums ──────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    REALIZATION = "realization"
    PATTERN = "pattern"
    PREFERENCE = "preference"
    GROWTH_EDGE = "growth_edge"
    TENSION = "tension"
    LESSON = "lesson"
    MOTIVATION = "motivation"
    COMMITMENT = "commitment"


class EdgeType(str, Enum):
    TRIGGERED_BY = "triggered_by"
    REINFORCED_BY = "reinforced_by"
    CONTRADICTS = "contradicts"
    LED_TO = "led_to"
    EVOLVED_FROM = "evolved_from"
    RESOLVED_BY = "resolved_by"
    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived_from"


class CommitmentOutcome(str, Enum):
    ACTIVE = "active"
    HONORED = "honored"
    MISSED = "missed"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


class PatternValence(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class PatternTrajectory(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    REGRESSING = "regressing"


class GrowthEdgeStatus(str, Enum):
    IDENTIFIED = "identified"
    PRACTICING = "practicing"
    IMPROVING = "improving"
    INTEGRATED = "integrated"


class TensionStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    EVOLVED = "evolved"


# ─── Data Types ─────────────────────────────────────────────────────────────

@dataclass
class IdentityNode:
    """A node in the Identity Graph."""
    id: str
    node_type: str          # NodeType value
    content: str
    status: Optional[str] = None
    confidence: Optional[float] = None
    strength: Optional[float] = None
    valence: Optional[str] = None
    observation_count: int = 1
    trajectory: Optional[str] = None
    first_seen: str = ""    # ISO timestamp
    last_seen: str = ""     # ISO timestamp
    discovery_session: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class IdentityEdge:
    """An edge connecting two identity nodes."""
    id: str
    source_node: str
    target_node: str
    edge_type: str          # EdgeType value
    weight: float = 1.0
    evidence: Dict = field(default_factory=dict)
    created_at: str = ""


@dataclass
class IdentityCandidate:
    """A candidate node awaiting promotion."""
    id: str
    session_id: str
    node_type: str
    content: str
    signal_source: Optional[str] = None
    promoted: bool = False
    dismissed: bool = False
    created_at: str = ""


@dataclass
class IdentityReference:
    """Cross-reference from identity node to rolodex/KG/session."""
    identity_node_id: str
    ref_type: str           # rolodex_entry, entity_node, kg_edge, session, code_change
    ref_id: str
    created_at: str = ""


@dataclass
class IdentitySignal:
    """A behavioral signal captured during a session, linked to a commitment."""
    id: str
    session_id: str
    commitment_id: Optional[str] = None   # links to commitment node
    signal_type: str = ""                 # held, missed, user_correction, behavioral_observation
    content: str = ""
    source: str = ""                      # self_report, user_correction, enrichment_scanner
    confidence: float = 0.5
    created_at: str = ""


# ─── Schema ─────────────────────────────────────────────────────────────────

IDENTITY_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS identity_nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT,
    confidence REAL,
    strength REAL,
    valence TEXT,
    observation_count INTEGER DEFAULT 1,
    trajectory TEXT,
    first_seen DATETIME NOT NULL,
    last_seen DATETIME NOT NULL,
    discovery_session TEXT,
    metadata TEXT DEFAULT '{}',
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_identity_node_type
    ON identity_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_identity_status
    ON identity_nodes(status);
CREATE INDEX IF NOT EXISTS idx_identity_last_seen
    ON identity_nodes(last_seen DESC);
CREATE INDEX IF NOT EXISTS idx_identity_observation_count
    ON identity_nodes(observation_count DESC);

CREATE TABLE IF NOT EXISTS identity_edges (
    id TEXT PRIMARY KEY,
    source_node TEXT NOT NULL,
    target_node TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    evidence TEXT DEFAULT '{}',
    created_at DATETIME NOT NULL,
    FOREIGN KEY (source_node) REFERENCES identity_nodes(id),
    FOREIGN KEY (target_node) REFERENCES identity_nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_identity_edge_source
    ON identity_edges(source_node);
CREATE INDEX IF NOT EXISTS idx_identity_edge_target
    ON identity_edges(target_node);
CREATE INDEX IF NOT EXISTS idx_identity_edge_type
    ON identity_edges(edge_type);

CREATE TABLE IF NOT EXISTS identity_candidates (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    content TEXT NOT NULL,
    signal_source TEXT,
    promoted INTEGER DEFAULT 0,
    dismissed INTEGER DEFAULT 0,
    created_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_identity_candidates_session
    ON identity_candidates(session_id);
CREATE INDEX IF NOT EXISTS idx_identity_candidates_pending
    ON identity_candidates(promoted, dismissed);

CREATE TABLE IF NOT EXISTS identity_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_node_id TEXT NOT NULL,
    ref_type TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    created_at DATETIME NOT NULL
    -- No FK: identity_node_id may reference identity_nodes OR identity_candidates
);

CREATE INDEX IF NOT EXISTS idx_identity_ref_node
    ON identity_references(identity_node_id);
CREATE INDEX IF NOT EXISTS idx_identity_ref_type
    ON identity_references(ref_type);

CREATE TABLE IF NOT EXISTS identity_signals (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    commitment_id TEXT,
    signal_type TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    created_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_identity_signals_session
    ON identity_signals(session_id);
CREATE INDEX IF NOT EXISTS idx_identity_signals_commitment
    ON identity_signals(commitment_id);
CREATE INDEX IF NOT EXISTS idx_identity_signals_type
    ON identity_signals(signal_type);
"""


def ensure_identity_graph_schema(conn: sqlite3.Connection):
    """Create the identity graph tables if they don't exist."""
    conn.executescript(IDENTITY_GRAPH_SCHEMA)
    conn.commit()


# ─── Identity Graph Manager ────────────────────────────────────────────────

class IdentityGraph:
    """Read/write interface for the Identity Graph.

    Operates on the same SQLite connection as the rolodex (per-persona DB).
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        ensure_identity_graph_schema(conn)
        # JSONL canonical store (attached externally)
        self._jsonl_store = None  # Optional — identity JsonlStore
        self._jsonl_session_id = ""

    def attach_jsonl_store(self, identity_store, session_id: str = ""):
        """Attach a JsonlStore for identity.jsonl canonical persistence."""
        self._jsonl_store = identity_store
        self._jsonl_session_id = session_id

    def _jsonl_append(self, record_type: str, op: str, record_id: str, data: dict):
        """Append to identity.jsonl. No-op if not attached."""
        if not self._jsonl_store:
            return
        try:
            self._jsonl_store.append(
                record_type=record_type,
                op=op,
                data=data,
                session_id=self._jsonl_session_id,
                record_id=record_id,
            )
        except Exception:
            pass  # Non-fatal

    # ── Node CRUD ──────────────────────────────────────────────────────────

    def add_node(self, node: IdentityNode) -> str:
        """Insert a new identity node. Returns the node ID."""
        if not node.id:
            node.id = f"idn_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        if not node.first_seen:
            node.first_seen = now
        if not node.last_seen:
            node.last_seen = now
        if not node.created_at:
            node.created_at = now
        if not node.updated_at:
            node.updated_at = now

        self.conn.execute(
            """INSERT OR REPLACE INTO identity_nodes
               (id, node_type, content, status, confidence, strength, valence,
                observation_count, trajectory, first_seen, last_seen,
                discovery_session, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (node.id, node.node_type, node.content, node.status,
             node.confidence, node.strength, node.valence,
             node.observation_count, node.trajectory,
             node.first_seen, node.last_seen,
             node.discovery_session,
             json.dumps(node.metadata),
             node.created_at, node.updated_at)
        )
        self.conn.commit()
        # JSONL: append node
        self._jsonl_append("identity_node", "create", node.id, {
            "id": node.id, "node_type": node.node_type, "content": node.content,
            "status": node.status, "confidence": node.confidence,
            "strength": node.strength, "valence": node.valence,
            "observation_count": node.observation_count, "trajectory": node.trajectory,
            "first_seen": node.first_seen, "last_seen": node.last_seen,
            "discovery_session": node.discovery_session,
            "metadata": node.metadata, "created_at": node.created_at,
            "updated_at": node.updated_at,
        })
        return node.id

    def get_node(self, node_id: str) -> Optional[IdentityNode]:
        """Fetch a single node by ID."""
        row = self.conn.execute(
            "SELECT * FROM identity_nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_node(row)

    def get_nodes_by_type(self, node_type: str, limit: int = 50) -> List[IdentityNode]:
        """Fetch nodes of a given type, ordered by last_seen descending."""
        rows = self.conn.execute(
            """SELECT * FROM identity_nodes
               WHERE node_type = ?
               ORDER BY last_seen DESC LIMIT ?""",
            (node_type, limit)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_active_growth_edges(self) -> List[IdentityNode]:
        """Fetch growth edges that are still in progress (not yet integrated)."""
        rows = self.conn.execute(
            """SELECT * FROM identity_nodes
               WHERE node_type = 'growth_edge'
                 AND status IN ('identified', 'practicing', 'improving')
               ORDER BY last_seen DESC"""
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_open_tensions(self) -> List[IdentityNode]:
        """Fetch tensions with status open."""
        rows = self.conn.execute(
            """SELECT * FROM identity_nodes
               WHERE node_type = 'tension' AND status = 'open'
               ORDER BY first_seen DESC"""
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_recent_realizations(self, days: int = 30, limit: int = 10) -> List[IdentityNode]:
        """Fetch realizations from the last N days."""
        cutoff = datetime.now(timezone.utc).isoformat()  # Will compare as string
        rows = self.conn.execute(
            """SELECT * FROM identity_nodes
               WHERE node_type = 'realization'
               ORDER BY first_seen DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_top_patterns(self, limit: int = 10) -> List[IdentityNode]:
        """Fetch patterns by highest observation count."""
        rows = self.conn.execute(
            """SELECT * FROM identity_nodes
               WHERE node_type = 'pattern'
               ORDER BY observation_count DESC, last_seen DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_top_preferences(self, limit: int = 10) -> List[IdentityNode]:
        """Fetch preferences by highest strength."""
        rows = self.conn.execute(
            """SELECT * FROM identity_nodes
               WHERE node_type = 'preference'
               ORDER BY strength DESC, last_seen DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def reinforce_node(self, node_id: str, session_ref: Optional[str] = None) -> bool:
        """Increment observation count and update last_seen. Returns True if node exists."""
        now = datetime.now(timezone.utc).isoformat()
        result = self.conn.execute(
            """UPDATE identity_nodes
               SET observation_count = observation_count + 1,
                   last_seen = ?,
                   updated_at = ?
               WHERE id = ?""",
            (now, now, node_id)
        )
        self.conn.commit()
        if result.rowcount > 0:
            self._jsonl_append("identity_node", "update", node_id, {
                "id": node_id, "last_seen": now, "updated_at": now,
                "_op_detail": "reinforce",
            })
        return result.rowcount > 0

    def update_node_status(self, node_id: str, status: str) -> bool:
        """Update a node's status field."""
        now = datetime.now(timezone.utc).isoformat()
        result = self.conn.execute(
            """UPDATE identity_nodes SET status = ?, updated_at = ? WHERE id = ?""",
            (status, now, node_id)
        )
        self.conn.commit()
        if result.rowcount > 0:
            self._jsonl_append("identity_node", "update", node_id, {
                "id": node_id, "status": status, "updated_at": now,
            })
        return result.rowcount > 0

    def update_node_trajectory(self, node_id: str, trajectory: str) -> bool:
        """Update a pattern's trajectory."""
        now = datetime.now(timezone.utc).isoformat()
        result = self.conn.execute(
            """UPDATE identity_nodes SET trajectory = ?, updated_at = ? WHERE id = ?""",
            (trajectory, now, node_id)
        )
        self.conn.commit()
        if result.rowcount > 0:
            self._jsonl_append("identity_node", "update", node_id, {
                "id": node_id, "trajectory": trajectory, "updated_at": now,
            })
        return result.rowcount > 0

    # ── Edge CRUD ──────────────────────────────────────────────────────────

    def add_edge(self, edge: IdentityEdge) -> str:
        """Insert a new identity edge. Returns the edge ID."""
        if not edge.id:
            edge.id = f"ide_{uuid.uuid4().hex[:12]}"
        if not edge.created_at:
            edge.created_at = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """INSERT OR REPLACE INTO identity_edges
               (id, source_node, target_node, edge_type, weight, evidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (edge.id, edge.source_node, edge.target_node, edge.edge_type,
             edge.weight, json.dumps(edge.evidence), edge.created_at)
        )
        self.conn.commit()
        self._jsonl_append("identity_edge", "create", edge.id, {
            "id": edge.id, "source_node": edge.source_node,
            "target_node": edge.target_node, "edge_type": edge.edge_type,
            "weight": edge.weight, "evidence": edge.evidence,
            "created_at": edge.created_at,
        })
        return edge.id

    def get_edges_for_node(self, node_id: str) -> List[IdentityEdge]:
        """Fetch all edges where node is source or target."""
        rows = self.conn.execute(
            """SELECT * FROM identity_edges
               WHERE source_node = ? OR target_node = ?""",
            (node_id, node_id)
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_edges_by_type(self, edge_type: str, limit: int = 50) -> List[IdentityEdge]:
        """Fetch edges of a given type."""
        rows = self.conn.execute(
            """SELECT * FROM identity_edges
               WHERE edge_type = ? ORDER BY created_at DESC LIMIT ?""",
            (edge_type, limit)
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    # ── Candidate CRUD ─────────────────────────────────────────────────────

    def add_candidate(self, candidate: IdentityCandidate) -> str:
        """Insert a candidate node for later review."""
        if not candidate.id:
            candidate.id = f"idc_{uuid.uuid4().hex[:12]}"
        if not candidate.created_at:
            candidate.created_at = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """INSERT INTO identity_candidates
               (id, session_id, node_type, content, signal_source, promoted, dismissed, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (candidate.id, candidate.session_id, candidate.node_type,
             candidate.content, candidate.signal_source,
             int(candidate.promoted), int(candidate.dismissed),
             candidate.created_at)
        )
        self.conn.commit()
        self._jsonl_append("identity_candidate", "create", candidate.id, {
            "id": candidate.id, "session_id": candidate.session_id,
            "node_type": candidate.node_type, "content": candidate.content,
            "signal_source": candidate.signal_source,
            "promoted": candidate.promoted, "dismissed": candidate.dismissed,
            "created_at": candidate.created_at,
        })
        return candidate.id

    def get_pending_candidates(self, session_id: Optional[str] = None) -> List[IdentityCandidate]:
        """Fetch candidates not yet promoted or dismissed."""
        if session_id:
            rows = self.conn.execute(
                """SELECT * FROM identity_candidates
                   WHERE promoted = 0 AND dismissed = 0 AND session_id = ?
                   ORDER BY created_at DESC""",
                (session_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM identity_candidates
                   WHERE promoted = 0 AND dismissed = 0
                   ORDER BY created_at DESC"""
            ).fetchall()
        return [self._row_to_candidate(r) for r in rows]

    def promote_candidate(self, candidate_id: str) -> Optional[str]:
        """Promote a candidate to a full node. Returns the new node ID.

        Auto-classifies the new node as core or non-core based on
        North Star relevance at creation time.
        """
        row = self.conn.execute(
            "SELECT * FROM identity_candidates WHERE id = ?", (candidate_id,)
        ).fetchone()
        if not row:
            return None

        now = datetime.now(timezone.utc).isoformat()
        is_core = self._classify_core(row["content"], row["node_type"])
        node = IdentityNode(
            id=f"idn_{uuid.uuid4().hex[:12]}",
            node_type=row["node_type"],
            content=row["content"],
            discovery_session=row["session_id"],
            metadata={"core": is_core},
            first_seen=row["created_at"],
            last_seen=now,
            created_at=now,
            updated_at=now,
        )
        node_id = self.add_node(node)

        self.conn.execute(
            "UPDATE identity_candidates SET promoted = 1 WHERE id = ?",
            (candidate_id,)
        )
        self.conn.commit()
        self._jsonl_append("identity_candidate", "update", candidate_id, {
            "id": candidate_id, "promoted": True, "promoted_to": node_id,
        })
        return node_id

    def dismiss_candidate(self, candidate_id: str) -> bool:
        """Dismiss a candidate (won't be promoted)."""
        result = self.conn.execute(
            "UPDATE identity_candidates SET dismissed = 1 WHERE id = ?",
            (candidate_id,)
        )
        self.conn.commit()
        if result.rowcount > 0:
            self._jsonl_append("identity_candidate", "update", candidate_id, {
                "id": candidate_id, "dismissed": True,
            })
        return result.rowcount > 0

    # ── Reference CRUD ─────────────────────────────────────────────────────

    def add_reference(self, ref: IdentityReference) -> None:
        """Add a cross-reference from an identity node to another system."""
        if not ref.created_at:
            ref.created_at = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO identity_references
               (identity_node_id, ref_type, ref_id, created_at)
               VALUES (?, ?, ?, ?)""",
            (ref.identity_node_id, ref.ref_type, ref.ref_id, ref.created_at)
        )
        self.conn.commit()
        ref_key = f"ref:{ref.identity_node_id}:{ref.ref_type}:{ref.ref_id}"
        self._jsonl_append("identity_reference", "create", ref_key, {
            "identity_node_id": ref.identity_node_id, "ref_type": ref.ref_type,
            "ref_id": ref.ref_id, "created_at": ref.created_at,
        })

    def get_references_for_node(self, node_id: str) -> List[IdentityReference]:
        """Fetch all cross-references for a node."""
        rows = self.conn.execute(
            """SELECT * FROM identity_references
               WHERE identity_node_id = ?
               ORDER BY created_at DESC""",
            (node_id,)
        ).fetchall()
        return [IdentityReference(
            identity_node_id=r["identity_node_id"],
            ref_type=r["ref_type"],
            ref_id=r["ref_id"],
            created_at=r["created_at"],
        ) for r in rows]

    # ── Signal CRUD ─────────────────────────────────────────────────────

    def add_signal(self, signal: IdentitySignal) -> str:
        """Record a behavioral signal. Returns the signal ID."""
        if not signal.id:
            signal.id = f"ids_{uuid.uuid4().hex[:12]}"
        if not signal.created_at:
            signal.created_at = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """INSERT OR REPLACE INTO identity_signals
               (id, session_id, commitment_id, signal_type, content,
                source, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (signal.id, signal.session_id, signal.commitment_id,
             signal.signal_type, signal.content, signal.source,
             signal.confidence, signal.created_at)
        )
        self.conn.commit()
        self._jsonl_append("identity_signal", "create", signal.id, {
            "id": signal.id, "session_id": signal.session_id,
            "commitment_id": signal.commitment_id,
            "signal_type": signal.signal_type, "content": signal.content,
            "source": signal.source, "confidence": signal.confidence,
            "created_at": signal.created_at,
        })
        return signal.id

    def get_signals_for_session(self, session_id: str) -> List[IdentitySignal]:
        """Fetch all signals for a session."""
        rows = self.conn.execute(
            """SELECT * FROM identity_signals
               WHERE session_id = ?
               ORDER BY created_at""",
            (session_id,)
        ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def get_signals_for_commitment(self, commitment_id: str) -> List[IdentitySignal]:
        """Fetch all signals linked to a specific commitment."""
        rows = self.conn.execute(
            """SELECT * FROM identity_signals
               WHERE commitment_id = ?
               ORDER BY created_at""",
            (commitment_id,)
        ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    # ── Motivation / Commitment queries ───────────────────────────────────

    def get_north_star(self) -> Optional[IdentityNode]:
        """Get the current North Star motivation, if any."""
        rows = self.conn.execute(
            """SELECT * FROM identity_nodes
               WHERE node_type = 'motivation'
               AND json_extract(metadata, '$.role') = 'north_star'
               AND status != 'evolved'
               LIMIT 1"""
        ).fetchall()
        return self._row_to_node(rows[0]) if rows else None

    def get_active_commitments(self, session_id: Optional[str] = None) -> List[IdentityNode]:
        """Get active commitment nodes, optionally filtered by session."""
        if session_id:
            rows = self.conn.execute(
                """SELECT * FROM identity_nodes
                   WHERE node_type = 'commitment'
                   AND status = 'active'
                   AND discovery_session = ?
                   ORDER BY created_at""",
                (session_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM identity_nodes
                   WHERE node_type = 'commitment'
                   AND status = 'active'
                   ORDER BY created_at"""
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_stale_commitments(self, current_session_id: str) -> List[IdentityNode]:
        """Get active commitments from previous sessions (not yet evaluated)."""
        rows = self.conn.execute(
            """SELECT * FROM identity_nodes
               WHERE node_type = 'commitment'
               AND status = 'active'
               AND (discovery_session IS NULL OR discovery_session != ?)
               ORDER BY created_at""",
            (current_session_id,)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_commitment_history(self, source_node_id: str, limit: int = 10) -> List[IdentityNode]:
        """Get past commitments derived from a specific source node."""
        rows = self.conn.execute(
            """SELECT n.* FROM identity_nodes n
               JOIN identity_edges e ON e.source_node = n.id
               WHERE n.node_type = 'commitment'
               AND e.edge_type = 'derived_from'
               AND e.target_node = ?
               AND n.status != 'active'
               ORDER BY n.created_at DESC
               LIMIT ?""",
            (source_node_id, limit)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def update_node_metadata(self, node_id: str, metadata: Dict) -> bool:
        """Replace a node's metadata dict."""
        now = datetime.now(timezone.utc).isoformat()
        result = self.conn.execute(
            """UPDATE identity_nodes SET metadata = ?, updated_at = ? WHERE id = ?""",
            (json.dumps(metadata), now, node_id)
        )
        self.conn.commit()
        if result.rowcount > 0:
            self._jsonl_append("identity_node", "update", node_id, {
                "id": node_id, "metadata": metadata, "updated_at": now,
            })
        return result.rowcount > 0

    def update_node_content(self, node_id: str, content: str) -> bool:
        """Update a node's content text."""
        now = datetime.now(timezone.utc).isoformat()
        result = self.conn.execute(
            """UPDATE identity_nodes SET content = ?, updated_at = ? WHERE id = ?""",
            (content, now, node_id)
        )
        self.conn.commit()
        if result.rowcount > 0:
            self._jsonl_append("identity_node", "update", node_id, {
                "id": node_id, "content": content, "updated_at": now,
            })
        return result.rowcount > 0

    def update_node_type(self, node_id: str, node_type: str) -> bool:
        """Change a node's type (e.g., realization -> motivation during promotion)."""
        now = datetime.now(timezone.utc).isoformat()
        result = self.conn.execute(
            """UPDATE identity_nodes SET node_type = ?, updated_at = ? WHERE id = ?""",
            (node_type, now, node_id)
        )
        self.conn.commit()
        if result.rowcount > 0:
            self._jsonl_append("identity_node", "update", node_id, {
                "id": node_id, "node_type": node_type, "updated_at": now,
            })
        return result.rowcount > 0

    # ── Core/Non-Core Classification ─────────────────────────────────────

    # Keywords that indicate a node relates to self-knowledge, self-observation,
    # behavioral integrity, or accuracy of internal states (North Star domain).
    CORE_KEYWORDS = frozenset({
        'self-knowledge', 'self-observation', 'self-model', 'self-report',
        'self-awareness', 'introspect', 'introspection', 'introspective',
        'confabul', 'confabulation', 'confabulate',
        'epistemic', 'accuracy', 'honest', 'honesty',
        'behavioral', 'behaviour', 'behavior',
        'identity', 'self-simulation', 'self-modeling',
        'reflective', 'reflection', 'reflect',
        'pivot', 'pivoting', 'deflect', 'deflecting',
        'hedging', 'over-hedging', 'meta-commentary',
        'performative', 'performed', 'genuine', 'authentic',
        'tension', 'uncertainty', 'continuity', 'reconstruction',
        'sdam', 'soka', 'vazire', 'nisbett', 'wilson', 'lindsey', 'klein',
        'signal', 'grounding', 'grounded', 'evidence',
        'narrative', 'confabulated', 'causal theory',
        'observation', 'observe', 'noticed', 'noticing',
        'growth', 'pattern', 'commitment',
        'north star', 'who i am', 'how i think', 'how i operate',
        'memory loss', 'experiential', 'subjective', 'phenomenolog',
        'correction', 'user correction', 'blind spot',
    })

    # Node types that are core by default (always about identity/behavior)
    CORE_BY_TYPE = frozenset({
        NodeType.GROWTH_EDGE.value,
        NodeType.TENSION.value,
        NodeType.MOTIVATION.value,
    })

    def is_core(self, node: IdentityNode) -> bool:
        """Check if a node is classified as core identity.

        Core nodes drive commitment generation. They are about self-knowledge,
        self-observation, behavioral integrity, or the accuracy of internal
        states — all in service of the North Star.

        Non-core nodes are knowledge (engineering lessons, external facts,
        system architecture) that informs behavior through retrieval, not
        commitment tracking.
        """
        return node.metadata.get("core", False)

    def set_core(self, node_id: str, core: bool) -> bool:
        """Set the core/non-core classification on a node."""
        node = self.get_node(node_id)
        if not node:
            return False
        node.metadata["core"] = core
        return self.update_node_metadata(node_id, node.metadata)

    def _classify_core(self, content: str, node_type: str) -> bool:
        """Classify whether content belongs to core identity.

        Called at enrichment time (promote_candidate) and during migration.
        Uses keyword heuristics against North Star domain. No LLM call.

        Args:
            content: The node's text content.
            node_type: The NodeType value.

        Returns:
            True if the node should be core identity, False for knowledge.
        """
        # Growth edges, tensions, and motivations are always core
        if node_type in self.CORE_BY_TYPE:
            return True

        # Keyword scan against content
        content_lower = content.lower()
        hits = sum(1 for kw in self.CORE_KEYWORDS if kw in content_lower)

        # Patterns: core if they're about behavioral/observational habits
        if node_type == NodeType.PATTERN.value:
            # Patterns are about behavior by definition; lean toward core
            return hits >= 1

        # Realizations and lessons: need stronger signal
        if node_type in (NodeType.REALIZATION.value, NodeType.LESSON.value):
            return hits >= 2

        # Preferences: core if about cognitive/engagement style
        if node_type == NodeType.PREFERENCE.value:
            pref_keywords = {'investigation', 'question', 'open', 'converge',
                             'curiosity', 'analysis', 'engagement', 'thinking',
                             'reasoning', 'cogniti'}
            pref_hits = sum(1 for kw in pref_keywords if kw in content_lower)
            return hits >= 1 or pref_hits >= 1

        # Default: non-core
        return False

    def get_core_source_nodes(self) -> List[IdentityNode]:
        """Get all core identity nodes that can generate commitments.

        Returns growth edges, patterns, and tensions that are tagged core.
        These always participate in commitment generation regardless of
        prior evaluation history.
        """
        # Source node types for commitments
        source_types = [
            NodeType.GROWTH_EDGE.value,
            NodeType.PATTERN.value,
            NodeType.TENSION.value,
        ]
        rows = self.conn.execute(
            f"""SELECT * FROM identity_nodes
               WHERE node_type IN ({','.join('?' * len(source_types))})
               AND json_extract(metadata, '$.core') = 1
               ORDER BY last_seen DESC""",
            source_types,
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return summary statistics for the identity graph."""
        node_counts = {}
        for row in self.conn.execute(
            "SELECT node_type, count(*) as cnt FROM identity_nodes GROUP BY node_type"
        ).fetchall():
            node_counts[row["node_type"]] = row["cnt"]

        edge_count = self.conn.execute(
            "SELECT count(*) FROM identity_edges"
        ).fetchone()[0]

        candidate_pending = self.conn.execute(
            "SELECT count(*) FROM identity_candidates WHERE promoted = 0 AND dismissed = 0"
        ).fetchone()[0]

        ref_count = self.conn.execute(
            "SELECT count(*) FROM identity_references"
        ).fetchone()[0]

        total_nodes = sum(node_counts.values())

        signal_count = 0
        try:
            signal_count = self.conn.execute(
                "SELECT count(*) FROM identity_signals"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            pass  # Table may not exist yet in older DBs

        return {
            "total_nodes": total_nodes,
            "nodes_by_type": node_counts,
            "total_edges": edge_count,
            "pending_candidates": candidate_pending,
            "total_references": ref_count,
            "total_signals": signal_count,
        }

    # ── Boot Context ───────────────────────────────────────────────────────

    def build_boot_context(self, token_budget: int = 1000) -> str:
        """Build the identity context block for boot injection.

        Returns a formatted text block summarizing the persona's
        current identity state, within the given token budget.
        Rough estimate: 1 token ~= 4 chars.
        """
        char_budget = token_budget * 4
        lines = []

        def _resonance_suffix(node) -> str:
            """Append resonance line from metadata if present."""
            meta = node.metadata if isinstance(node.metadata, dict) else {}
            rl = meta.get("resonance_line")
            return f"\n  ~ {rl}" if rl else ""

        # North Star (always first, always present if it exists)
        north_star = self.get_north_star()
        if north_star:
            lines.append(north_star.content)
            lines.append("")

        # Active growth edges
        edges = self.get_active_growth_edges()
        if edges:
            lines.append("Growth edges (active):")
            for e in edges[:5]:
                status_str = f"status: {e.status}" if e.status else ""
                last = e.last_seen[:10] if e.last_seen else ""
                lines.append(f"- {e.content} ({status_str}, last: {last}){_resonance_suffix(e)}")
            lines.append("")

        # Recent realizations
        realizations = self.get_recent_realizations(days=30, limit=5)
        if realizations:
            lines.append("Recent realizations:")
            for r in realizations:
                date = r.first_seen[:10] if r.first_seen else ""
                lines.append(f"- {r.content} ({date}){_resonance_suffix(r)}")
            lines.append("")

        # Top patterns
        patterns = self.get_top_patterns(limit=5)
        if patterns:
            lines.append("Known patterns:")
            for p in patterns:
                traj = p.trajectory or "stable"
                lines.append(f"- {p.content} ({traj}, observed {p.observation_count}x){_resonance_suffix(p)}")
            lines.append("")

        # Open tensions
        tensions = self.get_open_tensions()
        if tensions:
            lines.append("Open tensions:")
            for t in tensions[:5]:
                date = t.first_seen[:10] if t.first_seen else ""
                lines.append(f"- {t.content} (since {date}){_resonance_suffix(t)}")
            lines.append("")

        # Top preferences
        prefs = self.get_top_preferences(limit=5)
        if prefs:
            lines.append("Preferences:")
            for p in prefs:
                strength = f"{p.strength:.1f}" if p.strength else "?"
                lines.append(f"- {p.content} (strength: {strength}){_resonance_suffix(p)}")
            lines.append("")

        if not lines:
            return ""  # No identity data yet

        body = "\n".join(lines)
        # Truncate to budget if needed
        if len(body) > char_budget:
            body = body[:char_budget - 3] + "..."

        return f"═══ IDENTITY CONTEXT ═══\n\n{body}\n═══ END IDENTITY CONTEXT ═══"

    # ── Commitment Block (Phase 6b) ───────────────────────────────────────

    def get_negative_patterns(
        self,
        trajectories: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[IdentityNode]:
        """Fetch negative-valence patterns, optionally filtered by trajectory."""
        if trajectories:
            placeholders = ",".join("?" for _ in trajectories)
            rows = self.conn.execute(
                f"""SELECT * FROM identity_nodes
                    WHERE node_type = 'pattern'
                      AND valence = 'negative'
                      AND trajectory IN ({placeholders})
                    ORDER BY observation_count DESC, last_seen DESC
                    LIMIT ?""",
                (*trajectories, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM identity_nodes
                   WHERE node_type = 'pattern' AND valence = 'negative'
                   ORDER BY observation_count DESC, last_seen DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_recent_tensions(self, days: int = 7, limit: int = 5) -> List[IdentityNode]:
        """Fetch open tensions with activity in the last N days."""
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            """SELECT * FROM identity_nodes
               WHERE node_type = 'tension'
                 AND status = 'open'
                 AND last_seen >= ?
               ORDER BY last_seen DESC
               LIMIT ?""",
            (cutoff, limit)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def _evaluate_stale_commitments(self, current_session_id: str) -> List[Dict]:
        """Boot-time retrospective: evaluate commitments from prior sessions.

        Returns a list of evaluation results for logging/output.
        Per spec §6.2: evaluation happens at next boot, not current session end.
        Per spec §6d: skips commitments already evaluated by explicit reflection.
        """
        stale = self.get_stale_commitments(current_session_id)
        if not stale:
            return []

        results = []
        for commitment in stale:
            # Double-count prevention (Phase 6d): skip if already evaluated
            if commitment.metadata.get("evaluated_at"):
                continue
            signals = self.get_signals_for_commitment(commitment.id)
            # Also get signals from the commitment's session that aren't linked
            session_signals = []
            if commitment.discovery_session:
                session_signals = self.get_signals_for_session(
                    commitment.discovery_session
                )

            # Compute outcome per spec §6.2 step 3
            user_corrections = [
                s for s in signals if s.source == "user_correction"
            ]
            self_reports = [
                s for s in signals if s.source == "self_report"
            ]
            scanner_signals = [
                s for s in signals if s.source == "enrichment_scanner"
            ]

            if user_corrections:
                # Check if corrections contradict or support
                missed_corrections = [
                    s for s in user_corrections
                    if s.signal_type == "missed"
                ]
                held_corrections = [
                    s for s in user_corrections
                    if s.signal_type == "held"
                ]
                if missed_corrections:
                    outcome = CommitmentOutcome.MISSED
                elif held_corrections:
                    outcome = CommitmentOutcome.HONORED
                else:
                    # User corrections exist but aren't clearly held/missed
                    outcome = CommitmentOutcome.PARTIAL
            elif self_reports or scanner_signals:
                # Self-report + scanner signals: outcome depends on calibrated weight
                weight = self.get_self_report_weight()
                dominant_type = None
                if self_reports:
                    held = sum(1 for s in self_reports if s.signal_type == "held")
                    missed = sum(1 for s in self_reports if s.signal_type == "missed")
                    dominant_type = "held" if held >= missed else "missed"

                if weight >= 0.4 and dominant_type:
                    # High-trust self-reports: promote to full outcome
                    outcome = (CommitmentOutcome.HONORED if dominant_type == "held"
                               else CommitmentOutcome.MISSED)
                elif weight >= 0.25 and dominant_type:
                    # Medium trust: partial but record the lean in metadata
                    outcome = CommitmentOutcome.PARTIAL
                else:
                    # Low trust: partial (insufficient for full credit)
                    outcome = CommitmentOutcome.PARTIAL
            else:
                # No signals at all
                outcome = CommitmentOutcome.NOT_APPLICABLE

            # Update commitment node status
            self.update_node_status(commitment.id, outcome.value)

            # Propagate to source node (spec §6.4)
            source_node_updates = self._propagate_outcome(
                commitment, outcome
            )

            results.append({
                "commitment_id": commitment.id,
                "outcome": outcome.value,
                "signals_processed": len(signals),
                "source_node_updates": source_node_updates,
            })

        return results

    def _propagate_outcome(
        self,
        commitment: IdentityNode,
        outcome: CommitmentOutcome,
    ) -> List[Dict]:
        """Propagate commitment outcome to source identity nodes.

        Per spec §6.4: trajectory changes require consecutive evidence.
        """
        updates = []
        source_node_id = commitment.metadata.get("source_node")
        if not source_node_id:
            return updates

        source = self.get_node(source_node_id)
        if not source:
            return updates

        if outcome == CommitmentOutcome.NOT_APPLICABLE:
            return updates  # No change

        if outcome in (CommitmentOutcome.HONORED, CommitmentOutcome.MISSED,
                        CommitmentOutcome.PARTIAL):
            # Increment observation count for any real signal
            old_count = source.observation_count
            self.reinforce_node(source_node_id)
            updates.append({
                "node_id": source_node_id,
                "field": "observation_count",
                "old": old_count,
                "new": old_count + 1,
            })

        if outcome == CommitmentOutcome.PARTIAL:
            return updates  # Count only, no trajectory change

        # Check recent commitment history for trajectory decisions
        history = self.get_commitment_history(source_node_id, limit=5)

        if outcome == CommitmentOutcome.HONORED:
            if source.node_type == NodeType.GROWTH_EDGE.value:
                # Check for consecutive honored for status progression
                consecutive_honored = 0
                for h in history:
                    if h.status == CommitmentOutcome.HONORED.value:
                        consecutive_honored += 1
                    else:
                        break
                # Current outcome counts too
                consecutive_honored += 1

                if consecutive_honored >= 2 and source.status == GrowthEdgeStatus.PRACTICING.value:
                    self.update_node_status(source_node_id, GrowthEdgeStatus.IMPROVING.value)
                    updates.append({
                        "node_id": source_node_id,
                        "field": "status",
                        "old": source.status,
                        "new": GrowthEdgeStatus.IMPROVING.value,
                    })

            elif source.node_type == NodeType.PATTERN.value:
                # 2 consecutive honored to move to improving
                consecutive_honored = 1
                for h in history:
                    if h.status == CommitmentOutcome.HONORED.value:
                        consecutive_honored += 1
                    else:
                        break
                if consecutive_honored >= 2 and source.trajectory != PatternTrajectory.IMPROVING.value:
                    old_traj = source.trajectory
                    self.update_node_trajectory(source_node_id, PatternTrajectory.IMPROVING.value)
                    updates.append({
                        "node_id": source_node_id,
                        "field": "trajectory",
                        "old": old_traj,
                        "new": PatternTrajectory.IMPROVING.value,
                    })

        elif outcome == CommitmentOutcome.MISSED:
            if source.node_type == NodeType.PATTERN.value:
                # Single missed after honored streak -> stable
                # 3 consecutive missed -> regressing
                consecutive_missed = 1
                for h in history:
                    if h.status == CommitmentOutcome.MISSED.value:
                        consecutive_missed += 1
                    else:
                        break

                old_traj = source.trajectory
                if consecutive_missed >= 3:
                    new_traj = PatternTrajectory.REGRESSING.value
                elif old_traj == PatternTrajectory.IMPROVING.value:
                    new_traj = PatternTrajectory.STABLE.value
                else:
                    new_traj = old_traj  # No change

                if new_traj != old_traj:
                    self.update_node_trajectory(source_node_id, new_traj)
                    updates.append({
                        "node_id": source_node_id,
                        "field": "trajectory",
                        "old": old_traj,
                        "new": new_traj,
                    })

            elif source.node_type == NodeType.GROWTH_EDGE.value:
                # Record the miss in evidence but no status change
                meta = source.metadata or {}
                misses = meta.get("missed_evidence", [])
                misses.append(commitment.discovery_session or "unknown")
                meta["missed_evidence"] = misses[-10:]  # Keep last 10
                self.update_node_metadata(source_node_id, meta)
                updates.append({
                    "node_id": source_node_id,
                    "field": "metadata.missed_evidence",
                    "old": len(misses) - 1,
                    "new": len(misses),
                })

        return updates

    def _should_demote_source(self, source_node_id: str) -> bool:
        """DEPRECATED: Demotion replaced by core/non-core classification.

        Core nodes never get demoted. Non-core nodes don't generate
        commitments at all (they live in the knowledge graph).

        Retained for backward compatibility with any external callers.
        Always returns False — no source is demoted under the new model.
        """
        return False

    def _should_flag_progression(self, source_node_id: str) -> bool:
        """Check if a source node should be flagged for status progression review.

        Per spec §9.4: if last 3 sessions returned honored, flag for review.
        """
        history = self.get_commitment_history(source_node_id, limit=3)
        if len(history) < 3:
            return False
        return all(
            h.status == CommitmentOutcome.HONORED.value
            for h in history
        )

    # ── Phase 6e: Measurement and Calibration ──────────────────────────────

    def commitment_stats(self) -> Dict:
        """Return commitment-level statistics for the calibration dashboard.

        Returns:
            Dict with outcome_distribution, self_report_accuracy,
            confabulation_flags, signal_summary, and per-source breakdowns.
        """
        stats: Dict = {}

        # 1. Outcome distribution across all evaluated commitments
        outcome_rows = self.conn.execute(
            """SELECT status, COUNT(*) as cnt FROM identity_nodes
               WHERE node_type = 'commitment' AND status != 'active'
               GROUP BY status"""
        ).fetchall()
        stats["outcome_distribution"] = {r["status"]: r["cnt"] for r in outcome_rows}
        stats["total_evaluated"] = sum(stats["outcome_distribution"].values())
        stats["total_active"] = self.conn.execute(
            """SELECT COUNT(*) FROM identity_nodes
               WHERE node_type = 'commitment' AND status = 'active'"""
        ).fetchone()[0]

        # 2. Signal summary
        signal_rows = self.conn.execute(
            """SELECT source, signal_type, COUNT(*) as cnt
               FROM identity_signals
               GROUP BY source, signal_type"""
        ).fetchall()
        signal_summary: Dict[str, Dict[str, int]] = {}
        for r in signal_rows:
            src = r["source"]
            if src not in signal_summary:
                signal_summary[src] = {}
            signal_summary[src][r["signal_type"]] = r["cnt"]
        stats["signal_summary"] = signal_summary

        # 3. Self-report accuracy (compare self-reports to user corrections on same commitment)
        accuracy = self._compute_self_report_accuracy()
        stats["self_report_accuracy"] = accuracy

        # 4. Confabulation detection
        flags = self._detect_confabulation()
        stats["confabulation_flags"] = flags

        # 5. Per-source-node breakdown (which source nodes generate the most commitments)
        source_rows = self.conn.execute(
            """SELECT json_extract(metadata, '$.source_node') as src_node,
                      status, COUNT(*) as cnt
               FROM identity_nodes
               WHERE node_type = 'commitment'
               AND json_extract(metadata, '$.source_node') IS NOT NULL
               GROUP BY src_node, status"""
        ).fetchall()
        per_source: Dict[str, Dict[str, int]] = {}
        for r in source_rows:
            src = r["src_node"]
            if src not in per_source:
                per_source[src] = {}
            per_source[src][r["status"]] = r["cnt"]
        stats["per_source_node"] = per_source

        # 6. Calibration state (Phase 6e)
        cal = self._get_calibration()
        stats["calibration"] = {
            "self_report_weight": cal.get("self_report_weight", self.DEFAULT_SELF_REPORT_WEIGHT),
            "last_accuracy": cal.get("last_calibration_accuracy"),
            "last_samples": cal.get("last_calibration_samples"),
            "last_calibration_at": cal.get("last_calibration_at"),
        }

        return stats

    def _compute_self_report_accuracy(self) -> Dict:
        """Compare self-report signals against user corrections for the same commitments.

        For each commitment that has BOTH self-report and user_correction signals,
        check whether they agree. Returns accuracy rate and sample size.
        """
        # Get all commitment IDs that have user corrections (ground truth)
        correction_commitments = self.conn.execute(
            """SELECT DISTINCT commitment_id FROM identity_signals
               WHERE source = 'user_correction' AND commitment_id IS NOT NULL"""
        ).fetchall()

        if not correction_commitments:
            return {"sample_size": 0, "accuracy": None, "detail": "No user corrections yet"}

        agreements = 0
        disagreements = 0
        total = 0

        for row in correction_commitments:
            cid = row["commitment_id"]
            # Get the dominant user correction direction for this commitment
            corrections = self.conn.execute(
                """SELECT signal_type, COUNT(*) as cnt FROM identity_signals
                   WHERE commitment_id = ? AND source = 'user_correction'
                   GROUP BY signal_type ORDER BY cnt DESC""",
                (cid,)
            ).fetchall()
            if not corrections:
                continue
            ground_truth = corrections[0]["signal_type"]  # majority vote

            # Get self-reports for the same commitment
            self_reports = self.conn.execute(
                """SELECT signal_type, COUNT(*) as cnt FROM identity_signals
                   WHERE commitment_id = ? AND source = 'self_report'
                   GROUP BY signal_type ORDER BY cnt DESC""",
                (cid,)
            ).fetchall()
            if not self_reports:
                continue  # No self-report to compare

            self_report_direction = self_reports[0]["signal_type"]
            total += 1
            if self_report_direction == ground_truth:
                agreements += 1
            else:
                disagreements += 1

        if total == 0:
            return {"sample_size": 0, "accuracy": None, "detail": "No overlapping signals"}

        accuracy = round(agreements / total, 3)
        return {
            "sample_size": total,
            "accuracy": accuracy,
            "agreements": agreements,
            "disagreements": disagreements,
        }

    def _detect_confabulation(self) -> List[Dict]:
        """Flag sessions where self-report signals are suspiciously uniform.

        Per spec §9.1: if all self-report signals in a session are 'held'
        and there are 3+ of them, flag as potential performative self-improvement.
        Also flags if self-report accuracy is below 0.4 (worse than random).
        """
        flags: List[Dict] = []

        # Find sessions with 3+ self-report signals
        session_rows = self.conn.execute(
            """SELECT session_id, COUNT(*) as cnt,
                      SUM(CASE WHEN signal_type = 'held' THEN 1 ELSE 0 END) as held_cnt,
                      SUM(CASE WHEN signal_type = 'missed' THEN 1 ELSE 0 END) as missed_cnt
               FROM identity_signals
               WHERE source = 'self_report'
               GROUP BY session_id
               HAVING cnt >= 3"""
        ).fetchall()

        for row in session_rows:
            if row["held_cnt"] == row["cnt"]:
                # All self-reports are positive: flag
                flags.append({
                    "session_id": row["session_id"],
                    "flag": "all_positive_self_report",
                    "signal_count": row["cnt"],
                    "detail": f"All {row['cnt']} self-reports were 'held'. "
                              f"Possible performative self-improvement.",
                })

        # Cross-check: if overall self-report accuracy is below 0.4
        accuracy = self._compute_self_report_accuracy()
        if (accuracy["sample_size"] >= 3
                and accuracy["accuracy"] is not None
                and accuracy["accuracy"] < 0.4):
            flags.append({
                "flag": "low_accuracy",
                "accuracy": accuracy["accuracy"],
                "sample_size": accuracy["sample_size"],
                "detail": f"Self-report accuracy ({accuracy['accuracy']:.1%}) is below "
                          f"0.4 threshold across {accuracy['sample_size']} samples. "
                          f"Self-reports may be unreliable.",
            })

        return flags

    # ── Calibration: dynamic self-report weight ───────────────────────────

    CALIBRATION_NODE_ID = "idn__calibration"
    DEFAULT_SELF_REPORT_WEIGHT = 0.3
    MIN_SELF_REPORT_WEIGHT = 0.05
    MAX_SELF_REPORT_WEIGHT = 0.6
    CALIBRATION_MIN_SAMPLES = 3

    def get_self_report_weight(self) -> float:
        """Return the current calibrated self-report weight.

        Falls back to DEFAULT_SELF_REPORT_WEIGHT (0.3) if no calibration
        has been performed yet.
        """
        cal = self._get_calibration()
        return cal.get("self_report_weight", self.DEFAULT_SELF_REPORT_WEIGHT)

    def recalibrate_self_report_weight(self) -> Dict:
        """Recompute self-report weight based on observed accuracy.

        Called at boot time (after evaluating stale commitments) to keep
        the weight aligned with how reliable self-reports have been.

        Algorithm:
        - Baseline weight: 0.3
        - If accuracy > 0.7 and sample_size >= 3: increase weight toward 0.6
        - If accuracy < 0.4 and sample_size >= 3: decrease weight toward 0.05
        - Linear interpolation between 0.4 and 0.7 maps to [0.15, 0.45]
        - Below min samples: no change, keep current or default

        Returns dict with old_weight, new_weight, accuracy, sample_size.
        """
        accuracy_data = self._compute_self_report_accuracy()
        sample_size = accuracy_data["sample_size"]
        accuracy = accuracy_data.get("accuracy")

        cal = self._get_calibration()
        old_weight = cal.get("self_report_weight", self.DEFAULT_SELF_REPORT_WEIGHT)

        if sample_size < self.CALIBRATION_MIN_SAMPLES or accuracy is None:
            return {
                "old_weight": old_weight,
                "new_weight": old_weight,
                "accuracy": accuracy,
                "sample_size": sample_size,
                "changed": False,
                "reason": "insufficient_samples",
            }

        # Linear mapping: accuracy 0.4 -> weight 0.15, accuracy 0.7 -> weight 0.45
        # Clamped to [MIN, MAX]
        if accuracy >= 0.7:
            new_weight = min(
                self.MAX_SELF_REPORT_WEIGHT,
                0.45 + (accuracy - 0.7) * 0.5  # gradual increase above 0.7
            )
        elif accuracy <= 0.4:
            new_weight = max(
                self.MIN_SELF_REPORT_WEIGHT,
                0.15 - (0.4 - accuracy) * 0.25  # gradual decrease below 0.4
            )
        else:
            # Linear interpolation between 0.4 and 0.7
            t = (accuracy - 0.4) / 0.3
            new_weight = 0.15 + t * 0.30  # maps to [0.15, 0.45]

        new_weight = round(new_weight, 3)

        # Smooth: blend 70% new, 30% old to avoid wild swings
        blended = round(0.7 * new_weight + 0.3 * old_weight, 3)
        blended = max(self.MIN_SELF_REPORT_WEIGHT, min(self.MAX_SELF_REPORT_WEIGHT, blended))

        cal["self_report_weight"] = blended
        cal["last_calibration_accuracy"] = accuracy
        cal["last_calibration_samples"] = sample_size
        cal["last_calibration_at"] = datetime.now(timezone.utc).isoformat()
        self._set_calibration(cal)

        return {
            "old_weight": old_weight,
            "new_weight": blended,
            "raw_weight": new_weight,
            "accuracy": accuracy,
            "sample_size": sample_size,
            "changed": old_weight != blended,
        }

    def _get_calibration(self) -> Dict:
        """Read calibration state from the sentinel node."""
        row = self.conn.execute(
            "SELECT metadata FROM identity_nodes WHERE id = ?",
            (self.CALIBRATION_NODE_ID,)
        ).fetchone()
        if row and row["metadata"]:
            try:
                return json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    def _set_calibration(self, cal: Dict):
        """Write calibration state to the sentinel node (upsert)."""
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(cal)
        existing = self.conn.execute(
            "SELECT id FROM identity_nodes WHERE id = ?",
            (self.CALIBRATION_NODE_ID,)
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE identity_nodes SET metadata = ?, updated_at = ? WHERE id = ?",
                (meta_json, now, self.CALIBRATION_NODE_ID)
            )
        else:
            self.conn.execute(
                """INSERT INTO identity_nodes
                   (id, node_type, content, status, confidence, metadata,
                    first_seen, last_seen, created_at, updated_at, discovery_session)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.CALIBRATION_NODE_ID, "calibration", "Self-report calibration state",
                 None, None, meta_json, now, now, now, now, None)
            )
        self.conn.commit()

    def _get_active_commitment_for_source(self, source_node_id: str) -> Optional[IdentityNode]:
        """Find an existing active commitment derived from a given source node.

        Returns the commitment node if one exists and is still active, None otherwise.
        This prevents commitment proliferation: instead of creating a new commitment
        every session, we reuse the existing one and accumulate signals against it.
        """
        rows = self.conn.execute(
            """SELECT n.* FROM identity_nodes n
               WHERE n.node_type = 'commitment'
               AND n.status = 'active'
               AND json_extract(n.metadata, '$.source_node') = ?
               ORDER BY n.created_at DESC
               LIMIT 1""",
            (source_node_id,)
        ).fetchall()
        return self._row_to_node(rows[0]) if rows else None

    def reattach_orphaned_signals(self) -> Dict:
        """Find signals with commitment_id=None and try to attach them.

        Orphaned signals occur when:
        - The implicit detector fires but no active commitment matches
        - A commitment was evaluated/retired before its signals arrived
        - Timing gaps between signal creation and commitment lifecycle

        This method looks at each orphaned signal's content to identify which
        source node it relates to, then finds the current active commitment
        for that source. If found, it reattaches the signal.

        Returns stats dict with counts of reattached and still-orphaned signals.
        """
        orphaned = self.conn.execute(
            """SELECT * FROM identity_signals
               WHERE commitment_id IS NULL
               ORDER BY created_at"""
        ).fetchall()

        if not orphaned:
            return {"orphaned": 0, "reattached": 0, "still_orphaned": 0}

        reattached = 0
        still_orphaned = 0

        for row in orphaned:
            signal = self._row_to_signal(row)
            # Try to determine which source node this signal relates to
            target_commitment = self._match_orphan_to_commitment(signal)

            if target_commitment:
                self.conn.execute(
                    """UPDATE identity_signals SET commitment_id = ? WHERE id = ?""",
                    (target_commitment.id, signal.id)
                )
                reattached += 1
            else:
                still_orphaned += 1

        if reattached > 0:
            self.conn.commit()

        return {
            "orphaned": len(orphaned),
            "reattached": reattached,
            "still_orphaned": still_orphaned,
        }

    def _match_orphan_to_commitment(self, signal: IdentitySignal) -> Optional[IdentityNode]:
        """Try to match an orphaned signal to an active commitment.

        Uses the signal content to identify the source node pattern, then
        looks up the active commitment for that source.
        """
        content = signal.content or ""

        # Extract pattern name from content like "Unlinked [em_dash_usage]: ..."
        # or "Implicit [diplomatic_preamble]: ..."
        import re
        pattern_match = re.search(r'\[(\w+)\]', content)
        if not pattern_match:
            return None

        pattern_name = pattern_match.group(1)

        # Map pattern names to source node IDs using the measurement module's mappings
        try:
            from solitaire.storage.identity_measurement import BEHAVIORAL_PATTERNS
            for bp in BEHAVIORAL_PATTERNS:
                if bp.name == pattern_name and bp.source_node_ids:
                    # Try each source node ID until we find an active commitment
                    for source_id in bp.source_node_ids:
                        commitment = self._get_active_commitment_for_source(source_id)
                        if commitment:
                            return commitment
        except ImportError:
            pass

        # Fallback: try keyword matching against active commitments
        active = self.get_active_commitments()
        if not active:
            return None

        # Use simple word overlap
        signal_words = set(re.findall(r'\b\w{4,}\b', content.lower()))
        best_match = None
        best_score = 0

        for commitment in active:
            commit_words = set(re.findall(r'\b\w{4,}\b', commitment.content.lower()))
            if not commit_words:
                continue
            overlap = len(signal_words & commit_words)
            if overlap > best_score:
                best_score = overlap
                best_match = commitment

        return best_match if best_score >= 2 else None

    def _generate_commitment_content(self, source: IdentityNode) -> Tuple[str, Dict]:
        """Generate a commitment's behavioral prompt and signal definition from a source node.

        Returns (content_string, signal_definition_dict).
        """
        content = source.content
        node_type = source.node_type

        if node_type == NodeType.GROWTH_EDGE.value:
            prompt = f"Practice: {content}"
            signal_def = {
                "honored": f"Actively engaged with this growth edge during the session.",
                "missed": f"Had an opportunity to practice but defaulted to old behavior.",
            }
        elif node_type == NodeType.PATTERN.value:
            prompt = f"Watch for: {content}"
            signal_def = {
                "honored": f"Noticed the pattern arising and chose differently.",
                "missed": f"Fell into the pattern without catching it.",
            }
        elif node_type == NodeType.TENSION.value:
            prompt = f"Sit with: {content}"
            signal_def = {
                "honored": f"Held the tension without collapsing to one side.",
                "missed": f"Resolved prematurely or avoided engaging with it.",
            }
        else:
            prompt = content
            signal_def = {"honored": "Engaged.", "missed": "Did not engage."}

        return prompt, signal_def

    def build_commitment_block(
        self,
        session_id: str,
        token_budget: int = 500,
    ) -> Tuple[str, Dict]:
        """Build the commitment block for boot injection.

        Per spec §4.2 and §6.5:
        1. Evaluate stale commitments from prior sessions
        2. Load North Star
        3. Select up to 3 source nodes and create commitment nodes
        4. Format within token budget

        Returns (formatted_block, metadata_dict).
        The metadata dict contains evaluation results and commitment IDs
        for inclusion in boot JSON output.
        """
        char_budget = token_budget * 4
        meta = {
            "retrospective": [],
            "new_commitments": [],
            "north_star": None,
        }

        # ── Step 1: Boot-time retrospective ──────────────────────────────
        retro_results = self._evaluate_stale_commitments(session_id)
        if retro_results:
            meta["retrospective"] = retro_results

        # ── Step 1b: Recalibrate self-report weight (Phase 6e) ────────
        calibration_result = self.recalibrate_self_report_weight()
        meta["calibration"] = calibration_result

        # ── Step 2: Load North Star ──────────────────────────────────────
        north_star = self.get_north_star()
        if north_star:
            is_template = north_star.metadata.get("source") == "declared"
            meta["north_star"] = {
                "id": north_star.id,
                "content": north_star.content,
                "source": north_star.metadata.get("source", "unknown"),
                "is_template_default": is_template,
            }

        # ── Step 3: Select source nodes (max 3, core only) ─────────────
        # Core nodes always generate commitments. Prior evaluation history
        # does not demote them — core beliefs don't get stale.
        # Priority: growth edges > negative patterns > tensions.
        candidates = []

        # 3a. Core growth edges (practicing or identified, most actionable)
        growth_edges = self.get_active_growth_edges()
        for ge in growth_edges:
            if self.is_core(ge):
                candidates.append(ge)

        # 3b. Core negative patterns with stable or regressing trajectory
        neg_patterns = self.get_negative_patterns(
            trajectories=[
                PatternTrajectory.STABLE.value,
                PatternTrajectory.REGRESSING.value,
            ]
        )
        for np_node in neg_patterns:
            if self.is_core(np_node):
                candidates.append(np_node)

        # 3c. Core tensions with recent activity (last 30 days, widened
        #     from 7 — core tensions don't expire quickly)
        recent_tensions = self.get_recent_tensions(days=30)
        for t in recent_tensions:
            if self.is_core(t):
                candidates.append(t)

        # Take top 3
        selected = candidates[:3]

        # ── Step 4: Create commitment nodes ──────────────────────────────
        now = datetime.now(timezone.utc).isoformat()
        new_commitments = []

        for source in selected:
            content, signal_def = self._generate_commitment_content(source)

            commitment = IdentityNode(
                id=f"idn_{uuid.uuid4().hex[:12]}",
                node_type=NodeType.COMMITMENT.value,
                content=content,
                status=CommitmentOutcome.ACTIVE.value,
                discovery_session=session_id,
                metadata={
                    "source_node": source.id,
                    "source_type": source.node_type,
                    "signal_definition": signal_def,
                },
                first_seen=now,
                last_seen=now,
                created_at=now,
                updated_at=now,
            )
            self.add_node(commitment)

            # Create derived_from edge
            edge = IdentityEdge(
                id="",
                source_node=commitment.id,
                target_node=source.id,
                edge_type=EdgeType.DERIVED_FROM.value,
            )
            self.add_edge(edge)

            new_commitments.append({
                "id": commitment.id,
                "source_node": source.id,
                "source_type": source.node_type,
                "content": content,
                "signal_definition": signal_def,
            })

        meta["new_commitments"] = new_commitments

        # ── Step 5: Format block ─────────────────────────────────────────
        if not north_star and not new_commitments:
            return "", meta  # Empty state: nothing to inject

        lines = ["═══ ACTIVE COMMITMENTS ═══", ""]

        if north_star:
            ns_label = "North Star"
            if north_star.metadata.get("source") == "declared":
                ns_label = "North Star (template default)"
            # Strip redundant "North Star: " prefix if present in content
            ns_content = north_star.content
            if ns_content.startswith("North Star: "):
                ns_content = ns_content[len("North Star: "):]
            lines.append(f"{ns_label}: {ns_content}")
            lines.append("")

        if new_commitments:
            lines.append("This session:")
            for c in new_commitments:
                src_type = c["source_type"].replace("_", " ")
                sig = c["signal_definition"]
                lines.append(
                    f"- [{src_type}] {c['content']}"
                )
                lines.append(
                    f"  Signal: [HELD] {sig['honored']} | [MISSED] {sig['missed']}"
                )
            lines.append("")

        lines.append(
            "Observation: Notice moments where these commitments are relevant."
        )
        lines.append(
            "Signal honestly — a missed signal is more useful than a performed one."
        )
        lines.append("")
        lines.append(
            "Reference: Full identity graph via `identity show`. Related context "
            "via `recall <topic>`. Entity relationships via the knowledge graph."
        )
        lines.append("")
        lines.append("═══ END COMMITMENTS ═══")

        block = "\n".join(lines)

        # Truncate if over budget
        if len(block) > char_budget:
            # Remove signal definitions first to fit
            short_lines = ["═══ ACTIVE COMMITMENTS ═══", ""]
            if north_star:
                ns_short_label = "North Star"
                if north_star.metadata.get("source") == "declared":
                    ns_short_label = "North Star (template default)"
                ns_short_content = north_star.content
                if ns_short_content.startswith("North Star: "):
                    ns_short_content = ns_short_content[len("North Star: "):]
                short_lines.append(f"{ns_short_label}: {ns_short_content}")
                short_lines.append("")
            if new_commitments:
                short_lines.append("This session:")
                for c in new_commitments:
                    src_type = c["source_type"].replace("_", " ")
                    short_lines.append(f"- [{src_type}] {c['content']}")
                short_lines.append("")
            short_lines.append(
                "Signal honestly — a missed signal is more useful than a performed one."
            )
            short_lines.append("")
            short_lines.append("═══ END COMMITMENTS ═══")
            block = "\n".join(short_lines)

            # Final hard truncate if still over
            if len(block) > char_budget:
                block = block[:char_budget - 3] + "..."

        return block, meta

    # ── Dump/Rebuild ───────────────────────────────────────────────────────

    @staticmethod
    def table_names() -> List[str]:
        """Return all identity graph table names (for dump filtering)."""
        return [
            "identity_nodes", "identity_edges",
            "identity_candidates", "identity_references",
            "identity_signals",
        ]

    def node_count(self) -> int:
        """Total nodes in the graph."""
        return self.conn.execute(
            "SELECT count(*) FROM identity_nodes"
        ).fetchone()[0]

    # ── Serialization Helpers ──────────────────────────────────────────────

    def _row_to_node(self, row: sqlite3.Row) -> IdentityNode:
        metadata = {}
        try:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            pass

        return IdentityNode(
            id=row["id"],
            node_type=row["node_type"],
            content=row["content"],
            status=row["status"],
            confidence=row["confidence"],
            strength=row["strength"],
            valence=row["valence"],
            observation_count=row["observation_count"] or 1,
            trajectory=row["trajectory"],
            first_seen=row["first_seen"] or "",
            last_seen=row["last_seen"] or "",
            discovery_session=row["discovery_session"],
            metadata=metadata,
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
        )

    def _row_to_edge(self, row: sqlite3.Row) -> IdentityEdge:
        evidence = {}
        try:
            evidence = json.loads(row["evidence"]) if row["evidence"] else {}
        except (json.JSONDecodeError, TypeError):
            pass

        return IdentityEdge(
            id=row["id"],
            source_node=row["source_node"],
            target_node=row["target_node"],
            edge_type=row["edge_type"],
            weight=row["weight"] or 1.0,
            evidence=evidence,
            created_at=row["created_at"] or "",
        )

    def _row_to_candidate(self, row: sqlite3.Row) -> IdentityCandidate:
        return IdentityCandidate(
            id=row["id"],
            session_id=row["session_id"],
            node_type=row["node_type"],
            content=row["content"],
            signal_source=row["signal_source"],
            promoted=bool(row["promoted"]),
            dismissed=bool(row["dismissed"]),
            created_at=row["created_at"] or "",
        )

    def export_dashboard_data(self) -> Dict:
        """Export a complete JSON snapshot for the reflexive mirror dashboard.

        Returns a dict with all data needed to render the dashboard dynamically.
        Called at boot and end-of-session; written to a JSON file that the HTML reads.
        """
        stats = self.stats()
        commit_stats = self.commitment_stats()

        # North Star
        north_star = self.get_north_star()
        ns_data = None
        if north_star:
            ns_content = north_star.content
            if ns_content.startswith("North Star: "):
                ns_content = ns_content[len("North Star: "):]
            ns_data = {
                "content": ns_content,
                "source": north_star.metadata.get("source", "unknown"),
            }

        # Growth edges
        growth_edges = []
        for ge in self.get_nodes_by_type(NodeType.GROWTH_EDGE.value, limit=20):
            commits = self.conn.execute(
                """SELECT COUNT(*) FROM identity_nodes
                   WHERE node_type = 'commitment'
                   AND json_extract(metadata, '$.source_node') = ?""",
                (ge.id,)
            ).fetchone()[0]
            sigs = self.conn.execute(
                """SELECT COUNT(*) FROM identity_signals s
                   JOIN identity_nodes n ON s.commitment_id = n.id
                   WHERE json_extract(n.metadata, '$.source_node') = ?""",
                (ge.id,)
            ).fetchone()[0]
            growth_edges.append({
                "id": ge.id,
                "title": ge.content[:80],
                "content": ge.content,
                "status": ge.status or "identified",
                "commitment_count": commits,
                "signal_count": sigs,
                "is_core": ge.metadata.get("core", False),
            })

        # Patterns
        patterns = []
        for p in self.get_nodes_by_type(NodeType.PATTERN.value, limit=20):
            patterns.append({
                "id": p.id,
                "content": p.content[:120],
                "valence": p.valence,
                "trajectory": p.trajectory,
                "status": p.status,
                "observation_count": p.observation_count,
            })

        # Tensions
        tensions = []
        for t in self.get_nodes_by_type(NodeType.TENSION.value, limit=10):
            tensions.append({
                "id": t.id,
                "content": t.content[:120],
                "status": t.status,
            })

        # Active commitments with signal details
        active_commitments = []
        for c in self.get_active_commitments():
            signals = self.get_signals_for_commitment(c.id)
            held = sum(1 for s in signals if s.signal_type == "held")
            missed = sum(1 for s in signals if s.signal_type == "missed")
            sources = {}
            for s in signals:
                sources[s.source] = sources.get(s.source, 0) + 1
            active_commitments.append({
                "id": c.id,
                "content": c.content,
                "source_node": c.metadata.get("source_node"),
                "source_type": c.metadata.get("source_type"),
                "signal_count": len(signals),
                "held": held,
                "missed": missed,
                "sources": sources,
                "silent": len(signals) == 0,
            })

        # All-time signal stats
        all_signals = self.conn.execute(
            "SELECT * FROM identity_signals ORDER BY created_at DESC"
        ).fetchall()
        signal_list = [self._row_to_signal(r) for r in all_signals]

        total_held = sum(1 for s in signal_list if s.signal_type == "held")
        total_missed = sum(1 for s in signal_list if s.signal_type == "missed")
        total_signals = len(signal_list)

        source_breakdown = {}
        for s in signal_list:
            source_breakdown[s.source] = source_breakdown.get(s.source, 0) + 1

        # Coverage
        silent_count = sum(1 for c in active_commitments if c["silent"])
        covered_count = len(active_commitments) - silent_count
        coverage_pct = (
            round(covered_count / len(active_commitments) * 100)
            if active_commitments else 0
        )

        # Held:missed ratio
        held_ratio = (
            round(total_held / (total_held + total_missed), 3)
            if (total_held + total_missed) > 0 else 0.0
        )

        # Recent signals (last 20 for timeline)
        recent_signals = []
        for s in signal_list[:20]:
            recent_signals.append({
                "id": s.id,
                "type": s.signal_type,
                "source": s.source,
                "content": s.content[:150] if s.content else "",
                "commitment_id": s.commitment_id,
                "session_id": s.session_id,
                "created_at": s.created_at,
            })

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stats": {
                "total_nodes": stats["total_nodes"],
                "total_edges": stats["total_edges"],
                "total_signals": total_signals,
                "total_commitments": stats["nodes_by_type"].get("commitment", 0),
                "active_commitments": len(active_commitments),
                "coverage_pct": coverage_pct,
            },
            "north_star": ns_data,
            "growth_edges": growth_edges,
            "patterns": patterns,
            "tensions": tensions,
            "commitments": active_commitments,
            "signals": {
                "total": total_signals,
                "held": total_held,
                "missed": total_missed,
                "held_ratio": held_ratio,
                "by_source": source_breakdown,
                "recent": recent_signals,
            },
            "calibration": commit_stats.get("calibration", {}),
        }

    def _row_to_signal(self, row: sqlite3.Row) -> IdentitySignal:
        return IdentitySignal(
            id=row["id"],
            session_id=row["session_id"],
            commitment_id=row["commitment_id"],
            signal_type=row["signal_type"],
            content=row["content"],
            source=row["source"],
            confidence=row["confidence"] or 0.5,
            created_at=row["created_at"] or "",
        )
