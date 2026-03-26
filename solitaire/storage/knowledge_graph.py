"""
The Librarian — Knowledge Graph (SQLite-native)

Lightweight entity-relationship graph stored in the same SQLite DB as the
rolodex. No external graph database required.

Schema:
    entity_nodes: Named entities extracted from ingested content
    entity_edges: Relationships between entities with temporal validity

Design principles:
    - Single-file: lives alongside rolodex_entries in the same DB
    - Bi-temporal: tracks both event_time (when it happened) and
      ingestion_time (when we learned about it)
    - Incremental: entities and edges are added during ingestion,
      no batch recomputation needed
    - Portable: pure SQLite, no Neo4j/FalkorDB dependency
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field


# ─── Data Types ──────────────────────────────────────────────────────────────

@dataclass
class EntityNode:
    """A named entity in the knowledge graph."""
    id: str
    name: str                    # Canonical name (e.g., "The Librarian")
    entity_type: str             # person, project, tool, concept, org, file
    aliases: List[str]           # Alternative names (e.g., ["Librarian", "TL"])
    first_seen: str              # ISO timestamp
    last_seen: str               # ISO timestamp
    mention_count: int = 1
    metadata: Dict = field(default_factory=dict)


@dataclass
class EntityEdge:
    """A relationship between two entities."""
    id: str
    source_id: str               # Entity node ID
    target_id: str               # Entity node ID
    relationship: str            # e.g., "created_by", "depends_on", "part_of"
    weight: float = 1.0          # Strength (incremented on re-observation)
    event_time: Optional[str] = None      # When the relationship was established
    ingestion_time: Optional[str] = None  # When we learned about it
    invalidated_at: Optional[str] = None  # When superseded (temporal invalidation)
    source_entry_id: Optional[str] = None # Rolodex entry that established this
    metadata: Dict = field(default_factory=dict)


@dataclass
class GraphNeighbor:
    """A neighbor in a graph traversal result."""
    entity: EntityNode
    relationship: str
    direction: str  # "outgoing" or "incoming"
    weight: float


# ─── Schema ──────────────────────────────────────────────────────────────────

KNOWLEDGE_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    name_lower TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    aliases TEXT NOT NULL DEFAULT '[]',
    first_seen DATETIME NOT NULL,
    last_seen DATETIME NOT NULL,
    mention_count INTEGER DEFAULT 1,
    metadata TEXT DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_name_lower ON entity_nodes(name_lower);
CREATE INDEX IF NOT EXISTS idx_entity_type ON entity_nodes(entity_type);

CREATE TABLE IF NOT EXISTS entity_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relationship TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    event_time DATETIME,
    ingestion_time DATETIME NOT NULL,
    invalidated_at DATETIME,
    source_entry_id TEXT,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (source_id) REFERENCES entity_nodes(id),
    FOREIGN KEY (target_id) REFERENCES entity_nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_edge_source ON entity_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edge_target ON entity_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edge_relationship ON entity_edges(relationship);
CREATE INDEX IF NOT EXISTS idx_edge_valid ON entity_edges(invalidated_at);

-- Partial unique index: prevent duplicate active edges (same source, target, relationship)
-- where the edge has not been invalidated. Concurrent writers cannot create duplicates.
CREATE UNIQUE INDEX IF NOT EXISTS idx_edge_active_unique
    ON entity_edges(source_id, target_id, relationship)
    WHERE invalidated_at IS NULL;
"""


def ensure_knowledge_graph_schema(conn: sqlite3.Connection):
    """Create the knowledge graph tables if they don't exist."""
    conn.executescript(KNOWLEDGE_GRAPH_SCHEMA)
    conn.commit()


# ─── Entity Type Detection ───────────────────────────────────────────────────

# Known entity type mappings (bootstrap — grows via ingestion)
_TYPE_HINTS = {
    # People
    "owner": "person", "mycompany": "org",
    # Projects
    "librarian": "project", "token alchemy": "project",
    "example-project": "project", "example project": "project",
    # Tools/tech
    "sqlite": "tool", "claude": "tool", "cowork": "tool",
    "anthropic": "org", "neo4j": "tool", "graphiti": "tool",
    "huggingface": "tool", "github": "tool",
    "python": "tool", "fastapi": "tool", "pydantic": "tool",
}


def _infer_entity_type(name: str, context: str = "") -> str:
    """Infer entity type from name and context."""
    name_lower = name.lower()

    # Check known mappings
    for hint_name, hint_type in _TYPE_HINTS.items():
        if hint_name in name_lower:
            return hint_type

    # Heuristics
    if name_lower.endswith(('.py', '.js', '.ts', '.md', '.json', '.yaml', '.sql')):
        return "file"
    if '/' in name or '\\' in name:
        return "file"
    if '_' in name or name[0].islower():
        return "tool"  # snake_case = likely a tool/module
    if name[0].isupper() and len(name.split()) == 1:
        return "concept"  # Single capitalized word

    return "concept"


# ─── Relationship Extraction ─────────────────────────────────────────────────

import re

# Relationship patterns: (regex, relationship_type, direction)
# direction: "forward" means subject→object, "reverse" means object→subject
_RELATIONSHIP_PATTERNS = [
    # Creation/authorship
    (r'(\b\w+\b)\s+(?:created|built|wrote|developed|implemented|designed|shipped)\s+(.+?)(?:\.|,|$)', 'created', 'forward'),
    (r'(\b\w+\b)\s+(?:is|was)\s+(?:created|built|developed)\s+by\s+(.+?)(?:\.|,|$)', 'created_by', 'reverse'),

    # Dependencies
    (r'(\b\w+\b)\s+(?:uses|requires|depends\s+on|relies\s+on|needs)\s+(.+?)(?:\.|,|$)', 'depends_on', 'forward'),
    (r'(\b\w+\b)\s+(?:is\s+part\s+of|belongs\s+to|lives\s+in)\s+(.+?)(?:\.|,|$)', 'part_of', 'forward'),

    # Decisions/preferences
    (r'(\b\w+\b)\s+(?:decided|chose|picked|selected|prefers)\s+(.+?)(?:\.|,|$)', 'decided_on', 'forward'),

    # Ownership
    (r"(\b\w+\b)'s\s+(\b\w+(?:\s+\w+)?)\b", 'owns', 'forward'),
]


def extract_relationships(
    content: str,
    known_entities: Dict[str, str],  # name_lower → entity_id
) -> List[Tuple[str, str, str]]:
    """
    Extract entity relationships from content text.

    Returns list of (source_name, relationship, target_name) tuples.
    Heuristic-only, no LLM calls.
    """
    relationships = []

    # Strategy: find pairs of known entities that co-occur in the same
    # sentence, and infer a relationship based on proximity and patterns.
    sentences = re.split(r'[.!?]\s+|\n', content)

    for sentence in sentences:
        sentence_lower = sentence.lower()

        # Find all known entities in this sentence
        found_entities = []
        for name_lower, entity_id in known_entities.items():
            if name_lower in sentence_lower:
                # Get position for ordering
                pos = sentence_lower.index(name_lower)
                found_entities.append((name_lower, entity_id, pos))

        # Sort by position
        found_entities.sort(key=lambda x: x[2])

        # For each adjacent pair, try to determine relationship
        for i in range(len(found_entities)):
            for j in range(i + 1, min(i + 3, len(found_entities))):  # Look at next 2
                source_name = found_entities[i][0]
                target_name = found_entities[j][0]

                # Skip self-references
                if source_name == target_name:
                    continue

                # Check pattern matches between the two entities
                between_text = sentence_lower[
                    found_entities[i][2] + len(source_name):
                    found_entities[j][2]
                ].strip()

                rel = _infer_relationship(between_text, sentence_lower)
                if rel:
                    relationships.append((source_name, rel, target_name))
                elif len(found_entities) == 2:
                    # Only two entities in sentence — they're likely related
                    relationships.append((source_name, "related_to", target_name))

    return relationships


def _infer_relationship(between_text: str, full_sentence: str) -> Optional[str]:
    """Infer relationship type from text between two entities."""
    bt = between_text.strip().lower()

    # Direct verb patterns
    if any(w in bt for w in ['created', 'built', 'wrote', 'developed', 'implemented', 'shipped']):
        return 'created'
    if any(w in bt for w in ['uses', 'requires', 'depends on', 'relies on', 'needs']):
        return 'depends_on'
    if any(w in bt for w in ['is part of', 'belongs to', 'lives in', 'inside']):
        return 'part_of'
    if any(w in bt for w in ['decided', 'chose', 'selected', 'prefers']):
        return 'decided_on'
    if any(w in bt for w in ['manages', 'handles', 'controls', 'owns']):
        return 'manages'
    if any(w in bt for w in ['replaces', 'supersedes', 'replaced']):
        return 'replaces'
    if any(w in bt for w in ['conflicts with', 'contradicts']):
        return 'conflicts_with'
    if "'s" in bt:
        return 'associated_with'
    if any(w in bt for w in ['and', 'with', 'for', 'in', 'on', 'about']):
        return 'related_to'

    return None


# ─── Knowledge Graph Store ───────────────────────────────────────────────────

class KnowledgeGraph:
    """
    SQLite-native knowledge graph for entity-relationship storage.

    Usage:
        kg = KnowledgeGraph(conn)
        kg.ensure_schema()

        # During ingestion
        kg.process_content("The user built the system using SQLite", entry_id="abc123")

        # During recall
        neighbors = kg.get_neighbors("the librarian", depth=1)
        subgraph = kg.get_subgraph("the librarian", depth=2)
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._entity_cache: Dict[str, str] = {}  # name_lower → id
        self._schema_ensured = False

    def ensure_schema(self):
        """Create tables if needed."""
        if not self._schema_ensured:
            ensure_knowledge_graph_schema(self.conn)
            self._load_entity_cache()
            self._schema_ensured = True

    def _load_entity_cache(self):
        """Load existing entities into memory for fast lookup."""
        try:
            rows = self.conn.execute(
                "SELECT id, name_lower FROM entity_nodes"
            ).fetchall()
            for row in rows:
                name = row["name_lower"] if isinstance(row, sqlite3.Row) else row[1]
                eid = row["id"] if isinstance(row, sqlite3.Row) else row[0]
                self._entity_cache[name] = eid
        except Exception:
            pass

    # ─── Entity Operations ───────────────────────────────────────────────

    def upsert_entity(
        self,
        name: str,
        entity_type: str = "concept",
        aliases: Optional[List[str]] = None,
    ) -> str:
        """Create or update an entity node. Returns the entity ID."""
        self.ensure_schema()
        name_lower = name.lower().strip()

        # Check cache first
        if name_lower in self._entity_cache:
            entity_id = self._entity_cache[name_lower]
            # Update mention count and last_seen
            now = datetime.now(timezone.utc).isoformat()
            self.conn.execute(
                """UPDATE entity_nodes
                   SET mention_count = mention_count + 1,
                       last_seen = ?
                   WHERE id = ?""",
                (now, entity_id)
            )
            return entity_id

        # Check aliases
        if aliases:
            for alias in aliases:
                alias_lower = alias.lower().strip()
                if alias_lower in self._entity_cache:
                    entity_id = self._entity_cache[alias_lower]
                    self.conn.execute(
                        """UPDATE entity_nodes
                           SET mention_count = mention_count + 1,
                               last_seen = ?
                           WHERE id = ?""",
                        (datetime.now(timezone.utc).isoformat(), entity_id)
                    )
                    # Add new name as alias
                    self._entity_cache[name_lower] = entity_id
                    return entity_id

        # Create new entity (UPSERT to handle concurrent writers safely)
        entity_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO entity_nodes
               (id, name, name_lower, entity_type, aliases, first_seen, last_seen, mention_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1)
               ON CONFLICT(name_lower) DO UPDATE SET
                   mention_count = mention_count + 1,
                   last_seen = excluded.last_seen""",
            (entity_id, name, name_lower, entity_type,
             json.dumps(aliases or []), now, now)
        )
        # If conflict occurred, fetch the existing ID
        row = self.conn.execute(
            "SELECT id FROM entity_nodes WHERE name_lower = ?", (name_lower,)
        ).fetchone()
        if row:
            entity_id = row[0]
        self._entity_cache[name_lower] = entity_id
        if aliases:
            for alias in aliases:
                self._entity_cache[alias.lower().strip()] = entity_id
        return entity_id

    def get_entity(self, name: str) -> Optional[EntityNode]:
        """Look up an entity by name or alias."""
        self.ensure_schema()
        name_lower = name.lower().strip()

        entity_id = self._entity_cache.get(name_lower)
        if not entity_id:
            return None

        row = self.conn.execute(
            "SELECT * FROM entity_nodes WHERE id = ?", (entity_id,)
        ).fetchone()
        if not row:
            return None

        return EntityNode(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            aliases=json.loads(row["aliases"]) if row["aliases"] else [],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            mention_count=row["mention_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    # ─── Edge Operations ─────────────────────────────────────────────────

    def add_edge(
        self,
        source_name: str,
        target_name: str,
        relationship: str,
        source_entry_id: Optional[str] = None,
        event_time: Optional[str] = None,
    ) -> Optional[str]:
        """Add or strengthen a relationship edge. Returns edge ID."""
        self.ensure_schema()

        source_lower = source_name.lower().strip()
        target_lower = target_name.lower().strip()

        source_id = self._entity_cache.get(source_lower)
        target_id = self._entity_cache.get(target_lower)

        if not source_id or not target_id:
            return None

        now = datetime.now(timezone.utc).isoformat()

        # Upsert edge: the partial unique index idx_edge_active_unique
        # prevents duplicate active edges. ON CONFLICT strengthens the
        # existing edge instead of creating a duplicate.
        edge_id = str(uuid.uuid4())[:8]
        result = self.conn.execute(
            """INSERT INTO entity_edges
               (id, source_id, target_id, relationship, weight,
                event_time, ingestion_time, source_entry_id)
               VALUES (?, ?, ?, ?, 1.0, ?, ?, ?)
               ON CONFLICT(source_id, target_id, relationship)
                   WHERE invalidated_at IS NULL
               DO UPDATE SET
                   weight = entity_edges.weight + 1.0,
                   ingestion_time = excluded.ingestion_time""",
            (edge_id, source_id, target_id, relationship,
             event_time or now, now, source_entry_id)
        )
        # If conflict occurred, return the existing edge's ID
        if result.rowcount == 0 or result.lastrowid is None:
            existing = self.conn.execute(
                """SELECT id FROM entity_edges
                   WHERE source_id = ? AND target_id = ? AND relationship = ?
                   AND invalidated_at IS NULL""",
                (source_id, target_id, relationship)
            ).fetchone()
            if existing:
                return existing[0]
        return edge_id

    def invalidate_edge(self, edge_id: str):
        """Temporally invalidate an edge (mark as no longer current)."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE entity_edges SET invalidated_at = ? WHERE id = ?",
            (now, edge_id)
        )

    # ─── Graph Traversal ─────────────────────────────────────────────────

    def get_neighbors(
        self,
        entity_name: str,
        depth: int = 1,
        valid_only: bool = True,
    ) -> List[GraphNeighbor]:
        """Get neighboring entities connected to the given entity."""
        self.ensure_schema()
        entity = self.get_entity(entity_name)
        if not entity:
            return []

        valid_clause = "AND e.invalidated_at IS NULL" if valid_only else ""

        neighbors = []

        # Outgoing edges (this entity → others)
        rows = self.conn.execute(
            f"""SELECT e.relationship, e.weight, n.id, n.name, n.entity_type,
                       n.mention_count, n.aliases, n.first_seen, n.last_seen
                FROM entity_edges e
                JOIN entity_nodes n ON e.target_id = n.id
                WHERE e.source_id = ? {valid_clause}
                ORDER BY e.weight DESC""",
            (entity.id,)
        ).fetchall()

        for row in rows:
            neighbors.append(GraphNeighbor(
                entity=EntityNode(
                    id=row[2], name=row[3], entity_type=row[4],
                    aliases=json.loads(row[6]) if row[6] else [],
                    first_seen=row[7], last_seen=row[8],
                    mention_count=row[5],
                ),
                relationship=row[0],
                direction="outgoing",
                weight=row[1],
            ))

        # Incoming edges (others → this entity)
        rows = self.conn.execute(
            f"""SELECT e.relationship, e.weight, n.id, n.name, n.entity_type,
                       n.mention_count, n.aliases, n.first_seen, n.last_seen
                FROM entity_edges e
                JOIN entity_nodes n ON e.source_id = n.id
                WHERE e.target_id = ? {valid_clause}
                ORDER BY e.weight DESC""",
            (entity.id,)
        ).fetchall()

        for row in rows:
            neighbors.append(GraphNeighbor(
                entity=EntityNode(
                    id=row[2], name=row[3], entity_type=row[4],
                    aliases=json.loads(row[6]) if row[6] else [],
                    first_seen=row[7], last_seen=row[8],
                    mention_count=row[5],
                ),
                relationship=row[0],
                direction="incoming",
                weight=row[1],
            ))

        return neighbors

    def get_subgraph(
        self,
        entity_name: str,
        depth: int = 2,
        max_nodes: int = 20,
    ) -> Dict:
        """Get a subgraph centered on an entity, up to N hops deep.

        Returns a dict with 'nodes' and 'edges' suitable for serialization.
        """
        self.ensure_schema()
        entity = self.get_entity(entity_name)
        if not entity:
            return {"nodes": [], "edges": []}

        visited_ids: Set[str] = {entity.id}
        frontier = [entity.id]
        all_nodes = [entity]
        all_edges = []

        for hop in range(depth):
            next_frontier = []
            for node_id in frontier:
                if len(all_nodes) >= max_nodes:
                    break

                # Get edges from this node
                rows = self.conn.execute(
                    """SELECT e.id, e.source_id, e.target_id, e.relationship, e.weight,
                              n_src.name as src_name, n_tgt.name as tgt_name
                       FROM entity_edges e
                       JOIN entity_nodes n_src ON e.source_id = n_src.id
                       JOIN entity_nodes n_tgt ON e.target_id = n_tgt.id
                       WHERE (e.source_id = ? OR e.target_id = ?)
                       AND e.invalidated_at IS NULL
                       ORDER BY e.weight DESC
                       LIMIT 10""",
                    (node_id, node_id)
                ).fetchall()

                for row in rows:
                    edge_info = {
                        "id": row[0],
                        "source": row[5],
                        "target": row[6],
                        "relationship": row[3],
                        "weight": row[4],
                    }
                    all_edges.append(edge_info)

                    # Add unvisited neighbor
                    neighbor_id = row[2] if row[1] == node_id else row[1]
                    if neighbor_id not in visited_ids and len(all_nodes) < max_nodes:
                        visited_ids.add(neighbor_id)
                        next_frontier.append(neighbor_id)
                        # Fetch neighbor node
                        n_row = self.conn.execute(
                            "SELECT * FROM entity_nodes WHERE id = ?",
                            (neighbor_id,)
                        ).fetchone()
                        if n_row:
                            all_nodes.append(EntityNode(
                                id=n_row["id"], name=n_row["name"],
                                entity_type=n_row["entity_type"],
                                aliases=json.loads(n_row["aliases"]) if n_row["aliases"] else [],
                                first_seen=n_row["first_seen"],
                                last_seen=n_row["last_seen"],
                                mention_count=n_row["mention_count"],
                            ))

            frontier = next_frontier

        # Deduplicate edges
        seen_edges = set()
        unique_edges = []
        for e in all_edges:
            key = (e["source"], e["target"], e["relationship"])
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)

        return {
            "center": entity.name,
            "nodes": [
                {"name": n.name, "type": n.entity_type, "mentions": n.mention_count}
                for n in all_nodes
            ],
            "edges": unique_edges,
        }

    # ─── Content Processing (called during ingestion) ────────────────────

    def process_content(
        self,
        content: str,
        entry_id: Optional[str] = None,
        event_time: Optional[str] = None,
    ) -> Dict:
        """
        Extract entities and relationships from content and add to the graph.

        Called during ingestion. Returns stats about what was extracted.

        Args:
            content: The text to process
            entry_id: The rolodex entry ID (for edge provenance)
            event_time: When the content describes (for bi-temporal tracking)
        """
        self.ensure_schema()

        from .entity_extractor_kg import extract_entities_for_graph

        # Extract entities from content
        extracted = extract_entities_for_graph(content)

        entities_added = 0
        edges_added = 0

        # Upsert all extracted entities
        for name, etype in extracted["entities"]:
            self.upsert_entity(name, entity_type=etype)
            entities_added += 1

        # Extract and add relationships
        relationships = extract_relationships(content, self._entity_cache)
        for source, rel, target in relationships:
            edge_id = self.add_edge(
                source_name=source,
                target_name=target,
                relationship=rel,
                source_entry_id=entry_id,
                event_time=event_time,
            )
            if edge_id:
                edges_added += 1

        if entities_added > 0 or edges_added > 0:
            self.conn.commit()

        return {
            "entities_processed": entities_added,
            "edges_added": edges_added,
            "entity_cache_size": len(self._entity_cache),
        }

    # ─── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return graph statistics."""
        self.ensure_schema()
        nodes = self.conn.execute("SELECT COUNT(*) FROM entity_nodes").fetchone()[0]
        edges = self.conn.execute(
            "SELECT COUNT(*) FROM entity_edges WHERE invalidated_at IS NULL"
        ).fetchone()[0]
        invalid_edges = self.conn.execute(
            "SELECT COUNT(*) FROM entity_edges WHERE invalidated_at IS NOT NULL"
        ).fetchone()[0]

        # Top entities by mention
        top = self.conn.execute(
            "SELECT name, entity_type, mention_count FROM entity_nodes ORDER BY mention_count DESC LIMIT 10"
        ).fetchall()

        return {
            "total_nodes": nodes,
            "active_edges": edges,
            "invalidated_edges": invalid_edges,
            "top_entities": [
                {"name": r[0], "type": r[1], "mentions": r[2]}
                for r in top
            ],
        }
