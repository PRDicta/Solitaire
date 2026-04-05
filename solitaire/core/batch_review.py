"""
Solitaire — Batch Review Engine

Scheduled background job where the active model reviews accumulated heuristic
output and applies semantic judgment to correct, upgrade, or validate it.
No API calls; the model running the scheduled task IS the judge.

Two-phase CLI pattern:
  Phase 1: `solitaire review run` gathers items and outputs structured JSON
  Phase 2: `solitaire review apply` accepts decisions and commits them

Five review categories rotate across runs:
  1. commitment_signals — retroactive scorer output accuracy
  2. identity_candidates — pending promotion decisions
  3. disposition_drift — low-confidence signal validation
  4. growth_edge_evolution — Tier C: growth edge lifecycle
  5. lifecycle_validation — thermostat recommendation review
"""
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Constants ────────────────────────────────────────────────────────────

BATCH_REVIEW_SOURCE = "batch_review"
BATCH_REVIEW_WEIGHT = 0.8

# Rotation: commitment_signals appears twice (highest volume)
CATEGORY_ROTATION = [
    "commitment_signals",
    "identity_candidates",
    "commitment_signals",
    "disposition_drift",
    "growth_edge_evolution",
    "lifecycle_validation",
]


# ── Schema ───────────────────────────────────────────────────────────────

REVIEW_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_log (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    started_at DATETIME NOT NULL,
    completed_at DATETIME,
    items_reviewed INTEGER DEFAULT 0,
    confirmed INTEGER DEFAULT 0,
    corrected INTEGER DEFAULT 0,
    upgraded INTEGER DEFAULT 0,
    dismissed INTEGER DEFAULT 0,
    deferred INTEGER DEFAULT 0,
    override_rate REAL DEFAULT 0.0,
    watermark TEXT,
    rotation_index INTEGER,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_review_log_completed
    ON review_log(completed_at DESC);

CREATE INDEX IF NOT EXISTS idx_review_log_category
    ON review_log(category, completed_at DESC);
"""


def ensure_review_schema(conn: sqlite3.Connection):
    """Create the review_log table if it doesn't exist."""
    conn.executescript(REVIEW_LOG_SCHEMA)


# ── Data Structures ──────────────────────────────────────────────────────

class ReviewCategory(str, Enum):
    COMMITMENT_SIGNALS = "commitment_signals"
    IDENTITY_CANDIDATES = "identity_candidates"
    DISPOSITION_DRIFT = "disposition_drift"
    GROWTH_EDGE_EVOLUTION = "growth_edge_evolution"
    LIFECYCLE_VALIDATION = "lifecycle_validation"


class ReviewVerdict(str, Enum):
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    UPGRADED = "upgraded"
    DISMISSED = "dismissed"
    DEFERRED = "deferred"


@dataclass
class ReviewItem:
    """An item gathered for model review."""
    item_id: str
    category: str
    content: str
    context: str
    heuristic_source: str
    heuristic_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "content": self.content,
            "context": self.context,
            "heuristic_source": self.heuristic_source,
            "heuristic_confidence": self.heuristic_confidence,
            "metadata": self.metadata,
        }


@dataclass
class ReviewDecision:
    """The model's judgment on a review item."""
    item_id: str
    category: str
    verdict: str
    reasoning: str
    new_confidence: Optional[float] = None
    corrections: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "item_id": self.item_id,
            "category": self.category,
            "verdict": self.verdict,
            "reasoning": self.reasoning,
        }
        if self.new_confidence is not None:
            d["new_confidence"] = self.new_confidence
        if self.corrections:
            d["corrections"] = self.corrections
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReviewDecision":
        return cls(
            item_id=d["item_id"],
            category=d.get("category", ""),
            verdict=d["verdict"],
            reasoning=d.get("reasoning", ""),
            new_confidence=d.get("new_confidence"),
            corrections=d.get("corrections", {}),
        )


@dataclass
class ReviewRunResult:
    """Aggregated outcome from a batch review run."""
    category: str = ""
    items_reviewed: int = 0
    confirmed: int = 0
    corrected: int = 0
    upgraded: int = 0
    dismissed: int = 0
    deferred: int = 0
    override_rate: float = 0.0
    actions: List[Dict[str, Any]] = field(default_factory=list)
    analyzed_at: str = ""

    @property
    def has_activity(self) -> bool:
        return self.items_reviewed > 0

    def compute_override_rate(self):
        denominator = self.confirmed + self.corrected + self.upgraded
        self.override_rate = self.corrected / denominator if denominator > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "items_reviewed": self.items_reviewed,
            "confirmed": self.confirmed,
            "corrected": self.corrected,
            "upgraded": self.upgraded,
            "dismissed": self.dismissed,
            "deferred": self.deferred,
            "override_rate": round(self.override_rate, 3),
            "actions": self.actions,
            "analyzed_at": self.analyzed_at,
        }

    def format_readable(self) -> str:
        lines = [f"Batch Review: {self.category}"]
        lines.append(f"  Items reviewed: {self.items_reviewed}")
        if self.confirmed:
            lines.append(f"  Confirmed: {self.confirmed}")
        if self.corrected:
            lines.append(f"  Corrected: {self.corrected}")
        if self.upgraded:
            lines.append(f"  Upgraded: {self.upgraded}")
        if self.dismissed:
            lines.append(f"  Dismissed: {self.dismissed}")
        if self.deferred:
            lines.append(f"  Deferred: {self.deferred}")
        lines.append(f"  Override rate: {self.override_rate:.0%}")
        return "\n".join(lines)


# ── Guiding Questions per Category ───────────────────────────────────────

GUIDING_QUESTIONS = {
    "commitment_signals": [
        "Does the signal content actually demonstrate honoring or missing this commitment?",
        "Is the heuristic's direction (held/missed) correct, or did keyword matching mislead?",
        "Would a different signal_type be more accurate given the full context?",
    ],
    "identity_candidates": [
        "Is this genuinely a realization/lesson/tension/preference, or casual usage of the keyword?",
        "Is the content substantive enough to become a permanent identity node?",
        "Does a similar node already exist that this would duplicate?",
    ],
    "disposition_drift": [
        "Did the detected signal actually occur in this exchange?",
        "Was the user's tone/intent correctly classified by the pattern matcher?",
        "Would you classify this differently after reading the full exchange?",
    ],
    "growth_edge_evolution": [
        "Is this growth edge being actively practiced in recent sessions?",
        "Does experiential data suggest the behavior is integrating naturally?",
        "Should the status advance (identified -> practicing -> improving -> integrated)?",
        "Should a new growth edge emerge from patterns seen in recent data?",
    ],
    "lifecycle_validation": [
        "Is the thermostat's recommendation (retire/escalate/sharpen) correct?",
        "Does the experiential data support this commitment being integrated (retire)?",
        "Is the commitment genuinely stuck, or were the recent misses situational?",
    ],
}


# ── Batch Review Engine ──────────────────────────────────────────────────

class BatchReviewEngine:
    """Two-phase review engine: gather items, then apply model decisions.

    Phase 1 (gather): reads heuristic output, assembles review items with
    context, outputs structured JSON for the model to judge.

    Phase 2 (apply): takes the model's decisions, writes corrections/upgrades
    back to the identity graph, logs the run.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        identity_graph=None,
        limit: int = 20,
        dry_run: bool = False,
    ):
        self.conn = conn
        self.ig = identity_graph
        self.limit = limit
        self.dry_run = dry_run
        ensure_review_schema(conn)

    # ── Phase 1: Gather ──────────────────────────────────────────────

    def gather(self, category: str) -> Dict[str, Any]:
        """Gather items for review in a specific category.

        Returns structured JSON with items and guiding questions.
        """
        gatherers = {
            "commitment_signals": self._gather_commitment_signals,
            "identity_candidates": self._gather_identity_candidates,
            "disposition_drift": self._gather_disposition_drift,
            "growth_edge_evolution": self._gather_growth_edges,
            "lifecycle_validation": self._gather_lifecycle_recommendations,
        }

        gatherer = gatherers.get(category)
        if not gatherer:
            return {"error": f"Unknown category: {category}"}

        items = gatherer()

        return {
            "category": category,
            "items": [item.to_dict() for item in items],
            "item_count": len(items),
            "guiding_questions": GUIDING_QUESTIONS.get(category, []),
            "verdicts": [v.value for v in ReviewVerdict],
        }

    def gather_auto(self) -> Dict[str, Any]:
        """Gather items for the next category in rotation."""
        category = self._get_next_category()
        return self.gather(category)

    # ── Phase 2: Apply ───────────────────────────────────────────────

    def apply_decisions(self, decisions: List[Dict]) -> ReviewRunResult:
        """Apply model decisions from Phase 2.

        Args:
            decisions: List of decision dicts with item_id, verdict, reasoning, etc.

        Returns:
            ReviewRunResult with counts and actions.
        """
        result = ReviewRunResult(
            analyzed_at=datetime.now(timezone.utc).isoformat(),
        )

        if not decisions:
            return result

        # Infer category from first decision
        first = decisions[0]
        result.category = first.get("category", "unknown")

        appliers = {
            "commitment_signals": self._apply_commitment_signal,
            "identity_candidates": self._apply_identity_candidate,
            "disposition_drift": self._apply_disposition_drift,
            "growth_edge_evolution": self._apply_growth_edge,
            "lifecycle_validation": self._apply_lifecycle,
        }

        applier = appliers.get(result.category)
        max_watermark = ""

        for d in decisions:
            decision = ReviewDecision.from_dict(d)
            result.items_reviewed += 1

            # Track verdict counts
            verdict = decision.verdict
            if verdict == ReviewVerdict.CONFIRMED:
                result.confirmed += 1
            elif verdict == ReviewVerdict.CORRECTED:
                result.corrected += 1
            elif verdict == ReviewVerdict.UPGRADED:
                result.upgraded += 1
            elif verdict == ReviewVerdict.DISMISSED:
                result.dismissed += 1
            elif verdict == ReviewVerdict.DEFERRED:
                result.deferred += 1

            # Apply the decision
            action = {"item_id": decision.item_id, "verdict": verdict, "reasoning": decision.reasoning}
            if applier and not self.dry_run:
                try:
                    apply_detail = applier(decision)
                    if apply_detail:
                        action.update(apply_detail)
                except Exception as e:
                    action["error"] = str(e)

            result.actions.append(action)

            # Track watermark from metadata
            wm = d.get("metadata", {}).get("created_at", "")
            if wm > max_watermark:
                max_watermark = wm

        result.compute_override_rate()

        # Write review log
        if not self.dry_run:
            self._write_review_log(result, max_watermark)

        return result

    # ── Category Gatherers ───────────────────────────────────────────

    def _gather_commitment_signals(self) -> List[ReviewItem]:
        """Gather enrichment_scanner signals for review."""
        watermark = self._get_watermark("commitment_signals")
        query = """
            SELECT s.id, s.session_id, s.commitment_id, s.signal_type,
                   s.content, s.confidence, s.created_at
            FROM identity_signals s
            WHERE s.source = 'enrichment_scanner'
        """
        params = []
        if watermark:
            query += " AND s.created_at > ?"
            params.append(watermark)
        query += " ORDER BY s.created_at ASC LIMIT ?"
        params.append(self.limit)

        items = []
        try:
            rows = self.conn.execute(query, params).fetchall()
        except Exception:
            return items

        for row in rows:
            sig_id, session_id, commitment_id, signal_type, content, confidence, created_at = row

            # Load commitment context
            commitment_text = ""
            if commitment_id:
                try:
                    crow = self.conn.execute(
                        "SELECT content FROM identity_nodes WHERE id = ?",
                        (commitment_id,),
                    ).fetchone()
                    if crow:
                        commitment_text = crow[0]
                except Exception:
                    pass

            items.append(ReviewItem(
                item_id=sig_id,
                category="commitment_signals",
                content=content,
                context=f"Commitment: {commitment_text}\nSignal type: {signal_type}\nSession: {session_id}",
                heuristic_source="enrichment_scanner",
                heuristic_confidence=confidence,
                metadata={"commitment_id": commitment_id, "signal_type": signal_type, "created_at": created_at},
            ))

        return items

    def _gather_identity_candidates(self) -> List[ReviewItem]:
        """Gather pending identity candidates for review."""
        query = """
            SELECT id, session_id, node_type, content, signal_source, created_at
            FROM identity_candidates
            WHERE promoted = 0 AND dismissed = 0
            ORDER BY created_at ASC
            LIMIT ?
        """
        items = []
        try:
            rows = self.conn.execute(query, (self.limit,)).fetchall()
        except Exception:
            return items

        for row in rows:
            cand_id, session_id, node_type, content, signal_source, created_at = row
            items.append(ReviewItem(
                item_id=cand_id,
                category="identity_candidates",
                content=content,
                context=f"Node type: {node_type}\nDetected by: {signal_source or 'unknown'}\nSession: {session_id}",
                heuristic_source=signal_source or "enrichment_scanner",
                heuristic_confidence=0.5,
                metadata={"node_type": node_type, "created_at": created_at},
            ))

        return items

    def _gather_disposition_drift(self) -> List[ReviewItem]:
        """Gather low-confidence disposition drift entries for review."""
        watermark = self._get_watermark("disposition_drift")
        query = """
            SELECT id, conversation_id, content, created_at
            FROM rolodex_entries
            WHERE category = 'disposition_drift'
        """
        params = []
        if watermark:
            query += " AND created_at > ?"
            params.append(watermark)
        query += " ORDER BY created_at ASC LIMIT ?"
        params.append(self.limit * 3)  # fetch extra, filter by confidence

        items = []
        try:
            rows = self.conn.execute(query, params).fetchall()
        except Exception:
            return items

        for row in rows:
            entry_id, conv_id, content_json, created_at = row
            try:
                data = json.loads(content_json) if content_json else {}
            except (json.JSONDecodeError, TypeError):
                continue

            confidence = data.get("confidence", 1.0)
            if confidence >= 0.7:
                continue  # only review low-confidence entries

            # Check if already reviewed
            if data.get("reviewed"):
                continue

            signal_key = data.get("signal", "unknown")
            traits = data.get("traits_affected", {})

            items.append(ReviewItem(
                item_id=entry_id,
                category="disposition_drift",
                content=f"Signal: {signal_key}, Traits: {json.dumps(traits)}",
                context=f"Confidence: {confidence}\nSession: {conv_id}",
                heuristic_source="disposition_filter",
                heuristic_confidence=confidence,
                metadata={"signal": signal_key, "created_at": created_at},
            ))

            if len(items) >= self.limit:
                break

        return items

    def _gather_growth_edges(self) -> List[ReviewItem]:
        """Gather active growth edges for evolution review."""
        query = """
            SELECT id, content, status, metadata, observation_count, last_seen
            FROM identity_nodes
            WHERE node_type = 'growth_edge'
              AND status IN ('identified', 'practicing', 'improving')
            ORDER BY last_seen ASC
            LIMIT ?
        """
        items = []
        try:
            rows = self.conn.execute(query, (self.limit,)).fetchall()
        except Exception:
            return items

        for row in rows:
            node_id, content, status, meta_json, obs_count, last_seen = row
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}

            items.append(ReviewItem(
                item_id=node_id,
                category="growth_edge_evolution",
                content=content,
                context=f"Status: {status}\nObservation count: {obs_count}\nLast seen: {last_seen}",
                heuristic_source="identity_graph",
                heuristic_confidence=0.5,
                metadata={"status": status, "observation_count": obs_count, "last_seen": last_seen},
            ))

        return items

    def _gather_lifecycle_recommendations(self) -> List[ReviewItem]:
        """Gather thermostat recommendations for validation."""
        items = []
        try:
            from .adaptation_engine import AdaptationEngine
            from .trend_analyzer import TrendAnalyzer

            analyzer = TrendAnalyzer(self.conn)
            trend_report = analyzer.analyze(max_sessions=30)

            engine = AdaptationEngine(self.conn)
            result = engine.evaluate(trend_report)

            for rec in result.commitment_recommendations:
                items.append(ReviewItem(
                    item_id=rec.commitment_id,
                    category="lifecycle_validation",
                    content=rec.commitment_content,
                    context=(
                        f"Recommendation: {rec.action}\n"
                        f"Confidence: {rec.confidence:.2f}\n"
                        f"Reasoning: {rec.reasoning}"
                    ),
                    heuristic_source="adaptation_engine",
                    heuristic_confidence=rec.confidence,
                    metadata={"action": rec.action, "evidence": rec.evidence},
                ))

                if len(items) >= self.limit:
                    break
        except Exception:
            pass

        return items

    # ── Category Appliers ────────────────────────────────────────────

    def _apply_commitment_signal(self, decision: ReviewDecision) -> Optional[Dict]:
        """Apply a commitment signal review decision.

        Creates a new signal with source='batch_review' rather than
        modifying the original. Non-destructive.
        """
        # Load the original signal
        try:
            row = self.conn.execute(
                "SELECT session_id, commitment_id, signal_type, content FROM identity_signals WHERE id = ?",
                (decision.item_id,),
            ).fetchone()
        except Exception:
            return None
        if not row:
            return None

        session_id, commitment_id, original_type, content = row
        new_type = original_type

        if decision.verdict == ReviewVerdict.CORRECTED:
            # Flip the signal type
            new_type = decision.corrections.get("signal_type", "")
            if not new_type:
                new_type = "missed" if original_type == "held" else "held"

        if decision.verdict in (ReviewVerdict.CONFIRMED, ReviewVerdict.UPGRADED, ReviewVerdict.CORRECTED):
            new_confidence = decision.new_confidence or BATCH_REVIEW_WEIGHT
            new_id = f"ids_{uuid.uuid4().hex[:12]}"
            self.conn.execute(
                """INSERT OR REPLACE INTO identity_signals
                   (id, session_id, commitment_id, signal_type, content, source, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (new_id, session_id, commitment_id, new_type,
                 f"{content} [reviewed: {decision.reasoning[:100]}]",
                 BATCH_REVIEW_SOURCE, new_confidence,
                 datetime.now(timezone.utc).isoformat()),
            )
            self.conn.commit()
            return {"new_signal_id": new_id, "signal_type": new_type, "original_type": original_type}

        return None

    def _apply_identity_candidate(self, decision: ReviewDecision) -> Optional[Dict]:
        """Apply an identity candidate review decision."""
        if decision.verdict in (ReviewVerdict.CONFIRMED, ReviewVerdict.UPGRADED):
            # Promote the candidate
            if self.ig:
                node_id = self.ig.promote_candidate(decision.item_id)
                if node_id:
                    # Enrich metadata with review reasoning
                    self.ig.update_node_metadata(node_id, {
                        "reviewed_by": BATCH_REVIEW_SOURCE,
                        "review_reasoning": decision.reasoning[:200],
                    })
                    return {"promoted_node_id": node_id}
            else:
                # Direct SQL fallback
                self.conn.execute(
                    "UPDATE identity_candidates SET promoted = 1 WHERE id = ?",
                    (decision.item_id,),
                )
                self.conn.commit()
                return {"promoted": True}

        elif decision.verdict in (ReviewVerdict.DISMISSED, ReviewVerdict.CORRECTED):
            if self.ig:
                self.ig.dismiss_candidate(decision.item_id)
            else:
                self.conn.execute(
                    "UPDATE identity_candidates SET dismissed = 1 WHERE id = ?",
                    (decision.item_id,),
                )
                self.conn.commit()
            return {"dismissed": True}

        return None

    def _apply_disposition_drift(self, decision: ReviewDecision) -> Optional[Dict]:
        """Apply a disposition drift review decision."""
        try:
            row = self.conn.execute(
                "SELECT content FROM rolodex_entries WHERE id = ?",
                (decision.item_id,),
            ).fetchone()
            if not row:
                return None

            data = json.loads(row[0]) if row[0] else {}

            if decision.verdict in (ReviewVerdict.CONFIRMED, ReviewVerdict.UPGRADED):
                data["reviewed"] = True
                data["review_confidence"] = decision.new_confidence or 0.85
                data["review_result"] = "confirmed"
            elif decision.verdict in (ReviewVerdict.CORRECTED, ReviewVerdict.DISMISSED):
                data["reviewed"] = True
                data["review_result"] = "incorrect"
                data["review_reasoning"] = decision.reasoning[:200]

            self.conn.execute(
                "UPDATE rolodex_entries SET content = ? WHERE id = ?",
                (json.dumps(data), decision.item_id),
            )
            self.conn.commit()
            return {"review_result": data.get("review_result", "unknown")}
        except Exception:
            return None

    def _apply_growth_edge(self, decision: ReviewDecision) -> Optional[Dict]:
        """Apply a growth edge evolution decision."""
        if decision.verdict == ReviewVerdict.CONFIRMED:
            # No change needed, edge stays at current status
            return {"status": "unchanged"}

        if decision.verdict == ReviewVerdict.UPGRADED:
            # Advance status
            new_status = decision.corrections.get("new_status", "")
            if new_status and self.ig:
                self.ig.update_node_status(decision.item_id, new_status)
                return {"new_status": new_status}
            elif new_status:
                self.conn.execute(
                    "UPDATE identity_nodes SET status = ?, updated_at = ? WHERE id = ?",
                    (new_status, datetime.now(timezone.utc).isoformat(), decision.item_id),
                )
                self.conn.commit()
                return {"new_status": new_status}

        if decision.verdict == ReviewVerdict.CORRECTED:
            # Status change or content refinement
            new_status = decision.corrections.get("new_status", "")
            if new_status:
                self.conn.execute(
                    "UPDATE identity_nodes SET status = ?, updated_at = ? WHERE id = ?",
                    (new_status, datetime.now(timezone.utc).isoformat(), decision.item_id),
                )
                self.conn.commit()
                return {"new_status": new_status}

        return None

    def _apply_lifecycle(self, decision: ReviewDecision) -> Optional[Dict]:
        """Apply a lifecycle validation decision."""
        if decision.verdict == ReviewVerdict.CONFIRMED:
            # Execute the thermostat's recommendation
            action = decision.corrections.get("action", "")
            if action == "retire" and self.ig:
                self.ig.update_node_status(decision.item_id, "retired_honored")
                return {"executed": "retire"}
            elif action == "escalate":
                return {"validated": "escalate", "note": "Requires manual intervention"}

        if decision.verdict == ReviewVerdict.CORRECTED:
            # Override the thermostat
            return {"overridden": True, "reasoning": decision.reasoning[:200]}

        return None

    # ── Checkpoint Management ────────────────────────────────────────

    def _get_watermark(self, category: str) -> Optional[str]:
        """Get the last processed timestamp for a category."""
        try:
            row = self.conn.execute(
                """SELECT watermark FROM review_log
                   WHERE category = ? AND completed_at IS NOT NULL
                   ORDER BY completed_at DESC LIMIT 1""",
                (category,),
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def _get_next_category(self) -> str:
        """Determine the next category in rotation."""
        try:
            row = self.conn.execute(
                """SELECT rotation_index FROM review_log
                   WHERE completed_at IS NOT NULL
                   ORDER BY completed_at DESC LIMIT 1""",
            ).fetchone()
            if row and row[0] is not None:
                next_idx = (row[0] + 1) % len(CATEGORY_ROTATION)
            else:
                next_idx = 0
            return CATEGORY_ROTATION[next_idx]
        except Exception:
            return CATEGORY_ROTATION[0]

    def _get_rotation_index(self, category: str) -> int:
        """Find the current rotation index for a category."""
        try:
            row = self.conn.execute(
                """SELECT rotation_index FROM review_log
                   WHERE completed_at IS NOT NULL
                   ORDER BY completed_at DESC LIMIT 1""",
            ).fetchone()
            if row and row[0] is not None:
                return (row[0] + 1) % len(CATEGORY_ROTATION)
        except Exception:
            pass
        # Find first occurrence of category in rotation
        try:
            return CATEGORY_ROTATION.index(category)
        except ValueError:
            return 0

    def _write_review_log(self, result: ReviewRunResult, watermark: str):
        """Write a review log entry."""
        try:
            log_id = f"rev_{uuid.uuid4().hex[:12]}"
            rotation_idx = self._get_rotation_index(result.category)
            self.conn.execute(
                """INSERT INTO review_log
                   (id, category, started_at, completed_at, items_reviewed,
                    confirmed, corrected, upgraded, dismissed, deferred,
                    override_rate, watermark, rotation_index, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (log_id, result.category, result.analyzed_at, result.analyzed_at,
                 result.items_reviewed, result.confirmed, result.corrected,
                 result.upgraded, result.dismissed, result.deferred,
                 result.override_rate, watermark or "", rotation_idx,
                 json.dumps({"actions_count": len(result.actions)})),
            )
            self.conn.commit()
        except Exception:
            pass

    # ── Status ───────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get review run history and override rates."""
        status = {"runs": [], "category_stats": {}}
        try:
            rows = self.conn.execute(
                """SELECT category, completed_at, items_reviewed,
                          confirmed, corrected, upgraded, dismissed, deferred,
                          override_rate
                   FROM review_log
                   WHERE completed_at IS NOT NULL
                   ORDER BY completed_at DESC LIMIT 20""",
            ).fetchall()

            for row in rows:
                status["runs"].append({
                    "category": row[0],
                    "completed_at": row[1],
                    "items_reviewed": row[2],
                    "confirmed": row[3],
                    "corrected": row[4],
                    "upgraded": row[5],
                    "dismissed": row[6],
                    "deferred": row[7],
                    "override_rate": row[8],
                })

            # Aggregate override rates per category
            cat_rows = self.conn.execute(
                """SELECT category,
                          SUM(items_reviewed) as total_reviewed,
                          SUM(corrected) as total_corrected,
                          SUM(confirmed + corrected + upgraded) as total_decisive
                   FROM review_log
                   WHERE completed_at IS NOT NULL
                   GROUP BY category""",
            ).fetchall()

            for row in cat_rows:
                cat, total, corrected, decisive = row
                status["category_stats"][cat] = {
                    "total_reviewed": total,
                    "total_corrected": corrected,
                    "aggregate_override_rate": round(corrected / decisive, 3) if decisive else 0.0,
                }
        except Exception:
            pass

        return status


# ── Pipeline Entry Points ────────────────────────────────────────────────

def run_review_gather(
    conn: sqlite3.Connection,
    category: str = "auto",
    identity_graph=None,
    limit: int = 20,
) -> Optional[Dict]:
    """Phase 1: gather items for review. Returns structured JSON or None."""
    try:
        engine = BatchReviewEngine(conn, identity_graph=identity_graph, limit=limit)
        if category == "auto":
            return engine.gather_auto()
        return engine.gather(category)
    except Exception:
        return None


def run_review_apply(
    conn: sqlite3.Connection,
    decisions: List[Dict],
    identity_graph=None,
    dry_run: bool = False,
) -> Optional[Dict]:
    """Phase 2: apply decisions. Returns result dict or None."""
    try:
        engine = BatchReviewEngine(conn, identity_graph=identity_graph, dry_run=dry_run)
        result = engine.apply_decisions(decisions)
        if result.has_activity:
            return result.to_dict()
        return None
    except Exception:
        return None


def get_review_status(conn: sqlite3.Connection) -> Optional[Dict]:
    """Get review history and override rates."""
    try:
        engine = BatchReviewEngine(conn)
        return engine.get_status()
    except Exception:
        return None
