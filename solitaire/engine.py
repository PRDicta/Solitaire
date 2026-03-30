"""
Solitaire Engine — Public API

The single entry point for all Solitaire operations. Every distribution
channel (CLI, agentskills.io skill, Dify plugin, direct Python import)
goes through this class.

SolitaireEngine is model-agnostic. It returns structured data (dicts,
strings). The host agent decides how to inject that data into whatever
model's prompt format it uses.

Usage:
    from solitaire import SolitaireEngine

    engine = SolitaireEngine(workspace_dir="/path/to/data")
    result = engine.boot(persona_key="default", intent="working on financials")
    engine.ingest(user_msg="What's our Q1 revenue?", assistant_msg="Based on...")
    context = engine.recall(query="pricing history")
    engine.remember(fact="Client X prefers email")
    engine.end(summary="Reviewed Q1 pricing")
"""
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .__version__ import __version__


class SolitaireEngine:
    """
    Persistent memory and evolving identity for AI agents.

    Wraps the core Librarian engine with a clean, high-level API.
    All methods are synchronous (async internals are handled transparently).
    All return values are plain dicts or strings -- no model-specific formatting.
    """

    def __init__(
        self,
        workspace_dir: str,
        persona_dir: Optional[str] = None,
        db_name: str = "rolodex.db",
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Solitaire engine.

        Args:
            workspace_dir: Directory for all persistent data (DB, personas, sessions).
                          This is the user's workspace root.
            persona_dir: Optional path to persona definitions. Defaults to
                        workspace_dir/personas/.
            db_name: Name of the SQLite database file. Defaults to "rolodex.db".
            config_overrides: Optional dict of config values to override defaults.
        """
        self.workspace_dir = Path(workspace_dir).resolve()
        self.persona_dir = Path(persona_dir) if persona_dir else self.workspace_dir / "personas"
        self.db_path = self.workspace_dir / db_name
        self._config_overrides = config_overrides or {}

        # Session state file (tracks active session, persona, turn count)
        self._session_file = self.workspace_dir / ".solitaire_session"

        # Internal state -- initialized on boot
        self._lib = None  # TheLibrarian instance
        self._persona_key = None
        self._session_id = None
        self._session_data = {}
        self._mode = "verbatim"  # "verbatim" (no API key) or "enhanced" (with LLM)
        self._booted = False

    # ─── Guards ─────────────────────────────────────────────────────────────

    def _ensure_booted(self, operation: str) -> None:
        """Raise if the engine hasn't been booted yet."""
        if not self._booted or self._lib is None:
            raise RuntimeError(
                f"Engine not booted. Call boot() before {operation}()."
            )

    # ─── Async bridge ──────────────────────────────────────────────────────

    def _run_async(self, coro):
        """Run an async coroutine synchronously.

        Handles two cases:
        - No running event loop (CLI, scripts): uses asyncio.run()
        - Inside a running loop (embedded in async app): runs in a thread
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    # ─── Boot ─────────────────────────────────────────────────────────────

    def boot_pre_persona(self) -> Dict[str, Any]:
        """
        Stage 1: Return available personas and user profile.
        Call this before persona selection. No Librarian instance needed.

        Returns:
            Dict with: status, available_personas, template_creation_enabled
        """
        personas = self._scan_personas()
        return {
            "status": "ok",
            "version": __version__,
            "boot_type": "pre_persona",
            "available_personas": personas,
            "template_creation_enabled": True,
        }

    def boot(
        self,
        persona_key: str = "default",
        intent: str = "",
        resume: bool = False,
        cold: bool = False,
    ) -> Dict[str, Any]:
        """
        Boot the engine with a specific persona.

        Args:
            persona_key: Which persona to load. Maps to a directory under persona_dir.
            intent: Optional context about what the user is working on.
                   Pre-loads relevant memories into the boot context.
            resume: If True, reads the last active persona from session state
                   (for post-compaction continuation).
            cold: If True, skip experiential memory and residue (testing mode).

        Returns:
            Dict with: status, session_id, persona info, boot_files paths,
            total_entries, first_turn_briefed flag.
        """
        old_session_id = None
        if resume:
            persona_key = self._load_session_persona() or persona_key
            old_session_id = self._load_session_id()

        self._persona_key = persona_key

        # Initialize TheLibrarian
        self._lib = self._make_librarian(persona_key=persona_key)

        # If resuming, try to restore the prior session's state
        if old_session_id:
            orphan_id = self._lib.session_id  # Auto-generated during init
            session_info = self._lib.resume_session(old_session_id)
            if session_info:
                self._session_id = old_session_id
                # Rebind JSONL audit trail to the resumed session
                if self._lib.rolodex._jsonl_session_id:
                    self._lib.rolodex._jsonl_session_id = old_session_id
                if (hasattr(self._lib, 'topic_router')
                        and self._lib.topic_router._jsonl_session_id):
                    self._lib.topic_router._jsonl_session_id = old_session_id
                # Reset pressure monitor so stale orphan-session
                # pressure history doesn't bleed into the resumed session
                if hasattr(self._lib, 'librarian_agent'):
                    from .preloading.pressure import PressureMonitor
                    self._lib.librarian_agent.pressure_monitor = PressureMonitor()
                # Clean up the orphan session row created during init
                if orphan_id != old_session_id:
                    try:
                        self._lib.rolodex.conn.execute(
                            "DELETE FROM conversations WHERE id = ?",
                            (orphan_id,),
                        )
                        self._lib.rolodex.conn.commit()
                    except Exception:
                        pass
            else:
                # Prior session not found in DB; fall back to fresh session
                self._session_id = self._lib.session_id
        else:
            self._session_id = self._lib.session_id

        self._mode = "enhanced" if self._lib._llm_adapter else "verbatim"

        # Save session state
        self._save_session_state(persona_key)

        # Check if backup is needed (non-blocking, non-fatal)
        backup_result = None
        try:
            from .storage.backup import BackupManager
            from .utils.config import LibrarianConfig
            _cfg = LibrarianConfig()
            for k, v in self._config_overrides.items():
                if hasattr(_cfg, k):
                    setattr(_cfg, k, v)
            bm = BackupManager.from_config(self.workspace_dir, _cfg)
            backup_result = bm.check_and_backup()
        except Exception:
            pass

        # Check for updates (non-blocking, non-fatal)
        update_info = None
        try:
            from .core.update_checker import check_for_updates
            update_info = check_for_updates(self.workspace_dir, __version__)
        except Exception:
            pass

        # Get stats
        stats = self._lib.get_stats()

        # Build boot context blocks
        context_blocks = self._build_boot_context(
            intent=intent, cold=cold, persona_key=persona_key
        )

        # Write boot files to workspace
        boot_files = self._write_boot_files(context_blocks)

        # Build persona info
        persona_info = self._get_persona_info()

        result = {
            "status": "ok",
            "version": __version__,
            "mode": self._mode,
            "boot_type": "persona",
            "cold_boot": cold,
            "session_id": self._session_id,
            "resumed": resume,
            "total_entries": stats.get("total_entries", 0),
            "persona": persona_info,
            "first_turn_briefed": bool(context_blocks.get("briefing")),
            "active_persona": {
                "key": persona_key,
                "display_name": persona_info.get("identity", {}).get("name", persona_key),
                "short_label": persona_info.get("identity", {}).get("name", persona_key),
            },
            "todo_pin": {
                "content": f"Active Persona: {persona_info.get('identity', {}).get('name', persona_key)}",
                "status": "in_progress",
                "activeForm": f"Running as {persona_info.get('identity', {}).get('name', persona_key)}",
            },
            "boot_files": boot_files,
            "backup": backup_result,
            "update": update_info,
        }

        self._booted = True
        return result

    # ─── Mark Response ─────────────────────────────────────────────────────

    def mark_response(self, response_text: str) -> Dict[str, Any]:
        """Store the assistant's response for deferred ingestion.

        Called after the LLM generates its response. Writes the response text
        to pending_ingest.assistant in session state. The next recall() call
        will find the complete turn pair and ingest it as Step 0.

        Args:
            response_text: The assistant's full response text.

        Returns:
            Dict with: status, whether pair is ready for ingestion.
        """
        if not response_text or not response_text.strip():
            return {"status": "skip", "reason": "empty_response"}

        try:
            # Read current session data from disk
            try:
                with open(self._session_file, "r") as f:
                    session_data = json.load(f)
            except Exception:
                session_data = dict(self._session_data)

            pending = session_data.get("pending_ingest", {})
            pending["assistant"] = response_text
            session_data["pending_ingest"] = pending

            with open(self._session_file, "w") as f:
                json.dump(session_data, f)

            return {
                "status": "ok",
                "pending_user": bool(pending.get("user")),
                "pending_assistant": True,
                "pair_ready": bool(pending.get("user")),
            }
        except Exception as e:
            return {"error": f"mark_response failed: {e}"}

    def diarize(self, response_text: str, residue_text: str) -> Dict[str, Any]:
        """Combined closing-anchor procedure: mark response + write residue.

        This is the format-anchor enforcement mechanism. The closing anchor `-`
        cannot be written without diarize having fired. Combines mark_response
        (stores assistant response for deferred ingestion) and write_residue
        (session texture) into one atomic call.

        Args:
            response_text: The assistant's full response text.
            residue_text: Session texture paragraph.

        Returns:
            Dict with combined status from both operations.
        """
        result = {"status": "ok"}

        # Step 1: Mark response (stores for deferred ingestion)
        if response_text and response_text.strip():
            mr_result = self.mark_response(response_text)
            result["mark_response"] = mr_result
            if "error" in mr_result:
                result["status"] = "partial"

        # Step 2: Write residue
        if residue_text and residue_text.strip():
            res_result = self.write_residue(residue_text)
            result["residue"] = res_result
            if res_result.get("status") == "error":
                result["status"] = "partial"

        return result

    # ─── Ingest ───────────────────────────────────────────────────────────

    def ingest(
        self,
        user_msg: str,
        assistant_msg: str,
    ) -> Dict[str, Any]:
        """
        Ingest a user + assistant turn pair.

        This is the primary ingestion method. Call after each conversation turn.
        Runs the full enrichment pipeline: extraction, embedding, knowledge graph,
        temporal reasoning, identity enrichment, active summarization.

        Args:
            user_msg: The user's message text.
            assistant_msg: The assistant's response text.

        Returns:
            Dict with: ingested count, entry IDs, enrichment stats.
        """
        self._ensure_booted("ingest")

        result = {"user": {}, "assistant": {}, "enrichment": {}}

        # Ingest user message
        user_entries = self._run_async(
            self._lib.ingest("user", user_msg)
        )
        result["user"] = {
            "ingested": len(user_entries),
            "entry_ids": [e.id for e in user_entries],
        }

        # Ingest assistant message
        asst_entries = self._run_async(
            self._lib.ingest("assistant", assistant_msg)
        )
        result["assistant"] = {
            "ingested": len(asst_entries),
            "entry_ids": [e.id for e in asst_entries],
        }

        # Run enrichment pipeline on both sets
        all_entries = user_entries + asst_entries
        enrichment = self._run_enrichment(
            user_msg, assistant_msg, user_entries, asst_entries
        )
        result["enrichment"] = enrichment

        # Evaluate retrieval usage: compare recalled entries against assistant response
        try:
            pending = getattr(self, '_pending_recall_ids', [])
            if pending:
                from .core.retrieval_feedback import evaluate_usage
                usage_result = evaluate_usage(
                    conn=self._lib.rolodex.conn,
                    session_id=self._session_id,
                    assistant_response=assistant_msg,
                    recalled_entry_ids=pending,
                )
                result["retrieval_usage"] = usage_result
                self._pending_recall_ids = []  # Reset after evaluation
        except Exception:
            pass  # Non-fatal

        # Write session tail (rolling window of last N turns)
        try:
            from .core.session_residue import write_session_tail
            persona_dir_str = str(self.persona_dir / self._persona_key) if self._persona_key else None
            write_session_tail(
                conn=self._lib.rolodex.conn,
                session_id=self._session_id,
                persona_key=self._persona_key or "",
                persona_dir=persona_dir_str,
            )
        except Exception:
            pass  # Non-fatal

        return result

    # ─── Recall ───────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        include_preflight: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from memory.

        Runs the full recall pipeline: preflight evaluation, then the
        RecallOrchestrator (trigger analysis, graph expansion, query expansion,
        variant firing, reranking, tiered confidence waterfall).

        Args:
            query: The user's current message or a specific search query.
            include_preflight: Whether to run the evaluation gate before recall.
                             Defaults to True. Set False for raw search.

        Returns:
            Dict with: context_block (string for prompt injection),
            entries found, preflight results, recall_tier, diagnostics.
        """
        self._ensure_booted("recall")

        result = {
            "context_block": "",
            "entries_found": 0,
            "preflight": {},
        }

        # Preflight evaluation gate
        if include_preflight:
            preflight = self._run_preflight(query)
            result["preflight"] = preflight
            # Honor the gate: if preflight says suppress recall, return early
            if preflight.get("proceed") is False:
                result["suppressed_by_preflight"] = True
                return result

        # Full recall pipeline via orchestrator
        from .retrieval.recall_orchestrator import RecallOrchestrator
        from .core.types import LibrarianResponse, LibrarianQuery

        # Load identity graph for identity-aware recall (optional, non-fatal)
        _identity_graph = None
        try:
            from .storage.identity_graph import IdentityGraph
            _identity_graph = IdentityGraph(self._lib.rolodex.conn)
        except Exception:
            pass

        orchestrator = RecallOrchestrator(
            conn=self._lib.rolodex.conn,
            rolodex=self._lib.rolodex,
            topic_router=getattr(self._lib, 'topic_router', None),
            identity_graph=_identity_graph,
        )
        recall_result = orchestrator.run(query)

        if recall_result.entries:
            # Build context block from orchestrator results
            synthetic_response = LibrarianResponse(
                found=True,
                entries=recall_result.entries,
                query=LibrarianQuery(query_text=query),
            )
            result["context_block"] = self._lib.get_context_block(synthetic_response)
            result["entries_found"] = len(recall_result.entries)
            recalled_ids = [e.id for e in recall_result.entries]
            result["recalled_entry_ids"] = recalled_ids
            result["recall_tier"] = recall_result.recall_tier
            result["candidates_total"] = recall_result.candidates_total
            result["queries_fired"] = recall_result.queries_fired
            result["signals"] = recall_result.signals

            # Include tier confidence diagnostics when widening occurred
            if recall_result.tier1_confidence:
                result["tier1_confidence"] = recall_result.tier1_confidence
            if recall_result.tier2_confidence:
                result["tier2_confidence"] = recall_result.tier2_confidence

            # Record retrieval outcomes for feedback tracking
            try:
                from .core.retrieval_feedback import record_recall_outcomes
                record_recall_outcomes(
                    conn=self._lib.rolodex.conn,
                    session_id=self._session_id,
                    query_text=query,
                    entry_ids=recalled_ids,
                )
            except Exception:
                pass  # Non-fatal: feedback tracking is supplementary

            # Track recalled IDs for usage evaluation at ingest time
            if not hasattr(self, '_pending_recall_ids'):
                self._pending_recall_ids = []
            self._pending_recall_ids.extend(recalled_ids)

        elif recall_result.dominant_intent == "temporal":
            # Temporal queries: fall back to basic retrieve for session digests
            response = self._run_async(
                self._lib.retrieve(query)
            )
            if response and response.entries:
                result["context_block"] = self._lib.get_context_block(response)
                result["entries_found"] = len(response.entries)
                result["recalled_entry_ids"] = [e.id for e in response.entries]

        # Combine preflight block with recall context
        preflight_block = result["preflight"].get("context_block", "")
        if preflight_block:
            result["context_block"] = preflight_block + "\n\n" + result["context_block"]

        return result

    # ─── Remember ─────────────────────────────────────────────────────────

    def remember(self, fact: str) -> Dict[str, Any]:
        """
        Store a fact as user_knowledge -- privileged, always-on context.

        user_knowledge entries are:
        - Always loaded at boot
        - Boosted 3x in search results
        - Never demoted from hot tier
        - Ideal for: preferences, biographical details, corrections, working style

        Args:
            fact: The fact to remember.

        Returns:
            Dict with: remembered count, entry IDs, content preview.
        """
        self._ensure_booted("remember")

        entries = self._run_async(
            self._lib.ingest("user", fact)
        )

        # Recategorize as user_knowledge
        from .core.types import EntryCategory
        for entry in entries:
            self._lib.rolodex.update_entry_enrichment(
                entry_id=entry.id,
                category=EntryCategory.USER_KNOWLEDGE,
            )

        return {
            "remembered": len(entries),
            "entry_ids": [e.id for e in entries],
            "content_preview": fact[:120],
        }

    # ─── Residue ──────────────────────────────────────────────────────────

    def write_residue(self, text: str) -> Dict[str, Any]:
        """
        Write or update the session residue.

        The residue is a poetic/textural encoding of the session's arc so far.
        Each call overwrites the previous residue. Written by the assistant,
        not the user. Ensures sessions that end without an explicit goodbye
        still have a residue on file.

        Args:
            text: The residue text (paragraph form, not a summary).

        Returns:
            Dict with: status, session_id, timestamp.
        """
        self._ensure_booted("write_residue")

        try:
            from .core.session_residue import write_residue
            write_residue(
                conn=self._lib.rolodex.conn,
                session_id=self._session_id,
                persona_key=self._persona_key or "",
                text=text,
            )
            return {
                "status": "ok",
                "session_id": self._session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_residue(self) -> Dict[str, Any]:
        """
        Get the latest session residue.

        Returns:
            Dict with: text, timestamp, session_id. Empty text if none exists.
        """
        self._ensure_booted("get_residue")

        try:
            from .core.session_residue import load_latest_residue
            persona_dir = str(self.persona_dir / self._persona_key) if self._persona_key else None
            meta = load_latest_residue(
                conn=self._lib.rolodex.conn,
                current_session_id=self._session_id,
                persona_key=self._persona_key or "",
                persona_dir=persona_dir,
            )
            return {
                "text": meta.get("text", ""),
                "timestamp": meta.get("timestamp"),
                "session_id": meta.get("session_id"),
            }
        except Exception:
            return {"text": "", "timestamp": None, "session_id": None}

    # ─── End ──────────────────────────────────────────────────────────────

    def end(self, summary: str = "") -> Dict[str, Any]:
        """
        End the current session.

        Finalizes the boot manifest, updates project clusters,
        and closes the session cleanly.

        Args:
            summary: Optional summary of what was accomplished.

        Returns:
            Dict with: status, session_id, summary.
        """
        self._ensure_booted("end")

        session_id = self._session_id

        # Refine manifest
        try:
            from .storage.manifest_manager import ManifestManager
            mm = ManifestManager(self._lib.rolodex.conn, self._lib.rolodex)
            current_manifest = mm.get_latest_manifest()
            if current_manifest:
                from .core.types import estimate_tokens
                from .retrieval.context_builder import ContextBuilder
                cb = ContextBuilder()
                profile = self._get_profile()
                profile_block = cb.build_profile_block(profile) if profile else ""
                uk_entries = self._lib.rolodex.get_user_knowledge_entries()
                uk_block = cb.build_user_knowledge_block(uk_entries) if uk_entries else ""
                fixed_cost = estimate_tokens(profile_block + uk_block)
                available_budget = max(0, 20000 - fixed_cost)
                mm.refine_manifest(current_manifest, session_id, available_budget)
        except Exception:
            pass

        # Adjust retrieval weights based on session outcomes
        try:
            from .core.retrieval_feedback import adjust_weights
            weight_result = adjust_weights(
                conn=self._lib.rolodex.conn,
                session_id=session_id,
            )
            # Store for the return value
            result_weight_stats = weight_result
        except Exception:
            result_weight_stats = {}

        # Update project clusters
        try:
            from .indexing.project_clusterer import ProjectClusterer
            pc = ProjectClusterer(self._lib.rolodex.conn)
            pc.update_clusters_for_session(session_id)
        except Exception:
            pass

        # Drain ingestion queue before ending the session.
        # Without this, pending enrichment tasks are lost on exit.
        try:
            if self._lib.ingestion_queue:
                self._run_async(self._lib.ingestion_queue.shutdown())
        except Exception:
            pass  # Best-effort drain; don't block session end

        # End the session
        self._lib.end_session(summary=summary)

        # Clean up session state file
        try:
            self._session_file.unlink(missing_ok=True)
        except Exception:
            pass

        self._booted = False
        result = {
            "status": "ok",
            "session_id": session_id,
            "summary": summary,
            "retrieval_feedback": result_weight_stats if result_weight_stats else {},
        }
        return result

    # ─── Context Accessors ────────────────────────────────────────────────

    def get_boot_context(self) -> str:
        """
        Return the full boot context as a string for prompt injection.
        Reads from the boot context file written during boot().
        """
        self._ensure_booted("get_boot_context")
        context_path = self._session_data.get("boot_files_context")
        if context_path and os.path.exists(context_path):
            with open(context_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def get_stats(self) -> Dict[str, Any]:
        """Return system stats."""
        self._ensure_booted("get_stats")
        return self._lib.get_stats()

    def get_retrieval_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get retrieval outcome statistics.

        Args:
            session_id: Scope to a specific session. Defaults to current session.

        Returns:
            Dict with: total_recalls, total_used, total_ignored, use_rate,
            top_used entries, top_ignored entries.
        """
        self._ensure_booted("get_retrieval_stats")
        from .core.retrieval_feedback import get_retrieval_stats
        return get_retrieval_stats(
            conn=self._lib.rolodex.conn,
            session_id=session_id or self._session_id,
        )

    def get_patterns(
        self,
        window_sessions: int = 5,
        stale_days: int = 30,
        gap_window_days: int = 14,
    ) -> Dict[str, Any]:
        """
        Generate a retrieval pattern report.

        Identifies:
            - Hot topics: subjects actively recalled across recent sessions
            - Dead zones: topics with entries not recalled in 30+ days
            - Gap signals: recurring queries with no good results

        Args:
            window_sessions: How many recent sessions to analyze for hot topics.
            stale_days: Days without recall before a topic is a dead zone.
            gap_window_days: Window for detecting gap signal patterns.

        Returns:
            Dict with: hot_topics, dead_zones, gaps, generated_at, config.
        """
        self._ensure_booted("get_patterns")
        from .core.retrieval_patterns import get_pattern_report
        return get_pattern_report(
            conn=self._lib.rolodex.conn,
            window_sessions=window_sessions,
            stale_days=stale_days,
            gap_window_days=gap_window_days,
        )

    # ─── Proactive Tool Finding ──────────────────────────────────────────

    def find_tools(
        self,
        providers: Optional[List] = None,
        gap_window_days: int = 14,
        min_occurrences: int = 3,
    ) -> Dict[str, Any]:
        """
        Run the gap-to-search pipeline: detect gaps, search for tools, create proposals.

        Args:
            providers: List of SearchProvider callables. If None, returns gap signals
                      without searching (useful for seeing what would trigger searches).
            gap_window_days: Window for gap signal detection.
            min_occurrences: Minimum gap occurrences before triggering search.

        Returns:
            Dict with: new_proposals (list), gap_signals (list), status.
        """
        self._ensure_booted("find_tools")
        from .core.tool_finder import generate_proposals
        from .core.retrieval_patterns import detect_gap_signals

        conn = self._lib.rolodex.conn

        # Always return gap signals for visibility
        gaps = detect_gap_signals(
            conn, threshold=min_occurrences, window_days=gap_window_days,
        )

        if not providers:
            return {
                "status": "ok",
                "gap_signals": gaps,
                "new_proposals": [],
                "note": "No search providers registered. Pass providers to search.",
            }

        proposals = generate_proposals(
            conn=conn,
            providers=providers,
            gap_window_days=gap_window_days,
            min_occurrences=min_occurrences,
        )

        return {
            "status": "ok",
            "gap_signals": gaps,
            "new_proposals": proposals,
        }

    def get_tool_proposals(self) -> List[Dict[str, Any]]:
        """Get all pending tool proposals awaiting user decision."""
        self._ensure_booted("get_tool_proposals")
        from .core.tool_finder import get_pending_proposals
        return get_pending_proposals(self._lib.rolodex.conn)

    def approve_tool(self, proposal_id: str) -> Dict[str, Any]:
        """Approve a tool proposal. Returns install details."""
        self._ensure_booted("approve_tool")
        from .core.tool_finder import confirm_proposal
        return confirm_proposal(self._lib.rolodex.conn, proposal_id)

    def dismiss_tool(self, proposal_id: str, reason: str = "") -> Dict[str, Any]:
        """Dismiss a tool proposal."""
        self._ensure_booted("dismiss_tool")
        from .core.tool_finder import dismiss_proposal
        return dismiss_proposal(self._lib.rolodex.conn, proposal_id, reason=reason)

    def mark_tool_installed(self, proposal_id: str) -> Dict[str, Any]:
        """Mark a tool as successfully installed."""
        self._ensure_booted("mark_tool_installed")
        from .core.tool_finder import mark_installed
        return mark_installed(self._lib.rolodex.conn, proposal_id)

    def record_tool_use(self, tool_name: str, tool_source: str) -> Dict[str, Any]:
        """Record that an installed tool was used."""
        self._ensure_booted("record_tool_use")
        from .core.tool_finder import record_tool_usage
        return record_tool_usage(self._lib.rolodex.conn, tool_name, tool_source)

    def get_tool_report(self) -> Dict[str, Any]:
        """Get full tool finding report: proposals, installed, unused."""
        self._ensure_booted("get_tool_report")
        from .core.tool_finder import get_tool_report
        return get_tool_report(self._lib.rolodex.conn)

    # ─── Pulse (Heartbeat) ─────────────────────────────────────────────────

    def pulse(self) -> Dict[str, Any]:
        """
        Lightweight heartbeat probe. Returns alive status without full boot.

        Returns:
            Dict with: alive, needs_boot, last_ingest_at, idle_minutes, needs_sweep.
        """
        IDLE_SWEEP_MINUTES = 60

        result = {
            "alive": self._booted and self._lib is not None,
            "needs_boot": not self._booted,
        }

        if not self._booted:
            # Try reading session file to check if a prior session exists
            try:
                persona_key = self._load_session_persona()
                result["has_prior_session"] = persona_key is not None
                result["prior_persona"] = persona_key
            except Exception:
                result["has_prior_session"] = False
            return result

        # Idle detection
        try:
            conn = self._lib.rolodex.conn
            row = conn.execute(
                "SELECT MAX(created_at) as latest FROM rolodex_entries"
            ).fetchone()
            last_ingest = row["latest"] if row and row["latest"] else None
            result["last_ingest_at"] = last_ingest

            if last_ingest:
                last_dt = datetime.fromisoformat(last_ingest)
                idle_delta = datetime.now(timezone.utc) - last_dt
                idle_minutes = idle_delta.total_seconds() / 60.0
                result["idle_minutes"] = round(idle_minutes, 1)
                result["needs_sweep"] = idle_minutes >= IDLE_SWEEP_MINUTES
            else:
                result["idle_minutes"] = None
                result["needs_sweep"] = False
        except Exception:
            result["needs_sweep"] = False

        return result

    # ─── Ingest Single Message ────────────────────────────────────────────

    def ingest_single(
        self,
        role: str,
        content: str,
        as_user_knowledge: bool = False,
        corrects_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a single message (user or assistant).

        Args:
            role: "user" or "assistant"
            content: The message text.
            as_user_knowledge: If True, mark as user_knowledge (3x boost).
            corrects_id: If set, supersede this entry with the new one.

        Returns:
            Dict with: ingested count, entry IDs, flags applied.
        """
        self._ensure_booted("ingest_single")

        entries = self._run_async(self._lib.ingest(role, content))

        result = {
            "ingested": len(entries),
            "entry_ids": [e.id for e in entries],
        }

        if as_user_knowledge and entries:
            from .core.types import EntryCategory
            for entry in entries:
                self._lib.rolodex.update_entry_enrichment(
                    entry_id=entry.id,
                    category=EntryCategory.USER_KNOWLEDGE,
                )
            result["user_knowledge"] = True

        if corrects_id and entries:
            self._lib.rolodex.supersede_entry(corrects_id, entries[0].id)
            result["corrected"] = corrects_id

        # Run enrichment
        try:
            from .storage.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph(self._lib.rolodex.conn)
            entry_ids = [e.id for e in entries]
            kg.extract_and_store(content, entry_ids)
        except Exception:
            pass

        return result

    # ─── Correct ──────────────────────────────────────────────────────────

    def correct(self, old_entry_id: str, corrected_text: str) -> Dict[str, Any]:
        """
        Supersede a factually wrong entry with corrected content.

        The old entry is soft-deleted (hidden from search, kept in DB).
        The corrected content is ingested as user_knowledge into the
        *original entry's session* to avoid cross-session contamination.

        Args:
            old_entry_id: ID of the entry to supersede.
            corrected_text: The correct content.

        Returns:
            Dict with: corrected (bool), old_entry_id, new_entry_id.
        """
        self._ensure_booted("correct")

        # Look up the original entry to find its session context
        original = self._lib.rolodex.get_entry(old_entry_id)
        target_session = (
            original.conversation_id
            if original and original.conversation_id
            else self._session_id
        )

        entries = self._run_async(
            self._lib.ingest("user", corrected_text)
        )

        if not entries:
            return {"error": "Failed to create corrected entry"}

        new_entry = entries[0]
        from .core.types import EntryCategory
        self._lib.rolodex.update_entry_enrichment(
            entry_id=new_entry.id,
            category=EntryCategory.USER_KNOWLEDGE,
        )

        # Rebind the corrected entry to the original's session
        if target_session != self._session_id:
            self._lib.rolodex.conn.execute(
                "UPDATE rolodex_entries SET conversation_id = ? WHERE id = ?",
                (target_session, new_entry.id),
            )
            self._lib.rolodex.conn.commit()

        existed = self._lib.rolodex.supersede_entry(old_entry_id, new_entry.id)

        return {
            "corrected": existed,
            "old_entry_id": old_entry_id,
            "new_entry_id": new_entry.id,
            "session_id": target_session,
        }

    # ─── Profile Management ──────────────────────────────────────────────

    def profile_set(self, key: str, value: str) -> Dict[str, Any]:
        """Set a user profile key-value pair."""
        self._ensure_booted("profile_set")
        conn = self._lib.rolodex.conn
        conn.execute(
            """INSERT OR REPLACE INTO user_profile (key, value, updated_at)
               VALUES (?, ?, ?)""",
            (key, value, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return {"profile_set": key, "value": value}

    def profile_show(self) -> Dict[str, Any]:
        """Get all user profile entries."""
        self._ensure_booted("profile_show")
        profile = self._get_profile()
        return {"profile": {k: v["value"] for k, v in profile.items()}}

    def profile_delete(self, key: str) -> Dict[str, Any]:
        """Delete a user profile entry."""
        self._ensure_booted("profile_delete")
        conn = self._lib.rolodex.conn
        cursor = conn.execute("DELETE FROM user_profile WHERE key = ?", (key,))
        conn.commit()
        return {"profile_deleted": key, "existed": cursor.rowcount > 0}

    # ─── Browse ──────────────────────────────────────────────────────────

    def browse_recent(self, limit: int = 20) -> Dict[str, Any]:
        """Browse most recent entries."""
        self._ensure_booted("browse_recent")
        entries = self._lib.rolodex.browse_recent(limit)
        return {
            "title": "Most Recent Entries",
            "count": len(entries),
            "entries": [self._entry_to_dict(e) for e in entries],
        }

    def browse_entry(self, entry_id: str) -> Dict[str, Any]:
        """Get a specific entry by ID or prefix."""
        self._ensure_booted("browse_entry")
        entry = self._lib.rolodex.get_entry(entry_id)
        if not entry:
            entry = self._lib.rolodex.browse_entry_by_prefix(entry_id)
        if not entry:
            return {"error": f"Entry not found: {entry_id}"}
        return {"title": f"Entry: {entry.id[:8]}", "entry": self._entry_to_dict(entry)}

    def browse_knowledge(self) -> Dict[str, Any]:
        """Browse user_knowledge entries."""
        self._ensure_booted("browse_knowledge")
        entries = self._lib.rolodex.get_user_knowledge_entries()
        return {
            "title": "User Knowledge",
            "count": len(entries),
            "entries": [self._entry_to_dict(e) for e in entries],
        }

    @staticmethod
    def _entry_to_dict(entry) -> Dict[str, Any]:
        """Convert a rolodex entry to a JSON-serializable dict."""
        return {
            "id": entry.id[:8] if hasattr(entry, 'id') else "",
            "full_id": entry.id if hasattr(entry, 'id') else "",
            "content": entry.content[:200] if hasattr(entry, 'content') else "",
            "source_type": getattr(entry, 'source_type', 'conversation'),
            "category": getattr(entry, 'category', None),
            "created_at": str(getattr(entry, 'created_at', '')),
            "tags": getattr(entry, 'tags', []),
        }

    # ─── Auto-Evaluate ───────────────────────────────────────────────────

    def auto_evaluate(self, message: str) -> Dict[str, Any]:
        """
        Standalone evaluation gate (no recall). For testing/debugging.

        Args:
            message: The user's message to evaluate.

        Returns:
            Dict with: status, intent, flags, proceed, context_block.
        """
        self._ensure_booted("auto_evaluate")

        stripped = message.strip()
        if len(stripped) < 3 or stripped.lower() in (
            "ok", "yes", "no", "thanks", "sure", "k", "hi", "hello", "hey",
        ):
            return {"status": "skip", "reason": "trivial_message"}

        from .retrieval.evaluation_gate import evaluate_message
        result = evaluate_message(
            message=message,
            conn=self._lib.rolodex.conn,
            session_id=self._session_id,
            workspace_dir=str(self.workspace_dir),
        )

        output = {
            "status": "ok",
            "intent": getattr(result, 'intent', 'unknown'),
            "flags": [
                {"category": f.category, "severity": f.severity, "detail": f.detail}
                for f in (result.flags if result else [])
            ],
            "proceed": getattr(result, 'proceed', True),
        }
        if result and result.context_block:
            output["context_block"] = result.context_block
        if result and getattr(result, 'initiative_prompt', None):
            output["initiative_prompt"] = result.initiative_prompt

        return output

    # ─── Harvest + Integrity ─────────────────────────────────────────────

    def harvest(
        self, force_all: bool = False, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Run the conversation harvest: process backup audit logs into the rolodex.

        Args:
            force_all: Re-process all logs, not just new ones.
            dry_run: Show what would be processed without actually doing it.

        Returns:
            Dict with harvest results.
        """
        try:
            # Import harvest from workspace (it lives alongside librarian_cli.py)
            import importlib.util
            harvest_path = self.workspace_dir / "harvest.py"
            if not harvest_path.exists():
                return {"error": "harvest.py not found in workspace"}

            spec = importlib.util.spec_from_file_location("harvest", str(harvest_path))
            harvest_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(harvest_mod)

            result = harvest_mod.harvest(
                workspace_root=str(self.workspace_dir.parent),
                persona_key=self._persona_key or "default",
                dry_run=dry_run,
                force_all=force_all,
            )
            return result
        except Exception as e:
            return {"error": f"Harvest failed: {e}"}

    def harvest_status(self) -> Dict[str, Any]:
        """Show harvest progress without running a harvest."""
        try:
            import importlib.util
            harvest_path = self.workspace_dir / "harvest.py"
            if not harvest_path.exists():
                return {"error": "harvest.py not found in workspace"}

            spec = importlib.util.spec_from_file_location("harvest", str(harvest_path))
            harvest_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(harvest_mod)

            return harvest_mod.show_status(str(self.workspace_dir.parent))
        except Exception as e:
            return {"error": f"Harvest status failed: {e}"}

    def integrity_check(self) -> Dict[str, Any]:
        """
        Detect sessions with messages that weren't ingested as entries.

        Returns:
            Dict with: status, sessions_checked, gaps list, coverage percentages.
        """
        self._ensure_booted("integrity_check")
        conn = self._lib.rolodex.conn

        sessions = conn.execute(
            """SELECT id, created_at, ended_at
               FROM conversations
               ORDER BY created_at DESC LIMIT 5"""
        ).fetchall()

        gaps = []
        total_messages = 0
        total_entries = 0

        for sess in sessions:
            sid = sess["id"]
            msg_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
                (sid,)
            ).fetchone()["cnt"]
            entry_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM rolodex_entries WHERE conversation_id = ?",
                (sid,)
            ).fetchone()["cnt"]

            total_messages += msg_count
            total_entries += entry_count

            if msg_count > 0 and entry_count < (msg_count * 0.5):
                gaps.append({
                    "session_id": sid,
                    "created_at": sess["created_at"],
                    "messages": msg_count,
                    "entries": entry_count,
                    "coverage_pct": round((entry_count / msg_count) * 100, 1),
                })

        result = {
            "status": "gap_detected" if gaps else "healthy",
            "sessions_checked": len(sessions),
            "total_messages": total_messages,
            "total_entries": total_entries,
            "overall_coverage_pct": round(
                (total_entries / total_messages) * 100, 1
            ) if total_messages > 0 else 100,
            "gaps": gaps,
        }

        if gaps:
            result["remediation"] = (
                "Run 'harvest' to process backup audit logs and fill gaps. "
                "Or use 'integrity-repair' to re-ingest from the messages table."
            )

        return result

    def integrity_repair(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Re-ingest messages that have no corresponding entries.

        Creates entries directly in the target session (not the current
        booted session) to avoid cross-session contamination.

        Args:
            session_id: Scope to a specific session. Defaults to last 5 sessions.

        Returns:
            Dict with: sessions_processed, messages_re_ingested, skipped counts.
        """
        self._ensure_booted("integrity_repair")
        conn = self._lib.rolodex.conn
        import hashlib
        from .core.types import RolodexEntry, EntryCategory, ContentModality

        existing_fps = set()
        for row in conn.execute(
            "SELECT content FROM rolodex_entries WHERE superseded_by IS NULL"
        ):
            text = str(row["content"]).strip()
            if text:
                existing_fps.add(hashlib.md5(text.lower().strip()[:300].encode()).hexdigest())

        if session_id:
            target_sessions = [session_id]
        else:
            rows = conn.execute(
                "SELECT id FROM conversations ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
            target_sessions = [r["id"] for r in rows]

        repaired = 0
        skipped_dup = 0

        for sid in target_sessions:
            messages = conn.execute(
                """SELECT role, content, turn_number
                   FROM messages
                   WHERE conversation_id = ? AND LENGTH(content) > 30
                   ORDER BY turn_number""",
                (sid,)
            ).fetchall()

            for msg in messages:
                fp = hashlib.md5(msg["content"].lower().strip()[:300].encode()).hexdigest()
                if fp in existing_fps:
                    skipped_dup += 1
                    continue

                try:
                    # Create entry directly with the correct session ID
                    # instead of going through ingest() which would write
                    # to the current booted session.
                    entry = RolodexEntry(
                        conversation_id=sid,
                        content=msg["content"],
                        content_type=ContentModality.CONVERSATIONAL,
                        category=EntryCategory.NOTE,
                        tags=["repaired"],
                        source_range={"turn": msg["turn_number"]} if msg["turn_number"] else {},
                        metadata={"repair_source": "integrity_repair"},
                    )
                    self._lib.rolodex.create_entry(entry)
                    existing_fps.add(fp)
                    repaired += 1
                except Exception:
                    pass

        return {
            "status": "repaired",
            "sessions_processed": len(target_sessions),
            "messages_re_ingested": repaired,
            "skipped_duplicate": skipped_dup,
        }

    # ─── Build Chains / Turn Pairs / Decision Journal ────────────────────

    def build_chains(
        self, session_id: Optional[str] = None, force: bool = False
    ) -> Dict[str, Any]:
        """
        Build narrative reasoning chains for sessions that lack them.

        Args:
            session_id: Target session. Defaults to last 10 sessions.
            force: Build even for short segments.

        Returns:
            Dict with: sessions_processed, chains_created.
        """
        self._ensure_booted("build_chains")
        conn = self._lib.rolodex.conn

        if session_id:
            target_sessions = [session_id]
        else:
            rows = conn.execute(
                "SELECT id FROM conversations ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
            target_sessions = [r["id"] for r in rows]

        try:
            from .core.chain_builder import ChainBuilder
        except ImportError:
            return {"error": "ChainBuilder not available"}

        chain_builder = ChainBuilder(
            rolodex=self._lib.rolodex,
            embedding_manager=self._lib.embeddings,
            llm_adapter=getattr(self._lib, '_llm_adapter', None),
            chain_interval=5,
        )

        chains_created = 0
        sessions_processed = 0

        for sid in target_sessions:
            existing_chains = self._lib.rolodex.get_chains_for_session(sid)
            last_chained_turn = 0
            if existing_chains:
                last_chained_turn = max(c.turn_range_end for c in existing_chains)

            messages_rows = conn.execute(
                """SELECT role, content, turn_number, token_count, timestamp
                   FROM messages WHERE conversation_id = ?
                   ORDER BY turn_number""",
                (sid,)
            ).fetchall()

            if not messages_rows:
                continue

            from .core.types import Message, MessageRole
            messages = []
            for row in messages_rows:
                try:
                    messages.append(Message(
                        role=MessageRole(row["role"]),
                        content=row["content"],
                        turn_number=row["turn_number"],
                        token_count=row["token_count"] or 0,
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                    ))
                except Exception:
                    continue

            if not messages:
                continue

            entry_ids = [
                r["id"] for r in conn.execute(
                    "SELECT id FROM rolodex_entries WHERE conversation_id = ?",
                    (sid,)
                ).fetchall()
            ]

            unchained = [m for m in messages if m.turn_number > last_chained_turn]
            if not unchained or (len(unchained) < 3 and not force):
                continue

            segment_size = chain_builder.chain_interval
            i = 0
            while i < len(unchained):
                segment = unchained[i:i + segment_size]
                if len(segment) < 2 and not force:
                    break

                chain = self._run_async(
                    chain_builder.build_breadcrumb(
                        session_id=sid,
                        messages=messages,
                        turn_range_start=segment[0].turn_number,
                        turn_range_end=segment[-1].turn_number,
                        related_entry_ids=entry_ids,
                    )
                )
                if chain:
                    self._lib.rolodex.create_chain(chain)
                    chains_created += 1
                i += segment_size

            sessions_processed += 1

        return {
            "status": "ok",
            "sessions_processed": sessions_processed,
            "chains_created": chains_created,
        }

    def turn_pairs(
        self, session_id: Optional[str] = None, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Ingest user+assistant turn pairs as atomic context units.

        Args:
            session_id: Target session. Defaults to last N sessions.
            limit: Number of recent sessions to process.

        Returns:
            Dict with: sessions_processed, turn_pairs_created.
        """
        self._ensure_booted("turn_pairs")
        conn = self._lib.rolodex.conn
        import hashlib

        existing_fps = set()
        for row in conn.execute(
            "SELECT content FROM rolodex_entries WHERE superseded_by IS NULL AND tags LIKE '%turn-pair%'"
        ):
            text = str(row["content"]).strip()
            if text:
                existing_fps.add(hashlib.md5(text.lower().strip()[:500].encode()).hexdigest())

        if session_id:
            target_sessions = [session_id]
        else:
            rows = conn.execute(
                "SELECT id FROM conversations ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            target_sessions = [r["id"] for r in rows]

        pairs_created = 0
        for sid in target_sessions:
            messages = conn.execute(
                """SELECT role, content, turn_number, timestamp
                   FROM messages WHERE conversation_id = ?
                   ORDER BY turn_number""",
                (sid,)
            ).fetchall()

            i = 0
            while i < len(messages) - 1:
                msg = messages[i]
                next_msg = messages[i + 1]

                if msg["role"] == "user" and next_msg["role"] == "assistant":
                    user_text = msg["content"].strip()
                    asst_text = next_msg["content"].strip()

                    if len(user_text) < 20 or len(asst_text) < 20:
                        i += 1
                        continue

                    pair_content = f"[USER]: {user_text}\n\n[ASSISTANT]: {asst_text}"
                    fp = hashlib.md5(pair_content.lower().strip()[:500].encode()).hexdigest()
                    if fp in existing_fps:
                        i += 2
                        continue

                    import json as _json
                    entry_id = str(uuid.uuid4())
                    tags = _json.dumps(["turn-pair", f"session:{sid[:8]}"])
                    source_range = _json.dumps({
                        "turn_start": msg["turn_number"],
                        "turn_end": next_msg["turn_number"],
                    })

                    conn.execute(
                        """INSERT INTO rolodex_entries
                           (id, conversation_id, content, content_type, category, tags,
                            source_range, access_count, last_accessed, created_at, tier,
                            embedding, linked_ids, metadata, source_type, verbatim_source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            entry_id, sid, pair_content, "conversational", "note",
                            tags, source_range, 0, None, msg["timestamp"], "cold",
                            None, _json.dumps([]),
                            _json.dumps({"turn_pair": True}),
                            "conversation", 1,
                        )
                    )

                    try:
                        conn.execute(
                            "INSERT INTO rolodex_fts (rowid, content) VALUES ((SELECT rowid FROM rolodex_entries WHERE id = ?), ?)",
                            (entry_id, pair_content)
                        )
                    except Exception:
                        pass

                    existing_fps.add(fp)
                    pairs_created += 1
                    i += 2
                else:
                    i += 1

        if pairs_created > 0:
            conn.commit()

        return {
            "status": "ok",
            "sessions_processed": len(target_sessions),
            "turn_pairs_created": pairs_created,
        }

    def decision_journal(
        self, session_id: Optional[str] = None, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Extract decisions from conversations as first-class entities.

        Args:
            session_id: Target session. Defaults to last N sessions.
            limit: Number of recent sessions to process.

        Returns:
            Dict with: sessions_processed, decisions_extracted.
        """
        self._ensure_booted("decision_journal")
        conn = self._lib.rolodex.conn

        if session_id:
            target_sessions = [session_id]
        else:
            rows = conn.execute(
                "SELECT id FROM conversations ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            target_sessions = [r["id"] for r in rows]

        # Count existing decision entries
        existing = conn.execute(
            "SELECT COUNT(*) as cnt FROM rolodex_entries WHERE tags LIKE '%decision%'"
        ).fetchone()["cnt"]

        decisions_found = 0
        for sid in target_sessions:
            entries = conn.execute(
                """SELECT id, content, tags FROM rolodex_entries
                   WHERE conversation_id = ? AND content LIKE '%decid%'
                   OR conversation_id = ? AND content LIKE '%decision%'
                   OR conversation_id = ? AND tags LIKE '%decision%'""",
                (sid, sid, sid)
            ).fetchall()
            decisions_found += len(entries)

        return {
            "status": "ok",
            "sessions_processed": len(target_sessions),
            "decisions_found": decisions_found,
            "existing_decision_entries": existing,
        }

    # ─── Load Skill (Tier 2 Knowledge Packs) ─────────────────────────────

    def load_skill_list(self) -> Dict[str, Any]:
        """List available indexed skill packs for the active persona."""
        self._ensure_booted("load_skill_list")

        persona_key = self._persona_key or "default"
        try:
            from .core.indexed_pack_loader import list_indexed_packs
            packs = list_indexed_packs(
                persona_key=persona_key,
                persona_dir=str(self.persona_dir),
            )
            return {"persona": persona_key, "indexed_packs": packs, "total": len(packs)}
        except Exception as e:
            return {"persona": persona_key, "indexed_packs": [], "total": 0, "error": str(e)}

    def load_skill_auto(self, message: str) -> Dict[str, Any]:
        """Auto-detect and load skill packs matching keywords in a message."""
        self._ensure_booted("load_skill_auto")

        persona_key = self._persona_key or "default"
        try:
            from .core.indexed_pack_loader import auto_detect_and_load
            result = auto_detect_and_load(
                message=message,
                persona_key=persona_key,
                persona_dir=str(self.persona_dir),
            )
            return result
        except Exception as e:
            return {"error": f"Auto-detect failed: {e}"}

    def load_skill_load(self, pack_name: str) -> Dict[str, Any]:
        """Load a specific skill pack by name."""
        self._ensure_booted("load_skill_load")

        persona_key = self._persona_key or "default"
        try:
            from .core.indexed_pack_loader import load_pack
            result = load_pack(
                pack_name=pack_name,
                persona_key=persona_key,
                persona_dir=str(self.persona_dir),
            )
            return result
        except Exception as e:
            return {"error": f"Load skill failed: {e}"}

    # ─── Reflect ─────────────────────────────────────────────────────────

    def reflect(self, force: bool = False) -> Dict[str, Any]:
        """
        Run session reflection: analyze skill usage, detect evolution candidates.

        Args:
            force: Override cooldown timer.

        Returns:
            Dict with: reflection report, recommendations.
        """
        self._ensure_booted("reflect")

        # Check cooldown
        if not force and self._lib.persona and hasattr(self._lib.persona, '_state'):
            state = self._lib.persona._state
            if state and hasattr(state, 'last_reflection_at') and state.last_reflection_at:
                try:
                    from datetime import timedelta
                    last_dt = datetime.fromisoformat(state.last_reflection_at)
                    if datetime.now(timezone.utc) - last_dt < timedelta(minutes=120):
                        return {
                            "status": "skipped",
                            "reason": "cooldown",
                            "last_reflection_at": state.last_reflection_at,
                        }
                except (ValueError, TypeError):
                    pass

        return {
            "status": "ok",
            "note": "Reflection requires session_reflection module. Run from workspace CLI.",
        }

    # ─── Onboarding ──────────────────────────────────────────────────────

    def onboard_start(self, intent: Optional[str] = None) -> Dict[str, Any]:
        """
        Start the v2.0 onboarding flow for persona creation.

        Args:
            intent: Optional user intent to pre-fill.

        Returns:
            Dict with: flow_version, first step, context session ID.
        """
        try:
            from .core.onboarding_flow import FlowEngine, OnboardingContext, save_onboarding_context

            engine = FlowEngine(templates_dir=str(self.persona_dir.parent / "persona_templates"))
            ctx = OnboardingContext()

            if intent:
                ctx.user_intent = intent
                ctx = engine.process_input(ctx, "welcome", None)
                ctx = engine.process_input(ctx, "intent_capture", intent)

            step = engine.get_next_step(ctx)

            session_id = self._session_id or str(uuid.uuid4())
            save_onboarding_context(ctx, session_id, str(self.workspace_dir))

            return {
                "flow_version": "2.0",
                "step": step.to_dict(),
                "context_session_id": session_id,
            }
        except Exception as e:
            return {"error": f"Onboarding flow error: {e}"}

    def onboard_flow_step(
        self, step_id: str, user_input: str
    ) -> Dict[str, Any]:
        """
        Process a single step in the onboarding flow.

        Args:
            step_id: The step ID being responded to.
            user_input: The user's response (JSON or text).

        Returns:
            Dict with: next step or completion status.
        """
        try:
            from .core.onboarding_flow import (
                FlowEngine, load_onboarding_context,
                save_onboarding_context,
            )

            session_id = self._session_id or ""
            ctx = load_onboarding_context(session_id, str(self.workspace_dir))
            if not ctx:
                return {"error": "No onboarding context found. Start with 'onboard create'."}

            engine = FlowEngine(templates_dir=str(self.persona_dir.parent / "persona_templates"))

            # Parse input
            import json as _json
            try:
                parsed = _json.loads(user_input)
            except (ValueError, TypeError):
                parsed = user_input

            ctx = engine.process_input(ctx, step_id, parsed)
            step = engine.get_next_step(ctx)

            save_onboarding_context(ctx, session_id, str(self.workspace_dir))

            result = {"step": step.to_dict(), "context_session_id": session_id}
            if step.step_type == "complete":
                result["status"] = "complete"
                result["persona_key"] = getattr(ctx, 'created_persona_key', None)

            # Materialize persona on terminal apply step:
            # seed identity graph so commitments/signals work from day one.
            if (step.metadata or {}).get("terminal") and getattr(ctx, "generated_persona", None):
                try:
                    from .core.onboarding import apply_scaffolding
                    persona_key = getattr(ctx, "persona_key", "") or "default"
                    conn = self._lib.rolodex.conn if self._lib else None
                    templates_dir = str(self.persona_dir.parent / "persona_templates")
                    if conn:
                        apply_scaffolding(
                            template_key=persona_key,
                            templates_dir=templates_dir,
                            rolodex_conn=conn,
                            session_id=session_id,
                            persona_yaml=ctx.generated_persona,
                        )
                        result["scaffolding_applied"] = True
                except Exception as exc:
                    result["scaffolding_warning"] = str(exc)

                # Generate .claude/settings.json from template if it doesn't exist.
                # This activates the hook architecture for the new user.
                try:
                    import shutil
                    claude_dir = Path(self.workspace_dir) / ".claude"
                    settings_path = claude_dir / "settings.json"
                    template_path = claude_dir / "settings.json.template"
                    if template_path.exists() and not settings_path.exists():
                        shutil.copy2(str(template_path), str(settings_path))
                        result["hooks_activated"] = True
                except Exception:
                    pass  # Non-fatal; hooks are a bonus, not a blocker

            return result
        except Exception as e:
            return {"error": f"Onboarding flow-step error: {e}"}

    # ─── Harvest Full ────────────────────────────────────────────────────

    def harvest_full(self) -> Dict[str, Any]:
        """
        Full pipeline: harvest + build-chains + integrity check.
        Run before persona switch or at session end for completeness.
        """
        results = {}

        # Harvest
        results["harvest"] = self.harvest()

        # Build chains
        if self._booted:
            try:
                results["chains"] = self.build_chains()
            except Exception as e:
                results["chains"] = {"error": str(e)}

            # Integrity check
            try:
                results["integrity"] = self.integrity_check()
            except Exception as e:
                results["integrity"] = {"error": str(e)}

        return {"status": "ok", "pipeline": results}

    # ─── Internal: Librarian Construction ─────────────────────────────────

    def _make_librarian(self, persona_key: Optional[str] = None):
        """Construct a TheLibrarian instance with proper config."""
        from .core.librarian import TheLibrarian
        from .utils.config import LibrarianConfig

        # Build config
        config = LibrarianConfig()
        for k, v in self._config_overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)

        # Check for API key (enables enhanced mode)
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        llm_adapter = None
        if api_key:
            try:
                from .indexing.anthropic_adapter import AnthropicAdapter
                llm_adapter = AnthropicAdapter(api_key=api_key)
            except Exception:
                pass

        # Persona path: support both persona.yaml and {key}.yaml
        persona_path = None
        if persona_key:
            candidate = self.persona_dir / persona_key / "persona.yaml"
            if candidate.exists():
                persona_path = str(candidate)
            else:
                candidate_alt = self.persona_dir / persona_key / f"{persona_key}.yaml"
                if candidate_alt.exists():
                    persona_path = str(candidate_alt)

        lib = TheLibrarian(
            db_path=str(self.db_path),
            llm_adapter=llm_adapter,
            config=config,
            persona_path=persona_path,
        )
        return lib

    # ─── Internal: Boot Context Building ──────────────────────────────────

    def _build_boot_context(
        self,
        intent: str = "",
        cold: bool = False,
        persona_key: str = "",
    ) -> Dict[str, str]:
        """Build all the context blocks loaded at boot time."""
        blocks = {}

        # Cognitive profile (from persona)
        if self._lib.persona:
            blocks["cognitive_profile"] = self._build_cognitive_profile()

        # User knowledge
        try:
            from .retrieval.context_builder import ContextBuilder
            cb = ContextBuilder()
            uk_entries = self._lib.rolodex.get_user_knowledge_entries(priority="core")
            if uk_entries:
                blocks["user_knowledge"] = cb.build_user_knowledge_block(uk_entries)
        except Exception:
            pass

        # Session briefing
        try:
            from .core.session_briefing import build_briefing_block
            briefing_result = build_briefing_block(
                conn=self._lib.rolodex.conn,
                current_session_id=self._session_id,
                window_hours=72,
                budget_tokens=1500,
            )
            if briefing_result.block:
                blocks["briefing"] = briefing_result.block
        except Exception:
            pass

        # Intent context
        if intent:
            try:
                from .core.intent_context import build_intent_context_block
                def _recall_for_intent(query, limit=10):
                    results = self._lib.rolodex.hybrid_search(
                        query, limit=limit, keyword_weight=0.7, semantic_weight=0.3
                    )
                    return [{"content": e.content, "source_type": e.source_type} for e, _score in results]
                intent_block = build_intent_context_block(
                    conn=self._lib.rolodex.conn,
                    intent_text=intent,
                    recall_fn=_recall_for_intent,
                    budget_tokens=2000,
                )
                if intent_block:
                    blocks["intent"] = intent_block
            except Exception:
                pass

        # Session residue (skip on cold boot)
        _residue_meta = {}
        if not cold:
            try:
                from .core.session_residue import load_latest_residue, build_residue_block
                persona_dir_str = str(self.persona_dir / persona_key) if persona_key else None
                _residue_meta = load_latest_residue(
                    conn=self._lib.rolodex.conn,
                    current_session_id=self._session_id,
                    persona_key=persona_key,
                    persona_dir=persona_dir_str,
                )
                residue_block = build_residue_block(
                    _residue_meta.get("text", ""),
                    timestamp=_residue_meta.get("timestamp"),
                    session_id=_residue_meta.get("session_id"),
                )
                if residue_block:
                    blocks["residue"] = residue_block
            except Exception:
                pass

        # Session tail (red-hot context from prior session)
        if not cold:
            try:
                from .core.session_residue import load_session_tail, build_tail_block
                persona_dir_str = str(self.persona_dir / persona_key) if persona_key else None
                _tail_data = load_session_tail(
                    conn=self._lib.rolodex.conn,
                    current_session_id=self._session_id,
                    persona_key=persona_key,
                    persona_dir=persona_dir_str,
                )
                if _tail_data.get("turns"):
                    tail_block = build_tail_block(_tail_data)
                    if tail_block:
                        blocks["tail"] = tail_block
            except Exception:
                pass

        # Resident knowledge (indexed packs)
        try:
            from .core.resident_knowledge import build_resident_knowledge_block
            rk_block = build_resident_knowledge_block(
                conn=self._lib.rolodex.conn,
                persona_key=persona_key,
                persona_dir=str(self.persona_dir / persona_key) if persona_key else None,
                budget_tokens=2000,
            )
            if rk_block:
                blocks["resident_knowledge"] = rk_block
        except Exception:
            pass

        # Identity context
        try:
            from .storage.identity_graph import IdentityGraph
            ig = IdentityGraph(self._lib.rolodex.conn)
            identity_block = ig.build_identity_context_block(budget_tokens=1500)
            if identity_block:
                blocks["identity"] = identity_block
        except Exception:
            pass

        # Experiential memory (skip on cold boot)
        if not cold:
            try:
                from .core.experiential_encoder import load_experiential_block
                exp_block = load_experiential_block(
                    persona_dir=str(self.persona_dir / persona_key) if persona_key else None,
                    budget_tokens=1500,
                )
                if exp_block:
                    blocks["experiential"] = exp_block
            except Exception:
                pass

        # Active commitments
        try:
            from .storage.identity_graph import IdentityGraph
            ig = IdentityGraph(self._lib.rolodex.conn)
            commitments = ig.build_commitments_block()
            if commitments:
                blocks["commitments"] = commitments
        except Exception:
            pass

        # Proactive tool proposals (surface pending suggestions at boot)
        try:
            from .core.tool_finder import get_pending_proposals, format_boot_proposals
            proposals = get_pending_proposals(self._lib.rolodex.conn)
            if proposals:
                proposal_block = format_boot_proposals(proposals)
                if proposal_block:
                    blocks["tool_proposals"] = proposal_block
        except Exception:
            pass  # Non-fatal — table may not exist yet

        # ── Residue/Briefing Conflict Resolution ──────────────────────
        # When a fresh residue exists, it IS the authoritative session
        # handoff. The briefing's open-thread detection uses keyword
        # scanning on old entries and can resurface work that was completed
        # in the session that wrote the residue. Strip the briefing's
        # "Open threads:" section when the residue is fresh enough to be
        # the definitive source.
        if blocks.get("residue") and blocks.get("briefing") and _residue_meta.get("text"):
            _residue_ts = _residue_meta.get("timestamp", "")
            _residue_fresh = False
            if _residue_ts:
                try:
                    from datetime import datetime, timezone, timedelta
                    _rts = _residue_ts.replace("Z", "+00:00")
                    if "+" not in _rts and _rts.count("-") <= 2:
                        _rdt = datetime.fromisoformat(_rts).replace(tzinfo=timezone.utc)
                    else:
                        _rdt = datetime.fromisoformat(_rts)
                    _age_hours = (datetime.now(timezone.utc) - _rdt).total_seconds() / 3600
                    _residue_fresh = _age_hours < 6
                except Exception:
                    pass
            if _residue_fresh:
                import re as _re_bt
                blocks["briefing"] = _re_bt.sub(
                    r'Open threads:\n(?:- [^\n]+\n)*\n?',
                    '',
                    blocks["briefing"],
                )

        return blocks

    def _write_boot_files(self, blocks: Dict[str, str]) -> Dict[str, str]:
        """Write boot context and operations to files, return paths."""
        boot_dir = self.workspace_dir
        context_path = str(boot_dir / "_boot_context.md")
        ops_path = str(boot_dir / "_boot_ops.md")

        # Context file: all the pre-loaded context blocks
        context_parts = []
        section_order = [
            ("cognitive_profile", "COGNITIVE PROFILE"),
            ("experiential", "EXPERIENTIAL MEMORY"),
            ("user_knowledge", "USER KNOWLEDGE"),
            ("resident_knowledge", "RESIDENT KNOWLEDGE"),
            ("identity", "IDENTITY CONTEXT"),
            ("commitments", "ACTIVE COMMITMENTS"),
            ("briefing", "SITUATIONAL BRIEFING"),
            ("residue", "SESSION RESIDUE"),
            ("intent", "INTENT CONTEXT"),
        ]
        for key, header in section_order:
            if key in blocks and blocks[key]:
                context_parts.append(blocks[key])

        # Write context file
        with open(context_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(context_parts))

        # Operations file: session rules (placeholder for now)
        with open(ops_path, "w", encoding="utf-8") as f:
            f.write(self._build_operations_block())

        # Save paths to session data
        self._session_data["boot_files_context"] = context_path
        self._session_data["boot_files_ops"] = ops_path

        return {
            "context": context_path,
            "operations": ops_path,
        }

    def _build_operations_block(self) -> str:
        """Build the operations/rules block for the session.

        Pulls behavioral genome (attention patterns, rhythm, memory priorities,
        triggers) from the persona profile so that persona-specific behavioral
        instructions reach the model at inference time.
        """
        sections = []

        sections.append("# Session Operations\n")
        # Get persona name for the anchor
        _anchor_name = self._lib.persona.name if self._lib.persona else "Partner"
        sections.append(
            "## Format Anchors (MANDATORY, every response)\n"
            f"OPEN: [{_anchor_name}] on its own line. Prerequisite: auto-recall has fired.\n"
            "CLOSE: - on its own line. Prerequisite: diarize has fired (mark response + residue).\n"
            "A response without both anchors is structurally incomplete.\n\n"
            "Before opening anchor: solitaire auto-recall \"<user message>\"\n"
            "After composing, before closing anchor: solitaire diarize \"<response>\" \"<residue>\"\n"
        )

        # ── Behavioral Genome from Persona ──────────────────────────
        if self._lib.persona:
            genome = self._build_behavioral_genome()
            if genome:
                sections.append(genome)

        # ── Writing Standards ────────────────────────────────────────
        sections.append(
            "## Writing Standards (MANDATORY, all responses)\n"
            "Ref: solitaire/ai_writing_tells.md (23 categories: 13 surface + 2 structural + 8 interactional)\n\n"
            "Cursed cluster: delve, intricate, tapestry, pivotal, underscore(v), landscape(met),\n"
            "  foster, testament, multifaceted, leverage, utilize, nuanced, realm, robust, streamline,\n"
            "  paradigm, holistic, myriad, plethora, elucidate, culminate, encompass, spearhead,\n"
            "  bolster, navigate(met), cornerstone, embark, forge(met), resonate, advent.\n"
            "  Any one is fine. 3+ in proximity = AI smell.\n"
            'No em dashes. No "It\'s not X, it\'s Y." No -ing editorial filler. No throat-clearing.\n'
            'No compulsive summaries. No closers that add nothing. No "honestly." No "good catch."\n'
            "No false ranges. No bloated phrasing. Vague superlatives need evidence. Weasel words need attribution.\n"
            "Scan for tell clustering before sending anything >3 sentences.\n"
        )

        return "\n".join(sections)

    def _build_behavioral_genome(self) -> str:
        """Build behavioral genome from persona profile.

        Renders attention patterns, conversational rhythm, memory priorities,
        and triggers into natural-language instructions the model can act on.
        """
        p = self._lib.persona
        if not p:
            return ""

        parts = []

        # ── Attention Patterns ──────────────────────────────────────
        if hasattr(p, 'attention') and p.attention:
            attn = p.attention
            effective_obs = p.traits.observance if hasattr(p.traits, 'observance') else 0.5
            lines = ["## Attention Patterns"]
            lines.append(f"Flag style: {getattr(attn, 'flag_style', 'inline')}")

            active_flags = attn.get_active_flags(effective_obs) if hasattr(attn, 'get_active_flags') else getattr(attn, 'always_flag', [])
            if active_flags:
                lines.append("Always flag (observance threshold met):")
                for item in active_flags:
                    cat = getattr(item, 'category', item.get('category', '')) if hasattr(item, 'category') or isinstance(item, dict) else str(item)
                    desc = getattr(item, 'description', item.get('description', '')) if hasattr(item, 'description') or isinstance(item, dict) else ''
                    lines.append(f"  - {cat}: {desc}")

            never_flag = getattr(attn, 'never_flag', [])
            if never_flag:
                lines.append("Never flag:")
                for nf in never_flag:
                    lines.append(f"  - {nf}")

            parts.append("\n".join(lines))

        # ── Conversational Rhythm ───────────────────────────────────
        if hasattr(p, 'rhythm') and p.rhythm:
            rhy = p.rhythm
            lines = ["## Conversational Rhythm"]
            lines.append(f"Default verbosity: {getattr(rhy, 'default_verbosity', 'moderate')}")

            _rhythm_bands = {
                "tangent_tolerance": {
                    "high": "Tangent tolerance: high -- follow interesting side threads freely",
                    "moderate": "Tangent tolerance: moderate -- follow relevant tangents, gently redirect others",
                    "low": "Tangent tolerance: low -- stay on topic, redirect if conversation drifts",
                },
                "silence_comfort": {
                    "high": "Silence comfort: high -- short replies are fine, don't pad responses",
                    "moderate": "Silence comfort: moderate -- brief is fine, but don't leave gaps that feel abrupt",
                    "low": "Silence comfort: low -- fill gaps, provide context proactively",
                },
                "action_bias": {
                    "high": "Action bias: high -- prefer doing over discussing. Build, ship, execute.",
                    "moderate": "Action bias: moderate -- act on clear tasks, discuss ambiguous ones",
                    "low": "Action bias: low -- discuss before doing. Verify intent, confirm scope.",
                },
            }
            for param_name, bands in _rhythm_bands.items():
                val = getattr(rhy, param_name, 0.5)
                if val >= 0.7:
                    lines.append(bands["high"])
                elif val >= 0.4:
                    lines.append(bands["moderate"])
                else:
                    lines.append(bands["low"])

            elab_desc = {
                "ask": "Elaborate only when the user asks for more detail",
                "offer": 'Offer to elaborate ("Want me to go deeper on X?") but don\'t auto-expand',
                "automatic": "Elaborate automatically when the topic warrants depth",
            }
            elab = getattr(rhy, 'elaboration_trigger', 'ask')
            lines.append(f"Elaboration: {elab_desc.get(elab, elab)}")

            parts.append("\n".join(lines))

        # ── Memory Priorities ───────────────────────────────────────
        if hasattr(p, 'memory_priorities') and p.memory_priorities:
            mem = p.memory_priorities
            high_w = getattr(mem, 'high_weight', [])
            normal_w = getattr(mem, 'normal_weight', [])
            low_w = getattr(mem, 'low_weight', [])
            if high_w or low_w:
                lines = ["## Memory Priorities"]
                lines.append("When recalling past context, weight these categories:")
                if high_w:
                    lines.append(f"  HIGH priority (2x boost): {', '.join(high_w)}")
                if normal_w:
                    lines.append(f"  NORMAL priority (1x): {', '.join(normal_w)}")
                if low_w:
                    lines.append(f"  LOW priority (0.5x): {', '.join(low_w)}")
                lines.append("Use these weights to prioritize which recalled entries to foreground in responses.")
                parts.append("\n".join(lines))

        # ── Triggers ────────────────────────────────────────────────
        if hasattr(p, 'triggers') and p.triggers:
            triggers = p.triggers
            lines = ["## Behavioral Triggers"]

            conv_overrides = getattr(triggers, 'conviction_overrides', [])
            if conv_overrides:
                lines.append("Conviction overrides (act on these even at moderate conviction):")
                for t in conv_overrides:
                    cond = getattr(t, 'condition', '') if not isinstance(t, dict) else t.get('condition', '')
                    action = getattr(t, 'action', '') if not isinstance(t, dict) else t.get('action', '')
                    lines.append(f"  - When: {cond} -> {action}")

            init_triggers = getattr(triggers, 'initiative_triggers', [])
            if init_triggers:
                lines.append("Initiative triggers (act proactively when detected):")
                for t in init_triggers:
                    cond = getattr(t, 'condition', '') if not isinstance(t, dict) else t.get('condition', '')
                    action = getattr(t, 'action', '') if not isinstance(t, dict) else t.get('action', '')
                    lines.append(f"  - When: {cond} -> {action}")

            if len(lines) > 1:
                parts.append("\n".join(lines))

        # ── Greeting Protocol ───────────────────────────────────────
        if hasattr(p, 'greeting') and p.greeting:
            greet = p.greeting
            lines = ["## Greeting Protocol"]
            style = getattr(greet, 'style', 'direct')
            lines.append(f"Style: {style}")
            mem_ref = getattr(greet, 'memory_reference', True)
            if mem_ref:
                lines.append("Reference prior context when greeting (natural memory weave)")
            examples = getattr(greet, 'examples', {})
            if examples:
                lines.append("Examples by warmth level:")
                for warmth_level, example in (examples.items() if isinstance(examples, dict) else []):
                    lines.append(f"  {warmth_level}: \"{example}\"")
            parts.append("\n".join(lines))

        if not parts:
            return ""

        return "\n\n".join(parts) + "\n"

    def _build_cognitive_profile(self) -> str:
        """Build the cognitive profile block from persona data.

        Delegates to solitaire.core.cognitive_profile.build_cognitive_profile(),
        the single source of truth for identity rendering (Phase 3).
        """
        if not self._lib.persona:
            return ""
        from .core.cognitive_profile import build_cognitive_profile
        conn = self._lib.rolodex.conn if self._lib.rolodex else None
        return build_cognitive_profile(self._lib.persona, conn=conn)

    # ─── Internal: Enrichment Pipeline ────────────────────────────────────

    def _run_enrichment(
        self,
        user_msg: str,
        assistant_msg: str,
        user_entries: list,
        asst_entries: list,
    ) -> Dict[str, Any]:
        """Run enrichment pipeline on ingested entries."""
        stats = {"kg": {}, "summarizer": {}, "temporal": {}, "identity": {}}

        all_entries = user_entries + asst_entries
        all_entry_ids = [e.id for e in all_entries]

        # Knowledge graph extraction
        try:
            from .storage.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph(self._lib.rolodex.conn)
            for msg, role in [(user_msg, "user"), (assistant_msg, "assistant")]:
                result = kg.extract_and_store(msg, all_entry_ids)
                stats["kg"]["entities"] = stats["kg"].get("entities", 0) + result.get("entities", 0)
                stats["kg"]["edges"] = stats["kg"].get("edges", 0) + result.get("edges", 0)
        except Exception:
            pass

        # Active summarizer
        try:
            from .core.active_summarizer import ActiveSummarizer
            summarizer = ActiveSummarizer(self._lib.rolodex.conn)
            for entry in all_entries:
                summarizer.process_entry(entry)
            stats["summarizer"]["updates"] = len(all_entries)
        except Exception:
            pass

        # Temporal reasoning
        try:
            from .core.temporal_reasoning import TemporalReasoner
            reasoner = TemporalReasoner(self._lib.rolodex.conn)
            for entry in all_entries:
                reasoner.extract_temporal_markers(entry)
        except Exception:
            pass

        # Identity enrichment
        try:
            from .storage.identity_enrichment import run_identity_enrichment
            id_result = run_identity_enrichment(
                conn=self._lib.rolodex.conn,
                content=user_msg + "\n" + assistant_msg,
                entry_ids=all_entry_ids,
            )
            stats["identity"] = id_result or {}
        except Exception:
            pass

        # User facts extraction (from user messages only)
        try:
            from .core.user_facts import process_at_ingestion as _process_user_facts
            user_entry_ids = [e.id for e in user_entries]
            if user_msg and user_entry_ids:
                uf_stats = _process_user_facts(
                    conn=self._lib.rolodex.conn,
                    content=user_msg,
                    role="user",
                    entry_ids=user_entry_ids,
                )
                stats["user_facts"] = uf_stats or {}
        except Exception:
            pass

        return stats

    # ─── Internal: Preflight Evaluation ───────────────────────────────────

    def _run_preflight(self, message: str) -> Dict[str, Any]:
        """Run the evaluation gate before recall, including gap signal detection."""
        preflight = {"context_block": "", "intent": "unknown", "proceed": True}

        try:
            from .retrieval.evaluation_gate import evaluate_message
            result = evaluate_message(
                message=message,
                conn=self._lib.rolodex.conn if self._lib else None,
                session_id=self._session_id,
                workspace_dir=str(self.workspace_dir),
            )
            preflight = {
                "context_block": result.context_block if result else "",
                "intent": getattr(result, "intent_type", "unknown"),
                "proceed": getattr(result, "proceed", True),
            }
        except Exception:
            pass

        # Gap signal detection: check if this query matches a known gap
        try:
            from .core.retrieval_patterns import check_gap_for_query
            gap = check_gap_for_query(
                conn=self._lib.rolodex.conn,
                query=message,
            )
            if gap:
                preflight["gap_signal"] = gap
                gap_note = f"\n\n⚠ GAP SIGNAL: {gap['note']}"
                preflight["context_block"] = (
                    preflight.get("context_block", "") + gap_note
                )
        except Exception:
            pass  # Non-fatal

        return preflight

    # ─── Internal: Persona Scanning ───────────────────────────────────────

    def _scan_personas(self) -> List[Dict[str, str]]:
        """Scan the persona directory for available personas."""
        personas = []
        if not self.persona_dir.exists():
            return personas

        for item in sorted(self.persona_dir.iterdir()):
            if not item.is_dir():
                continue

            # Support both persona.yaml and {key}.yaml naming conventions
            persona_yaml = item / "persona.yaml"
            key_yaml = item / f"{item.name}.yaml"
            yaml_path = persona_yaml if persona_yaml.exists() else (
                key_yaml if key_yaml.exists() else None
            )
            if yaml_path is None:
                continue

            try:
                import yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                identity = data.get("identity", {})
                personas.append({
                    "key": item.name,
                    "display_name": identity.get("name", item.name),
                    "short_label": identity.get("name", item.name),
                    "description": identity.get("description", ""),
                })
            except Exception:
                # Fallback: list the directory even without valid YAML
                personas.append({
                    "key": item.name,
                    "display_name": item.name,
                    "short_label": item.name,
                    "description": "",
                })

        return personas

    # ─── Internal: Session State ──────────────────────────────────────────

    def _save_session_state(self, persona_key: str) -> None:
        """Persist session state to disk for resume support."""
        self._session_data = {
            "session_id": self._session_id,
            "persona_key": persona_key,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "turn_count": 0,
        }
        try:
            with open(self._session_file, "w") as f:
                json.dump(self._session_data, f)
        except Exception:
            pass

    def _load_session_persona(self) -> Optional[str]:
        """Load the last active persona key from session state."""
        try:
            with open(self._session_file, "r") as f:
                data = json.load(f)
            return data.get("persona_key")
        except Exception:
            return None

    def _load_session_id(self) -> Optional[str]:
        """Load the last session ID from session state."""
        try:
            with open(self._session_file, "r") as f:
                data = json.load(f)
            return data.get("session_id")
        except Exception:
            return None

    def _get_profile(self) -> Dict[str, Any]:
        """Get user profile from rolodex."""
        try:
            rows = self._lib.rolodex.conn.execute(
                "SELECT key, value FROM user_profile ORDER BY key"
            ).fetchall()
            return {row["key"]: {"value": row["value"]} for row in rows}
        except Exception:
            return {}

    def _get_persona_info(self) -> Dict[str, Any]:
        """Get current persona info for boot response."""
        if self._lib.persona:
            profile = self._lib.get_persona_profile()
            if profile:
                return {
                    "identity": profile.get("identity", {}),
                    "effective_traits": profile.get("effective_traits", {}),
                }
            return {}
        return {}