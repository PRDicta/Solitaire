"""
The Librarian — System 5: Narrative Continuity Tracker

Tracks narrative arcs, thematic chapters, beats, and open threads across
a persona's lifetime. Maintains a human-readable YAML file that bridges
discrete sessions into a coherent story arc.

Designed for:
- Capturing key moments (beats) that shape the narrative
- Tracking unresolved plot threads that span multiple sessions
- Managing narrative arcs (product development, personal growth, etc.)
- Closing chapters and summarizing eras
- Injecting narrative context into session prompts

The tracker keeps the YAML under 2KB through beat FIFO trimming.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import os
import yaml


# ─── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class Beat:
    """A key moment in the narrative — a turning point, milestone, or insight."""
    description: str
    timestamp: str  # ISO format: "2026-03-04T18:00:00"
    significance: float = 0.5  # 0.0 (minor) to 1.0 (pivotal)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Beat":
        """Reconstruct from dict."""
        return cls(**data)


@dataclass
class Thread:
    """An open narrative thread — a question, task, or unresolved plot point."""
    description: str
    status: str = "open"  # "open" | "resolved" | "parked"
    opened_at: str = ""  # ISO format, empty if not set
    resolved_at: str = ""  # ISO format, empty if not resolved

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Thread":
        """Reconstruct from dict."""
        return cls(**data)


@dataclass
class Arc:
    """A long-running narrative arc — product development, personal growth, etc."""
    name: str
    status: str = "active"  # "active" | "paused" | "completed"
    trajectory: str = "steady"  # "ascending" | "steady" | "declining"
    next_milestone: str = ""  # What's the next goal or waypoint?

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Arc":
        """Reconstruct from dict."""
        return cls(**data)


@dataclass
class Chapter:
    """A thematic chapter — a narrative unit with beats and threads."""
    title: str
    opened: str  # ISO date: "2026-03-04"
    theme: str
    key_beats: List[Beat] = field(default_factory=list)
    open_threads: List[Thread] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict, converting nested dataclasses."""
        return {
            "title": self.title,
            "opened": self.opened,
            "theme": self.theme,
            "key_beats": [b.to_dict() for b in self.key_beats],
            "open_threads": [t.to_dict() for t in self.open_threads],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chapter":
        """Reconstruct from dict."""
        return cls(
            title=data["title"],
            opened=data["opened"],
            theme=data["theme"],
            key_beats=[Beat.from_dict(b) for b in data.get("key_beats", [])],
            open_threads=[Thread.from_dict(t) for t in data.get("open_threads", [])],
        )


@dataclass
class Narrative:
    """The full narrative state: current chapter, history, and arcs."""
    current_chapter: Optional[Chapter] = None
    previous_chapters: List[dict] = field(default_factory=list)  # {title, period, summary}
    arcs: List[Arc] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "current_chapter": self.current_chapter.to_dict() if self.current_chapter else None,
            "previous_chapters": self.previous_chapters,
            "arcs": [a.to_dict() for a in self.arcs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Narrative":
        """Reconstruct from dict."""
        current = None
        if data.get("current_chapter"):
            current = Chapter.from_dict(data["current_chapter"])

        return cls(
            current_chapter=current,
            previous_chapters=data.get("previous_chapters", []),
            arcs=[Arc.from_dict(a) for a in data.get("arcs", [])],
        )


# ─── NarrativeTracker ──────────────────────────────────────────────────────

class NarrativeTracker:
    """
    Manages narrative state for a persona.

    Loads/saves from YAML at the given path. Enforces guardrails:
    - Max 10 beats per chapter (FIFO trimming when exceeded)
    - Max 8 open threads per chapter
    - YAML file stays under 2KB

    All dates/times are ISO format strings.
    """

    MAX_BEATS_PER_CHAPTER = 10
    MAX_THREADS_PER_CHAPTER = 8
    MAX_YAML_SIZE_BYTES = 2048

    def __init__(self, narrative_path: str):
        """
        Initialize tracker with path to narrative.yaml.

        Args:
            narrative_path: Absolute path to narrative.yaml (will be created if missing)
        """
        self.path = narrative_path
        self.narrative = self.load()

    def load(self) -> Narrative:
        """
        Load narrative from YAML file.
        If file doesn't exist, create a default (empty) narrative.
        Returns the loaded Narrative object.
        """
        if not os.path.exists(self.path):
            # Create default empty narrative
            return Narrative()

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                return Narrative()

            return Narrative.from_dict(data)
        except Exception as e:
            # If parse fails, return empty narrative
            print(f"Warning: Failed to load narrative from {self.path}: {e}")
            return Narrative()

    def save(self) -> None:
        """
        Persist narrative to YAML file.

        If the serialized YAML exceeds 2KB, trim beats from current chapter
        (oldest first, FIFO) until it fits.
        """
        # Serialize narrative to dict
        data = self.narrative.to_dict()

        # Convert to YAML
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)

        # Check size and trim if needed
        while len(yaml_str.encode("utf-8")) > self.MAX_YAML_SIZE_BYTES:
            if (
                self.narrative.current_chapter
                and len(self.narrative.current_chapter.key_beats) > 0
            ):
                # Remove oldest beat (FIFO)
                self.narrative.current_chapter.key_beats.pop(0)
                data = self.narrative.to_dict()
                yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
            else:
                # Can't trim further; save as-is (shouldn't happen in practice)
                break

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # Write to file
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(yaml_str)

    def add_beat(self, description: str, significance: float = 0.5) -> None:
        """
        Add a key beat to the current chapter.
        Enforces MAX_BEATS_PER_CHAPTER; removes oldest beat (FIFO) if exceeded.

        Args:
            description: What happened (1-2 sentences recommended)
            significance: 0.0 (minor) to 1.0 (pivotal)
        """
        if self.narrative.current_chapter is None:
            raise RuntimeError("Cannot add beat: no current chapter open.")

        beat = Beat(
            description=description,
            timestamp=datetime.now(timezone.utc).isoformat(),
            significance=significance,
        )
        self.narrative.current_chapter.key_beats.append(beat)

        # Enforce max beats; remove oldest (FIFO)
        if len(self.narrative.current_chapter.key_beats) > self.MAX_BEATS_PER_CHAPTER:
            self.narrative.current_chapter.key_beats.pop(0)

        self.save()

    def open_thread(self, description: str) -> None:
        """
        Add an open thread to the current chapter.
        Enforces MAX_THREADS_PER_CHAPTER.

        Args:
            description: The question, task, or unresolved point
        """
        if self.narrative.current_chapter is None:
            raise RuntimeError("Cannot open thread: no current chapter open.")

        if len(self.narrative.current_chapter.open_threads) >= self.MAX_THREADS_PER_CHAPTER:
            raise RuntimeError(
                f"Cannot open more than {self.MAX_THREADS_PER_CHAPTER} threads per chapter."
            )

        thread = Thread(
            description=description,
            status="open",
            opened_at=datetime.now(timezone.utc).isoformat(),
            resolved_at="",
        )
        self.narrative.current_chapter.open_threads.append(thread)
        self.save()

    def resolve_thread(self, keyword: str) -> bool:
        """
        Resolve the first thread matching the keyword.
        Updates the thread's status to "resolved" and sets resolved_at timestamp.

        Args:
            keyword: Search term (case-insensitive substring match)

        Returns:
            True if a thread was found and resolved, False otherwise
        """
        if self.narrative.current_chapter is None:
            return False

        for thread in self.narrative.current_chapter.open_threads:
            if keyword.lower() in thread.description.lower():
                thread.status = "resolved"
                thread.resolved_at = datetime.now(timezone.utc).isoformat()
                self.save()
                return True

        return False

    def park_thread(self, keyword: str) -> bool:
        """
        Park the first thread matching the keyword.
        Updates the thread's status to "parked" (on hold, not resolved).

        Args:
            keyword: Search term (case-insensitive substring match)

        Returns:
            True if a thread was found and parked, False otherwise
        """
        if self.narrative.current_chapter is None:
            return False

        for thread in self.narrative.current_chapter.open_threads:
            if keyword.lower() in thread.description.lower():
                thread.status = "parked"
                self.save()
                return True

        return False

    def close_chapter(self, summary: str) -> None:
        """
        Close the current chapter and move it to previous_chapters.
        Sets current_chapter to None.

        If no current chapter exists, raises RuntimeError.

        Args:
            summary: Brief summary of the chapter (2-3 sentences recommended)
        """
        if self.narrative.current_chapter is None:
            raise RuntimeError("No current chapter to close.")

        chapter = self.narrative.current_chapter

        # Create a summary dict with metadata
        chapter_summary = {
            "title": chapter.title,
            "period": chapter.opened,  # Just the opened date (end date could be added)
            "summary": summary,
        }

        self.narrative.previous_chapters.append(chapter_summary)
        self.narrative.current_chapter = None
        self.save()

    def open_chapter(self, title: str, theme: str) -> None:
        """
        Open a new chapter.
        If a chapter is currently open, closes it with an auto-generated summary first.

        Args:
            title: Chapter title
            theme: Thematic focus or context
        """
        # If a chapter is open, close it auto-summary
        if self.narrative.current_chapter is not None:
            auto_summary = f"Transitioned to next chapter. {len(self.narrative.current_chapter.key_beats)} beats recorded."
            self.close_chapter(auto_summary)

        # Open new chapter
        self.narrative.current_chapter = Chapter(
            title=title,
            opened=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            theme=theme,
            key_beats=[],
            open_threads=[],
        )
        self.save()

    def update_arc(
        self,
        name: str,
        status: Optional[str] = None,
        trajectory: Optional[str] = None,
        next_milestone: Optional[str] = None,
    ) -> None:
        """
        Update or create an arc.

        If the arc doesn't exist, creates it with the given parameters.
        If it does exist, updates only the provided fields (preserves others).

        Args:
            name: Arc name (used to identify/update)
            status: "active" | "paused" | "completed" (optional)
            trajectory: "ascending" | "steady" | "declining" (optional)
            next_milestone: Next goal or waypoint (optional)
        """
        # Look for existing arc by name
        existing_arc = None
        for arc in self.narrative.arcs:
            if arc.name == name:
                existing_arc = arc
                break

        if existing_arc is None:
            # Create new arc
            arc = Arc(
                name=name,
                status=status or "active",
                trajectory=trajectory or "steady",
                next_milestone=next_milestone or "",
            )
            self.narrative.arcs.append(arc)
        else:
            # Update fields if provided
            if status is not None:
                existing_arc.status = status
            if trajectory is not None:
                existing_arc.trajectory = trajectory
            if next_milestone is not None:
                existing_arc.next_milestone = next_milestone

        self.save()

    def get_context_summary(self) -> str:
        """
        Return a brief text summary of narrative state for prompt injection.

        Format: 'Chapter: [title] | Theme: [theme] | Beats: N | Open threads: N | Arcs: [names]'

        If no current chapter, returns a minimal summary of the arcs.
        """
        if self.narrative.current_chapter is None:
            arc_names = ", ".join([a.name for a in self.narrative.arcs])
            return f"No active chapter. Arcs: {arc_names or 'none'}"

        chapter = self.narrative.current_chapter
        arc_names = ", ".join([a.name for a in self.narrative.arcs])

        return (
            f"Chapter: {chapter.title} | Theme: {chapter.theme} | "
            f"Beats: {len(chapter.key_beats)} | "
            f"Open threads: {len(chapter.open_threads)} | "
            f"Arcs: {arc_names or 'none'}"
        )

    def to_dict(self) -> dict:
        """Full serialization to dict."""
        return self.narrative.to_dict()

    @classmethod
    def from_dict(cls, data: dict, path: str) -> "NarrativeTracker":
        """Reconstruct from dict and set path."""
        tracker = cls(path)
        tracker.narrative = Narrative.from_dict(data)
        return tracker
