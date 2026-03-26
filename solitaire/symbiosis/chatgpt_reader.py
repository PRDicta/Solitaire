"""
Reader for ChatGPT conversation export files.

ChatGPT exports produce a conversations.json file containing an array
of conversation objects. Each conversation has a tree structure of message
nodes (not a flat list).

Structure:
    [
        {
            "title": "Conversation Title",
            "create_time": 1709000000.0,
            "update_time": 1709001000.0,
            "mapping": {
                "<node_id>": {
                    "id": "<node_id>",
                    "message": {
                        "id": "<msg_id>",
                        "author": {"role": "user"|"assistant"|"system"},
                        "content": {"content_type": "text", "parts": ["..."]},
                        "create_time": 1709000000.0
                    },
                    "parent": "<parent_node_id>",
                    "children": ["<child_node_id>"]
                }
            }
        }
    ]

Each conversation becomes one IngestCandidate containing the full
conversation text (user + assistant turns concatenated). This is
intentional: Solitaire's enrichment pipeline is designed to extract
facts, preferences, and patterns from conversation text.

For very long conversations, the reader chunks at a configurable
turn count to keep individual entries from becoming unwieldy.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from .reader_base import ReaderBase
from ..core.types import (
    IngestCandidate,
    IngestContentType,
    EnrichmentHint,
)


def _extract_messages(mapping: Dict[str, Any]) -> List[Tuple[str, str, Optional[float]]]:
    """Walk the conversation tree and extract messages in order.

    Returns list of (role, content, create_time) tuples.
    The tree is traversed by following parent->children links.
    """
    if not mapping:
        return []

    # Find the root node (no parent or parent not in mapping)
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if not parent or parent not in mapping:
            root_id = node_id
            break

    if not root_id:
        # Fallback: just iterate all nodes
        root_id = next(iter(mapping))

    # BFS from root, following first child at each level
    messages = []
    current_id = root_id
    visited = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        node = mapping.get(current_id)
        if not node:
            break

        msg = node.get("message")
        if msg:
            author = msg.get("author", {})
            role = author.get("role", "unknown")
            content_obj = msg.get("content", {})

            # Extract text from parts
            parts = content_obj.get("parts", [])
            text_parts = []
            for part in parts:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict):
                    # Some parts are dicts (images, etc). Skip non-text.
                    text_val = part.get("text", "")
                    if text_val:
                        text_parts.append(text_val)

            content = "\n".join(text_parts).strip()
            if content and role in ("user", "assistant"):
                create_time = msg.get("create_time")
                messages.append((role, content, create_time))

        # Follow first child
        children = node.get("children", [])
        current_id = children[0] if children else None

    return messages


def _format_conversation(messages: List[Tuple[str, str, Optional[float]]]) -> str:
    """Format extracted messages into readable conversation text."""
    lines = []
    for role, content, _ in messages:
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"[{prefix}]\n{content}")
    return "\n\n".join(lines)


class ChatGPTExportReader(ReaderBase):
    """Reads ChatGPT conversation export files (conversations.json).

    Config:
        path: str - Path to conversations.json.
        max_turns_per_chunk: int - Split long conversations at this turn
            count (default 40). Set to 0 for no splitting.
        min_turns: int - Skip conversations shorter than this (default 2).
    """

    @property
    def source_id(self) -> str:
        return "chatgpt-export"

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        path = config.get("path", "")
        if not path:
            return {"valid": False, "error": "No path provided"}
        if not os.path.isfile(path):
            return {"valid": False, "error": f"File not found: {path}"}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return {"valid": False, "error": "Expected a JSON array of conversations"}
            if len(data) == 0:
                return {"valid": False, "error": "Empty conversation list"}
            # Quick sanity: first item should have 'mapping'
            if "mapping" not in data[0]:
                return {"valid": False, "error": "First conversation missing 'mapping' field"}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return {"valid": False, "error": f"Invalid JSON: {e}"}
        except MemoryError:
            return {"valid": False, "error": "File too large to load into memory"}

        return {"valid": True, "conversation_count": len(data)}

    def read(self, config: Dict[str, Any]) -> Iterator[IngestCandidate]:
        """Yield IngestCandidates from a ChatGPT export file.

        Each conversation (or chunk of a long conversation) becomes
        one IngestCandidate with content_type=CONVERSATION.
        """
        path = config["path"]
        max_turns = config.get("max_turns_per_chunk", 40)
        min_turns = config.get("min_turns", 2)

        try:
            with open(path, "r", encoding="utf-8") as f:
                conversations = json.load(f)
        except Exception as e:
            yield IngestCandidate(
                source_ref=f"file://{os.path.abspath(path)}",
                raw_content="",
                content_type=IngestContentType.OTHER,
                enrichment_hint=EnrichmentHint.SKIP,
                confidence=0.0,
                source_id=self.source_id,
                metadata={"reader_error": f"Failed to load file: {e}"},
                dedup_key=f"chatgpt:{path}:load-error",
            )
            return

        for conv_idx, conv in enumerate(conversations):
            try:
                title = conv.get("title", f"Conversation {conv_idx + 1}")
                mapping = conv.get("mapping", {})
                create_time = conv.get("create_time")

                messages = _extract_messages(mapping)

                if len(messages) < min_turns:
                    continue

                # Parse conversation timestamp
                timestamp = None
                if create_time:
                    try:
                        timestamp = datetime.fromtimestamp(float(create_time), tz=timezone.utc)
                    except (ValueError, OSError):
                        pass

                # Chunk long conversations
                if max_turns > 0 and len(messages) > max_turns:
                    chunks = [
                        messages[i:i + max_turns]
                        for i in range(0, len(messages), max_turns)
                    ]
                else:
                    chunks = [messages]

                for chunk_idx, chunk in enumerate(chunks):
                    text = _format_conversation(chunk)
                    if not text.strip():
                        continue

                    # Build a header with conversation context
                    header = f"ChatGPT Conversation: {title}"
                    if len(chunks) > 1:
                        header += f" (part {chunk_idx + 1}/{len(chunks)})"
                    full_content = f"{header}\n\n{text}"

                    # Dedup key: conversation index + chunk index + content hash
                    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                    conv_id = conv.get("id", str(conv_idx))
                    dedup_key = f"chatgpt:{conv_id}:{chunk_idx}:{content_hash}"

                    # Use earliest message timestamp in chunk if available
                    chunk_time = timestamp
                    for _, _, msg_time in chunk:
                        if msg_time:
                            try:
                                chunk_time = datetime.fromtimestamp(float(msg_time), tz=timezone.utc)
                                break
                            except (ValueError, OSError):
                                continue

                    yield IngestCandidate(
                        source_ref=f"chatgpt-conv://{conv_id}",
                        raw_content=full_content,
                        content_type=IngestContentType.CONVERSATION,
                        enrichment_hint=EnrichmentHint.FULL,
                        confidence=0.6,  # Conversations need heavy extraction
                        source_id=self.source_id,
                        timestamp=chunk_time,
                        metadata={
                            "conversation_title": title,
                            "conversation_index": conv_idx,
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks),
                            "message_count": len(chunk),
                            "total_messages": len(messages),
                        },
                        tags=[
                            f"chatgpt-title:{title}",
                            "source:chatgpt-export",
                        ],
                        dedup_key=dedup_key,
                    )

            except Exception as e:
                yield IngestCandidate(
                    source_ref=f"file://{os.path.abspath(path)}#conv{conv_idx}",
                    raw_content="",
                    content_type=IngestContentType.OTHER,
                    enrichment_hint=EnrichmentHint.SKIP,
                    confidence=0.0,
                    source_id=self.source_id,
                    metadata={"reader_error": f"Error processing conversation {conv_idx}: {e}"},
                    dedup_key=f"chatgpt:{path}:conv{conv_idx}:error",
                )
