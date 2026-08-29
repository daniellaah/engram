import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from engram.tools import Tool, ToolResult


class MemoryKind(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PERCEPTUAL = "perceptual"


class Memory(BaseModel):
    """One durable or short-lived memory with retrieval signals."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    content: str
    kind: MemoryKind = MemoryKind.SEMANTIC
    importance: float = Field(default=0.5, ge=0, le=1)
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryStore(Protocol):
    def add(self, memory: Memory) -> Memory: ...

    def get(self, memory_id: str) -> Memory | None: ...

    def remove(self, memory_id: str) -> bool: ...

    def list(self) -> tuple[Memory, ...]: ...


class InMemoryStore:
    def __init__(self, memories: Iterable[Memory] | None = None) -> None:
        self._memories = {memory.id: memory for memory in memories or ()}

    def add(self, memory: Memory) -> Memory:
        self._memories[memory.id] = memory
        return memory

    def get(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    def remove(self, memory_id: str) -> bool:
        return self._memories.pop(memory_id, None) is not None

    def list(self) -> tuple[Memory, ...]:
        return tuple(self._memories.values())


class JsonMemoryStore(InMemoryStore):
    """A small local store with explicit atomic persistence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__()
        if self.path.exists():
            self.load()

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(
                    [memory.model_dump(mode="json") for memory in self.list()],
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return self.path

    def load(self) -> None:
        with self.path.open(encoding="utf-8") as stream:
            payload: object = json.load(stream)
        if not isinstance(payload, list):
            raise ValueError("Memory store must contain a JSON array.")
        memories = [Memory.model_validate(item) for item in payload]
        self._memories = {memory.id: memory for memory in memories}


class MemoryManager:
    """Record, retrieve, and forget memories without imposing a database backend."""

    _token_pattern = re.compile(r"[\w]+|[^\x00-\x7F]", re.UNICODE)

    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store or InMemoryStore()

    def remember(
        self,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.SEMANTIC,
        importance: float = 0.5,
        tags: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        if not content.strip():
            raise ValueError("Memory content cannot be empty.")
        return self.store.add(
            Memory(
                content=content.strip(),
                kind=kind,
                importance=importance,
                tags=tuple(tags),
                metadata=metadata or {},
            )
        )

    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        kinds: Iterable[MemoryKind] | None = None,
    ) -> list[Memory]:
        if limit <= 0:
            return []
        allowed = set(kinds) if kinds is not None else None
        candidates = [
            memory for memory in self.store.list() if allowed is None or memory.kind in allowed
        ]
        query_terms = self._terms(query)
        now = datetime.now(UTC)

        def score(memory: Memory) -> float:
            memory_terms = self._terms(memory.content + " " + " ".join(memory.tags))
            overlap = len(query_terms & memory_terms) / math.sqrt(
                max(1, len(query_terms)) * max(1, len(memory_terms))
            )
            age_days = max(0.0, (now - memory.created_at).total_seconds() / 86_400)
            recency = 1 / (1 + age_days / 30)
            return 0.7 * overlap + 0.2 * memory.importance + 0.1 * recency

        return sorted(candidates, key=score, reverse=True)[:limit]

    def forget(self, memory_id: str) -> bool:
        return self.store.remove(memory_id)

    def prune(self, *, below_importance: float, older_than_days: int | None = None) -> int:
        now = datetime.now(UTC)
        removed = 0
        for memory in self.store.list():
            age_days = (now - memory.created_at).total_seconds() / 86_400
            old_enough = older_than_days is None or age_days >= older_than_days
            if memory.importance < below_importance and old_enough:
                removed += int(self.store.remove(memory.id))
        return removed

    def as_tool(self, *, name: str = "memory") -> Tool:
        manager = self

        def manage_memory(
            action: str,
            content: str = "",
            query: str = "",
            memory_id: str = "",
            kind: MemoryKind = MemoryKind.SEMANTIC,
            importance: float = 0.5,
            limit: int = 5,
        ) -> ToolResult:
            """Remember, recall, or forget information for future turns."""
            if action == "remember":
                memory = manager.remember(content, kind=kind, importance=importance)
                return ToolResult.success(
                    f"Stored memory {memory.id}.", data=memory.model_dump(mode="json")
                )
            if action == "recall":
                matches = manager.recall(query, limit=limit)
                text = "\n".join(f"- {item.content}" for item in matches) or "No memories matched."
                return ToolResult.success(
                    text, data=[item.model_dump(mode="json") for item in matches]
                )
            if action == "forget":
                removed = manager.forget(memory_id)
                return ToolResult.success("Memory removed." if removed else "Memory not found.")
            return ToolResult.error("invalid_action", "Action must be remember, recall, or forget.")

        generated = Tool.from_callable(manage_memory, name=name)
        generated.parameters = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["remember", "recall", "forget"]},
                "content": {"type": "string"},
                "query": {"type": "string"},
                "memory_id": {"type": "string"},
                "kind": {"type": "string", "enum": [item.value for item in MemoryKind]},
                "importance": {"type": "number", "minimum": 0, "maximum": 1},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["action"],
            "additionalProperties": False,
        }
        return generated

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        return {match.group(0).lower() for match in cls._token_pattern.finditer(text)}
