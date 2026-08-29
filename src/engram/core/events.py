import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class EventType(StrEnum):
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    MODEL_STARTED = "model.started"
    MODEL_COMPLETED = "model.completed"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"


@dataclass(frozen=True, slots=True)
class RunEvent:
    type: EventType
    source: str
    data: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


EventHandler = Callable[[RunEvent], None]


class EventBus:
    """A synchronous observer list with no logging policy of its own."""

    def __init__(self, handlers: list[EventHandler] | None = None) -> None:
        self._handlers = list(handlers or [])

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        self._handlers.remove(handler)

    def emit(self, event: RunEvent) -> None:
        for handler in tuple(self._handlers):
            handler(event)


class TraceRecorder:
    """Collect sanitized events and write them as JSON Lines on request."""

    _sensitive_key = re.compile(
        r"(^|[_-])(api[_-]?key|authorization|password|secret|access[_-]?token|"
        r"refresh[_-]?token|bearer)($|[_-])",
        re.I,
    )

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def __call__(self, event: RunEvent) -> None:
        self.events.append(
            RunEvent(
                type=event.type,
                source=event.source,
                data=self._sanitize(event.data),
                timestamp=event.timestamp,
            )
        )

    def write_jsonl(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            for event in self.events:
                stream.write(json.dumps(asdict(event), default=str, ensure_ascii=False) + "\n")
        return destination

    @classmethod
    def _sanitize(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): (
                    "[redacted]" if cls._sensitive_key.search(str(key)) else cls._sanitize(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list | tuple):
            return [cls._sanitize(item) for item in value]
        return value
