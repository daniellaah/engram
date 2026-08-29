import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engram.core.message import Message


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    agent_name: str
    messages: tuple[Message, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class JsonSessionStore:
    """Persist portable sessions as atomic JSON files."""

    _valid_id = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def save(self, session: Session) -> Path:
        path = self._path(session.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "id": session.id,
            "agent_name": session.agent_name,
            "messages": [message.model_dump(mode="json") for message in session.messages],
            "metadata": session.metadata,
            "updated_at": session.updated_at.isoformat(),
        }
        handle, temporary_name = tempfile.mkstemp(prefix=f".{session.id}.", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return path

    def load(self, session_id: str) -> Session:
        with self._path(session_id).open(encoding="utf-8") as stream:
            payload: dict[str, Any] = json.load(stream)
        if payload.get("version") != 1:
            raise ValueError("Unsupported session format version.")
        return Session(
            id=str(payload["id"]),
            agent_name=str(payload["agent_name"]),
            messages=tuple(Message.model_validate(item) for item in payload["messages"]),
            metadata=dict(payload.get("metadata", {})),
            updated_at=datetime.fromisoformat(str(payload["updated_at"])),
        )

    def list(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(path.stem for path in self.directory.glob("*.json") if path.is_file())

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def _path(self, session_id: str) -> Path:
        if not self._valid_id.fullmatch(session_id):
            raise ValueError("Session IDs may contain only letters, numbers, '.', '_', and '-'.")
        return self.directory / f"{session_id}.json"
