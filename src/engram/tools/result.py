import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ToolStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A tool result that is useful to both models and application code."""

    status: ToolStatus
    content: str
    data: Any = None
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        content: str,
        *,
        data: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(ToolStatus.SUCCESS, content, data, metadata=metadata or {})

    @classmethod
    def partial(
        cls,
        content: str,
        *,
        data: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(ToolStatus.PARTIAL, content, data, metadata=metadata or {})

    @classmethod
    def error(
        cls,
        code: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(ToolStatus.ERROR, message, error_code=code, metadata=metadata or {})

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "status": self.status.value,
            "content": self.content,
            "data": self.data,
        }
        if self.error_code is not None:
            value["error"] = {"code": self.error_code, "message": self.content}
        if self.metadata:
            value["metadata"] = self.metadata
        return value

    def to_model_output(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)
