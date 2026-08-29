from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MessageRole = Literal["user", "assistant", "system", "developer"]


class Message(BaseModel):
    """A provider-neutral conversation message."""

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_input_item(self) -> dict[str, Any]:
        """Return the minimal Responses-compatible input item."""
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
