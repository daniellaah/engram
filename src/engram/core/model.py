from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

type ModelInput = str | list[dict[str, Any]]
type ToolSchema = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A validated request from a model to execute a local tool."""

    id: str
    name: str
    arguments: Mapping[str, Any]
    raw_arguments: str = "{}"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized text, tool calls, usage, and continuation items."""

    text: str
    model: str
    response_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    output_items: tuple[dict[str, Any], ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)


class ModelClient(Protocol):
    """The small model interface consumed by all Engram agents."""

    model: str

    def complete(
        self,
        input_data: ModelInput,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        parallel_tool_calls: bool = True,
        max_output_tokens: int | None = None,
    ) -> ModelResponse: ...

    async def acomplete(
        self,
        input_data: ModelInput,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        parallel_tool_calls: bool = True,
        max_output_tokens: int | None = None,
    ) -> ModelResponse: ...
