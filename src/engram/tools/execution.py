from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from engram.tools.registry import ToolRegistry
from engram.tools.result import ToolResult, ToolStatus

ArgumentBuilder = Callable[[ToolResult | None], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    name: str
    arguments: Mapping[str, Any] | ArgumentBuilder


class ToolPipeline:
    """Execute an explicit sequence of tools, passing results between steps."""

    def __init__(self, registry: ToolRegistry, steps: Sequence[ToolInvocation]) -> None:
        self.registry = registry
        self.steps = tuple(steps)

    def run(self) -> list[ToolResult]:
        results: list[ToolResult] = []
        previous: ToolResult | None = None
        for step in self.steps:
            arguments = step.arguments(previous) if callable(step.arguments) else step.arguments
            previous = self.registry.invoke(step.name, arguments)
            results.append(previous)
            if previous.status is ToolStatus.ERROR:
                break
        return results
