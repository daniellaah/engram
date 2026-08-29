import asyncio
import inspect
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import monotonic
from typing import Any

from engram.tools.base import Tool, ToolFunction, ToolValidationError
from engram.tools.result import ToolResult


class ToolRegistry:
    """Register, describe, and safely invoke tools."""

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._lock = Lock()
        for item in tools or ():
            self.register_tool(item)

    def register_tool(self, tool: Tool, *, replace: bool = False) -> None:
        with self._lock:
            if tool.name in self._tools and not replace:
                raise ValueError(f"Tool already registered: {tool.name}")
            self._tools[tool.name] = tool

    def register(
        self,
        name: str,
        description: str,
        function: ToolFunction,
        *,
        replace: bool = False,
    ) -> Tool:
        registered = Tool(name, description, function)
        self.register_tool(registered, replace=replace)
        return registered

    def register_function(
        self,
        function: ToolFunction,
        *,
        name: str | None = None,
        description: str | None = None,
        replace: bool = False,
    ) -> Tool:
        registered = Tool.from_callable(function, name=name, description=description)
        self.register_tool(registered, replace=replace)
        return registered

    def unregister(self, name: str) -> Tool:
        with self._lock:
            try:
                return self._tools.pop(name)
            except KeyError as error:
                raise KeyError(f"Unknown tool: {name}") from error

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [item.to_schema() for item in self._tools.values()]

    def descriptions(self) -> str:
        return "\n".join(f"- {item.name}: {item.description}" for item in self._tools.values())

    def invoke(self, name: str, arguments: Mapping[str, Any] | str) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult.error("unknown_tool", f"Unknown tool: {name}")
        started = monotonic()
        try:
            result = tool.invoke(arguments)
        except ToolValidationError as error:
            result = ToolResult.error("invalid_arguments", str(error))
        except Exception as error:
            result = ToolResult.error("execution_failed", f"{type(error).__name__}: {error}")
        return _with_duration(result, started)

    def execute(self, name: str, argument: str) -> str:
        """Execute a legacy single-string tool and return its display text."""
        return self.invoke(name, argument).content

    async def ainvoke(self, name: str, arguments: Mapping[str, Any] | str) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult.error("unknown_tool", f"Unknown tool: {name}")
        started = monotonic()
        try:
            if inspect.iscoroutinefunction(tool.function):
                result = await tool.ainvoke(arguments)
            else:
                result = await asyncio.to_thread(tool.invoke, arguments)
        except ToolValidationError as error:
            result = ToolResult.error("invalid_arguments", str(error))
        except Exception as error:
            result = ToolResult.error("execution_failed", f"{type(error).__name__}: {error}")
        return _with_duration(result, started)

    def invoke_many(
        self,
        calls: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        max_workers: int | None = None,
    ) -> list[ToolResult]:
        if not calls:
            return []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.invoke, name, arguments) for name, arguments in calls]
            return [future.result() for future in futures]

    async def ainvoke_many(
        self,
        calls: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        max_concurrency: int = 8,
    ) -> list[ToolResult]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def guarded(name: str, arguments: Mapping[str, Any]) -> ToolResult:
            async with semaphore:
                return await self.ainvoke(name, arguments)

        return await asyncio.gather(*(guarded(name, arguments) for name, arguments in calls))

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def _with_duration(result: ToolResult, started: float) -> ToolResult:
    metadata = dict(result.metadata)
    metadata["duration_ms"] = round((monotonic() - started) * 1000, 3)
    return ToolResult(
        status=result.status,
        content=result.content,
        data=result.data,
        error_code=result.error_code,
        metadata=metadata,
    )
