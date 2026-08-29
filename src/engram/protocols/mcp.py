from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from engram.tools import Tool, ToolRegistry, ToolResult


class MCPSession(Protocol):
    """The subset of an MCP client session needed for tool adaptation."""

    async def list_tools(self) -> Any: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class MCPToolProvider:
    """Adapt tools from an already connected MCP session into Engram tools."""

    def __init__(self, session: MCPSession, tools: Sequence[Tool]) -> None:
        self.session = session
        self.tools = tuple(tools)

    @classmethod
    async def discover(cls, session: MCPSession) -> "MCPToolProvider":
        response = await session.list_tools()
        remote_tools = _value(response, "tools", response)
        if not isinstance(remote_tools, Sequence):
            raise ValueError("MCP list_tools response does not contain a tool sequence.")
        adapted: list[Tool] = []
        for remote in remote_tools:
            name = _value(remote, "name")
            if not isinstance(name, str) or not name:
                raise ValueError("MCP tool is missing a valid name.")
            description = _value(remote, "description", f"Call remote MCP tool {name}.")
            schema = _value(remote, "inputSchema", None)
            if schema is None:
                schema = _value(remote, "input_schema", None)
            if not isinstance(schema, Mapping):
                schema = {"type": "object", "properties": {}}

            async def call_remote(_name: str = name, **arguments: Any) -> ToolResult:
                result = await session.call_tool(_name, dict(arguments))
                return _convert_result(result)

            adapted.append(
                Tool(
                    name=name,
                    description=str(description),
                    function=call_remote,
                    parameters=dict(schema),
                )
            )
        return cls(session, adapted)

    def register(self, registry: ToolRegistry, *, replace: bool = False) -> None:
        for tool in self.tools:
            registry.register_tool(tool, replace=replace)


def _convert_result(result: Any) -> ToolResult:
    is_error = bool(_value(result, "isError", _value(result, "is_error", False)))
    content = _value(result, "content", result)
    if isinstance(content, Sequence) and not isinstance(content, str | bytes):
        parts: list[str] = []
        for block in content:
            text = _value(block, "text", None)
            if text is not None:
                parts.append(str(text))
        rendered = "\n".join(parts) if parts else str(content)
    else:
        rendered = str(content)
    if is_error:
        return ToolResult.error("mcp_tool_error", rendered)
    return ToolResult.success(rendered)


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)
