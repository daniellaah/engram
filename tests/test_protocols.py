from typing import Any

import pytest

from engram.protocols import MCPToolProvider
from engram.tools import ToolRegistry


class FakeSession:
    async def list_tools(self) -> Any:
        return {
            "tools": [
                {
                    "name": "remote_add",
                    "description": "Add values remotely.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                }
            ]
        }

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        assert name == "remote_add"
        return {"content": [{"text": str(arguments["a"] + arguments["b"])}]}


@pytest.mark.asyncio
async def test_mcp_provider_discovers_and_executes_tools() -> None:
    provider = await MCPToolProvider.discover(FakeSession())
    registry = ToolRegistry()
    provider.register(registry)

    result = await registry.ainvoke("remote_add", {"a": 2, "b": 3})

    assert result.content == "5"
    assert registry.schemas()[0]["parameters"]["required"] == ["a", "b"]
