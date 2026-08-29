from typing import cast

import pytest

from engram.tools import (
    Tool,
    ToolInvocation,
    ToolPipeline,
    ToolRegistry,
    ToolStatus,
    calculate,
    tool,
)


def add(a: int, b: int = 1) -> int:
    """Add two integers."""
    return a + b


def test_typed_tool_builds_schema_and_validates() -> None:
    created = Tool.from_callable(add)
    schema = created.to_schema()

    assert schema["name"] == "add"
    assert schema["parameters"]["properties"]["a"]["type"] == "integer"
    assert created.invoke({"a": "2", "b": 3}).content == "5"


def test_registry_returns_structured_errors() -> None:
    registry = ToolRegistry([Tool.from_callable(add)])

    missing = registry.invoke("missing", {})
    invalid = registry.invoke("add", {})

    assert missing.status is ToolStatus.ERROR
    assert missing.error_code == "unknown_tool"
    assert invalid.error_code == "invalid_arguments"


@pytest.mark.asyncio
async def test_async_tool_execution() -> None:
    async def uppercase(value: str) -> str:
        """Convert text to uppercase."""
        return value.upper()

    registry = ToolRegistry([Tool.from_callable(uppercase)])
    result = await registry.ainvoke("uppercase", {"value": "hello"})

    assert result.content == "HELLO"
    assert "duration_ms" in result.metadata


def test_pipeline_passes_previous_result() -> None:
    registry = ToolRegistry([Tool.from_callable(add)])

    def use_previous(previous: object) -> dict[str, int]:
        assert hasattr(previous, "content")
        return {"a": int(previous.content), "b": 4}

    pipeline = ToolPipeline(
        registry,
        [
            ToolInvocation("add", {"a": 2, "b": 3}),
            ToolInvocation("add", use_previous),
        ],
    )

    assert [result.content for result in pipeline.run()] == ["5", "9"]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2 + 3 * 4", "14"), ("sqrt(16) + 2", "6"), ("pi * 2", str(3.141592653589793 * 2))],
)
def test_safe_calculator(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


def test_safe_calculator_rejects_code() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        calculate("__import__('os').getcwd()")


def test_tool_decorator() -> None:
    def greeting_function(name: str) -> str:
        """Greet a person."""
        return f"Hello, {name}."

    created = cast(Tool, tool(greeting_function, name="greet"))
    assert created.invoke({"name": "Ada"}).content == "Hello, Ada."
