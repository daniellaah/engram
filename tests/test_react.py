import pytest

from engram import EventBus, ModelResponse, ReActAgent, Tool, ToolCall, TraceRecorder
from tests.fakes import FakeModel


def lookup(topic: str) -> str:
    """Look up a topic."""
    return f"Result for {topic}"


def test_react_agent_uses_native_tool_call_and_finishes() -> None:
    model = FakeModel(
        [
            ModelResponse(
                text="",
                model="fake",
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="lookup",
                        arguments={"topic": "weather"},
                        raw_arguments='{"topic":"weather"}',
                    ),
                ),
            ),
            ModelResponse(text="It is sunny.", model="fake"),
        ]
    )
    recorder = TraceRecorder()
    agent = ReActAgent(model, [Tool.from_callable(lookup)], events=EventBus([recorder]))

    assert agent.run("Check the weather.") == "It is sunny."
    assert [message.role for message in agent.get_history()] == ["user", "assistant"]
    assert isinstance(model.inputs[1], list)
    assert any(item.get("type") == "function_call_output" for item in model.inputs[1])
    assert [event.type.value for event in recorder.events].count("tool.completed") == 1


def test_react_agent_enforces_step_limit() -> None:
    call = ToolCall(id="call", name="lookup", arguments={"topic": "x"})
    model = FakeModel([ModelResponse(text="", model="fake", tool_calls=(call,))])
    agent = ReActAgent(model, [Tool.from_callable(lookup)], max_steps=1)

    with pytest.raises(RuntimeError, match="step limit"):
        agent.run("Never finish.")
