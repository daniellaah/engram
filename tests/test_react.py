from typing import cast
from unittest.mock import MagicMock

import pytest

from engram.agents.react import ReActAgent
from engram.llm import LLMClient
from engram.tools import ToolRegistry


def test_react_agent_uses_tool_and_finishes() -> None:
    mock_llm = MagicMock()
    mock_llm.respond.side_effect = [
        "Rationale: I need the data.\nAction: lookup[weather]",
        "Rationale: I have enough information.\nAction: Finish[It is sunny.]",
    ]
    tools = ToolRegistry()
    tools.register("lookup", "Look up a value.", lambda value: f"Result for {value}")
    agent = ReActAgent(cast(LLMClient, mock_llm), tools)

    assert agent.run("Check the weather.") == "It is sunny."
    assert mock_llm.respond.call_count == 2
    second_prompt = mock_llm.respond.call_args_list[1].args[0]
    assert "Observation: Result for weather" in second_prompt


def test_react_agent_rejects_invalid_action() -> None:
    mock_llm = MagicMock()
    mock_llm.respond.return_value = "No valid action"
    agent = ReActAgent(cast(LLMClient, mock_llm), ToolRegistry())

    with pytest.raises(ValueError, match="valid Action"):
        agent.run("Do something.")


def test_react_agent_enforces_step_limit() -> None:
    mock_llm = MagicMock()
    mock_llm.respond.return_value = "Rationale: Keep going.\nAction: lookup[value]"
    tools = ToolRegistry()
    tools.register("lookup", "Look up a value.", lambda value: value)
    agent = ReActAgent(cast(LLMClient, mock_llm), tools, max_steps=1)

    with pytest.raises(RuntimeError, match="step limit"):
        agent.run("Never finish.")
