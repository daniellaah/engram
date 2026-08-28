from typing import cast
from unittest.mock import MagicMock

from engram.agents.reflection import STOP_MARKER, ReflectionAgent
from engram.llm import LLMClient


def test_reflection_agent_refines_then_stops() -> None:
    mock_llm = MagicMock()
    mock_llm.respond.side_effect = [
        "initial solution",
        "Handle the empty input case.",
        "refined solution",
        STOP_MARKER,
    ]
    agent = ReflectionAgent(cast(LLMClient, mock_llm), max_iterations=2)

    assert agent.run("Create a solution.") == "refined solution"
    assert [record.kind for record in agent.memory.records] == [
        "execution",
        "reflection",
        "execution",
        "reflection",
    ]
    assert agent.memory.last_execution() == "refined solution"
    assert "Handle the empty input case." in agent.memory.trajectory()


def test_reflection_agent_can_skip_review() -> None:
    mock_llm = MagicMock()
    mock_llm.respond.return_value = "initial solution"
    agent = ReflectionAgent(cast(LLMClient, mock_llm), max_iterations=0)

    assert agent.run("Create a solution.") == "initial solution"
    mock_llm.respond.assert_called_once()
