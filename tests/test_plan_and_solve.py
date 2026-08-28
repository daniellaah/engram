from typing import cast
from unittest.mock import MagicMock

import pytest

from engram.agents.plan_and_solve import PlanAndSolveAgent, Planner
from engram.llm import LLMClient


def test_plan_and_solve_runs_every_step() -> None:
    mock_llm = MagicMock()
    mock_llm.respond.side_effect = [
        '["Compute the value", "Return the answer"]',
        "The computed value is 96.",
        "The final answer is 96.",
    ]
    agent = PlanAndSolveAgent(cast(LLMClient, mock_llm))

    assert agent.run("Solve the problem.") == "The final answer is 96."
    assert mock_llm.respond.call_count == 3
    final_prompt = mock_llm.respond.call_args_list[2].args[0]
    assert "Step 1:\nThe computed value is 96." in final_prompt


def test_planner_accepts_fenced_json() -> None:
    response = '```json\n["First step", "Second step"]\n```'

    assert Planner._parse_plan(response) == ["First step", "Second step"]


@pytest.mark.parametrize("response", ["not json", "[]", '["valid", ""]'])
def test_planner_rejects_invalid_plans(response: str) -> None:
    with pytest.raises(ValueError):
        Planner._parse_plan(response)
