import pytest

from engram import ModelResponse, PlanAndSolveAgent
from engram.agents.plan_and_solve import Planner
from tests.fakes import FakeModel


def test_plan_and_solve_runs_every_step() -> None:
    model = FakeModel(
        [
            ModelResponse(text='["Compute the value", "Return the answer"]', model="fake"),
            ModelResponse(text="The computed value is 96.", model="fake"),
            ModelResponse(text="The final answer is 96.", model="fake"),
        ]
    )
    agent = PlanAndSolveAgent(model)

    assert agent.run("Solve the problem.") == "The final answer is 96."
    assert agent.last_plan == ("Compute the value", "Return the answer")
    assert "Step 1:\\nThe computed value is 96." in repr(model.inputs[2])


def test_planner_accepts_fenced_json() -> None:
    assert Planner._parse_plan('```json\n["First step", "Second step"]\n```') == [
        "First step",
        "Second step",
    ]


@pytest.mark.parametrize("response", ["not json", "[]", '["valid", ""]'])
def test_planner_rejects_invalid_plans(response: str) -> None:
    with pytest.raises(ValueError):
        Planner._parse_plan(response)
