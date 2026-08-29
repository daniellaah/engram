from engram import ModelResponse, ReflectionAgent
from engram.agents.reflection import STOP_MARKER
from tests.fakes import FakeModel


def test_reflection_agent_refines_then_stops() -> None:
    model = FakeModel(
        [
            ModelResponse(text="initial solution", model="fake"),
            ModelResponse(text="Handle the empty input case.", model="fake"),
            ModelResponse(text="refined solution", model="fake"),
            ModelResponse(text=STOP_MARKER, model="fake"),
        ]
    )
    agent = ReflectionAgent(model, max_iterations=2)

    assert agent.run("Create a solution.") == "refined solution"
    assert [record.kind for record in agent.memory.records] == [
        "execution",
        "reflection",
        "execution",
        "reflection",
    ]


def test_reflection_agent_can_skip_review() -> None:
    model = FakeModel([ModelResponse(text="initial solution", model="fake")])
    agent = ReflectionAgent(model, max_iterations=0)

    assert agent.run("Create a solution.") == "initial solution"
