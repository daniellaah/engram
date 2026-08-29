from pathlib import Path

from engram import (
    ContextBuilder,
    ContextSource,
    HistoryManager,
    JsonSessionStore,
    Message,
    ModelResponse,
    SimpleAgent,
    agent_as_tool,
)
from engram.core.session import Session
from tests.fakes import FakeModel


def test_context_builder_prioritizes_sources_and_recent_history() -> None:
    history = HistoryManager(max_messages=10)
    history.extend(
        [
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="new question"),
        ]
    )
    builder = ContextBuilder(token_budget=80)
    result = builder.build(
        history,
        sources=[
            ContextSource("low", "less relevant", priority=1),
            ContextSource("high", "very relevant", priority=10),
        ],
    )

    assert result[0]["role"] == "developer"
    assert "very relevant" in str(result[0]["content"])
    assert result[-1]["content"] == "new question"


def test_history_can_be_compacted() -> None:
    history = HistoryManager(
        [Message(role="user", content="one"), Message(role="assistant", content="two")]
    )

    history.compact("The user counted to two.", retain_messages=1)

    assert history.messages[0].metadata["kind"] == "summary"
    assert history.messages[-1].content == "two"


def test_session_round_trip(tmp_path: Path) -> None:
    store = JsonSessionStore(tmp_path / "sessions")
    session = Session(
        id="demo-1",
        agent_name="assistant",
        messages=(Message(role="user", content="hello"),),
        metadata={"topic": "test"},
    )

    path = store.save(session)
    loaded = store.load("demo-1")

    assert path.exists()
    assert loaded.messages[0].content == "hello"
    assert store.list() == ["demo-1"]
    assert store.delete("demo-1") is True


def test_simple_agent_and_agent_tool() -> None:
    model = FakeModel([ModelResponse(text="primary answer", model="fake")])
    agent = SimpleAgent(model, name="researcher")

    wrapped = agent_as_tool(agent)

    assert wrapped.invoke({"task": "research this"}).content == "primary answer"
