from pathlib import Path

from engram.memory import JsonMemoryStore, MemoryKind, MemoryManager


def test_memory_recall_uses_content_and_importance() -> None:
    manager = MemoryManager()
    manager.remember("The project uses Python.", importance=0.9, tags=["project"])
    manager.remember("The office has coffee.", importance=0.2)

    matches = manager.recall("Which language does the project use?", limit=1)

    assert matches[0].content == "The project uses Python."


def test_memory_tool_supports_lifecycle() -> None:
    manager = MemoryManager()
    tool = manager.as_tool()

    stored = tool.invoke(
        {
            "action": "remember",
            "content": "Ada prefers concise reports.",
            "kind": MemoryKind.EPISODIC.value,
            "importance": 0.8,
        }
    )
    recalled = tool.invoke({"action": "recall", "query": "Ada reports", "limit": 3})

    assert stored.status.value == "success"
    assert "concise reports" in recalled.content


def test_json_memory_store_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    store = JsonMemoryStore(path)
    manager = MemoryManager(store)
    manager.remember("Persistent fact.")
    store.save()

    restored = JsonMemoryStore(path)

    assert restored.list()[0].content == "Persistent fact."
