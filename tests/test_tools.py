from engram.tools import ToolRegistry, _format_search_results


def test_registry_registers_describes_and_executes_tool() -> None:
    registry = ToolRegistry()
    registry.register("uppercase", "Convert text to uppercase.", str.upper)

    assert registry.execute("uppercase", "hello") == "HELLO"
    assert registry.descriptions() == "- uppercase: Convert text to uppercase."


def test_registry_reports_unknown_tool() -> None:
    registry = ToolRegistry()

    assert registry.execute("missing", "input") == "Unknown tool: missing"


def test_search_formatter_prefers_answer_box() -> None:
    payload = {
        "answer_box": {"answer": "42"},
        "organic_results": [{"title": "Ignored", "snippet": "Ignored"}],
    }

    assert _format_search_results(payload, "answer") == "42"


def test_search_formatter_compacts_organic_results() -> None:
    payload = {
        "organic_results": [
            {"title": "First", "snippet": "First result"},
            {"title": "Second", "snippet": "Second result"},
        ]
    }

    assert _format_search_results(payload, "query") == (
        "[1] First\nFirst result\n\n[2] Second\nSecond result"
    )
