import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

ToolFunction = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    function: ToolFunction


class ToolRegistry:
    """Store and execute simple single-input tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, function: ToolFunction) -> None:
        if not name.strip():
            raise ValueError("Tool name cannot be empty.")
        if not description.strip():
            raise ValueError("Tool description cannot be empty.")
        self._tools[name] = Tool(name=name, description=description, function=function)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, argument: str) -> str:
        tool = self.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        return tool.function(argument)

    def descriptions(self) -> str:
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self._tools.values())


def web_search(query: str, api_key: str | None = None, timeout: float = 10) -> str:
    """Search the web through SerpAPI and return a compact text summary."""
    load_dotenv()
    key = api_key or os.getenv("SERPAPI_API_KEY")
    if not key:
        raise ValueError("Missing SerpAPI key. Set SERPAPI_API_KEY or pass api_key.")
    if timeout <= 0:
        raise ValueError("Search timeout must be greater than zero.")

    query_string = urlencode(
        {
            "engine": "google",
            "q": query,
            "api_key": key,
            "gl": "us",
            "hl": "en",
        }
    )
    request = Request(
        f"https://serpapi.com/search.json?{query_string}",
        headers={"Accept": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload: object = json.load(response)

    if not isinstance(payload, dict):
        raise ValueError("SerpAPI returned an invalid response.")
    return _format_search_results(cast(dict[str, Any], payload), query)


def _format_search_results(payload: Mapping[str, Any], query: str) -> str:
    answer_box = payload.get("answer_box")
    if isinstance(answer_box, Mapping):
        for key in ("answer", "snippet", "result"):
            value = answer_box.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    knowledge_graph = payload.get("knowledge_graph")
    if isinstance(knowledge_graph, Mapping):
        description = knowledge_graph.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()

    organic_results = payload.get("organic_results")
    if isinstance(organic_results, list):
        summaries: list[str] = []
        for position, result in enumerate(organic_results[:3], start=1):
            if not isinstance(result, Mapping):
                continue
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            summaries.append(f"[{position}] {title}\n{snippet}".strip())
        if summaries:
            return "\n\n".join(summaries)

    return f"No search results found for: {query}"
