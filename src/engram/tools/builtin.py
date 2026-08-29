import ast
import json
import math
import operator
import os
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from engram.tools.base import Tool

_BINARY_OPERATORS: Mapping[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: Mapping[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: Mapping[str, Callable[..., float]] = {
    "abs": abs,
    "ceil": math.ceil,
    "cos": math.cos,
    "floor": math.floor,
    "log": math.log,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}
_CONSTANTS = {"e": math.e, "pi": math.pi}


def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression."""
    tree = ast.parse(expression, mode="eval")
    value = _evaluate_node(tree.body)
    if not math.isfinite(value):
        raise ValueError("The expression produced a non-finite result.")
    return str(int(value)) if value.is_integer() else str(value)


CalculatorTool = Tool(
    name="calculator",
    description="Evaluate arithmetic with common mathematical functions and constants.",
    function=calculate,
)


def web_search(query: str, api_key: str | None = None, timeout: float = 10) -> str:
    """Search the web through SerpAPI and return a compact text summary."""
    load_dotenv()
    key = api_key or os.getenv("SERPAPI_API_KEY")
    if not key:
        raise ValueError("Missing SerpAPI key. Set SERPAPI_API_KEY or pass api_key.")
    if timeout <= 0:
        raise ValueError("Search timeout must be greater than zero.")

    query_string = urlencode(
        {"engine": "google", "q": query, "api_key": key, "gl": "us", "hl": "en"}
    )
    request = Request(
        f"https://serpapi.com/search.json?{query_string}",
        headers={"Accept": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload: object = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("SerpAPI returned an invalid response.")
    return _format_search_results(payload, query)


def _evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return float(_FUNCTIONS[node.func.id](*(_evaluate_node(arg) for arg in node.args)))
    raise ValueError("Expression contains an unsupported operation.")


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
