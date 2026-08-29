# Engram

Engram is a small, typed Python framework for building understandable AI agents. It keeps the
model loop, tool execution, context selection, memory, retrieval, and evaluation visible instead
of hiding them behind a workflow engine.

The framework uses the OpenAI Responses API and also works with services that implement the same
endpoint. Core features have no database, vector-store, web-framework, or training dependency.

## Features

- A normalized sync and async model client with streaming, tool calls, continuation items, and
  token usage.
- `SimpleAgent`, `ReActAgent`, `PlanAndSolveAgent`, `ReflectionAgent`, and the general
  `ToolAgent`.
- Native function calling with typed argument validation, JSON Schema generation, structured tool
  results, parallel execution, and explicit pipelines.
- Token-aware history and a priority-based context builder for retrieved information.
- Atomic JSON session persistence and opt-in event tracing with secret redaction.
- Working, episodic, semantic, and perceptual memory records with a backend protocol.
- A compact RAG pipeline with paragraph-aware chunking, lexical retrieval, and a pluggable vector
  index.
- Adaptation of tools discovered from an existing MCP client session.
- Dataset-independent evaluation, deterministic metrics, and an optional model judge.
- Agent composition through `agent_as_tool()`.

## Requirements

- Python 3.12 or later
- `uv` 0.9 or later

## Installation

```bash
uv sync
cp .env.example .env
```

Configure any Responses-compatible endpoint:

```dotenv
LLM_API_KEY=your_api_key
LLM_MODEL=your_model
LLM_BASE_URL=https://api.openai.com/v1
LLM_TIMEOUT=60
```

## Quick Start

```python
from engram import LLMClient, ReActAgent, Tool


def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"The weather in {city} is sunny."


with LLMClient() as llm:
    agent = ReActAgent(llm, [Tool.from_callable(get_weather)])
    answer = agent.run("Should I take an umbrella in Seattle?")

print(answer)
```

`Tool.from_callable()` reads the function signature and type hints to create the tool schema. Tool
arguments are validated before execution, and exceptions become structured observations that the
agent can inspect.

## Agent Patterns

All agents share the same model interface and conversation history:

```python
from engram import LLMClient, PlanAndSolveAgent, ReflectionAgent, SimpleAgent

with LLMClient() as llm:
    assistant = SimpleAgent(llm)
    planned = PlanAndSolveAgent(llm)
    reviewer = ReflectionAgent(llm, max_iterations=2)

    print(assistant.run("Explain dependency inversion in one paragraph."))
    print(planned.run("Create a three-day migration plan."))
    print(reviewer.run("Draft a robust CSV parsing function."))
```

- `SimpleAgent` is the default conversational agent and can use tools when provided.
- `ReActAgent` uses the same native tool loop with instructions tuned for deliberate
  reason-act-observe iteration.
- `PlanAndSolveAgent` requires a valid JSON plan and executes each step in order.
- `ReflectionAgent` keeps a typed execution/review trajectory and stops early when no improvement
  is needed.
- `ToolAgent` is the reusable native function-calling implementation behind the first two agents.

## Tools

Tools can be created from callables or constructed explicitly:

```python
from engram import Tool, ToolRegistry


def convert_temperature(value: float, to_unit: str = "celsius") -> float:
    """Convert a temperature to Celsius or Fahrenheit."""
    if to_unit == "fahrenheit":
        return value * 9 / 5 + 32
    return (value - 32) * 5 / 9


registry = ToolRegistry([Tool.from_callable(convert_temperature)])
result = registry.invoke(
    "convert_temperature",
    {"value": 20, "to_unit": "fahrenheit"},
)
print(result.content)
```

`ToolResult` separates model-readable `content` from structured `data`, status, error code, and
runtime metadata. Use `ainvoke()` or `ainvoke_many()` for async tools. `ToolPipeline` handles an
explicit sequence whose later arguments depend on earlier results.

## Context, Memory, and Retrieval

The conversation history and external context remain separate. This prevents retrieved text from
silently becoming durable chat history.

```python
from engram import Document, KnowledgeBase, LLMClient, SimpleAgent

knowledge = KnowledgeBase()
knowledge.add(
    [
        Document(content="Engram uses native function calls for local tools."),
        Document(content="Conversation history is selected with a token budget."),
    ]
)

with LLMClient() as llm:
    agent = SimpleAgent(llm, tools=[knowledge.as_tool()])
    print(agent.run("How are local tools called?"))
```

`MemoryManager` stores `working`, `episodic`, `semantic`, and `perceptual` records. Its default
store is in memory; `JsonMemoryStore` provides explicit local persistence. `KnowledgeBase` uses a
dependency-free lexical index by default. Pass `VectorIndex` an object that implements
`embed(texts)` to use any embedding service without coupling Engram to that provider.

For pre-retrieved information, pass `ContextSource` objects to
`ToolAgent.run_with_context()`. Sources are selected by priority within a dedicated part of the
input budget.

## Sessions and Observability

Persistence and tracing are opt-in:

```python
from engram import EventBus, JsonSessionStore, TraceRecorder

recorder = TraceRecorder()
agent.events.subscribe(recorder)

store = JsonSessionStore(".engram/sessions")
session = agent.save_session(store, "demo")
agent.load_session(store, session.id)

recorder.write_jsonl(".engram/traces/demo.jsonl")
```

Session files are written atomically. Traces redact data under keys that look like credentials and
never write to disk unless `write_jsonl()` is called.

## MCP Tools

Engram does not implement a competing MCP transport. It adapts tools from a connected client
session supplied by the official MCP SDK or another compatible client:

```python
from engram import ReActAgent, ToolRegistry
from engram.protocols import MCPToolProvider

# `session` is an already connected MCP client session.
provider = await MCPToolProvider.discover(session)
registry = ToolRegistry()
provider.register(registry)

agent = ReActAgent(llm, registry)
answer = await agent.arun("Use the remote data source to answer the question.")
```

The caller remains responsible for opening and closing the MCP session. This keeps transport,
authentication, and process lifetime under application control.

## Evaluation

```python
from engram.evaluation import EvaluationCase, Evaluator, exact_match

cases = [
    EvaluationCase("2 + 2", "4"),
    EvaluationCase("3 + 5", "8"),
]
report = Evaluator({"exact_match": exact_match}).evaluate(agent.run, cases)
print(report.averages)
```

Metrics are regular callables. `LLMJudge` adds rubric-based scoring when deterministic comparison
is insufficient.

## Project Structure

```text
src/engram/
|-- core/          # Messages, model protocol, Agent base, sessions, events, configuration
|-- agents/        # Simple, tool-loop, ReAct, planning, and reflection agents
|-- tools/         # Tool schemas, validation, registry, results, pipelines, built-ins
|-- context/       # Token estimation, history selection, external context construction
|-- memory/        # Typed memories, stores, retrieval, and memory tool adapter
|-- rag/           # Documents, chunking, lexical/vector indexes, retrieval tool adapter
|-- protocols/     # MCP tool discovery adapter
|-- evaluation/    # Cases, metrics, reports, and model judge
`-- llm.py         # Responses API client
```

See [Framework Design](docs/design.md) for component boundaries, execution flows, and design
decisions.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

The test suite uses deterministic fake model clients and does not require network access.

## Design Boundaries

Engram deliberately excludes model training, benchmark datasets, hosted vector databases, unsafe
shell execution, and web-server concerns. Those are applications or integrations, not agent-loop
primitives. They can be added through model clients, tools, memory stores, search indexes, MCP, and
evaluation metrics without expanding the core abstraction surface.
