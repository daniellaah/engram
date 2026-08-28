# Engram

A foundation for building a multi-model agent framework in Python. Dependencies and the
virtual environment are managed with `uv`.

The project supports OpenAI and DeepSeek V4 Flash through OpenAI-compatible endpoints.

## Requirements

- Python 3.12
- uv 0.9 or later

## Setup

```bash
uv sync
cp .env.example .env
```

Add the model, API key, and API URL to `.env`. The file is ignored by Git and must not be
committed.

## Model Providers

OpenAI:

```dotenv
LLM_API_KEY=your_openai_api_key
LLM_MODEL=gpt-5.6
LLM_BASE_URL=https://api.openai.com/v1
```

DeepSeek V4 Flash:

```dotenv
LLM_API_KEY=your_deepseek_api_key
LLM_MODEL=deepseek-v4-flash
LLM_BASE_URL=https://api.deepseek.com
```

The application code is the same for either provider:

```python
from engram import LLMClient

with LLMClient() as llm:
    response = llm.respond(
        "Write a quicksort implementation.",
        instructions="You are a helpful assistant that writes Python code.",
    )
```

`LLMClient` uses the Responses API, streams generated text to stdout, and returns the complete
text from `respond()`. Other Responses-compatible services can be selected with the same three
environment variables.

## Classic Agent Patterns

The project includes three foundational agent patterns. They all use `LLMClient` and the
Responses API:

- `ReActAgent` alternates between tool calls and observations within a bounded number of steps.
- `PlanAndSolveAgent` creates a validated JSON plan and executes it sequentially.
- `ReflectionAgent` generates, reviews, and refines a solution iteratively.

Run the examples:

```bash
uv run python examples/react.py
uv run python examples/plan_and_solve.py
uv run python examples/reflection.py
```

The ReAct search example requires the optional `SERPAPI_API_KEY` variable in `.env`. The other
two examples only require the model configuration.

## Common Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Add runtime or development dependencies:

```bash
uv add <package>
uv add --dev <package>
```

## Included Dependencies

Runtime dependencies provide the OpenAI SDK, environment loading, typed settings and models,
retry support, and structured logging. Development dependencies provide pytest, asynchronous
testing, HTTP mocking, Ruff, and mypy. Exact versions are recorded in `uv.lock`.
