# Engram

一个使用 Python 构建多模型 Agent 框架的基础项目，依赖与虚拟环境由 `uv` 管理。
目前支持 OpenAI，以及通过 OpenAI 兼容接口调用 DeepSeek V4 Flash。

## 环境要求

- Python 3.12
- uv 0.9 或更高版本

## 初始化

```bash
uv sync
cp .env.example .env
```

随后在 `.env` 中填写模型、API Key 和 API 地址。`.env` 已被 Git 忽略，不要提交密钥。

## 模型提供商

OpenAI：

```dotenv
LLM_API_KEY=your_openai_api_key
LLM_MODEL=gpt-5.6
LLM_BASE_URL=https://api.openai.com/v1
```

DeepSeek V4 Flash：

```dotenv
LLM_API_KEY=your_deepseek_api_key
LLM_MODEL=deepseek-v4-flash
LLM_BASE_URL=https://api.deepseek.com
```

代码无需随提供商改变：

```python
from engram import LLMClient

with LLMClient() as llm:
    response = llm.respond(
        "Write a quicksort implementation.",
        instructions="You are a helpful assistant that writes Python code.",
    )
```

`LLMClient` 使用 Responses API，并在生成过程中流式打印文本；`respond()` 同时返回完整文本。
通过修改这三个通用变量，也可以接入其他兼容 Responses API 的服务。

## 经典 Agent 范式

项目包含第四章中的三个核心范式，并统一使用 `LLMClient` 调用 Responses API：

- `ReActAgent`：在工具调用和观察结果之间循环，达到步数上限时停止。
- `PlanAndSolveAgent`：先生成经过校验的 JSON 计划，再逐步执行。
- `ReflectionAgent`：生成结果、审查问题并按反馈迭代改进。

运行示例：

```bash
uv run python examples/react.py
uv run python examples/plan_and_solve.py
uv run python examples/reflection.py
```

ReAct 搜索示例需要在 `.env` 中设置可选的 `SERPAPI_API_KEY`。另外两个示例只需要模型配置。

## 常用命令

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

添加新的运行时或开发依赖：

```bash
uv add <package>
uv add --dev <package>
```

## 已配置依赖

运行时依赖包括 OpenAI SDK、环境变量加载、类型化配置与数据模型、重试和结构化日志。
开发依赖包括 pytest、异步测试、HTTP Mock、Ruff 和 mypy。精确版本记录在 `uv.lock` 中。
