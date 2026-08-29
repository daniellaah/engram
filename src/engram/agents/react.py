from collections.abc import Iterable

from engram.agents.tool_agent import ToolAgent
from engram.core.config import AgentConfig
from engram.core.events import EventBus
from engram.core.model import ModelClient
from engram.tools import Tool, ToolRegistry

REACT_INSTRUCTIONS = """\
Solve the user's task through a bounded cycle of reasoning, tool use, and observation.
Use tools only when they materially improve correctness. Inspect every tool result before
deciding whether to call another tool. Return a complete final answer when the task is done.
"""


class ReActAgent(ToolAgent):
    """A tool agent configured for deliberate reason-act-observe iteration."""

    def __init__(
        self,
        llm: ModelClient,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        max_steps: int = 8,
        *,
        name: str = "react-agent",
        instructions: str = REACT_INSTRUCTIONS,
        config: AgentConfig | None = None,
        events: EventBus | None = None,
    ) -> None:
        resolved_config = config or AgentConfig(max_steps=max_steps)
        super().__init__(
            llm,
            name=name,
            instructions=instructions,
            config=resolved_config,
            tools=tools,
            events=events,
        )
