from collections.abc import Iterable

from engram.agents.tool_agent import ToolAgent
from engram.core.config import AgentConfig
from engram.core.events import EventBus
from engram.core.model import ModelClient
from engram.tools import Tool, ToolRegistry

DEFAULT_SIMPLE_INSTRUCTIONS = "Answer the user accurately, clearly, and concisely."


class SimpleAgent(ToolAgent):
    """A conversational agent that can optionally use native function tools."""

    def __init__(
        self,
        llm: ModelClient,
        *,
        name: str = "assistant",
        instructions: str = DEFAULT_SIMPLE_INSTRUCTIONS,
        config: AgentConfig | None = None,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        events: EventBus | None = None,
    ) -> None:
        super().__init__(
            llm,
            name=name,
            instructions=instructions,
            config=config,
            tools=tools,
            events=events,
        )
