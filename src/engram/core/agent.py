from abc import ABC, abstractmethod
from collections.abc import Iterable
from uuid import uuid4

from engram.context import ContextBuilder, HistoryManager
from engram.core.config import AgentConfig
from engram.core.events import EventBus, EventType, RunEvent
from engram.core.message import Message
from engram.core.model import ModelClient
from engram.core.session import JsonSessionStore, Session
from engram.tools import Tool, ToolRegistry


class Agent(ABC):
    """Shared identity, history, tools, events, and session behavior."""

    def __init__(
        self,
        name: str,
        llm: ModelClient,
        *,
        instructions: str | None = None,
        config: AgentConfig | None = None,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        history: HistoryManager | None = None,
        events: EventBus | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("Agent name cannot be empty.")
        self.name = name
        self.llm = llm
        self.instructions = instructions
        self.config = config or AgentConfig()
        if isinstance(tools, ToolRegistry):
            self.tools = tools
        else:
            self.tools = ToolRegistry(tools)
        self.history = history or HistoryManager(max_messages=self.config.max_history_messages)
        self.context = ContextBuilder(token_budget=self.config.input_token_budget)
        self.events = events or EventBus()

    @abstractmethod
    def run(self, input_text: str) -> str:
        """Run the agent and return its final user-facing text."""

    async def arun(self, input_text: str) -> str:
        """Run asynchronously when implemented by the concrete agent."""
        raise NotImplementedError(f"{type(self).__name__} does not implement arun().")

    def add_tool(self, tool: Tool) -> None:
        self.tools.register_tool(tool)

    def add_message(self, message: Message) -> None:
        self.history.append(message)

    def get_history(self) -> tuple[Message, ...]:
        return self.history.messages

    def clear_history(self) -> None:
        self.history.clear()

    def save_session(
        self,
        store: JsonSessionStore,
        session_id: str | None = None,
    ) -> Session:
        session = Session(
            id=session_id or uuid4().hex,
            agent_name=self.name,
            messages=self.history.messages,
        )
        store.save(session)
        return session

    def load_session(self, store: JsonSessionStore, session_id: str) -> Session:
        session = store.load(session_id)
        self.history.replace(session.messages)
        return session

    def _emit(self, event_type: EventType, **data: object) -> None:
        self.events.emit(RunEvent(type=event_type, source=self.name, data=data))
