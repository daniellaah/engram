from collections.abc import Iterable, Sequence

from engram.context.tokens import HeuristicTokenCounter, TokenCounter
from engram.core.message import Message


class HistoryManager:
    """Own conversation history and expose token-aware recent context."""

    def __init__(
        self,
        messages: Iterable[Message] | None = None,
        *,
        max_messages: int = 100,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than zero.")
        self.max_messages = max_messages
        self.token_counter = token_counter or HeuristicTokenCounter()
        self._messages = list(messages or ())[-max_messages:]

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def append(self, message: Message) -> None:
        self._messages.append(message)
        overflow = len(self._messages) - self.max_messages
        if overflow > 0:
            del self._messages[:overflow]

    def extend(self, messages: Iterable[Message]) -> None:
        for message in messages:
            self.append(message)

    def replace(self, messages: Sequence[Message]) -> None:
        self._messages = list(messages)[-self.max_messages :]

    def clear(self) -> None:
        self._messages.clear()

    def recent(self, token_budget: int) -> tuple[Message, ...]:
        if token_budget <= 0:
            return ()
        selected: list[Message] = []
        used = 0
        for message in reversed(self._messages):
            cost = self.token_counter.count_message(message)
            if selected and used + cost > token_budget:
                break
            if cost <= token_budget:
                selected.append(message)
                used += cost
        selected.reverse()
        return tuple(selected)

    def compact(self, summary: str, *, retain_messages: int = 10) -> None:
        if not summary.strip():
            raise ValueError("Summary cannot be empty.")
        retained = self._messages[-retain_messages:] if retain_messages > 0 else []
        summary_message = Message(
            role="developer",
            content=f"Conversation summary:\n{summary.strip()}",
            metadata={"kind": "summary"},
        )
        self._messages = [summary_message, *retained][-self.max_messages :]
