from dataclasses import dataclass, field

from engram.context.history import HistoryManager
from engram.context.tokens import HeuristicTokenCounter, TokenCounter
from engram.core.message import Message


@dataclass(frozen=True, slots=True)
class ContextSource:
    name: str
    content: str
    priority: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


class ContextBuilder:
    """Gather, select, and structure history plus retrieved context."""

    def __init__(
        self,
        *,
        token_budget: int,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be greater than zero.")
        self.token_budget = token_budget
        self.token_counter = token_counter or HeuristicTokenCounter()

    def build(
        self,
        history: HistoryManager,
        *,
        sources: list[ContextSource] | None = None,
    ) -> list[dict[str, object]]:
        selected_sources, source_cost = self._select_sources(sources or [])
        messages: list[Message] = []
        if selected_sources:
            blocks = [f"[{source.name}]\n{source.content}" for source in selected_sources]
            messages.append(
                Message(
                    role="developer",
                    content="Relevant context:\n\n" + "\n\n".join(blocks),
                    metadata={"kind": "retrieved_context"},
                )
            )
        messages.extend(history.recent(max(1, self.token_budget - source_cost)))
        return [message.to_input_item() for message in messages]

    def _select_sources(
        self,
        sources: list[ContextSource],
    ) -> tuple[list[ContextSource], int]:
        selected: list[ContextSource] = []
        used = 0
        for source in sorted(sources, key=lambda item: item.priority, reverse=True):
            cost = self.token_counter.count_text(source.content) + 8
            if used + cost <= self.token_budget // 2:
                selected.append(source)
                used += cost
        return selected, used
