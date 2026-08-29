import re
from typing import Protocol

from engram.core.message import Message


class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...

    def count_message(self, message: Message) -> int: ...


class HeuristicTokenCounter:
    """Dependency-free token estimate that works across model providers."""

    _word_or_cjk = re.compile(r"[\w]+|[^\x00-\x7F]")

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        lexical = len(self._word_or_cjk.findall(text))
        return max(1, max(lexical, (len(text) + 3) // 4))

    def count_message(self, message: Message) -> int:
        return 4 + self.count_text(message.content)
