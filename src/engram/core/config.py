from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentConfig(BaseModel):
    """Runtime limits shared by Engram agents."""

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=8, gt=0)
    max_history_messages: int = Field(default=100, gt=0)
    context_window: int = Field(default=128_000, gt=0)
    output_token_reserve: int = Field(default=4_096, ge=0)
    parallel_tool_calls: bool = True

    @model_validator(mode="after")
    def validate_token_budget(self) -> Self:
        if self.output_token_reserve >= self.context_window:
            raise ValueError("output_token_reserve must be smaller than context_window.")
        return self

    @property
    def input_token_budget(self) -> int:
        return max(1, self.context_window - self.output_token_reserve)
