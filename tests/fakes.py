from collections.abc import Sequence

from engram.core.model import ModelInput, ModelResponse, ToolSchema


class FakeModel:
    model = "fake-model"

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.responses = list(responses)
        self.inputs: list[ModelInput] = []
        self.instructions: list[str | None] = []
        self.tool_schemas: list[Sequence[ToolSchema] | None] = []

    def complete(
        self,
        input_data: ModelInput,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        parallel_tool_calls: bool = True,
        max_output_tokens: int | None = None,
    ) -> ModelResponse:
        del parallel_tool_calls, max_output_tokens
        self.inputs.append(input_data)
        self.instructions.append(instructions)
        self.tool_schemas.append(tools)
        if not self.responses:
            raise RuntimeError("FakeModel has no response left.")
        return self.responses.pop(0)

    async def acomplete(
        self,
        input_data: ModelInput,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        parallel_tool_calls: bool = True,
        max_output_tokens: int | None = None,
    ) -> ModelResponse:
        return self.complete(
            input_data,
            instructions=instructions,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            max_output_tokens=max_output_tokens,
        )
