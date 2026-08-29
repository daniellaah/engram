import json
import os
from collections.abc import Iterator, Sequence
from types import TracebackType
from typing import Any, Self, cast

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, OpenAIError, omit
from openai.types.responses import Response, ResponseFunctionToolCall, ResponseInputParam
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from openai.types.responses.tool_param import ToolParam

from engram.core.model import (
    ModelInput,
    ModelResponse,
    TokenUsage,
    ToolCall,
    ToolSchema,
)


class LLMClient:
    """Call Responses-compatible model endpoints through one normalized interface."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        *,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
    ) -> None:
        load_dotenv()
        model_name = model or os.getenv("LLM_MODEL")
        api_key_value = api_key or os.getenv("LLM_API_KEY")
        base_url_value = base_url or os.getenv("LLM_BASE_URL")
        if not model_name:
            raise ValueError("Missing model name. Set LLM_MODEL or pass model.")
        if client is None and not api_key_value:
            raise ValueError("Missing API key. Set LLM_API_KEY or pass api_key.")
        if client is None and not base_url_value:
            raise ValueError("Missing API URL. Set LLM_BASE_URL or pass base_url.")

        timeout_value = timeout if timeout is not None else self._load_timeout()
        self.model = model_name
        self._client = client or OpenAI(
            api_key=api_key_value,
            base_url=base_url_value,
            timeout=timeout_value,
        )
        self._async_client = async_client or AsyncOpenAI(
            api_key=api_key_value or "unused",
            base_url=base_url_value,
            timeout=timeout_value,
        )

    @staticmethod
    def _load_timeout() -> float:
        raw_timeout = os.getenv("LLM_TIMEOUT", "60")
        try:
            timeout = float(raw_timeout)
        except ValueError as error:
            raise ValueError("LLM_TIMEOUT must be a valid number.") from error
        if timeout <= 0:
            raise ValueError("LLM_TIMEOUT must be greater than zero.")
        return timeout

    def complete(
        self,
        input_data: ModelInput,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        parallel_tool_calls: bool = True,
        max_output_tokens: int | None = None,
    ) -> ModelResponse:
        """Return normalized text, tool calls, continuation items, and token usage."""
        response = self._client.responses.create(
            model=self.model,
            input=cast(str | ResponseInputParam, input_data),
            instructions=instructions,
            tools=cast(Sequence[ToolParam], tools) if tools else omit,
            parallel_tool_calls=parallel_tool_calls if tools else omit,
            max_output_tokens=max_output_tokens if max_output_tokens is not None else omit,
        )
        return self._normalize(response)

    async def acomplete(
        self,
        input_data: ModelInput,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        parallel_tool_calls: bool = True,
        max_output_tokens: int | None = None,
    ) -> ModelResponse:
        response = await self._async_client.responses.create(
            model=self.model,
            input=cast(str | ResponseInputParam, input_data),
            instructions=instructions,
            tools=cast(Sequence[ToolParam], tools) if tools else omit,
            parallel_tool_calls=parallel_tool_calls if tools else omit,
            max_output_tokens=max_output_tokens if max_output_tokens is not None else omit,
        )
        return self._normalize(response)

    def stream(
        self,
        input_data: ModelInput,
        *,
        instructions: str | None = None,
    ) -> Iterator[str]:
        """Yield text deltas for calls that do not need local tool execution."""
        with self._client.responses.stream(
            model=self.model,
            instructions=instructions,
            input=cast(str | ResponseInputParam, input_data),
        ) as stream:
            for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    yield event.delta

    def respond(
        self,
        input_data: ModelInput,
        *,
        instructions: str | None = None,
        stream_to_stdout: bool = True,
    ) -> str:
        """Compatibility helper that streams text and returns the combined response."""
        if stream_to_stdout:
            print(f"Calling {self.model}...")
        parts: list[str] = []
        for delta in self.stream(input_data, instructions=instructions):
            parts.append(delta)
            if stream_to_stdout:
                print(delta, end="", flush=True)
        if stream_to_stdout:
            print()
        return "".join(parts)

    @staticmethod
    def _normalize(response: Response) -> ModelResponse:
        calls: list[ToolCall] = []
        output_items: list[dict[str, Any]] = []
        for item in response.output:
            output_items.append(item.model_dump(mode="json", exclude_none=True))
            if isinstance(item, ResponseFunctionToolCall):
                try:
                    arguments: object = json.loads(item.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                calls.append(
                    ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=cast(dict[str, Any], arguments),
                        raw_arguments=item.arguments,
                    )
                )
        usage = response.usage
        normalized_usage = TokenUsage(
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
        return ModelResponse(
            text=response.output_text or "",
            model=response.model,
            response_id=response.id,
            tool_calls=tuple(calls),
            output_items=tuple(output_items),
            usage=normalized_usage,
        )

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._async_client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def main() -> None:
    try:
        with LLMClient() as llm:
            llm.respond(
                "Write a quicksort implementation.",
                instructions="You are a helpful assistant that writes Python code.",
            )
    except (OpenAIError, ValueError) as error:
        print(f"Model call failed: {error}")


if __name__ == "__main__":
    main()
