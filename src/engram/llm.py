import os
from types import TracebackType
from typing import Self

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from openai.types.responses import ResponseInputParam, ResponseTextDeltaEvent


class LLMClient:
    """Call OpenAI-compatible models through the Responses API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        load_dotenv()

        model_name = model or os.getenv("LLM_MODEL")
        api_key_value = api_key or os.getenv("LLM_API_KEY")
        base_url_value = base_url or os.getenv("LLM_BASE_URL")

        if not model_name:
            raise ValueError("Missing model name. Set LLM_MODEL or pass model.")
        if not api_key_value:
            raise ValueError("Missing API key. Set LLM_API_KEY or pass api_key.")
        if not base_url_value:
            raise ValueError("Missing API URL. Set LLM_BASE_URL or pass base_url.")

        timeout_value = timeout if timeout is not None else self._load_timeout()

        self.model = model_name
        self._client = OpenAI(
            api_key=api_key_value,
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

    def respond(
        self,
        input_data: str | ResponseInputParam,
        *,
        instructions: str | None = None,
        stream_to_stdout: bool = True,
    ) -> str:
        """Stream model output to stdout and return the complete text."""
        if stream_to_stdout:
            print(f"Calling {self.model}...")
        text_parts: list[str] = []

        with self._client.responses.stream(
            model=self.model,
            instructions=instructions,
            input=input_data,
        ) as stream:
            for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    if stream_to_stdout:
                        print(event.delta, end="", flush=True)
                    text_parts.append(event.delta)

        if stream_to_stdout:
            print()
        return "".join(text_parts)

    def close(self) -> None:
        """Close the underlying HTTP connections."""
        self._client.close()

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
