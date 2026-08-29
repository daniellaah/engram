from unittest.mock import MagicMock, patch

import pytest
from openai.types.responses import Response, ResponseTextDeltaEvent
from pytest import MonkeyPatch

from engram import LLMClient


def text_delta(text: str, sequence_number: int) -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent(
        content_index=0,
        delta=text,
        item_id="message_1",
        logprobs=[],
        output_index=0,
        sequence_number=sequence_number,
        type="response.output_text.delta",
    )


def test_respond_collects_streamed_text() -> None:
    sdk_client = MagicMock()
    async_client = MagicMock()
    stream_manager = MagicMock()
    stream_manager.__enter__.return_value = iter([text_delta("Hello", 1), text_delta(", world", 2)])
    sdk_client.responses.stream.return_value = stream_manager

    with LLMClient(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        client=sdk_client,
        async_client=async_client,
    ) as llm:
        result = llm.respond("Test input", instructions="Reply briefly.", stream_to_stdout=False)

    assert result == "Hello, world"
    sdk_client.close.assert_called_once_with()


def test_complete_normalizes_function_calls() -> None:
    sdk_client = MagicMock()
    response = Response.model_validate(
        {
            "id": "resp_1",
            "created_at": 0,
            "model": "test-model",
            "object": "response",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "add",
                    "arguments": '{"a": 2, "b": 3}',
                    "status": "completed",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )
    sdk_client.responses.create.return_value = response
    llm = LLMClient(model="test-model", client=sdk_client, async_client=MagicMock())

    result = llm.complete("add values")

    assert result.response_id == "resp_1"
    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].arguments == {"a": 2, "b": 3}


def test_missing_model_has_clear_error(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    with patch("engram.llm.load_dotenv"), pytest.raises(ValueError, match="LLM_MODEL"):
        LLMClient()


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_invalid_timeout_is_rejected(monkeypatch: MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM_TIMEOUT", value)
    with pytest.raises(ValueError, match="LLM_TIMEOUT"):
        LLMClient(
            model="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
        )
