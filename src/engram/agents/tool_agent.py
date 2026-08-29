from collections.abc import Iterable

from engram.context import ContextSource
from engram.core.agent import Agent
from engram.core.config import AgentConfig
from engram.core.events import EventBus, EventType
from engram.core.message import Message
from engram.core.model import ModelClient, ModelResponse, ToolCall
from engram.tools import Tool, ToolRegistry, ToolResult


class ToolAgent(Agent):
    """Use native model function calls in a bounded model-tool loop."""

    def __init__(
        self,
        llm: ModelClient,
        *,
        name: str = "tool-agent",
        instructions: str | None = None,
        config: AgentConfig | None = None,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        events: EventBus | None = None,
    ) -> None:
        super().__init__(
            name,
            llm,
            instructions=instructions,
            config=config,
            tools=tools,
            events=events,
        )

    def run(self, input_text: str) -> str:
        return self.run_with_context(input_text)

    def run_with_context(
        self,
        input_text: str,
        *,
        sources: list[ContextSource] | None = None,
    ) -> str:
        if not input_text.strip():
            raise ValueError("Input cannot be empty.")
        self._emit(EventType.RUN_STARTED, input=input_text)
        self.history.append(Message(role="user", content=input_text))
        input_items = self.context.build(self.history, sources=sources)
        try:
            answer = self._run_loop(input_items)
        except Exception as error:
            self._emit(EventType.RUN_FAILED, error=f"{type(error).__name__}: {error}")
            raise
        self.history.append(Message(role="assistant", content=answer))
        self._emit(EventType.RUN_COMPLETED, output=answer)
        return answer

    async def arun(self, input_text: str) -> str:
        if not input_text.strip():
            raise ValueError("Input cannot be empty.")
        self._emit(EventType.RUN_STARTED, input=input_text)
        self.history.append(Message(role="user", content=input_text))
        input_items = self.context.build(self.history)
        try:
            answer = await self._arun_loop(input_items)
        except Exception as error:
            self._emit(EventType.RUN_FAILED, error=f"{type(error).__name__}: {error}")
            raise
        self.history.append(Message(role="assistant", content=answer))
        self._emit(EventType.RUN_COMPLETED, output=answer)
        return answer

    def _run_loop(self, input_items: list[dict[str, object]]) -> str:
        model_input = [dict(item) for item in input_items]
        schemas = self.tools.schemas()
        for step in range(1, self.config.max_steps + 1):
            self._emit(EventType.MODEL_STARTED, step=step)
            response = self.llm.complete(
                model_input,
                instructions=self.instructions,
                tools=schemas,
                parallel_tool_calls=self.config.parallel_tool_calls,
                max_output_tokens=self.config.output_token_reserve or None,
            )
            self._emit(
                EventType.MODEL_COMPLETED,
                step=step,
                response_id=response.response_id,
                tool_calls=len(response.tool_calls),
                usage=response.usage.total_tokens,
            )
            if not response.tool_calls:
                return self._final_text(response)
            self._append_model_output(model_input, response)
            results = self._invoke_calls(response.tool_calls)
            self._append_tool_results(model_input, response.tool_calls, results)
        raise RuntimeError(f"Agent exceeded its {self.config.max_steps}-step limit.")

    async def _arun_loop(self, input_items: list[dict[str, object]]) -> str:
        model_input = [dict(item) for item in input_items]
        schemas = self.tools.schemas()
        for step in range(1, self.config.max_steps + 1):
            self._emit(EventType.MODEL_STARTED, step=step)
            response = await self.llm.acomplete(
                model_input,
                instructions=self.instructions,
                tools=schemas,
                parallel_tool_calls=self.config.parallel_tool_calls,
                max_output_tokens=self.config.output_token_reserve or None,
            )
            self._emit(
                EventType.MODEL_COMPLETED,
                step=step,
                response_id=response.response_id,
                tool_calls=len(response.tool_calls),
                usage=response.usage.total_tokens,
            )
            if not response.tool_calls:
                return self._final_text(response)
            self._append_model_output(model_input, response)
            for call in response.tool_calls:
                self._emit(EventType.TOOL_STARTED, tool=call.name, call_id=call.id)
            calls = [(call.name, call.arguments) for call in response.tool_calls]
            results: list[ToolResult]
            if self.config.parallel_tool_calls:
                results = await self.tools.ainvoke_many(calls)
            else:
                results = [await self.tools.ainvoke(name, arguments) for name, arguments in calls]
            for call, result in zip(response.tool_calls, results, strict=True):
                self._emit(
                    EventType.TOOL_COMPLETED,
                    tool=call.name,
                    call_id=call.id,
                    status=result.status.value,
                )
            self._append_tool_results(model_input, response.tool_calls, results)
        raise RuntimeError(f"Agent exceeded its {self.config.max_steps}-step limit.")

    def _invoke_calls(self, calls: tuple[ToolCall, ...]) -> list[ToolResult]:
        for call in calls:
            self._emit(EventType.TOOL_STARTED, tool=call.name, call_id=call.id)
        invocations = [(call.name, call.arguments) for call in calls]
        results: list[ToolResult]
        if self.config.parallel_tool_calls:
            results = self.tools.invoke_many(invocations)
        else:
            results = [self.tools.invoke(name, arguments) for name, arguments in invocations]
        for call, result in zip(calls, results, strict=True):
            self._emit(
                EventType.TOOL_COMPLETED,
                tool=call.name,
                call_id=call.id,
                status=result.status.value,
            )
        return results

    @staticmethod
    def _append_model_output(
        model_input: list[dict[str, object]],
        response: ModelResponse,
    ) -> None:
        if response.output_items:
            model_input.extend(response.output_items)
            return
        model_input.extend(
            {
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": call.raw_arguments,
            }
            for call in response.tool_calls
        )

    @staticmethod
    def _append_tool_results(
        model_input: list[dict[str, object]],
        calls: tuple[ToolCall, ...],
        results: list[ToolResult],
    ) -> None:
        model_input.extend(
            {
                "type": "function_call_output",
                "call_id": call.id,
                "output": result.to_model_output(),
            }
            for call, result in zip(calls, results, strict=True)
        )

    @staticmethod
    def _final_text(response: ModelResponse) -> str:
        answer = response.text.strip()
        if not answer:
            raise ValueError("Model returned neither text nor tool calls.")
        return answer


def agent_as_tool(
    agent: Agent,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Expose another agent as one task-oriented tool."""

    def delegate(task: str) -> str:
        """Delegate a self-contained task to another agent."""
        return agent.run(task)

    return Tool.from_callable(
        delegate,
        name=name or agent.name.replace("-", "_"),
        description=description or f"Delegate a self-contained task to {agent.name}.",
    )
