from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from engram.core.agent import Agent
from engram.core.message import Message
from engram.core.model import ModelClient

RecordType = Literal["execution", "reflection"]
STOP_MARKER = "NO_IMPROVEMENT_NEEDED"

DEFAULT_PROMPTS = {
    "initial": "Complete the task accurately and return only the solution.",
    "reflect": (
        "Review the solution for correctness, completeness, clarity, and edge cases. "
        f"Return actionable feedback, or exactly {STOP_MARKER} if no improvement is needed."
    ),
    "refine": "Revise the solution using the review feedback. Return only the improved solution.",
}


@dataclass(frozen=True, slots=True)
class ReflectionRecord:
    kind: RecordType
    content: str


class ReflectionMemory:
    """Keep execution and review records for one reflection run."""

    def __init__(self) -> None:
        self._records: list[ReflectionRecord] = []

    @property
    def records(self) -> tuple[ReflectionRecord, ...]:
        return tuple(self._records)

    def add(self, kind: RecordType, content: str) -> None:
        self._records.append(ReflectionRecord(kind=kind, content=content))

    def trajectory(self) -> str:
        return "\n\n".join(f"{record.kind.title()}:\n{record.content}" for record in self._records)

    def last_execution(self) -> str:
        for record in reversed(self._records):
            if record.kind == "execution":
                return record.content
        raise RuntimeError("Reflection memory does not contain an execution result.")


class ReflectionAgent(Agent):
    """Generate, review, and refine a solution for a bounded number of iterations."""

    def __init__(
        self,
        llm: ModelClient,
        max_iterations: int = 3,
        *,
        name: str = "reflection-agent",
        prompts: Mapping[str, str] | None = None,
    ) -> None:
        if max_iterations < 0:
            raise ValueError("max_iterations cannot be negative.")
        super().__init__(name, llm)
        self.max_iterations = max_iterations
        self.prompts = {**DEFAULT_PROMPTS, **dict(prompts or {})}
        required = {"initial", "reflect", "refine"}
        if not required.issubset(self.prompts):
            raise ValueError("Reflection prompts must define initial, reflect, and refine.")
        self.memory = ReflectionMemory()

    def run(self, input_text: str) -> str:
        if not input_text.strip():
            raise ValueError("Task cannot be empty.")
        self.history.append(Message(role="user", content=input_text))
        self.memory = ReflectionMemory()
        solution = self._respond(input_text, self.prompts["initial"])
        self.memory.add("execution", solution)
        for _ in range(self.max_iterations):
            feedback = self._respond(
                f"Task:\n{input_text}\n\nProposed solution:\n{solution}",
                self.prompts["reflect"],
            )
            self.memory.add("reflection", feedback)
            if feedback.strip().upper() == STOP_MARKER:
                break
            solution = self._respond(
                (
                    f"Task:\n{input_text}\n\nCurrent solution:\n{solution}\n\n"
                    f"Review feedback:\n{feedback}"
                ),
                self.prompts["refine"],
            )
            self.memory.add("execution", solution)
        answer = self.memory.last_execution()
        self.history.append(Message(role="assistant", content=answer))
        return answer

    def _respond(self, prompt: str, instructions: str) -> str:
        result = self.llm.complete(prompt, instructions=instructions).text.strip()
        if not result:
            raise ValueError("Model returned an empty response.")
        return result
