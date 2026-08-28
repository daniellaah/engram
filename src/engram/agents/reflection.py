from dataclasses import dataclass
from typing import Literal

from engram.llm import LLMClient

RecordType = Literal["execution", "reflection"]
STOP_MARKER = "NO_IMPROVEMENT_NEEDED"

INITIAL_INSTRUCTIONS = """\
Produce a complete solution to the task.
Return only the solution without commentary or Markdown fences.
"""

REFLECTION_INSTRUCTIONS = f"""\
Review the proposed solution for correctness, completeness, clarity, and edge cases.
Return concise, actionable feedback only.
If no meaningful improvement is needed, return exactly {STOP_MARKER}.
"""

REFINEMENT_INSTRUCTIONS = """\
Revise the solution using the review feedback.
Return only the complete improved solution without commentary or Markdown fences.
"""


@dataclass(frozen=True, slots=True)
class ReflectionRecord:
    kind: RecordType
    content: str


class ReflectionMemory:
    """Keep the execution and review records for one reflection run."""

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


class ReflectionAgent:
    """Generate, review, and refine a solution for a bounded number of iterations."""

    def __init__(self, llm: LLMClient, max_iterations: int = 3) -> None:
        if max_iterations < 0:
            raise ValueError("max_iterations cannot be negative.")
        self._llm = llm
        self._max_iterations = max_iterations
        self.memory = ReflectionMemory()

    def run(self, task: str) -> str:
        if not task.strip():
            raise ValueError("Task cannot be empty.")

        self.memory = ReflectionMemory()
        solution = self._respond(task, INITIAL_INSTRUCTIONS)
        self.memory.add("execution", solution)

        for _ in range(self._max_iterations):
            feedback = self._respond(
                self._review_prompt(task, solution),
                REFLECTION_INSTRUCTIONS,
            )
            self.memory.add("reflection", feedback)
            if feedback.strip().upper() == STOP_MARKER:
                break

            solution = self._respond(
                self._refinement_prompt(task, solution, feedback),
                REFINEMENT_INSTRUCTIONS,
            )
            self.memory.add("execution", solution)

        return self.memory.last_execution()

    def _respond(self, prompt: str, instructions: str) -> str:
        result = self._llm.respond(
            prompt,
            instructions=instructions,
            stream_to_stdout=False,
        ).strip()
        if not result:
            raise ValueError("Model returned an empty response.")
        return result

    @staticmethod
    def _review_prompt(task: str, solution: str) -> str:
        return f"Task:\n{task}\n\nProposed solution:\n{solution}"

    @staticmethod
    def _refinement_prompt(task: str, solution: str, feedback: str) -> str:
        return f"Task:\n{task}\n\nCurrent solution:\n{solution}\n\nReview feedback:\n{feedback}"
