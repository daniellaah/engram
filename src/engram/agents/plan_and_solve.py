import json
from typing import cast

from engram.core.agent import Agent
from engram.core.message import Message
from engram.core.model import ModelClient

PLANNER_INSTRUCTIONS = """\
Create a concise, logically ordered plan for the task.
Return only a JSON array of non-empty strings. Do not solve the task.
"""

EXECUTOR_INSTRUCTIONS = """\
Execute exactly the current plan step using the original task and completed results.
Return only the step result. On the final step, return the complete final answer.
"""


class Planner:
    """Create a validated JSON plan."""

    def __init__(self, llm: ModelClient, instructions: str = PLANNER_INSTRUCTIONS) -> None:
        self.llm = llm
        self.instructions = instructions

    def plan(self, task: str) -> list[str]:
        if not task.strip():
            raise ValueError("Task cannot be empty.")
        response = self.llm.complete(task, instructions=self.instructions)
        return self._parse_plan(response.text)

    @staticmethod
    def _parse_plan(response: str) -> list[str]:
        content = response.strip()
        if content.startswith("```") and content.endswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1]).strip()
        try:
            value: object = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("Planner response must be a valid JSON array.") from error
        if not isinstance(value, list) or not value:
            raise ValueError("Planner response must contain at least one step.")
        if not all(isinstance(step, str) and step.strip() for step in value):
            raise ValueError("Every plan step must be a non-empty string.")
        return [step.strip() for step in cast(list[str], value)]


class Executor:
    """Execute a plan sequentially and return the final step result."""

    def __init__(self, llm: ModelClient, instructions: str = EXECUTOR_INSTRUCTIONS) -> None:
        self.llm = llm
        self.instructions = instructions

    def execute(self, task: str, plan: list[str]) -> str:
        if not plan:
            raise ValueError("Plan cannot be empty.")
        completed: list[str] = []
        for index, step in enumerate(plan, start=1):
            response = self.llm.complete(
                self._build_prompt(task, plan, completed, index, step),
                instructions=self.instructions,
            )
            result = response.text.strip()
            if not result:
                raise ValueError(f"Step {index} returned an empty result.")
            completed.append(result)
        return completed[-1]

    @staticmethod
    def _build_prompt(
        task: str,
        plan: list[str],
        completed: list[str],
        index: int,
        step: str,
    ) -> str:
        numbered_plan = "\n".join(
            f"{position}. {item}" for position, item in enumerate(plan, start=1)
        )
        completed_results = (
            "\n\n".join(
                f"Step {position}:\n{result}" for position, result in enumerate(completed, start=1)
            )
            or "No steps have been completed yet."
        )
        return (
            f"Task:\n{task}\n\nPlan:\n{numbered_plan}\n\n"
            f"Completed results:\n{completed_results}\n\nCurrent step {index}:\n{step}"
        )


class PlanAndSolveAgent(Agent):
    """Plan a task first, then execute each validated step."""

    def __init__(
        self,
        llm: ModelClient,
        *,
        name: str = "plan-and-solve-agent",
        planner_instructions: str = PLANNER_INSTRUCTIONS,
        executor_instructions: str = EXECUTOR_INSTRUCTIONS,
    ) -> None:
        super().__init__(name, llm)
        self.planner = Planner(llm, planner_instructions)
        self.executor = Executor(llm, executor_instructions)
        self.last_plan: tuple[str, ...] = ()

    def run(self, input_text: str) -> str:
        if not input_text.strip():
            raise ValueError("Task cannot be empty.")
        self.history.append(Message(role="user", content=input_text))
        plan = self.planner.plan(input_text)
        self.last_plan = tuple(plan)
        answer = self.executor.execute(input_text, plan)
        self.history.append(Message(role="assistant", content=answer))
        return answer
