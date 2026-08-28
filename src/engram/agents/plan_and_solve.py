import json
from typing import cast

from engram.llm import LLMClient

PLANNER_INSTRUCTIONS = """\
Create a concise step-by-step plan for the task.
Return only a JSON array of non-empty strings.
Do not solve the task and do not add Markdown fences.
"""

EXECUTOR_INSTRUCTIONS = """\
Execute exactly the current plan step using the task and completed results as context.
Return only the result of the current step.
For the final step, return the complete final answer.
"""


class Planner:
    """Create a validated JSON plan with the Responses API client."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def plan(self, task: str) -> list[str]:
        if not task.strip():
            raise ValueError("Task cannot be empty.")

        response = self._llm.respond(
            task,
            instructions=PLANNER_INSTRUCTIONS,
            stream_to_stdout=False,
        )
        return self._parse_plan(response)

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

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def execute(self, task: str, plan: list[str]) -> str:
        if not plan:
            raise ValueError("Plan cannot be empty.")

        completed: list[str] = []
        for index, step in enumerate(plan, start=1):
            result = self._llm.respond(
                self._build_prompt(task, plan, completed, index, step),
                instructions=EXECUTOR_INSTRUCTIONS,
                stream_to_stdout=False,
            ).strip()
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
            f"Completed results:\n{completed_results}\n\n"
            f"Current step {index}:\n{step}"
        )


class PlanAndSolveAgent:
    """Plan a task first, then execute each planned step."""

    def __init__(self, llm: LLMClient) -> None:
        self.planner = Planner(llm)
        self.executor = Executor(llm)

    def run(self, task: str) -> str:
        plan = self.planner.plan(task)
        return self.executor.execute(task, plan)
