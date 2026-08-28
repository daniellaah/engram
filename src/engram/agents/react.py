import re

from engram.llm import LLMClient
from engram.tools import ToolRegistry

REACT_INSTRUCTIONS = """\
Solve the task by alternating between an action and an observation.

Available tools:
{tool_descriptions}

For every turn, return exactly these two lines:
Rationale: <one short, user-facing explanation>
Action: <tool_name>[<tool input>]

When the task is complete, use this action instead:
Action: Finish[<final answer>]

Choose only one action per turn. Do not add Markdown fences or extra sections.
"""

_ACTION_PATTERN = re.compile(
    r"^Action:\s*(?P<name>[A-Za-z_][A-Za-z0-9_-]*)\[(?P<argument>.*)\]\s*$",
    re.MULTILINE | re.DOTALL,
)


class ReActAgent:
    """Run a compact ReAct loop with simple single-input tools."""

    def __init__(self, llm: LLMClient, tools: ToolRegistry, max_steps: int = 5) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero.")
        self._llm = llm
        self._tools = tools
        self._max_steps = max_steps

    def run(self, task: str) -> str:
        if not task.strip():
            raise ValueError("Task cannot be empty.")

        trajectory: list[str] = []
        descriptions = self._tools.descriptions() or "- No tools are registered."
        instructions = REACT_INSTRUCTIONS.format(tool_descriptions=descriptions)

        for _ in range(self._max_steps):
            response = self._llm.respond(
                self._build_prompt(task, trajectory),
                instructions=instructions,
                stream_to_stdout=False,
            ).strip()
            action_name, argument = self._parse_action(response)
            trajectory.append(response)

            if action_name == "Finish":
                if not argument:
                    raise ValueError("Finish action must include a final answer.")
                return argument

            observation = self._tools.execute(action_name, argument)
            trajectory.append(f"Observation: {observation}")

        raise RuntimeError(f"ReAct agent exceeded its {self._max_steps}-step limit.")

    @staticmethod
    def _build_prompt(task: str, trajectory: list[str]) -> str:
        history = "\n".join(trajectory) if trajectory else "No actions have been taken yet."
        return f"Task:\n{task}\n\nTrajectory:\n{history}\n\nChoose the next action."

    @staticmethod
    def _parse_action(response: str) -> tuple[str, str]:
        match = _ACTION_PATTERN.search(response)
        if match is None:
            raise ValueError("Model response does not contain a valid Action line.")
        return match.group("name"), match.group("argument").strip()
