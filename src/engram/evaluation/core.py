import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Protocol, cast

from engram.core.model import ModelClient


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    input: str
    expected: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    case: EvaluationCase
    output: str
    scores: Mapping[str, float]
    duration_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    results: tuple[EvaluationResult, ...]

    @property
    def averages(self) -> dict[str, float]:
        names = {name for result in self.results for name in result.scores}
        return {
            name: sum(result.scores.get(name, 0.0) for result in self.results)
            / max(1, len(self.results))
            for name in sorted(names)
        }

    @property
    def error_count(self) -> int:
        return sum(result.error is not None for result in self.results)


class Metric(Protocol):
    def __call__(self, case: EvaluationCase, output: str) -> float: ...


class Evaluator:
    """Run deterministic or model-based metrics over arbitrary agent tasks."""

    def __init__(self, metrics: Mapping[str, Metric]) -> None:
        if not metrics:
            raise ValueError("At least one evaluation metric is required.")
        self.metrics = dict(metrics)

    def evaluate(
        self,
        runner: Callable[[str], str],
        cases: Sequence[EvaluationCase],
    ) -> EvaluationReport:
        results: list[EvaluationResult] = []
        for case in cases:
            started = monotonic()
            try:
                output = runner(case.input)
                scores = {name: metric(case, output) for name, metric in self.metrics.items()}
                error = None
            except Exception as exception:
                output = ""
                scores = {name: 0.0 for name in self.metrics}
                error = f"{type(exception).__name__}: {exception}"
            results.append(
                EvaluationResult(
                    case=case,
                    output=output,
                    scores=scores,
                    duration_ms=(monotonic() - started) * 1000,
                    error=error,
                )
            )
        return EvaluationReport(tuple(results))


def exact_match(case: EvaluationCase, output: str) -> float:
    return float(output.strip() == str(case.expected).strip())


def contains_all(case: EvaluationCase, output: str) -> float:
    expected = case.expected
    if isinstance(expected, str):
        values = [expected]
    elif isinstance(expected, Sequence):
        values = [str(item) for item in expected]
    else:
        values = [str(expected)]
    return float(all(value.lower() in output.lower() for value in values))


class LLMJudge:
    """Score one output against an explicit rubric using a model."""

    def __init__(self, llm: ModelClient, rubric: str) -> None:
        if not rubric.strip():
            raise ValueError("Judge rubric cannot be empty.")
        self.llm = llm
        self.rubric = rubric

    def __call__(self, case: EvaluationCase, output: str) -> float:
        response = self.llm.complete(
            (
                f"Rubric:\n{self.rubric}\n\nInput:\n{case.input}\n\n"
                f"Expected reference:\n{case.expected}\n\nCandidate output:\n{output}"
            ),
            instructions=(
                "Act as an impartial evaluator. Return only JSON with a numeric score from "
                '0 to 1 and a brief reason: {"score": 0.0, "reason": "..."}.'
            ),
        )
        try:
            payload: object = json.loads(response.text)
        except json.JSONDecodeError as error:
            raise ValueError("Judge returned invalid JSON.") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("score"), int | float):
            raise ValueError("Judge response must contain a numeric score.")
        return min(1.0, max(0.0, float(cast(dict[str, Any], payload)["score"])))
