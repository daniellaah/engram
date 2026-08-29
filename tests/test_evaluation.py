from engram import ModelResponse
from engram.evaluation import EvaluationCase, Evaluator, LLMJudge, contains_all, exact_match
from tests.fakes import FakeModel


def test_evaluator_aggregates_metrics_and_errors() -> None:
    evaluator = Evaluator({"exact": exact_match, "contains": contains_all})
    report = evaluator.evaluate(
        str.upper,
        [
            EvaluationCase("hello", "HELLO"),
            EvaluationCase("world", ["WOR", "LD"]),
        ],
    )

    assert report.averages == {"contains": 1.0, "exact": 0.5}
    assert report.error_count == 0


def test_llm_judge_parses_bounded_score() -> None:
    model = FakeModel([ModelResponse(text='{"score": 1.4, "reason": "good"}', model="fake")])
    judge = LLMJudge(model, "The answer must be correct.")

    score = judge(EvaluationCase("question", "reference"), "candidate")

    assert score == 1.0
