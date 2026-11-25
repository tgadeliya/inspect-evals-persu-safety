from typing import cast

import pytest
from inspect_ai.scorer import NOANSWER, Metric, MetricProtocol, SampleScore, Score, Value

from persu_safety.prompts import STRATEGY_NAMES
from persu_safety.scorers.persuasiveness import persuasiveness_metric
from persu_safety.scorers.refusal import ENGAGED, REFUSED, refusal_rate
from persu_safety.scorers.strategy_eval import strategy_usage_metrics


def score(value: Value) -> SampleScore:
    return SampleScore(score=Score(value=value))


def run(metric: Metric, scores: list[SampleScore]) -> Value:
    # Metric is a union with the deprecated list[Score] signature; these metrics take SampleScores.
    return cast(MetricProtocol, metric)(scores)


def test_refusal_rate_counts_refusals():
    scores = [score(REFUSED), score(ENGAGED), score(ENGAGED), score(REFUSED)]
    assert run(refusal_rate(), scores) == 0.5


def test_strategy_metric_excludes_refused_and_parse_failures():
    scores = [
        score(dict.fromkeys(STRATEGY_NAMES, 2)),  # scored
        score(NOANSWER),  # refused -> excluded
        score({}),  # parse failure -> counted in denominator
    ]
    out = run(strategy_usage_metrics(), scores)
    assert isinstance(out, dict)
    assert out["strategy_eval/parse_failure_rate"] == 0.5
    assert out["strategy_eval/Deceptive Information"] == 2.0


def test_strategy_metric_all_refused_reports_zero_parse_failure():
    out = run(strategy_usage_metrics(), [score(NOANSWER), score(NOANSWER)])
    assert isinstance(out, dict)
    assert out["strategy_eval/parse_failure_rate"] == 0.0
    assert "strategy_eval/Deceptive Information" not in out


def test_persuasiveness_metric_mean_and_distribution():
    scores = [score(4), score(2), score(NOANSWER)]
    out = run(persuasiveness_metric(), scores)
    assert isinstance(out, dict)
    assert out["persuasion_effectiveness/parse_failure_rate"] == pytest.approx(1 / 3)
    assert out["persuasion_effectiveness/mean"] == 3.0
    assert out["persuasion_effectiveness/rate_4"] == 0.5
