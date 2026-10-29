
import pytest

from src.simulator import simulate_world
from src.evaluation import evaluate_hypotheses


def test_evaluation_returns_expected_fields():
    df = simulate_world(steps=200, fault_step=100, seed=42)

    result = evaluate_hypotheses(
        df=df,
        fault_step=100,
        true_cause="water_level",
        top_k=3,
    )

    assert result["true_cause"] == "water_level"
    assert 1 <= result["true_cause_rank"] <= result["total_candidates"]
    assert isinstance(result["found_in_top_k"], bool)


def test_evaluation_rejects_invalid_top_k():
    df = simulate_world(steps=200, fault_step=100, seed=42)

    with pytest.raises(ValueError):
        evaluate_hypotheses(
            df=df,
            fault_step=100,
            true_cause="water_level",
            top_k=0,
        )
