
import pytest

from src.simulator import simulate_world
from src.predictor import FEATURES
from src.hypothesis_generator import rank_candidates
from src.intervention_engine import (
    test_interventions as run_interventions,
    INTERVENTION_CANDIDATES,
)


@pytest.fixture
def sample_data():
    return simulate_world(steps=200, fault_step=100, seed=42)


def test_hypothesis_ranking_returns_all_features(sample_data):
    result = rank_candidates(sample_data, fault_step=100)

    assert len(result) == len(FEATURES)
    assert "candidate_cause" in result.columns
    assert "hypothesis_score" in result.columns
    assert result["hypothesis_score"].between(0, 1).all()


def test_hypotheses_are_ranked_highest_first(sample_data):
    result = rank_candidates(sample_data, fault_step=100)

    scores = result["hypothesis_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_hypothesis_ranking_rejects_empty_data():
    import pandas as pd

    with pytest.raises(ValueError):
        rank_candidates(pd.DataFrame(), fault_step=100)


def test_interventions_exclude_the_outcome_variable(sample_data):
    result = run_interventions(
        original_df=sample_data,
        fault_step=100,
        seed=42,
    )

    assert set(result["candidate_cause"]) == set(INTERVENTION_CANDIDATES)
    assert "animal_population" not in result["candidate_cause"].values


def test_intervention_results_have_expected_columns(sample_data):
    result = run_interventions(
        original_df=sample_data,
        fault_step=100,
        seed=42,
    )

    expected_columns = {
        "candidate_cause",
        "restoration_target",
        "faulty_final_population",
        "intervention_final_population",
        "improvement",
        "interpretation",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == len(INTERVENTION_CANDIDATES)
