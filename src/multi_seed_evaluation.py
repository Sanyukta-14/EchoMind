
import numpy as np
import pandas as pd

from src.simulator import simulate_world
from src.evaluation import evaluate_hypotheses


def run_multi_seed_evaluation(
    seeds=None,
    steps=200,
    fault_step=100,
    true_cause="water_level",
    top_k=3,
):
    """
    Evaluate hypothesis ranking across multiple random seeds.

    This measures ranking performance on simulated data.
    It does not prove causation in real-world systems.
    """
    if seeds is None:
        seeds = range(10, 30)

    seeds = list(seeds)

    if not seeds:
        raise ValueError("At least one random seed is required.")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    if any(not isinstance(seed, (int, np.integer)) for seed in seeds):
        raise ValueError("Every seed must be an integer.")

    if steps < 2:
        raise ValueError("steps must be at least 2.")

    if not 0 <= fault_step < steps:
        raise ValueError("fault_step must be between 0 and steps - 1.")

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    rows = []

    for seed in seeds:
        df = simulate_world(
            steps=steps,
            fault_step=fault_step,
            seed=int(seed),
        )

        evaluation = evaluate_hypotheses(
            df=df,
            fault_step=fault_step,
            true_cause=true_cause,
            top_k=top_k,
        )

        rank = evaluation["true_cause_rank"]

        rows.append({
            "seed": int(seed),
            "true_cause": true_cause,
            "predicted_cause": evaluation["predicted_cause"],
            "true_cause_rank": rank,
            "top_1_correct": rank == 1,
            "top_k_correct": rank <= min(top_k, evaluation["total_candidates"]),
            "reciprocal_rank": 1.0 / rank,
        })

    per_seed = pd.DataFrame(rows)

    summary = {
        "experiments": len(per_seed),
        "true_cause": true_cause,
        "top_k": top_k,
        "top_1_accuracy": float(per_seed["top_1_correct"].mean()),
        "top_k_accuracy": float(per_seed["top_k_correct"].mean()),
        "mean_reciprocal_rank": float(
            per_seed["reciprocal_rank"].mean()
        ),
        "mean_true_cause_rank": float(
            per_seed["true_cause_rank"].mean()
        ),
    }

    return summary, per_seed
