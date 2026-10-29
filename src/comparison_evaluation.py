import numpy as np
import pandas as pd

from src.simulator import simulate_world, FAULT_CAUSES
from src.evaluation import evaluate_hypotheses
from src.baseline import rank_baseline_candidates


def run_comparison_evaluation(
    seeds=None,
    steps=200,
    fault_step=100,
    true_cause="water_level",
    top_k=3,
    window=5,
):
    """
    Compare EchoMind with a simple baseline across random seeds.

    Both methods receive data from the same simulated fault scenario.

    Metrics:
    - Top-1 accuracy
    - Top-K accuracy
    - Mean reciprocal rank (MRR)

    Results describe the simulator, not real-world causation.
    """

    if seeds is None:
        seeds = range(10, 30)

    seeds = list(seeds)

    if not seeds:
        raise ValueError("Provide at least one seed.")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    if any(
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        for seed in seeds
    ):
        raise ValueError("Every seed must be an integer.")

    if (
        not isinstance(steps, (int, np.integer))
        or isinstance(steps, (bool, np.bool_))
        or steps < 2
    ):
        raise ValueError("steps must be an integer of at least 2.")

    if (
        not isinstance(fault_step, (int, np.integer))
        or isinstance(fault_step, (bool, np.bool_))
        or not 0 <= fault_step < steps
    ):
        raise ValueError("fault_step must be between 0 and steps - 1.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("window must be a positive integer.")

    if true_cause not in FAULT_CAUSES:
        raise ValueError(
            f"Unsupported true_cause: {true_cause}. "
            f"Choose one of {FAULT_CAUSES}."
        )

    rows = []

    for seed in seeds:
        # IMPORTANT: inject the cause being evaluated.
        df = simulate_world(
            steps=int(steps),
            fault_step=int(fault_step),
            seed=int(seed),
            fault_cause=true_cause,
        )

        # Confirm the simulator recorded the expected cause.
        if not (df["fault_cause"] == true_cause).all():
            raise RuntimeError(
                "Simulator fault metadata does not match true_cause."
            )

        # EchoMind ranking evaluation.
        evaluation = evaluate_hypotheses(
            df=df,
            fault_step=int(fault_step),
            true_cause=true_cause,
            top_k=top_k,
        )

        echomind_rank = int(evaluation["true_cause_rank"])

        # Baseline ranking evaluation.
        baseline = rank_baseline_candidates(
            df=df,
            fault_step=int(fault_step),
            window=window,
        )

        matches = baseline.index[
            baseline["candidate_cause"] == true_cause
        ].tolist()

        if not matches:
            raise ValueError(
                f"Cause '{true_cause}' was not found in baseline candidates."
            )

        baseline_rank = int(matches[0]) + 1
        effective_k = min(top_k, int(evaluation["total_candidates"]))

        rows.append({
            "seed": int(seed),
            "fault_cause": true_cause,
            "echomind_rank": echomind_rank,
            "baseline_rank": baseline_rank,
            "echomind_top_1": echomind_rank == 1,
            "baseline_top_1": baseline_rank == 1,
            "echomind_top_k": echomind_rank <= effective_k,
            "baseline_top_k": baseline_rank <= effective_k,
            "echomind_reciprocal_rank": 1.0 / echomind_rank,
            "baseline_reciprocal_rank": 1.0 / baseline_rank,
        })

    per_seed = pd.DataFrame(rows)

    summary = {
        "experiments": len(per_seed),
        "true_cause": true_cause,
        "top_k": top_k,
        "echomind_top_1_accuracy": float(
            per_seed["echomind_top_1"].mean()
        ),
        "baseline_top_1_accuracy": float(
            per_seed["baseline_top_1"].mean()
        ),
        "echomind_top_k_accuracy": float(
            per_seed["echomind_top_k"].mean()
        ),
        "baseline_top_k_accuracy": float(
            per_seed["baseline_top_k"].mean()
        ),
        "echomind_mrr": float(
            per_seed["echomind_reciprocal_rank"].mean()
        ),
        "baseline_mrr": float(
            per_seed["baseline_reciprocal_rank"].mean()
        ),
    }

    return summary, per_seed


def run_multi_cause_evaluation(
    seeds=None,
    steps=200,
    fault_step=100,
    top_k=3,
    window=5,
    fault_causes=None,
):
    """
    Compare EchoMind and the baseline across multiple injected causes.

    Each cause is genuinely injected into its own simulation.
    Results are grouped by fault cause.
    """

    if seeds is None:
        seeds = range(10, 20)

    seeds = list(seeds)

    if not seeds:
        raise ValueError("Provide at least one seed.")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    if any(
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        for seed in seeds
    ):
        raise ValueError("Every seed must be an integer.")

    if fault_causes is None:
        fault_causes = list(FAULT_CAUSES)
    else:
        fault_causes = list(fault_causes)

    if not fault_causes:
        raise ValueError("Provide at least one fault cause.")

    if len(set(fault_causes)) != len(fault_causes):
        raise ValueError("Fault causes must be unique.")

    unsupported = set(fault_causes) - set(FAULT_CAUSES)

    if unsupported:
        raise ValueError(
            f"Unsupported fault causes: {sorted(unsupported)}"
        )

    rows = []

    for cause in fault_causes:
        summary, per_seed = run_comparison_evaluation(
            seeds=seeds,
            steps=steps,
            fault_step=fault_step,
            true_cause=cause,
            top_k=top_k,
            window=window,
        )

        rows.extend(per_seed.to_dict(orient="records"))

    results = pd.DataFrame(rows)

    summary = (
        results.groupby("fault_cause")
        .agg(
            experiments=("seed", "count"),
            echomind_mean_rank=("echomind_rank", "mean"),
            baseline_mean_rank=("baseline_rank", "mean"),
            echomind_top_1_accuracy=("echomind_top_1", "mean"),
            baseline_top_1_accuracy=("baseline_top_1", "mean"),
            echomind_top_k_accuracy=("echomind_top_k", "mean"),
            baseline_top_k_accuracy=("baseline_top_k", "mean"),
            echomind_mrr=("echomind_reciprocal_rank", "mean"),
            baseline_mrr=("baseline_reciprocal_rank", "mean"),
        )
        .reset_index()
    )

    return summary, results