import pandas as pd


def build_evaluation_report(summary, results=None):
    """
    Build a readable report from EchoMind evaluation results.

    Parameters
    ----------
    summary : pandas.DataFrame
        Grouped results from run_multi_cause_evaluation.
    results : pandas.DataFrame, optional
        Individual experiment results.

    Returns
    -------
    dict
        Overall metrics, per-cause metrics, and interpretation.
    """
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        raise ValueError("summary must be a non-empty DataFrame.")

    required = {
        "fault_cause",
        "experiments",
        "echomind_top_1_accuracy",
        "baseline_top_1_accuracy",
        "echomind_top_k_accuracy",
        "baseline_top_k_accuracy",
        "echomind_mrr",
        "baseline_mrr",
    }

    missing = required - set(summary.columns)

    if missing:
        raise ValueError(
            f"Evaluation summary is missing columns: {sorted(missing)}"
        )

    total_experiments = int(summary["experiments"].sum())

    # Accuracy and MRR are weighted by the number of experiments
    # per fault cause, so unequal group sizes are handled correctly.
    weights = summary["experiments"].astype(float)
    total_weight = float(weights.sum())

    if total_weight <= 0:
        raise ValueError("The summary must contain positive experiment counts.")

    def weighted_average(column):
        values = pd.to_numeric(summary[column], errors="coerce")

        if values.isna().any():
            raise ValueError(
                f"Column '{column}' contains invalid metric values."
            )

        return float((values * weights).sum() / total_weight)

    echomind_accuracy = weighted_average("echomind_top_1_accuracy")
    baseline_accuracy = weighted_average("baseline_top_1_accuracy")
    echomind_top_k = weighted_average("echomind_top_k_accuracy")
    baseline_top_k = weighted_average("baseline_top_k_accuracy")
    echomind_mrr = weighted_average("echomind_mrr")
    baseline_mrr = weighted_average("baseline_mrr")

    report = {
        "total_experiments": total_experiments,
        "number_of_fault_causes": int(summary["fault_cause"].nunique()),
        "echomind_top_1_accuracy": echomind_accuracy,
        "baseline_top_1_accuracy": baseline_accuracy,
        "top_1_difference_percentage_points": (
            echomind_accuracy - baseline_accuracy
        ) * 100,
        "echomind_top_k_accuracy": echomind_top_k,
        "baseline_top_k_accuracy": baseline_top_k,
        "echomind_mrr": echomind_mrr,
        "baseline_mrr": baseline_mrr,
        "per_cause": summary.copy(),
    }

    if results is not None:
        if not isinstance(results, pd.DataFrame):
            raise TypeError("results must be a pandas DataFrame.")

        report["individual_results"] = results.copy()

    return report