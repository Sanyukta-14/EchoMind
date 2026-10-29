
import pandas as pd

from src.hypothesis_generator import rank_candidates


def evaluate_hypotheses(df, fault_step, true_cause, top_k=3):
    """
    Evaluate whether the known injected fault appears in the
    hypothesis ranking.

    This evaluates a simulated experiment with a known cause.
    It does not prove causation in real-world data.
    """
    if true_cause not in df.columns:
        raise ValueError(f"Unknown cause: {true_cause}")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    ranking = rank_candidates(df, fault_step=fault_step)

    if "candidate_cause" not in ranking.columns:
        raise ValueError("Ranking does not contain candidate_cause.")

    candidates = ranking["candidate_cause"].tolist()

    if true_cause not in candidates:
        raise ValueError(
            f"True cause '{true_cause}' is not in the hypothesis candidates."
        )

    rank = candidates.index(true_cause) + 1

    return {
        "true_cause": true_cause,
        "predicted_cause": candidates[0],
        "true_cause_rank": rank,
        "top_k": top_k,
        "found_in_top_k": rank <= min(top_k, len(candidates)),
        "total_candidates": len(candidates),
        "ranking": ranking,
    }
