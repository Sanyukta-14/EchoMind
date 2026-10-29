
import numpy as np
import pandas as pd

from src.predictor import FEATURES


# Known direct dependencies in the synthetic simulator.
# These describe the model's equations, not the injected fault label.
CAUSAL_PARENTS = {
    "temperature": [],
    "water_level": ["temperature"],
    "plant_health": ["temperature", "water_level"],
    "food_supply": ["plant_health"],
    "animal_population": ["food_supply"],
}

# Small adjustment used only when a parent and child both show
# sustained deviations at the same time or the parent changes first.
DOWNSTREAM_PENALTY = 0.04


def rank_candidates(df, fault_step, window=5):
    """
    Rank candidate causes using anomaly evidence and known causal order.

    The fault_cause metadata is never used to calculate rankings.
    Scores are ranking scores, not probabilities.
    """

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input data must be a non-empty DataFrame.")

    if "step" not in df.columns:
        raise ValueError("Input data must contain a 'step' column.")

    if not isinstance(window, (int, np.integer)) or isinstance(
        window, (bool, np.bool_)
    ) or window < 1:
        raise ValueError("window must be a positive integer.")

    if not FEATURES:
        raise ValueError("FEATURES must contain at least one variable.")

    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    data["step"] = pd.to_numeric(data["step"], errors="coerce")

    if data["step"].isna().any():
        raise ValueError("'step' must contain numeric values.")

    if data["step"].duplicated().any():
        raise ValueError("'step' must contain unique values.")

    data = data.sort_values("step").reset_index(drop=True)

    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    values = data[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(
            "Feature columns contain missing or infinite values."
        )

    steps = data["step"].to_numpy(dtype=np.float64)
    baseline_positions = np.flatnonzero(steps < fault_step)
    post_positions = np.flatnonzero(steps >= fault_step)

    if len(baseline_positions) < 10:
        raise ValueError("Need at least 10 observations before the fault.")

    if len(post_positions) == 0:
        raise ValueError("No observations exist at or after the fault.")

    early_positions = post_positions[:window]
    candidates = []

    for feature in FEATURES:
        feature_values = data[feature].to_numpy(dtype=np.float64)

        baseline_values = feature_values[baseline_positions]
        post_values = feature_values[post_positions]
        early_values = feature_values[early_positions]

        baseline_mean = float(np.mean(baseline_values))
        baseline_std = float(np.std(baseline_values))
        scale = max(baseline_std, 1.0)

        early_shift = (
            float(np.mean(early_values)) - baseline_mean
        ) / scale

        overall_shift = (
            float(np.mean(post_values)) - baseline_mean
        ) / scale

        early_shift_score = abs(early_shift)
        mean_shift_score = abs(overall_shift)

        post_deviations = np.abs(post_values - baseline_mean) / scale
        max_deviation = float(np.max(post_deviations))
        persistence_score = float(np.mean(post_deviations > 2.0))

        full_deviation = np.abs(feature_values - baseline_mean) / scale
        first_deviation = None

        for position in post_positions:
            end = position + window
            if end > len(data):
                break

            segment_positions = np.arange(position, end)

            if not np.all(
                np.diff(steps[segment_positions]) == 1
            ):
                continue

            if np.all(full_deviation[segment_positions] > 2.0):
                first_deviation = int(steps[position])
                break

        if first_deviation is None:
            delay = None
            timing_score = 0.0
        else:
            delay = max(0, first_deviation - fault_step)
            timing_score = 1.0 / (1.0 + delay)

        normalized_early = (
            early_shift_score / (1.0 + early_shift_score)
        )
        normalized_mean = (
            mean_shift_score / (1.0 + mean_shift_score)
        )
        normalized_max = (
            max_deviation / (1.0 + max_deviation)
        )

        raw_score = (
            0.35 * normalized_early
            + 0.25 * normalized_mean
            + 0.15 * normalized_max
            + 0.15 * persistence_score
            + 0.10 * timing_score
        )

        direction = (
            "Increased" if overall_shift > 0
            else "Decreased" if overall_shift < 0
            else "Unchanged"
        )

        candidates.append({
            "candidate_cause": feature,
            "first_sustained_deviation": (
                first_deviation if first_deviation is not None
                else "Not found"
            ),
            "deviation_delay": (
                delay if delay is not None else "Not found"
            ),
            "early_shift_score": round(early_shift_score, 3),
            "mean_shift_score": round(mean_shift_score, 3),
            "max_deviation_score": round(max_deviation, 3),
            "persistence_score": round(persistence_score, 3),
            "timing_score": round(timing_score, 3),
            "direction": direction,
            "_raw_score": float(raw_score),
            "_first_step": first_deviation,
        })

    # Apply the causal-order adjustment only after measuring all signals.
    onset_by_feature = {
        item["candidate_cause"]: item["_first_step"]
        for item in candidates
    }

    for item in candidates:
        feature = item["candidate_cause"]
        child_onset = onset_by_feature[feature]
        penalty = 0.0

        if child_onset is not None:
            for parent in CAUSAL_PARENTS.get(feature, []):
                parent_onset = onset_by_feature.get(parent)

                # If both variables change together, the parent is
                # treated as at least as early as its downstream effect.
                if (
                    parent_onset is not None
                    and parent_onset <= child_onset
                ):
                    penalty += DOWNSTREAM_PENALTY

        adjusted_score = max(0.0, item["_raw_score"] - penalty)
        item["hypothesis_score"] = round(adjusted_score, 3)
        del item["_raw_score"]
        del item["_first_step"]

    result = pd.DataFrame(candidates)

    return result.sort_values(
        by=["hypothesis_score", "early_shift_score"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)
