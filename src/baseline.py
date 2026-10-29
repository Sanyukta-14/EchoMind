import pandas as pd

from src.predictor import FEATURES


def rank_baseline_candidates(df, fault_step, window=5):
    """
    Simple baseline that ranks variables by their absolute
    change between pre-fault and post-fault averages.

    This is a comparison method, not proof of causation.
    """

    if df is None or df.empty:
        raise ValueError("df cannot be empty.")

    required_columns = {"step", *FEATURES}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer.")

    if not isinstance(fault_step, int) or fault_step < 1:
        raise ValueError("fault_step must be a positive integer.")

    before = df[df["step"] < fault_step].tail(window)
    after = df[df["step"] >= fault_step].head(window)

    if before.empty or after.empty:
        raise ValueError(
            "Not enough observations before or after the fault."
        )

    rows = []

    for feature in FEATURES:
        if feature == "animal_population":
            continue

        before_mean = float(before[feature].mean())
        after_mean = float(after[feature].mean())

        change = after_mean - before_mean

        rows.append({
            "candidate_cause": feature,
            "before_mean": round(before_mean, 3),
            "after_mean": round(after_mean, 3),
            "absolute_change": round(abs(change), 3),
            "direction": (
                "Increased" if change > 0
                else "Decreased" if change < 0
                else "Unchanged"
            ),
        })

    result = pd.DataFrame(rows)

    return result.sort_values(
        "absolute_change",
        ascending=False,
    ).reset_index(drop=True)