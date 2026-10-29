import numpy as np
import pandas as pd

from src.predictor import FEATURES
from src.simulator import simulate_world


# Variables that can be tested as possible causes.
# animal_population is excluded because it is the outcome we measure.
INTERVENTION_CANDIDATES = [
    feature
    for feature in FEATURES
    if feature != "animal_population"
]


def test_interventions(
    original_df,
    fault_step,
    seed=42,
    candidate_variables=None,
):
    """
    Test possible interventions against a faulty baseline simulation.

    For each candidate variable:
    1. Estimate its normal value using pre-fault observations.
    2. Simulate the world with that variable restored to its normal value.
    3. Compare the final animal population with the faulty baseline.
    4. Rank interventions by their observed improvement.

    Important:
    These results describe the simulator only. They do not prove
    real-world causation.
    """

    # --------------------------------------------------
    # 1. Validate the input DataFrame
    # --------------------------------------------------
    if original_df is None:
        raise ValueError("original_df cannot be None.")

    if not isinstance(original_df, pd.DataFrame):
        raise TypeError("original_df must be a pandas DataFrame.")

    if original_df.empty:
        raise ValueError("original_df cannot be empty.")

    required_columns = {"step", *FEATURES}
    missing_columns = required_columns - set(original_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Work on a copy so the original DataFrame is not modified.
    df = original_df.copy()
    df = df.sort_values("step").reset_index(drop=True)

    steps = len(df)

    # --------------------------------------------------
    # 2. Validate the fault step
    # --------------------------------------------------
    if (
        not isinstance(fault_step, (int, np.integer))
        or isinstance(fault_step, (bool, np.bool_))
    ):
        raise ValueError("fault_step must be an integer.")

    fault_step = int(fault_step)

    if not 0 <= fault_step < steps:
        raise ValueError(
            "fault_step must be between 0 and steps - 1."
        )

    # Confirm that steps are consecutive: 0, 1, 2, ...
    expected_steps = np.arange(steps)

    if not np.array_equal(
        df["step"].to_numpy(),
        expected_steps,
    ):
        raise ValueError(
            "The step column must contain consecutive values "
            "starting at 0."
        )

    # --------------------------------------------------
    # 3. Validate the simulation data
    # --------------------------------------------------
    numeric_columns = ["step", *FEATURES]

    for column in numeric_columns:
        try:
            df[column] = pd.to_numeric(
                df[column],
                errors="raise",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Column '{column}' must contain numeric values."
            ) from exc

        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise ValueError(
                f"Column '{column}' contains missing or "
                "non-finite numeric values."
            )

    # --------------------------------------------------
    # 4. Validate intervention candidates
    # --------------------------------------------------
    if candidate_variables is None:
        candidate_variables = INTERVENTION_CANDIDATES.copy()
    else:
        if isinstance(candidate_variables, str):
            candidate_variables = [candidate_variables]
        else:
            candidate_variables = list(candidate_variables)

    if len(candidate_variables) != len(set(candidate_variables)):
        raise ValueError(
            "candidate_variables must not contain duplicates."
        )

    invalid_candidates = (
        set(candidate_variables) - set(INTERVENTION_CANDIDATES)
    )

    if invalid_candidates:
        raise ValueError(
            "Unsupported intervention candidates: "
            f"{sorted(invalid_candidates)}. "
            "Direct interventions on animal_population are excluded "
            "because it is the outcome."
        )

    # --------------------------------------------------
    # 5. Get observations before the fault
    # --------------------------------------------------
    normal_df = df[df["step"] < fault_step]

    if len(normal_df) < 10:
        raise ValueError(
            "Need at least 10 observations before the fault."
        )

    # If no candidates are supplied, return an empty result
    # with a consistent set of columns.
    result_columns = [
        "candidate_cause",
        "restoration_target",
        "faulty_final_population",
        "intervention_final_population",
        "improvement",
        "interpretation",
    ]

    if not candidate_variables:
        return pd.DataFrame(columns=result_columns)

    # --------------------------------------------------
    # 6. Estimate normal operating levels
    # --------------------------------------------------
    # Use pre-fault observations only. This avoids using
    # post-fault measurements to estimate normal values.
    targets = {
        variable: float(normal_df[variable].mean())
        for variable in candidate_variables
    }

    # --------------------------------------------------
    # 7. Measure the faulty baseline outcome
    # --------------------------------------------------
    # Use the mean population during the final 10% of steps.
    tail_count = max(1, steps // 10)

    faulty_outcome = float(
        df["animal_population"].tail(tail_count).mean()
    )

    # --------------------------------------------------
    # 8. Test each candidate intervention
    # --------------------------------------------------
    rows = []

    for variable in candidate_variables:
        target_value = targets[variable]

        # Simulate the same scenario with one variable restored
        # to its estimated pre-fault operating level.
        intervention_df = simulate_world(
            steps=steps,
            fault_step=fault_step,
            seed=seed,
            intervention={
                "variable": variable,
                "value": target_value,
            },
        )

        if (
            intervention_df is None
            or intervention_df.empty
            or "animal_population" not in intervention_df.columns
        ):
            raise ValueError(
                "The simulator returned invalid intervention results "
                f"for '{variable}'."
            )

        intervention_outcome = float(
            intervention_df["animal_population"]
            .tail(tail_count)
            .mean()
        )

        # Positive improvement means a higher final population
        # than the faulty baseline.
        improvement = intervention_outcome - faulty_outcome

        if improvement > 0.01:
            interpretation = (
                "Population improved in this simulation."
            )
        elif improvement < -0.01:
            interpretation = (
                "Population declined in this simulation."
            )
        else:
            interpretation = (
                "Little or no change in this simulation."
            )

        rows.append({
            "candidate_cause": variable,
            "restoration_target": round(target_value, 3),
            "faulty_final_population": round(faulty_outcome, 3),
            "intervention_final_population": round(
                intervention_outcome,
                3,
            ),
            "improvement": round(improvement, 3),
            "interpretation": interpretation,
        })

    # --------------------------------------------------
    # 9. Rank interventions by improvement
    # --------------------------------------------------
    result = pd.DataFrame(
        rows,
        columns=result_columns,
    )

    result = result.sort_values(
        by="improvement",
        ascending=False,
    ).reset_index(drop=True)

    return result