
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(__file__).resolve().parent / "echomind_holdout_details.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "seed", "fault_cause",
        "echomind_rank", "baseline_rank",
        "echomind_top_1", "baseline_top_1",
        "echomind_top_k", "baseline_top_k",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Convert CSV boolean values safely.
    def as_bool(series):
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False)
            .astype(bool)
        )

    for col in [
        "echomind_top_1", "baseline_top_1",
        "echomind_top_k", "baseline_top_k",
    ]:
        df[col] = as_bool(df[col])

    # Cases where EchoMind did not rank the true cause first.
    misses = df.loc[~df["echomind_top_1"]].copy()

    def compare(row):
        e = row["echomind_top_1"]
        b = row["baseline_top_1"]

        if e and b:
            return "Both correct"
        if e and not b:
            return "EchoMind only correct"
        if not e and b:
            return "Baseline only correct"
        return "Both missed top-1"

    df["top_1_comparison"] = df.apply(compare, axis=1)

    # Summarize performance for each fault type.
    by_cause = (
        df.groupby("fault_cause")
        .agg(
            experiments=("seed", "count"),
            echomind_top_1_accuracy=("echomind_top_1", "mean"),
            baseline_top_1_accuracy=("baseline_top_1", "mean"),
            echomind_top_k_accuracy=("echomind_top_k", "mean"),
            baseline_top_k_accuracy=("baseline_top_k", "mean"),
            echomind_mean_rank=("echomind_rank", "mean"),
            baseline_mean_rank=("baseline_rank", "mean"),
            echomind_top_1_misses=(
                "echomind_top_1", lambda s: int((~s).sum())
            ),
            baseline_top_1_misses=(
                "baseline_top_1", lambda s: int((~s).sum())
            ),
        )
        .reset_index()
    )

    accuracy_columns = [
        "echomind_top_1_accuracy",
        "baseline_top_1_accuracy",
        "echomind_top_k_accuracy",
        "baseline_top_k_accuracy",
    ]

    for col in accuracy_columns:
        by_cause[col] = (by_cause[col] * 100).round(2)

    comparison_counts = (
        df["top_1_comparison"]
        .value_counts()
        .rename_axis("comparison")
        .reset_index(name="experiments")
    )

    # Save all reports beside this script.
    output_dir = INPUT_FILE.parent

    misses.to_csv(output_dir / "echomind_holdout_misses.csv", index=False)
    by_cause.to_csv(
        output_dir / "echomind_failure_analysis_by_cause.csv",
        index=False,
    )
    comparison_counts.to_csv(
        output_dir / "echomind_baseline_comparison_counts.csv",
        index=False,
    )

    print("=" * 60)
    print("ECHOMIND HOLDOUT FAILURE ANALYSIS")
    print("=" * 60)
    print(f"Total experiments: {len(df)}")
    print(f"EchoMind top-1 misses: {(~df['echomind_top_1']).sum()}")
    print(f"Baseline top-1 misses: {(~df['baseline_top_1']).sum()}")
    print(f"EchoMind top-3 misses: {(~df['echomind_top_k']).sum()}")

    print("\n=== MISSED CASES ===")
    if misses.empty:
        print("No EchoMind top-1 misses.")
    else:
        print(
            misses[
                [
                    "seed", "fault_cause",
                    "echomind_rank", "baseline_rank",
                    "baseline_top_1", "echomind_top_k",
                ]
            ].to_string(index=False)
        )

    print("\n=== RESULTS BY CAUSE ===")
    print(by_cause.to_string(index=False))

    print("\n=== BASELINE COMPARISON ===")
    print(comparison_counts.to_string(index=False))

    print("\nSaved reports:")
    print("echomind_holdout_misses.csv")
    print("echomind_failure_analysis_by_cause.csv")
    print("echomind_baseline_comparison_counts.csv")


if __name__ == "__main__":
    main()
