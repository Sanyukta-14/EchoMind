from pathlib import Path

from src.comparison_evaluation import run_multi_cause_evaluation


def main():
    # Save evaluation results in the EchoMind project root.
    project_root = Path(__file__).resolve().parent

    # Unseen seeds, separate from development seeds 10-29.
    seeds = range(100, 150)

    print("=" * 65)
    print("ECHOMIND — UNSEEN-SEED HOLDOUT EVALUATION")
    print("=" * 65)
    print("Fault scenarios: 4")
    print("Random seeds per scenario: 50")
    print("Expected experiments: 200")
    print()

    summary, results = run_multi_cause_evaluation(
        seeds=seeds,
        steps=200,
        fault_step=100,
        top_k=3,
        window=5,
    )

    if "fault_cause" not in results.columns:
        raise ValueError("Missing fault_cause column in evaluation results.")

    counts = results.groupby("fault_cause").size()

    print("\nEXPERIMENT COUNTS")
    print("-" * 65)
    print(counts.to_string())
    print(f"\nTotal experiments: {len(results)}")

    if len(results) != 200:
        raise ValueError(
            f"Expected 200 experiments, received {len(results)}."
        )

    if len(counts) != 4 or not (counts == 50).all():
        raise ValueError(
            "Expected 50 experiments for each of four fault scenarios."
        )

    summary_path = project_root / "echomind_holdout_summary.csv"
    details_path = project_root / "echomind_holdout_details.csv"

    summary.to_csv(summary_path, index=False)
    results.to_csv(details_path, index=False)

    print("\nHOLDOUT SUMMARY")
    print("-" * 65)
    print(summary.to_string(index=False))

    print("\nSAVED FILES")
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")

    print("\nHoldout evaluation completed successfully.")


if __name__ == "__main__":
    main()
