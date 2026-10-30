from pathlib import Path

from src.comparison_evaluation import run_multi_cause_evaluation


def main():
    output_dir = Path(__file__).resolve().parent

    # Unseen seeds: separate from the multi-cause evaluation.
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

    summary_path = output_dir / "echomind_holdout_summary.csv"
    results_path = output_dir / "echomind_holdout_details.csv"

    summary.to_csv(summary_path, index=False)
    results.to_csv(results_path, index=False)

    print("\nHOLDOUT SUMMARY")
    print("-" * 65)
    print(summary.to_string(index=False))

    print("\nEXPERIMENT COUNTS")
    print("-" * 65)

    print(results.groupby("fault_cause").size().to_string())
    print(f"\nTotal experiments: {len(results)}")

    print("\nSAVED FILES")
    print(f"Summary: {summary_path}")
    print(f"Details: {results_path}")

    print("\nHoldout evaluation completed.")


if __name__ == "__main__":
    main()
