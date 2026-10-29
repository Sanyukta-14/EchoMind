from pathlib import Path

from src.comparison_evaluation import run_multi_cause_evaluation


def main():
    output_dir = Path(__file__).resolve().parent

    # Run 20 seeds for each of the four fault scenarios.
    seeds = range(10, 30)

    print("=" * 65)
    print("ECHOMIND — MULTI-CAUSE EVALUATION")
    print("=" * 65)
    print("Fault scenarios: 4")
    print("Random seeds per scenario: 20")
    print("Expected experiments: 80")
    print()

    summary, results = run_multi_cause_evaluation(
        seeds=seeds,
        steps=200,
        fault_step=100,
        top_k=3,
        window=5,
    )

    summary_path = output_dir / "echomind_multi_cause_summary.csv"
    results_path = output_dir / "echomind_multi_cause_details.csv"

    summary.to_csv(summary_path, index=False)
    results.to_csv(results_path, index=False)

    print("\nSUMMARY")
    print("-" * 65)
    print(summary.to_string(index=False))

    print("\nEXPERIMENT COUNTS")
    print("-" * 65)

    if "fault_cause" in results.columns:
        print(results.groupby("fault_cause").size().to_string())
        print(f"\nTotal experiments: {len(results)}")

    print("\nSAVED FILES")
    print(f"Summary: {summary_path}")
    print(f"Details: {results_path}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()