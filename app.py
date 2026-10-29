import pandas as pd
import streamlit as st

from src.reporting import build_evaluation_report
from src.simulator import simulate_world
from src.predictor import train_predictor
from src.anomaly_detector import detect_anomalies
from src.hypothesis_generator import rank_candidates
from src.intervention_engine import test_interventions, INTERVENTION_CANDIDATES
from src.evaluation import evaluate_hypotheses
from src.baseline import rank_baseline_candidates
from src.comparison_evaluation import (
    run_comparison_evaluation,
    run_multi_cause_evaluation,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="EchoMind | Causal Investigation",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 EchoMind")
st.subheader("Offline Causal Investigation Laboratory")
st.markdown(
    """
    EchoMind investigates failures in a simulated ecosystem.

    **Pipeline:** Simulation → Prediction → Anomaly Detection →
    Hypothesis Generation → Intervention Testing → Evaluation.
    """
)
st.caption(
    "This is a controlled synthetic experiment. Rankings and interventions "
    "generate evidence about this simulator; they do not prove real-world causation."
)
st.divider()


# ============================================================
# SIDEBAR SETTINGS
# ============================================================
st.sidebar.header("⚙️ Experiment Settings")

steps = st.sidebar.slider(
    "Simulation steps", min_value=100, max_value=500, value=200, step=25
)
fault_step = st.sidebar.slider(
    "Fault begins at step",
    min_value=20,
    max_value=steps - 1,
    value=min(100, steps - 1),
)
seed = st.sidebar.number_input(
    "Random seed", min_value=0, max_value=999999, value=42, step=1
)
st.sidebar.caption(
    "Runs locally with Python and scikit-learn. No external AI API is required."
)


# ============================================================
# RUN INVESTIGATION
# ============================================================
if st.button("🔍 Run Investigation", type="primary", key="run_investigation"):
    try:
        with st.spinner(
            "Simulating the ecosystem and running the investigation pipeline..."
        ):
            df = simulate_world(
                steps=int(steps),
                fault_step=int(fault_step),
                seed=int(seed),
            )

            model = train_predictor(
                df,
                window=5,
                normal_end=int(fault_step),
            )

            anomaly_results, threshold = detect_anomalies(
                df,
                model,
                normal_end=int(fault_step),
                window=5,
            )

            candidates = rank_candidates(df, fault_step=int(fault_step))

            candidate_variables = [
                variable
                for variable in candidates["candidate_cause"].tolist()
                if variable in INTERVENTION_CANDIDATES
            ]

            interventions = test_interventions(
                original_df=df,
                fault_step=int(fault_step),
                seed=int(seed),
                candidate_variables=candidate_variables,
            )

        # Persist every result required by the display sections.
        st.session_state["echo_df"] = df
        st.session_state["echo_results"] = anomaly_results
        st.session_state["echo_candidates"] = candidates
        st.session_state["echo_interventions"] = interventions
        st.session_state["echo_threshold"] = float(threshold)
        st.session_state["echo_fault_step"] = int(fault_step)
        st.session_state["echo_seed"] = int(seed)

        # Invalidate single-run evaluation results after a new simulation.
        st.session_state.pop("echo_evaluation", None)
        st.session_state.pop("baseline_comparison", None)

        st.success("Investigation completed successfully.")
    except Exception as exc:
        st.error(f"Investigation failed: {exc}")
        st.exception(exc)


# ============================================================
# MAIN INVESTIGATION RESULTS
# ============================================================
if "echo_df" not in st.session_state:
    st.info("Choose settings in the sidebar, then click **Run Investigation**.")
    st.markdown(
        """
        ### Investigation pipeline
        1. **Simulate** a world with a controlled fault.
        2. **Predict** expected behavior from historical observations.
        3. **Detect** unusually large prediction errors.
        4. **Rank** possible explanations.
        5. **Test interventions** and measure outcome changes.
        6. **Evaluate** ranking quality across one or more random seeds.
        """
    )
else:
    df = st.session_state["echo_df"]
    anomaly_results = st.session_state["echo_results"]
    candidates = st.session_state["echo_candidates"]
    interventions = st.session_state["echo_interventions"]
    threshold = st.session_state["echo_threshold"]
    actual_fault_step = st.session_state["echo_fault_step"]
    actual_seed = st.session_state["echo_seed"]

    detected = anomaly_results[anomaly_results["is_anomaly"]]

    st.header("📊 Investigation Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Injected fault step", f"Step {actual_fault_step}")
    col2.metric("Random seed", actual_seed)
    col3.metric("Anomalous observations", int(anomaly_results["is_anomaly"].sum()))
    col4.metric("Candidate variables", len(candidates))

    if not detected.empty:
        col5, col6 = st.columns(2)
        col5.metric("First detected step", int(detected["step"].iloc[0]))
        col6.metric("Anomaly threshold", f"{threshold:.3f}")
    else:
        st.metric("Anomaly threshold", f"{threshold:.3f}")

    st.caption(
        "The fault step is known to the experiment controller. The anomaly detector "
        "uses prediction errors and does not use the fault label as its anomaly score."
    )
    st.divider()

    # --------------------------------------------------------
    # ECOSYSTEM
    # --------------------------------------------------------
    st.header("🌱 Ecosystem Behavior")
    st.write(
        "Track temperature, water level, plant health, food supply, and animal "
        "population over the simulated timeline."
    )
    plot_columns = [
        column
        for column in [
            "temperature",
            "water_level",
            "plant_health",
            "food_supply",
            "animal_population",
        ]
        if column in df.columns
    ]
    if "step" in df.columns and plot_columns:
        st.line_chart(df.set_index("step")[plot_columns], height=350)

    with st.expander("View ecosystem data"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # --------------------------------------------------------
    # ANOMALY DETECTION
    # --------------------------------------------------------
    st.header("🚨 AI Anomaly Detection")
    st.write(
        "The predictor estimates expected observations from recent history. "
        "Larger prediction errors receive higher anomaly scores."
    )
    st.metric("Current anomaly threshold", f"{threshold:.3f}")

    if "step" in anomaly_results.columns and "anomaly_score" in anomaly_results.columns:
        st.line_chart(
            anomaly_results.set_index("step")[["anomaly_score"]],
            height=250,
        )

    if detected.empty:
        st.warning(
            "No observations crossed the current anomaly threshold. This does not "
            "necessarily mean that no fault occurred."
        )
    else:
        st.subheader("Detected anomalous observations")
        st.dataframe(detected, use_container_width=True, hide_index=True)

    with st.expander("View all anomaly scores"):
        display_columns = [
            column
            for column in ["step", "anomaly_score", "is_anomaly"]
            if column in anomaly_results.columns
        ]
        st.dataframe(
            anomaly_results[display_columns],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # --------------------------------------------------------
    # HYPOTHESIS GENERATION
    # --------------------------------------------------------
    st.header("🕵️ Possible Hidden Causes")
    st.write(
        "Candidates are ranked by changes in ecosystem variables around the fault."
    )
    st.warning(
        "A high hypothesis score means a variable changed in a way that may be "
        "relevant. It is not, by itself, proof of causation."
    )
    st.dataframe(candidates, use_container_width=True, hide_index=True)

    if not candidates.empty:
        top_candidate = candidates.iloc[0]
        st.metric(
            "Highest-ranked candidate",
            str(top_candidate.get("candidate_cause", "Unavailable")),
        )
        if "hypothesis_score" in candidates.columns:
            st.write(f"Hypothesis score: {float(top_candidate['hypothesis_score']):.3f}")

    st.divider()

    # --------------------------------------------------------
    # SINGLE-RUN EVALUATION
    # --------------------------------------------------------
    st.header("🎯 Hypothesis Ranking Evaluation")
    st.write(
        "Because this is a controlled simulation, the injected cause is known to "
        "the evaluator. Select the cause used for the experiment."
    )

    available_causes = [
        "temperature",
        "water_level",
        "plant_health",
        "food_supply",
    ]
    true_cause = st.selectbox(
        "Known injected fault",
        options=available_causes,
        index=1,
        key="echo_true_cause",
    )
    max_k = max(1, len(candidates))
    top_k = st.slider(
        "Evaluate top K hypotheses",
        min_value=1,
        max_value=max_k,
        value=min(3, max_k),
        key="echo_top_k",
    )

    if st.button("🎯 Evaluate Hypothesis Ranking", key="echo_evaluate"):
        try:
            evaluation = evaluate_hypotheses(
                df=df,
                fault_step=actual_fault_step,
                true_cause=true_cause,
                top_k=top_k,
            )
            eval_col1, eval_col2, eval_col3 = st.columns(3)
            eval_col1.metric("Known cause rank", evaluation["true_cause_rank"])
            eval_col2.metric("Top-ranked prediction", evaluation["predicted_cause"])
            eval_col3.metric(
                f"Found in Top {top_k}",
                "Yes" if evaluation["found_in_top_k"] else "No",
            )
            st.subheader("Evaluation ranking")
            st.dataframe(
                evaluation["ranking"],
                use_container_width=True,
                hide_index=True,
            )
            st.session_state["echo_evaluation"] = {
                "true_cause": evaluation["true_cause"],
                "predicted_cause": evaluation["predicted_cause"],
                "true_cause_rank": evaluation["true_cause_rank"],
                "top_k": evaluation["top_k"],
                "found_in_top_k": evaluation["found_in_top_k"],
                "total_candidates": evaluation["total_candidates"],
            }
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Evaluation failed: {exc}")

    saved_evaluation = st.session_state.get("echo_evaluation")
    if saved_evaluation:
        st.caption(
            f"Last evaluation: known cause = {saved_evaluation['true_cause']}; "
            f"rank = {saved_evaluation['true_cause_rank']} of "
            f"{saved_evaluation['total_candidates']}; found in top "
            f"{saved_evaluation['top_k']} = {saved_evaluation['found_in_top_k']}."
        )

    st.divider()

    # --------------------------------------------------------
    # INTERVENTION TESTING
    # --------------------------------------------------------
    st.header("🧪 Counterfactual Intervention Testing")
    st.write(
        "The intervention engine tests restoring candidate variables toward their "
        "pre-fault averages, then compares final animal population with the faulty run."
    )
    st.caption(
        "This tests a specific restoration policy using the same random seed. It "
        "does not test every possible intervention or establish real-world causation."
    )

    if interventions is None or interventions.empty:
        st.info("No intervention results are available for this run.")
    else:
        st.dataframe(interventions, use_container_width=True, hide_index=True)
        best = interventions.iloc[0]
        st.subheader("Best tested intervention")
        col1, col2, col3 = st.columns(3)
        col1.metric("Variable restored", str(best.get("candidate_cause", "Unknown")))
        if "faulty_final_population" in interventions.columns:
            col2.metric(
                "Faulty final population",
                f"{float(best['faulty_final_population']):.2f}",
            )
        if "improvement" in interventions.columns:
            improvement = float(best["improvement"])
            col3.metric("Intervention improvement", f"{improvement:+.2f}")
            if improvement > 0:
                st.success("This intervention improved the measured outcome in this run.")
            elif improvement < 0:
                st.warning("This intervention reduced the measured outcome in this run.")
            else:
                st.info("No outcome change was measured at the displayed precision.")

    st.divider()

    # --------------------------------------------------------
    # EXPORT SINGLE-RUN RESULTS
    # --------------------------------------------------------
    st.header("💾 Export Investigation Results")
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            "Download ecosystem data",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="echomind_ecosystem.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download anomaly results",
            data=anomaly_results.to_csv(index=False).encode("utf-8"),
            file_name="echomind_anomalies.csv",
            mime="text/csv",
        )
    with export_col2:
        st.download_button(
            "Download candidate causes",
            data=candidates.to_csv(index=False).encode("utf-8"),
            file_name="echomind_candidates.csv",
            mime="text/csv",
        )
        if interventions is not None:
            st.download_button(
                "Download intervention results",
                data=interventions.to_csv(index=False).encode("utf-8"),
                file_name="echomind_interventions.csv",
                mime="text/csv",
            )


# ============================================================
# BASELINE COMPARISON FOR THE CURRENT INVESTIGATION
# ============================================================
st.divider()
st.header("⚖️ EchoMind vs Simple Baseline")
st.write(
    "The baseline ranks variables by the magnitude of their changes around the "
    "fault. Compare its ranking with EchoMind's hypothesis ranking."
)

if "echo_df" not in st.session_state:
    st.info("Run an investigation above before comparing the rankings.")
elif st.button("Compare EchoMind vs Baseline", key="compare_baseline"):
    try:
        current_df = st.session_state["echo_df"]
        current_fault_step = st.session_state["echo_fault_step"]
        current_candidates = st.session_state["echo_candidates"]

        baseline_results = rank_baseline_candidates(
            df=current_df,
            fault_step=current_fault_step,
            window=5,
        )

        echo_ranking = current_candidates[["candidate_cause"]].copy()
        echo_ranking["echomind_rank"] = range(1, len(echo_ranking) + 1)

        baseline_ranking = baseline_results[["candidate_cause"]].copy()
        baseline_ranking["baseline_rank"] = range(1, len(baseline_ranking) + 1)

        comparison = baseline_results.merge(
            baseline_ranking,
            on="candidate_cause",
            how="left",
        )
        comparison = comparison.merge(
            echo_ranking,
            on="candidate_cause",
            how="outer",
        ).sort_values("baseline_rank", na_position="last")

        st.session_state["baseline_comparison"] = comparison.reset_index(drop=True)
        st.session_state["baseline_results"] = baseline_results
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not compare rankings: {exc}")

if "baseline_comparison" in st.session_state:
    st.subheader("Ranking comparison")
    st.dataframe(
        st.session_state["baseline_comparison"],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download baseline comparison CSV",
        data=st.session_state["baseline_comparison"].to_csv(index=False).encode("utf-8"),
        file_name="echomind_baseline_comparison.csv",
        mime="text/csv",
        key="download_baseline_comparison",
    )


# ============================================================
# MULTI-SEED COMPARISON: ECHOMIND VS BASELINE
# ============================================================
st.divider()
st.header("📈 Multi-Seed Model Evaluation")
st.write(
    "Repeat the comparison with different random seeds. Top-1 accuracy, Top-K "
    "accuracy, and mean reciprocal rank (MRR) summarize how highly each method "
    "ranks the known simulated cause."
)

with st.expander("Configure multi-seed evaluation", expanded=False):
    comparison_runs = st.slider(
        "Number of experiments",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        key="comparison_runs",
    )
    comparison_top_k = st.slider(
        "Top-K threshold",
        min_value=1,
        max_value=5,
        value=3,
        key="comparison_top_k",
    )
    comparison_window = st.slider(
        "Baseline comparison window",
        min_value=1,
        max_value=10,
        value=5,
        key="comparison_window",
    )

if st.button("Run Model Comparison", type="primary", key="run_model_comparison"):
    try:
        with st.spinner("Running comparison experiments..."):
            summary, per_seed = run_comparison_evaluation(
                seeds=range(10, 10 + comparison_runs),
                steps=int(steps),
                fault_step=int(fault_step),
                true_cause="water_level",
                top_k=int(comparison_top_k),
                window=int(comparison_window),
            )
        st.session_state["comparison_summary"] = summary
        st.session_state["comparison_results"] = per_seed
    except Exception as exc:
        st.error(f"Model comparison failed: {exc}")
        st.exception(exc)

if "comparison_summary" in st.session_state:
    summary = st.session_state["comparison_summary"]
    per_seed = st.session_state["comparison_results"]

    st.subheader("Performance summary")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.markdown("**EchoMind**")
        st.metric("Top-1 accuracy", f"{summary['echomind_top_1_accuracy']:.1%}")
        st.metric(
            f"Top-{summary['top_k']} accuracy",
            f"{summary['echomind_top_k_accuracy']:.1%}",
        )
        st.metric("Mean reciprocal rank", f"{summary['echomind_mrr']:.3f}")
    with metric_col2:
        st.markdown("**Baseline**")
        st.metric("Top-1 accuracy", f"{summary['baseline_top_1_accuracy']:.1%}")
        st.metric(
            f"Top-{summary['top_k']} accuracy",
            f"{summary['baseline_top_k_accuracy']:.1%}",
        )
        st.metric("Mean reciprocal rank", f"{summary['baseline_mrr']:.3f}")

    accuracy_chart = pd.DataFrame(
        {
            "Method": ["EchoMind", "Baseline"],
            "Top-1 Accuracy": [
                summary["echomind_top_1_accuracy"],
                summary["baseline_top_1_accuracy"],
            ],
            f"Top-{summary['top_k']} Accuracy": [
                summary["echomind_top_k_accuracy"],
                summary["baseline_top_k_accuracy"],
            ],
        }
    )
    st.subheader("Accuracy comparison")
    st.bar_chart(accuracy_chart.set_index("Method"))
    st.subheader("Results for each seed")
    st.dataframe(per_seed, use_container_width=True, hide_index=True)
    st.download_button(
        "Download multi-seed results CSV",
        data=per_seed.to_csv(index=False).encode("utf-8"),
        file_name="echomind_comparison_evaluation.csv",
        mime="text/csv",
        key="download_comparison_evaluation",
    )
    st.caption(
        f"Completed {summary['experiments']} experiments. Results apply to this "
        "synthetic simulator only."
    )


# ============================================================
# MULTI-CAUSE EVALUATION
# ============================================================
st.divider()
st.header("🧪 Multi-Cause Evaluation")
st.write(
    "Evaluate EchoMind and the baseline across several separately simulated fault "
    "causes. Each cause is injected into its own simulation."
)

with st.expander("Configure multi-cause evaluation", expanded=False):
    multi_cause_runs = st.slider(
        "Seeds per fault cause",
        min_value=3,
        max_value=20,
        value=5,
        key="multi_cause_runs",
    )
    multi_cause_top_k = st.slider(
        "Multi-cause Top-K threshold",
        min_value=1,
        max_value=5,
        value=3,
        key="multi_cause_top_k",
    )
    multi_cause_window = st.slider(
        "Multi-cause baseline window",
        min_value=1,
        max_value=10,
        value=5,
        key="multi_cause_window",
    )
    selected_causes = st.multiselect(
        "Fault causes to evaluate",
        options=["temperature", "water_level", "plant_health", "food_supply"],
        default=["temperature", "water_level", "plant_health", "food_supply"],
        key="multi_cause_selected",
    )

if st.button("Run Multi-Cause Evaluation", type="primary", key="run_multi_cause"):
    if not selected_causes:
        st.warning("Select at least one fault cause.")
    else:
        try:
            with st.spinner("Running multi-cause experiments..."):
                summary_df, results_df = run_multi_cause_evaluation(
                    seeds=range(10, 10 + multi_cause_runs),
                    steps=int(steps),
                    fault_step=int(fault_step),
                    top_k=int(multi_cause_top_k),
                    window=int(multi_cause_window),
                    fault_causes=selected_causes,
                )
            st.session_state["multi_cause_summary"] = summary_df
            st.session_state["multi_cause_results"] = results_df
        except Exception as exc:
            st.error(f"Multi-cause evaluation failed: {exc}")
            st.exception(exc)

if "multi_cause_summary" in st.session_state:
    summary_df = st.session_state["multi_cause_summary"]
    results_df = st.session_state["multi_cause_results"]

    st.subheader("Performance by fault cause")
    st.dataframe(
        summary_df.round(3),
        use_container_width=True,
        hide_index=True,
    )

    if "fault_cause" in summary_df.columns:
        if {
            "echomind_top_k_accuracy",
            "baseline_top_k_accuracy",
        }.issubset(summary_df.columns):
            top_k_chart = summary_df.set_index("fault_cause")[
                ["echomind_top_k_accuracy", "baseline_top_k_accuracy"]
            ].rename(
                columns={
                    "echomind_top_k_accuracy": "EchoMind",
                    "baseline_top_k_accuracy": "Baseline",
                }
            )
            st.subheader("Top-K accuracy comparison")
            st.bar_chart(top_k_chart)

        if {"echomind_mrr", "baseline_mrr"}.issubset(summary_df.columns):
            mrr_chart = summary_df.set_index("fault_cause")[
                ["echomind_mrr", "baseline_mrr"]
            ].rename(
                columns={
                    "echomind_mrr": "EchoMind",
                    "baseline_mrr": "Baseline",
                }
            )
            st.subheader("Mean reciprocal rank comparison")
            st.bar_chart(mrr_chart)

    st.subheader("Individual experiment results")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download multi-cause results CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="echomind_multi_cause_results.csv",
        mime="text/csv",
        key="download_multi_cause_results",
    )
    st.download_button(
        "Download multi-cause summary CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="echomind_multi_cause_summary.csv",
        mime="text/csv",
        key="download_multi_cause_summary",
    )

st.divider()
st.caption(
    "EchoMind | Offline AI research project | Simulation-based causal hypothesis "
    "investigation. Evaluation scores describe synthetic experiments, not proof "
    "of real-world causation."
)


# ============================================================
# ECHOMIND — EVALUATION REPORT
# ============================================================

st.divider()
st.header("📊 Model Evaluation Report")

st.write(
    "Review EchoMind's performance against the baseline "
    "using your saved unseen-seed evaluation results."
)

holdout_path = "echomind_holdout_summary.csv"
holdout_details_path = "echomind_holdout_details.csv"

if st.button("Load Holdout Evaluation Report"):
    try:
        summary_df = pd.read_csv(holdout_path)

        details_df = None
        if __import__("os").path.exists(holdout_details_path):
            details_df = pd.read_csv(holdout_details_path)

        report = build_evaluation_report(summary_df, details_df)
        st.session_state["echomind_holdout_report"] = report

    except FileNotFoundError:
        st.error(
            "Holdout results were not found. "
            "Run the unseen-seed evaluation first."
        )
    except (ValueError, TypeError) as error:
        st.error(f"Could not build the report: {error}")

report = st.session_state.get("echomind_holdout_report")

if report is not None:
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Experiments",
        report["total_experiments"],
    )

    col2.metric(
        "EchoMind Top-1 Accuracy",
        f'{report["echomind_top_1_accuracy"]:.1%}',
    )

    col3.metric(
        "Baseline Top-1 Accuracy",
        f'{report["baseline_top_1_accuracy"]:.1%}',
    )

    st.subheader("Accuracy Difference")

    st.metric(
        "EchoMind minus Baseline",
        f'{report["top_1_difference_percentage_points"]:+.2f} percentage points',
    )

    st.subheader("Performance Metrics")

    metric_df = pd.DataFrame(
        {
            "Metric": [
                "Top-1 Accuracy",
                "Top-3 Accuracy",
                "Mean Reciprocal Rank",
            ],
            "EchoMind": [
                report["echomind_top_1_accuracy"],
                report["echomind_top_k_accuracy"],
                report["echomind_mrr"],
            ],
            "Baseline": [
                report["baseline_top_1_accuracy"],
                report["baseline_top_k_accuracy"],
                report["baseline_mrr"],
            ],
        }
    )

    st.dataframe(metric_df, use_container_width=True)

    st.subheader("Performance by Fault Cause")

    st.dataframe(
        report["per_cause"],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Evaluation Report CSV",
        data=report["per_cause"].to_csv(index=False),
        file_name="echomind_holdout_report.csv",
        mime="text/csv",
    )
else:
    st.info(
        "Click 'Load Holdout Evaluation Report' to display "
        "your saved evaluation metrics."
    )
