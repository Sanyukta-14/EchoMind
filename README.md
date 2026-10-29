# EchoMind — An AI That Discovers Invisible Causes

**EchoMind** is an AI-powered causal investigation prototype that identifies possible root causes behind abnormal behaviour in a simulated ecosystem.

Rather than simply detecting that something has gone wrong, EchoMind analyses changes across connected variables, ranks possible fault causes, and evaluates its predictions using controlled experiments.

## 🚀 Live Demo

**Live Website:** https://echomind-causal-ai.streamlit.app/

> Note: The live deployment is being configured. The website may not be accessible until deployment issues are resolved.

## 📌 Project Overview

In complex systems, an abnormal outcome may be caused by an earlier failure elsewhere in the system. Identifying the root cause requires analysing relationships, changes over time, and the sequence in which anomalies appear.

EchoMind explores this problem through a simulated ecosystem involving environmental conditions, plant health, food availability, and animal population.

### 🎯 Objectives

- Detect abnormal behaviour in simulated system variables.
- Generate and rank possible fault causes.
- Analyse the timing and persistence of deviations.
- Compare the AI's predictions against a baseline method.
- Evaluate performance across multiple random seeds.
- Present investigation results through an interactive web interface.

## ✨ Key Features

- **Anomaly Detection:** Identifies deviations from expected system behaviour.
- **Hypothesis Generation:** Ranks candidate causes using multiple scoring factors.
- **Causal Investigation:** Uses known simulator relationships to help interpret downstream effects.
- **Intervention Simulation:** Explores how interventions affect the simulated system.
- **Baseline Comparison:** Compares EchoMind predictions with a baseline approach.
- **Multi-Seed Evaluation:** Tests performance across multiple experiment seeds.
- **Holdout Evaluation:** Supports evaluation on a separate set of experiment seeds.
- **Interactive Dashboard:** Displays investigation results, tables, and evaluation summaries.

## 🧠 Simulated Ecosystem

The simulator models these variables:

- Temperature
- Water level
- Plant health
- Food supply
- Animal population

The supported fault causes are temperature, water level, plant health, and food supply. Animal population is treated as a downstream outcome.

### Simplified relationship structure

```text
Temperature ──→ Water Level ──→ Plant Health
      │                             │
      └────────→ Plant Health       ↓
                                Food Supply
                                     │
                                     ↓
                             Animal Population
```

This graph represents the relationships implemented in the simulator. It is not a claim that EchoMind has discovered these relationships from real-world data.

## ⚙️ Technology Stack

- **Language:** Python
- **Web Framework:** Streamlit
- **Data Processing:** NumPy, Pandas
- **Machine Learning Utilities:** Scikit-learn
- **Visualisation:** Matplotlib, Seaborn, NetworkX
- **Testing:** Pytest
- **Version Control:** Git and GitHub

## 🏗️ Project Structure

```text
echomind/
├── app.py
├── requirements.txt
├── evaluate_all_causes.py
├── analyze_echomind_failures.py
├── src/
│   ├── simulator.py
│   ├── predictor.py
│   ├── anomaly_detector.py
│   ├── hypothesis_generator.py
│   ├── intervention_engine.py
│   ├── evaluation.py
│   ├── multi_seed_evaluation.py
│   ├── baseline.py
│   ├── comparison_evaluation.py
│   └── reporting.py
├── tests/
│   ├── test_ai_modules.py
│   ├── test_evaluation.py
│   ├── test_fault_scenarios.py
│   └── test_simulator.py
└── echomind_causal_dev_summary.csv
```

Additional evaluation scripts, tests, and CSV files may also be present in the repository.

## 💻 Installation and Setup

### Prerequisites

- Python installed on your computer
- Git (optional, if cloning the repository)
- Internet access to install the required packages

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd echomind
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the HTTPS URL of your actual GitHub repository.

### 2. Create a virtual environment

On Windows:

```bash
py -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
py -m pip install -r requirements.txt
```

### 4. Run the application

```bash
py -m streamlit run app.py
```

Streamlit will provide a local URL to open the dashboard in your browser.

## 🧪 Testing

Run the test suite from the project root:

```bash
py -m pytest -v
```

Alternatively, run tests from the tests directory:

```bash
py -m pytest tests -v
```

## 📊 Evaluation Results

EchoMind has been evaluated using controlled experiments in its synthetic simulator.

| Evaluation | Seeds | Experiments | EchoMind Top-1 Accuracy | Baseline Top-1 Accuracy |
|---|---|---:|---:|---:|
| Development | 10–29 | 80 | 100% | 100% |
| Independent validation | 200–299 | 400 | 100% | 100% |

In the reported experiments, both EchoMind and the baseline achieved a mean rank of 1.0, Top-k accuracy of 1.0, and mean reciprocal rank (MRR) of 1.0.

**Interpretation:** These results demonstrate performance on the current synthetic test scenarios. They do not establish superiority over the baseline or prove effectiveness on real-world systems.

## ⚠️ Limitations

- The evaluation environment is synthetic.
- The simulator contains predefined relationships between variables.
- Some hypothesis-ranking adjustments use simulator-informed heuristics.
- Perfect results on these controlled experiments do not guarantee performance on unseen real-world problems.
- Further evaluation on realistic datasets and less predictable causal relationships is required.

## 🔮 Future Enhancements

- Evaluate on real-world or publicly available datasets.
- Investigate causal discovery without relying on predefined simulator relationships.
- Test robustness under noisy, missing, and conflicting observations.
- Compare against additional causal inference and root-cause analysis methods.
- Improve explainability and uncertainty estimation.
- Expand intervention experiments and reproducible benchmarks.

## 🎓 Academic Context

EchoMind is an individual MCA project exploring AI-assisted root-cause investigation, anomaly detection, hypothesis ranking, and controlled evaluation.

## 📄 Disclaimer

EchoMind is a research-oriented prototype. Its current evaluation results apply to the implemented synthetic simulator and should not be interpreted as evidence of validated real-world causal discovery.
