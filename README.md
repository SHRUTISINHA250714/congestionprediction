<div align="center">

# 🌐 ML-Based Predictive Congestion Control in NS2

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![NS2 Version](https://img.shields.io/badge/NS--2-v2.35-orange.svg)](#)
[![Machine Learning](https://img.shields.io/badge/scikit--learn-v1.3-yellowgreen.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)

*A machine-learning integrated approach to network congestion prediction and adaptive intelligent routing in NS2.*
</div>

---

## 📖 Table of Contents
1. [Overview](#1-overview)
2. [Dataset Collection & Feature Engineering](#2-dataset-collection--feature-engineering)
3. [Models and Training](#3-models-and-training)
4. [ML Prediction and Inference](#4-ml-prediction-and-inference)
5. [NS2 Network Integration](#5-ns2-network-integration)
6. [Final Performance Impact vs Baseline](#6-final-performance-impact-vs-baseline)
7. [Installation & Setup](#7-installation--setup)
8. [How to Run](#8-how-to-run)
9. [Conclusion](#9-conclusion)

---

## 1. Overview 🚀

This project integrates Machine Learning (ML)-based predictive intelligence into an **NS2-simulated network** to proactively manage congestion. Rather than relying on traditional reactive congestion control mechanisms (e.g., TCP loss-based control such as Reno/NewReno), this pipeline **predicts congestion in advance** using windowed temporal network metrics and applies **dynamic routing** (by modifying link delays) to balance network load seamlessly.

The architecture bridges discrete-event network simulation (NS2) with supervised machine learning, demonstrating cross-domain integration between networking and predictive analytics.

---

## 2. Dataset Collection & Feature Engineering 📊

`(File: feature_engineering.py)`

The dataset is generated from multiple NS2 simulations under varying traffic intensities:
- **LOW**
- **MEDIUM**
- **HIGH**

For realistic data structuring, a custom `TraceParser` processes NS2 trace logs into **1-second time windows**, enabling real-time predictive modeling.

### 🚨 Congestion Definition

A time window is labeled as congested if any of the following thresholds are exceeded:
- **Link utilization** > `85.0%`
- **Packet loss** > `10.0%`
- **Average delay** > `0.08 ms`

*(Ensure unit consistency — 0.08 ms corresponds to 80 microseconds.)*

This creates a **binary classification target**:
- `1` → Congested
- `0` → Non-Congested

### ⚙️ Engineered ML Features

True predictive capability is achieved using temporal lag-based modeling, where:

$$ Features(t) \rightarrow Predict \rightarrow Congestion(t+1) $$

Engineered features include:
- `throughput_lag1` — Previous window throughput
- `delay_lag1` — Previous window delay
- `loss_lag1` — Previous window loss rate
- `utilization_lag1` — Previous window utilization
- `rolling_mean_util` — 3-window moving average of utilization
- `utilization_change` — First-order difference of utilization

These lag and rolling features allow the model to anticipate congestion **before it manifests**, rather than detecting it post-factum.

---

## 3. Models and Training 🤖

`(Files: train_models.py & evaluate_models.py)`

Multiple supervised classification models were evaluated to predict the binary congestion label. All input features were normalized using `StandardScaler` to ensure consistent feature scaling.

### 🧠 Model Configurations

- 🔹 **Logistic Regression** (`penalty='l2'`, `C=1.0`)
- 🌲 **Random Forest** (`n_estimators=100`, `max_depth=10`, `min_samples_split=5`)
- ⚡ **XGBoost** (`learning_rate=0.1`, `max_depth=6`, `n_estimators=100`, `eval_metric='logloss'`)
- 📉 **SVM (RBF Kernel)** *(evaluated in earlier versions, deprecated)* (`C=1.0`, `gamma='scale'`)

### 🧮 Evaluation Setup

- **80-20 Train-Test Split**
- **5-Fold Cross Validation** (`cv=5`) on training data
- Performance logged in `results/` directory

### 📈 Model Results (`detailed_metrics.csv`)

| Model               | Accuracy | Precision | Recall  | F1-Score | ROC-AUC   |
|---------------------|----------|-----------|---------|----------|-----------|
| Random Forest       | `0.822`  | `0.863`   | `0.791` | `0.826`  | **`0.894`**|
| XGBoost             | `0.844`  | `0.886`   | `0.812` | `0.847`  | `0.885`   |
| Logistic Regression | `0.855`  | `0.926`   | `0.791` | `0.854`  | `0.861`   |
| SVM                 | `0.866`  | `0.950`   | `0.791` | `0.863`  | `0.831`   |

> **Model Selection Rationale:**  
> **Random Forest** was selected because ROC-AUC evaluates ranking capability across all classification thresholds, making it more suitable for probabilistic congestion prediction than accuracy alone. It demonstrated superior discrimination power and stable generalization performance.

---

## 4. ML Prediction and Inference 📡

`(File: predict.py)`

The deployed predictive module operates on continuous streaming data.

### 🔄 Inference Workflow

1. Maintains and updates a **persistent runtime state** (`runtime_state.pkl`) to preserve temporal continuity.
2. Computes lag and rolling features consistently across time windows.
3. Applies the exact `StandardScaler.transform()` used during training.
4. Generates probabilistic output using `predict_proba()`.

The output represents:

$$ P(\text{congestion} \mid \text{current features}) $$

This probabilistic formulation enables flexible **threshold-based decision making.**

---

## 5. NS2 Network Integration 🔌

`(Files: apply_ml_routing.py & run_demo.py)`

The predicted congestion probability is integrated directly into **NS2 `.tcl` topology configurations.**

### 🛑 Action Threshold

If **$ P(\text{congestion}) > 50.0\% $**, dynamic routing adaptation is triggered.

#### 🟢 Baseline Network Configuration
- **Link delay:** `20 ms`
- **Bandwidth:** `2 Mb`
- Static shortest-path routing

#### 🔴 Congestion Mitigation Mechanism
When congestion probability exceeds the threshold:
1. Primary link delay is synthetically increased to **`50 ms`**
2. This inflates the routing cost
3. NS2’s shortest-path algorithm recomputes routes
4. Traffic is peacefully redistributed to alternate lower-cost paths

This simulates **cost-aware adaptive routing** without redesigning the routing protocol itself.

---

## 6. Final Performance Impact vs Baseline 🏆

`(Results located in: ns2/results/comparison_metrics.csv)`

Side-by-side trace analysis (`baseline.tr` vs `ml.tr`) natively inside the NS2 environment demonstrates massive efficiency gains.

| Performance Metric | Baseline Routing | **ML-Based Routing** | Improvement |
|--------------------|------------------|----------------------|-------------|
| ❌ Total Drops     | 3473 packets     | **151 packets**      | 🔥 **-95.6%** |
| 📉 Loss Rate (%)   | 21.52 %          | **0.62 %**           | 🔥 **-97.1%** |
| ⏱️ Avg Delay (ms)  | 1.18 ms          | **0.62 ms**          | 🚀 **-47.4%** |
| 🏁 Throughput (Mbps)| 6.76 Mbps       | **12.94 Mbps**       | 🚀 **+91.4%** |

---

## 7. Installation & Setup 🛠️

### Prerequisites

You need both NS2 installed on your shell environment, and Python 3.8+ to support Scikit-Learn.

1. **Install NS2 (Network Simulator 2) & NAM:**
   ```bash
   # Ubuntu / Debian
   sudo apt-get update
   sudo apt-get install ns2 nam
   ```

2. **Verify Python 3.8+ exists:**
   ```bash
   python3 --version
   ```

3. **Install Machine Learning Dependencies via PIP:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost
   ```

---

## 8. How to Run ▶️

The project executes through an easy multi-step pipeline from start to finish.

### Step 1: Train Models from Dataset
Extract the base dataset and train the Machine Learning Models:
```bash
python scripts/train_models.py
```
*(This produces evaluations inside `/results/` and outputs `.pkl` models to `/models/`)*

### Step 2: Apply the ML Router to NS2
Run the adaptive integration script. This analyzes the models, determines routing, and generates an optimized `.tcl` topology (`topology_ml_modified.tcl`):
```bash
python apply_ml_routing.py
```

### Step 3: Run the Congestion Demonstration
The demo handles compiling both the standard un-optimized simulation network vs. the newly scaled, ML-optimized simulation network. 

Run this to launch both simulations synchronously and pull up side-by-side NAM viewer visual simulations!
```bash
python run_demo.py
```

### Step 4: Compare Analytical Results
Extract pure packet insights comparing both trace files:
```bash
cd ns2/
python analyze_results.py
```
*(Check `ns2/results/` for the generated `comparison_metrics.csv` and graph `.png`)*

---

## 9. Conclusion 🎉

By integrating **Random Forest-based predictive modeling** with lag-driven temporal features, the system successfully anticipates congestion before packet drops natively occur. 

The resulting proactive routing adaptation **virtually eliminates packet drops (95%+ reduction)**, **halves delay time**, and **nearly doubles volumetric throughput**. This successfully proves that machine learning-driven predictive congestion control significantly outperforms static routing approaches within the NS2 simulation environment network layers.