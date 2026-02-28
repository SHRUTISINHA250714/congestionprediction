"""
============================================================================
Final Year Project: ML-Based Predictive Congestion Control Using NS2
File: evaluate_models.py
Description: Combined evaluation visualization for trained models
============================================================================
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)


class ModelEvaluator:
    def __init__(self, models_dir, results_dir, dataset_path):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.dataset_path = dataset_path

        os.makedirs(self.results_dir, exist_ok=True)

        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.X_test = None
        self.y_test = None

    # ==========================================================
    # LOAD MODELS + DATASET
    # ==========================================================
    def load_resources(self):

        print("[INFO] Loading dataset...")
        df = pd.read_csv(self.dataset_path)
        print(f"[SUCCESS] Loaded dataset: {df.shape}")

        with open(os.path.join(self.models_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

        with open(os.path.join(self.models_dir, 'feature_names.pkl'), 'rb') as f:
            self.feature_names = pickle.load(f)

        model_files = [
            f for f in os.listdir(self.models_dir)
            if f.endswith(".pkl") and f not in ["scaler.pkl", "feature_names.pkl"]
        ]

        for mf in model_files:
            name = mf.replace(".pkl", "").replace("_", " ").title()

            if name.lower() == "xgboost":
                name = "XGBoost"
            elif name.lower() == "random forest":
                name = "Random Forest"
            elif name.lower() == "logistic regression":
                name = "Logistic Regression"

            with open(os.path.join(self.models_dir, mf), "rb") as f:
                self.models[name] = pickle.load(f)

        X = df[self.feature_names]
        y = df["congestion"]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.X_test = self.scaler.transform(X_test)
        self.y_test = y_test

        print(f"[SUCCESS] Test set loaded with {len(self.y_test)} samples.\n")

    # ==========================================================
    # FULL EVALUATION + COMBINED VISUALIZATION
    # ==========================================================
    def evaluate_and_generate_combined_plots(self):

        print("============= MODEL PERFORMANCE =============\n")

        summary_rows = []

        # ===============================
        # 1️⃣ Metric Calculation
        # ===============================
        for name, model in self.models.items():

            y_pred = model.predict(self.X_test)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(self.X_test)[:, 1]
            else:
                y_prob = model.decision_function(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_prob)

            print(f"[SUCCESS] {name} Evaluated -> AUC: {roc_auc:.4f}, Acc: {accuracy:.4f}")
            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")
            print(f"F1-Score  : {f1:.4f}")
            print("-" * 50)

            summary_rows.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "ROC-AUC": roc_auc
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(
            os.path.join(self.results_dir, "detailed_metrics.csv"),
            index=False
        )

        print("[SUCCESS] Detailed metrics saved.\n")

        # ===============================
        # 2️⃣ Combined ROC Curve
        # ===============================
        plt.figure(figsize=(10, 8))

        for name, model in self.models.items():

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(self.X_test)[:, 1]
            else:
                y_prob = model.decision_function(self.X_test)

            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2,
                     label=f"{name} (AUC = {roc_auc:.4f})")

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate", fontweight="bold")
        plt.ylabel("True Positive Rate", fontweight="bold")
        plt.title("ROC Curve Comparison", fontweight="bold")
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "combined_roc_curve.png"), dpi=300)
        plt.close()

        print("[SUCCESS] Combined ROC curve saved.")

        # ===============================
        # 3️⃣ Combined Confusion Matrix
        # ===============================
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        for ax, (name, model) in zip(axes, self.models.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(name, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "combined_confusion_matrix.png"), dpi=300)
        plt.close()

        print("[SUCCESS] Combined confusion matrix saved.")

        # ===============================
        # 4️⃣ Combined Metrics Bar Chart
        # ===============================
        summary_df.set_index("Model").plot(kind="bar", figsize=(12, 6))

        plt.title("Model Performance Comparison", fontweight="bold")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "metrics_bar_chart.png"), dpi=300)
        plt.close()

        print("[SUCCESS] Metrics comparison bar chart saved.\n")


def main():

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    dataset_path = os.path.join(project_root, "dataset", "congestion_dataset.csv")

    if not os.path.exists(models_dir):
        print("[ERROR] Run train_models.py first.")
        sys.exit(1)

    evaluator = ModelEvaluator(models_dir, results_dir, dataset_path)
    evaluator.load_resources()
    evaluator.evaluate_and_generate_combined_plots()


if __name__ == "__main__":
    main()