"""
Model Training & Artifact Generation Pipeline for MaintenAI.
Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost for binary failure prediction,
benchmarks SMOTE vs Class Weighting, trains the Multilabel Classifier for failure modes,
and persists all trained artifacts to `models/`.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

from src.data_preprocessing import (
    load_dataset,
    prepare_features_and_targets,
    create_preprocessor,
    get_stratified_split,
    apply_smote_resampling,
)
from src.model_evaluation import evaluate_binary_model, evaluate_multilabel_model
from src.feature_engineering import FAILURE_TYPE_COLS


def train_all_models(data_path: str = "data/ai4i2020.csv", models_dir: str = "models"):
    """Executes the complete training, evaluation, and serialization workflow."""
    os.makedirs(models_dir, exist_ok=True)
    print("=" * 60)
    print("MaintenAI Industrial ML Training Pipeline Started")
    print("=" * 60)

    # 1. Load Data
    df = load_dataset(data_path)
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns.")

    # 2. Extract Features & Targets
    X, y_binary, y_multilabel = prepare_features_and_targets(df)
    print(f"Feature set extracted: {X.shape[1]} features.")
    print(f"Binary failure count: {y_binary.sum()} / {len(y_binary)} ({y_binary.mean() * 100:.2f}%)")

    # 3. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = get_stratified_split(X, y_binary, test_size=0.2, random_state=42)
    # Align multilabel test targets
    y_train_multi = y_multilabel.loc[X_train.index]
    y_test_multi = y_multilabel.loc[X_test.index]

    # 4. Preprocessing Pipeline
    preprocessor = create_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Extract transformed feature names
    feature_names = preprocessor.get_feature_names_out().tolist()
    print(f"Transformed features ({len(feature_names)}): {feature_names}")

    # 5. Define Model Architectures
    # Calculate positive weight ratio for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / max(pos_count, 1)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            scale_pos_weight=scale_pos,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    # 6. Train and Evaluate Binary Models
    metrics_report = {}
    saved_model_paths = {}

    print("\n--- Training Binary Machine Failure Classifiers ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_transformed, y_train)

        # Evaluation
        eval_res = evaluate_binary_model(model, X_test_transformed, y_test.values, model_name=name)
        metrics_report[name] = eval_res
        print(
            f"  -> {name} | Recall: {eval_res['recall']:.4f} | "
            f"Precision: {eval_res['precision']:.4f} | F1: {eval_res['f1_score']:.4f} | "
            f"ROC-AUC: {eval_res['roc_auc']:.4f} | PR-AUC: {eval_res['pr_auc']:.4f}"
        )

        # Save model
        filename = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join(models_dir, filename)
        joblib.dump(model, path)
        saved_model_paths[name] = path

    # 7. Compare SMOTE impact on training data
    print("\n--- Benchmarking SMOTE on Training Data ---")
    X_train_smote, y_train_smote = apply_smote_resampling(X_train_transformed, y_train, random_state=42)
    print(f"SMOTE Train Resampled: {X_train_smote.shape[0]} samples (Failures: {y_train_smote.sum()})")

    smote_metrics = {}
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if name == "Logistic Regression":
            m = LogisticRegression(max_iter=1500, random_state=42)
        elif name == "Random Forest":
            m = RandomForestClassifier(n_estimators=180, max_depth=12, random_state=42, n_jobs=-1)
        else:
            m = XGBClassifier(
                n_estimators=180, max_depth=5, learning_rate=0.06, random_state=42, eval_metric="logloss"
            )
        m.fit(X_train_smote, y_train_smote)
        eval_smote = evaluate_binary_model(m, X_test_transformed, y_test.values, model_name=f"{name} (SMOTE)")
        smote_metrics[f"{name} (SMOTE)"] = eval_smote
        print(
            f"  -> {name} (SMOTE) | Recall: {eval_smote['recall']:.4f} | "
            f"Precision: {eval_smote['precision']:.4f} | F1: {eval_smote['f1_score']:.4f} | "
            f"ROC-AUC: {eval_smote['roc_auc']:.4f}"
        )

    # 8. Train Multilabel Failure Mode Classifier
    print("\n--- Training Multilabel Failure Mode Classifier ---")
    # Using MultiOutput with Random Forest
    base_rf = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_split=4, class_weight="balanced", random_state=42, n_jobs=-1
    )
    multilabel_model = MultiOutputClassifier(base_rf, n_jobs=-1)
    multilabel_model.fit(X_train_transformed, y_train_multi.values)

    multi_eval = evaluate_multilabel_model(
        multilabel_model, X_test_transformed, y_test_multi, FAILURE_TYPE_COLS
    )
    print(f"Multilabel Classifier Hamming Loss: {multi_eval['hamming_loss']:.5f}")
    for lbl, m_dict in multi_eval["per_label"].items():
        print(
            f"  Label {lbl:4s} | Precision: {m_dict['precision']:.4f} | "
            f"Recall: {m_dict['recall']:.4f} | F1: {m_dict['f1_score']:.4f} | Support: {m_dict['support']}"
        )

    multilabel_path = os.path.join(models_dir, "multilabel_classifier.pkl")
    joblib.dump(multilabel_model, multilabel_path)

    # 9. Save Preprocessor & Feature Names
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessing_pipeline.pkl"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))

    # 10. Save Complete Comparison Evaluation Summary
    # Determine best model based on F1-score & PR-AUC
    best_model_name = max(
        metrics_report.keys(),
        key=lambda k: (metrics_report[k]["f1_score"] * 0.6 + metrics_report[k]["pr_auc"] * 0.4),
    )
    print(f"\nBest binary model selected: {best_model_name}")

    evaluation_summary = {
        "binary_models": metrics_report,
        "smote_models": smote_metrics,
        "multilabel": multi_eval,
        "best_model_name": best_model_name,
        "test_sample_size": len(y_test),
        "test_failure_count": int(y_test.sum()),
        "feature_names": feature_names,
    }
    joblib.dump(evaluation_summary, os.path.join(models_dir, "evaluation_metrics.pkl"))

    print("\nAll models and evaluation artifacts successfully saved to:", models_dir)
    print("=" * 60)
    return evaluation_summary


if __name__ == "__main__":
    train_all_models()
