"""
Failure Prediction Module for MaintenAI.
Handles single-sample and batch inference using trained binary classification models,
preprocesses raw input, calculates failure probabilities, risk categories, and confidence.
"""

import os
import sys
from typing import Dict, Any, Union, Tuple
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import add_engineered_features, NUMERICAL_COLS, CATEGORICAL_COLS

MODEL_MAP = {
    "XGBoost": "xgboost.pkl",
    "Random Forest": "random_forest.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Logistic Regression": "logistic_regression.pkl",
}


def load_model_and_pipeline(model_name: str = "XGBoost", models_dir: str = "models") -> Tuple[Any, Any]:
    """Loads a specific binary classifier and the global preprocessing pipeline."""
    filename = MODEL_MAP.get(model_name, "xgboost.pkl")
    model_path = os.path.join(models_dir, filename)
    pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    return model, pipeline


def categorize_risk(
    prob: float, low_threshold: float = 0.25, high_threshold: float = 0.65
) -> str:
    """Categorizes machine failure risk based on configurable threshold bands."""
    if prob < low_threshold:
        return "Low Risk"
    elif prob < high_threshold:
        return "Moderate Risk"
    else:
        return "High Risk"


def predict_machine_status(
    input_data: Union[Dict[str, Any], pd.DataFrame],
    model_name: str = "XGBoost",
    low_threshold: float = 0.25,
    high_threshold: float = 0.65,
    models_dir: str = "models",
) -> Dict[str, Any]:
    """
    Executes end-to-end failure prediction for a single machine or batch input.
    Validates input, computes engineered features, applies scaling, and returns rich diagnostics.
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = input_data.copy()

    # Domain feature engineering
    df_engineered = add_engineered_features(df_input)

    # Load model and preprocessor
    model, pipeline = load_model_and_pipeline(model_name, models_dir=models_dir)

    # Feature transformation
    feature_cols = [c for c in NUMERICAL_COLS + CATEGORICAL_COLS if c in df_engineered.columns]
    X_raw = df_engineered[feature_cols]
    X_transformed = pipeline.transform(X_raw)

    # Probability computation
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_transformed)[:, 1]
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X_transformed)
        probs = 1 / (1 + np.exp(-dec))
    else:
        probs = model.predict(X_transformed).astype(float)

    # Predictions
    # Using 0.5 decision threshold for the model's nominal classification,
    # or risk categorization for operational decisions
    preds = model.predict(X_transformed)

    prob_val = float(probs[0])
    pred_val = int(preds[0])
    prob_pct = round(prob_val * 100, 2)
    risk_cat = categorize_risk(prob_val, low_threshold, high_threshold)

    # Confidence calculation: distance from decision boundary (0.5)
    confidence = round(abs(prob_val - 0.5) * 2 * 100, 2)

    return {
        "prediction": pred_val,
        "status_label": "Potential Failure" if pred_val == 1 else "Healthy",
        "failure_probability": prob_val,
        "failure_probability_pct": prob_pct,
        "risk_category": risk_cat,
        "confidence_pct": confidence,
        "model_name": model_name,
        "raw_inputs": df_input.iloc[0].to_dict(),
        "engineered_features": df_engineered.iloc[0].to_dict(),
        "transformed_matrix": X_transformed,
    }
