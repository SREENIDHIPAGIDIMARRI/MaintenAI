"""
Risk Assessment Module for MaintenAI.
Provides fleet-wide risk scoring, risk tier segmentation (Low, Moderate, High),
distribution statistics, and machine-level risk ranking.
"""

import os
import sys
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.failure_prediction import load_model_and_pipeline, categorize_risk
from src.feature_engineering import add_engineered_features, NUMERICAL_COLS, CATEGORICAL_COLS


def compute_fleet_risk(
    df: pd.DataFrame,
    model_name: str = "XGBoost",
    low_thresh: float = 0.25,
    high_thresh: float = 0.65,
    models_dir: str = "models",
) -> pd.DataFrame:
    """
    Evaluates failure probabilities and risk tiers across an entire machine fleet.
    Adds 'Failure_Probability', 'Failure_Probability_Pct', 'Risk_Category', and 'Predicted_Status'.
    """
    df_fleet = df.copy()
    df_engineered = add_engineered_features(df_fleet)

    model, pipeline = load_model_and_pipeline(model_name, models_dir=models_dir)

    feature_cols = [c for c in NUMERICAL_COLS + CATEGORICAL_COLS if c in df_engineered.columns]
    X_transformed = pipeline.transform(df_engineered[feature_cols])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_transformed)[:, 1]
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X_transformed)
        probs = 1 / (1 + np.exp(-dec))
    else:
        probs = model.predict(X_transformed).astype(float)

    preds = (probs >= 0.5).astype(int)

    df_fleet["Failure_Probability"] = probs
    df_fleet["Failure_Probability_Pct"] = np.round(probs * 100, 2)
    df_fleet["Predicted_Status"] = np.where(preds == 1, "Potential Failure", "Healthy")
    df_fleet["Risk_Category"] = [categorize_risk(p, low_thresh, high_thresh) for p in probs]

    return df_fleet


def get_risk_summary(df_fleet_scored: pd.DataFrame) -> Dict[str, Any]:
    """Generates distribution KPI metrics from scored fleet DataFrame."""
    total = len(df_fleet_scored)
    low_count = int((df_fleet_scored["Risk_Category"] == "Low Risk").sum())
    mod_count = int((df_fleet_scored["Risk_Category"] == "Moderate Risk").sum())
    high_count = int((df_fleet_scored["Risk_Category"] == "High Risk").sum())

    return {
        "total_machines": total,
        "low_risk_count": low_count,
        "low_risk_pct": round((low_count / max(total, 1)) * 100, 2),
        "mod_risk_count": mod_count,
        "mod_risk_pct": round((mod_count / max(total, 1)) * 100, 2),
        "high_risk_count": high_count,
        "high_risk_pct": round((high_count / max(total, 1)) * 100, 2),
        "predicted_failures": int((df_fleet_scored["Predicted_Status"] == "Potential Failure").sum()),
        "average_failure_prob_pct": round(float(df_fleet_scored["Failure_Probability"].mean() * 100), 2),
    }
