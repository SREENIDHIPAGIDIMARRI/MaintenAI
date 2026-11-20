"""
Multilabel Failure Type Prediction Module for MaintenAI.
Classifies specific component failure modes:
- TWF: Tool Wear Failure
- HDF: Heat Dissipation Failure
- PWF: Power Failure
- OSF: Overstrain Failure
- RNF: Random Failure
"""

import os
import sys
from typing import Dict, Any, Union
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import (
    add_engineered_features,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    FAILURE_TYPE_COLS,
)

FAILURE_TYPE_DESCRIPTIONS = {
    "TWF": {
        "name": "Tool Wear Failure",
        "description": "Gradual degradation and dulling of the cutting edge, leading to excessive friction and surface roughness.",
        "primary_sensor": "Tool wear [min] & Torque [Nm]",
    },
    "HDF": {
        "name": "Heat Dissipation Failure",
        "description": "Insufficient thermodynamic heat transfer between tool and ambient environment.",
        "primary_sensor": "Air & Process Temperature difference [K] & Rotational speed [rpm]",
    },
    "PWF": {
        "name": "Power Failure",
        "description": "Excessive or insufficient mechanical power delivered through the drive spindle.",
        "primary_sensor": "Torque [Nm] & Rotational speed [rpm]",
    },
    "OSF": {
        "name": "Overstrain Failure",
        "description": "Product of tool wear and applied cutting torque exceeds structural mechanical limits.",
        "primary_sensor": "Tool wear [min] x Torque [Nm]",
    },
    "RNF": {
        "name": "Random Failure",
        "description": "Stochastic process failure occurring independently of physical sensor parameter thresholds.",
        "primary_sensor": "Non-deterministic stochastic events",
    },
}


def load_multilabel_model(models_dir: str = "models"):
    """Loads the serialized multilabel classifier and preprocessor."""
    model_path = os.path.join(models_dir, "multilabel_classifier.pkl")
    pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
    if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
        raise FileNotFoundError("Multilabel model or preprocessing pipeline not found.")
    return joblib.load(model_path), joblib.load(pipeline_path)


def predict_failure_types(
    input_data: Union[Dict[str, Any], pd.DataFrame], models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Predicts presence and probabilities of all 5 component failure modes.
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = input_data.copy()

    df_engineered = add_engineered_features(df_input)
    model, pipeline = load_multilabel_model(models_dir=models_dir)

    feature_cols = [c for c in NUMERICAL_COLS + CATEGORICAL_COLS if c in df_engineered.columns]
    X_transformed = pipeline.transform(df_engineered[feature_cols])

    # Predictions
    preds = model.predict(X_transformed)
    if isinstance(preds, list):
        preds = np.column_stack(preds)

    # Probabilities per estimator
    probs = {}
    for i, col in enumerate(FAILURE_TYPE_COLS):
        estimator = model.estimators_[i]
        if hasattr(estimator, "predict_proba"):
            p = estimator.predict_proba(X_transformed)[0, 1]
        else:
            p = float(preds[0, i])
        probs[col] = float(p)

    results = {}
    detected_types = []
    for i, col in enumerate(FAILURE_TYPE_COLS):
        is_positive = bool(preds[0, i] == 1)
        p_val = probs[col]
        results[col] = {
            "code": col,
            "name": FAILURE_TYPE_DESCRIPTIONS[col]["name"],
            "description": FAILURE_TYPE_DESCRIPTIONS[col]["description"],
            "primary_sensor": FAILURE_TYPE_DESCRIPTIONS[col]["primary_sensor"],
            "predicted_positive": is_positive,
            "probability": round(p_val, 4),
            "probability_pct": round(p_val * 100, 2),
        }
        if is_positive:
            detected_types.append(FAILURE_TYPE_DESCRIPTIONS[col]["name"])

    return {
        "detected_failure_types": detected_types,
        "is_multilabel": len(detected_types) > 1,
        "per_type_details": results,
    }
