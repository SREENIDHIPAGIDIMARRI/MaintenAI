"""
Model Explainability & Interpretability Module for MaintenAI.
Provides global feature importance comparisons (Random Forest MDI, XGBoost gain/weight),
and local instance-level SHAP explanations for individual machine predictions.
"""

import os
import sys
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def get_global_feature_importance(models_dir: str = "models") -> Dict[str, Any]:
    """
    Extracts normalized global feature importances from Random Forest and XGBoost.
    """
    feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
    rf_model = joblib.load(os.path.join(models_dir, "random_forest.pkl"))
    xgb_model = joblib.load(os.path.join(models_dir, "xgboost.pkl"))

    # Random forest MDI (Mean Decrease in Impurity)
    rf_importances = rf_model.feature_importances_
    rf_df = pd.DataFrame({"feature": feature_names, "importance": rf_importances}).sort_values(
        by="importance", ascending=False
    )

    # XGBoost feature importances
    xgb_importances = xgb_model.feature_importances_
    xgb_df = pd.DataFrame({"feature": feature_names, "importance": xgb_importances}).sort_values(
        by="importance", ascending=False
    )

    return {
        "features": feature_names,
        "random_forest": rf_df.to_dict(orient="records"),
        "xgboost": xgb_df.to_dict(orient="records"),
    }


def explain_single_prediction(
    transformed_vector: np.ndarray,
    raw_feature_values: Dict[str, Any] = None,
    model_name: str = "XGBoost",
    models_dir: str = "models",
) -> Dict[str, Any]:
    """
    Computes local SHAP values for a single machine instance.
    Identifies the top risk-increasing and risk-decreasing parameters.
    """
    feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))

    filename = "xgboost.pkl" if model_name == "XGBoost" else "random_forest.pkl"
    model = joblib.load(os.path.join(models_dir, filename))

    explainer = shap.TreeExplainer(model)
    shap_res = explainer(transformed_vector)

    # In shap >= 0.40, TreeExplainer for binary classification returns (1, n_features) or (1, n_features, 2)
    values = shap_res.values
    if len(values.shape) == 3:
        # Take class 1 (Failure)
        local_shap = values[0, :, 1]
        base_val = float(shap_res.base_values[0, 1])
    elif len(values.shape) == 2:
        local_shap = values[0]
        base_val = float(shap_res.base_values[0]) if hasattr(shap_res.base_values, "__getitem__") else float(shap_res.base_values)
    else:
        local_shap = values
        base_val = float(shap_res.base_values)

    records = []
    for i, name in enumerate(feature_names):
        val = float(local_shap[i])
        actual_val = raw_feature_values.get(name, transformed_vector[0, i]) if raw_feature_values else transformed_vector[0, i]
        records.append({
            "feature": name,
            "shap_value": val,
            "abs_shap": abs(val),
            "actual_value": actual_val,
            "direction": "Increases Failure Risk" if val > 0 else "Supports Healthy State",
        })

    df_shap = pd.DataFrame(records).sort_values(by="abs_shap", ascending=False)

    top_risk_drivers = df_shap[df_shap["shap_value"] > 0].head(5).to_dict(orient="records")
    top_healthy_factors = df_shap[df_shap["shap_value"] <= 0].head(5).to_dict(orient="records")

    return {
        "base_value": base_val,
        "feature_contributions": df_shap.to_dict(orient="records"),
        "top_risk_drivers": top_risk_drivers,
        "top_healthy_factors": top_healthy_factors,
    }
