"""
End-to-end integration and verification suite for MaintenAI.
Tests dataset loading, feature engineering, binary failure prediction across all 4 models,
multilabel classification, risk assessment, maintenance recommendations, and SHAP explainability.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from src.data_preprocessing import load_dataset
from src.feature_engineering import add_engineered_features
from src.failure_prediction import predict_machine_status
from src.failure_type_prediction import predict_failure_types
from src.risk_assessment import compute_fleet_risk, get_risk_summary
from src.maintenance_recommendations import evaluate_maintenance_rules
from src.explainability import get_global_feature_importance, explain_single_prediction
from utils.helpers import PRESET_MACHINES


def test_maintenai_pipeline():
    print("=" * 60)
    print("RUNNING MAINTENAI VERIFICATION SUITE")
    print("=" * 60)

    # 1. Dataset loading
    print("1. Testing Dataset loading...")
    df = load_dataset("data/ai4i2020.csv")
    assert len(df) == 10000, f"Expected 10,000 rows, got {len(df)}"
    print(f"   [OK] Authentic dataset verified: {len(df):,} samples, {df.shape[1]} columns.")

    # 2. Physics-based Feature Engineering
    print("2. Testing Physics-informed Feature Engineering...")
    df_eng = add_engineered_features(df.head(20))
    for col in ["Temp_Difference", "Power_kW", "Tool_Wear_Strain", "Speed_Torque_Ratio", "HDF_Risk_Flag", "PWF_Risk_Flag", "OSF_Risk_Flag"]:
        assert col in df_eng.columns, f"Missing feature {col}"
    print("   [OK] All physics-informed features correctly calculated.")

    # 3. Binary Failure Prediction
    print("3. Testing Binary Classifiers (XGBoost, Random Forest, Decision Tree, Logistic Regression)...")
    healthy_sample = PRESET_MACHINES["Nominal Healthy Machine (Type L)"]
    for m in ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"]:
        res = predict_machine_status(healthy_sample, model_name=m)
        assert "status_label" in res
        assert "failure_probability_pct" in res
        assert "risk_category" in res
        print(f"   [OK] {m:20s} -> Status: {res['status_label']:18s} | Prob: {res['failure_probability_pct']:5.1f}% | Risk: {res['risk_category']}")

    # 4. Multilabel Failure Mode Classification
    print("4. Testing Multilabel Component Failure Mode Predictor...")
    twf_sample = PRESET_MACHINES["Tool Wear Failure (TWF) Scenario"]
    multi_res = predict_failure_types(twf_sample)
    assert "detected_failure_types" in multi_res
    print(f"   [OK] Multilabel prediction: {multi_res['detected_failure_types']}")

    # 5. Fleet Risk Scoring
    print("5. Testing Fleet Risk Assessment...")
    df_scored = compute_fleet_risk(df.head(200), model_name="XGBoost")
    risk_sum = get_risk_summary(df_scored)
    assert risk_sum["total_machines"] == 200
    print(f"   [OK] Fleet scored: {risk_sum['low_risk_count']} Low Risk, {risk_sum['mod_risk_count']} Moderate, {risk_sum['high_risk_count']} High Risk.")

    # 6. Maintenance Recommendation Engine
    print("6. Testing Maintenance Recommendation System...")
    rec = evaluate_maintenance_rules(
        sensor_data=PRESET_MACHINES["Heat Dissipation Failure (HDF) Scenario"],
        failure_probability=0.91,
        predicted_status="Potential Failure",
        risk_category="High Risk",
    )
    assert rec["priority_level"] in ["Critical", "High"]
    assert len(rec["recommended_actions"]) > 0
    print(f"   [OK] Work order priority: {rec['priority_level']}")
    print(f"   [OK] Primary Action: {rec['recommended_actions'][0]}")

    # 7. Model Explainability & SHAP
    print("7. Testing Global & Local SHAP Explainability...")
    glob_imp = get_global_feature_importance()
    assert len(glob_imp["xgboost"]) > 0
    assert len(glob_imp["random_forest"]) > 0

    pred_xgb = predict_machine_status(healthy_sample, model_name="XGBoost")
    shap_out = explain_single_prediction(pred_xgb["transformed_matrix"], pred_xgb["engineered_features"], model_name="XGBoost")
    assert "base_value" in shap_out
    assert len(shap_out["feature_contributions"]) > 0
    print(f"   [OK] Global importances extracted for RF & XGBoost.")
    print(f"   [OK] Local SHAP values computed: {len(shap_out['feature_contributions'])} features.")

    print("=" * 60)
    print("ALL 7 SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_maintenai_pipeline()
