"""
Module 8: Model Insights and Explainability for MaintenAI.
Provides global feature importance rankings (Random Forest & XGBoost),
and local instance-level SHAP (SHapley Additive exPlanations) for individual machine predictions.
Updates dynamically whenever telemetry parameters or operational presets are changed.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import (
    load_cached_dataset,
    apply_custom_industrial_css,
    render_header,
    render_kpi,
    PRESET_MACHINES,
)
from utils.visualizations import (
    create_feature_importance_chart,
    create_shap_waterfall_chart,
)
from src.explainability import get_global_feature_importance, explain_single_prediction
from src.failure_prediction import predict_machine_status


def render_model_explainability_page():
    apply_custom_industrial_css()

    render_header(
        title="MaintenAI – Model Insights & Interpretability (SHAP)",
        subtitle="Global feature importance analysis and local SHAP instance attribution to explain predictive decisions.",
        module_tag="Module 8: Model Explainability",
    )

    tab_local, tab_global = st.tabs([
        "🔬 Individual Machine Explanation (Local SHAP)",
        "🌐 Global Feature Importance (RF & XGBoost)",
    ])

    # -------------------------------------------------------------
    # TAB 1: LOCAL INSTANCE-LEVEL SHAP EXPLANATION
    # -------------------------------------------------------------
    with tab_local:
        st.markdown("#### 🎯 Real-Time SHAP Attribution Workbench")
        st.caption(
            "Uses SHAP (SHapley Additive exPlanations) grounded in cooperative game theory to allocate exact "
            "marginal credit and blame to each sensor reading for increasing or decreasing failure probability."
        )

        # Session state synchronization for presets and custom parameter edits
        default_shap_preset = "Power Failure (PWF) Scenario"
        if "shap_preset_name" not in st.session_state:
            st.session_state["shap_preset_name"] = default_shap_preset
            init_data = PRESET_MACHINES[default_shap_preset]
            st.session_state["shap_type"] = init_data["Type"]
            st.session_state["shap_air"] = float(init_data["Air temperature"])
            st.session_state["shap_proc"] = float(init_data["Process temperature"])
            st.session_state["shap_speed"] = int(init_data["Rotational speed"])
            st.session_state["shap_torque"] = float(init_data["Torque"])
            st.session_state["shap_tool"] = int(init_data["Tool wear"])

        def on_shap_preset_change():
            chosen = st.session_state["shap_preset_selector"]
            st.session_state["shap_preset_name"] = chosen
            p = PRESET_MACHINES[chosen]
            st.session_state["shap_type"] = p["Type"]
            st.session_state["shap_air"] = float(p["Air temperature"])
            st.session_state["shap_proc"] = float(p["Process temperature"])
            st.session_state["shap_speed"] = int(p["Rotational speed"])
            st.session_state["shap_torque"] = float(p["Torque"])
            st.session_state["shap_tool"] = int(p["Tool wear"])

        st.selectbox(
            "⚡ Quick-Load Operational Test Scenario (updates all telemetry below):",
            options=list(PRESET_MACHINES.keys()),
            key="shap_preset_selector",
            index=list(PRESET_MACHINES.keys()).index(st.session_state["shap_preset_name"]),
            on_change=on_shap_preset_change,
            help="Select a preset scenario, or manually tweak any sensor parameter below.",
        )
        st.caption(f"ℹ️ **Scenario Focus**: {PRESET_MACHINES[st.session_state['shap_preset_name']]['description']}")

        st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

        # Interactive 2-Column Layout: Left Controls, Right Live Diagnostics
        col_ctrl, col_diag = st.columns([1, 1.4])

        with col_ctrl:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("##### 🎛️ Operational Telemetry Controls")
            st.caption("Change any parameter to observe real-time SHAP attributions.")

            s_type = st.selectbox(
                "Machine Variant (Type)",
                ["L", "M", "H"],
                key="shap_type",
                help="Type L (Low), M (Medium), H (Heavy duty)",
            )
            s_air = st.number_input(
                "Air Temperature [K]",
                min_value=290.0,
                max_value=310.0,
                step=0.1,
                key="shap_air",
            )
            s_proc = st.number_input(
                "Process Temperature [K]",
                min_value=300.0,
                max_value=320.0,
                step=0.1,
                key="shap_proc",
            )
            s_speed = st.number_input(
                "Rotational Speed [rpm]",
                min_value=1000,
                max_value=3000,
                step=10,
                key="shap_speed",
            )
            s_torque = st.number_input(
                "Spindle Torque [Nm]",
                min_value=1.0,
                max_value=90.0,
                step=0.5,
                key="shap_torque",
            )
            s_tool = st.number_input(
                "Tool Wear [min]",
                min_value=0,
                max_value=300,
                step=1,
                key="shap_tool",
            )

            # Live calculated physics parameters preview
            dt_calc = s_proc - s_air
            p_calc = (s_torque * s_speed * 2 * np.pi) / (60 * 1000)
            strain_calc = s_torque * s_tool
            strain_limit = {"L": 11000, "M": 12000, "H": 13000}.get(s_type, 11000)

            st.markdown(
                f"""
                <div style='background:#F8FAFC; border:1px solid #E2E8F0; padding:0.6rem 0.8rem; border-radius:6px; font-size:0.8rem; margin-top:0.8rem;'>
                    <b>Active Physics State:</b><br>
                    • Temp Delta: <b>{dt_calc:.1f} K</b> {'(⚠️ HDF Risk)' if (dt_calc < 8.6 and s_speed < 1380) else '✅'}<br>
                    • Spindle Power: <b>{p_calc:.2f} kW</b> {'(⚠️ PWF Risk)' if (p_calc < 3.5 or p_calc > 9.0) else '✅'}<br>
                    • Cutting Strain: <b>{strain_calc:.0f} / {strain_limit}</b> {'(⚠️ OSF Risk)' if strain_calc > strain_limit else '✅'}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        input_dict = {
            "Type": s_type,
            "Air temperature": s_air,
            "Process temperature": s_proc,
            "Rotational speed": s_speed,
            "Torque": s_torque,
            "Tool wear": s_tool,
        }

        # Real-time inference & SHAP computation
        pred = predict_machine_status(input_dict, model_name="XGBoost")
        shap_output = explain_single_prediction(
            transformed_vector=pred["transformed_matrix"],
            raw_feature_values=pred["engineered_features"],
            model_name="XGBoost",
        )

        with col_diag:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3)
            render_kpi(
                k1,
                "Predicted Outcome",
                pred["status_label"],
                f"{pred['failure_probability_pct']:.1f}% failure probability",
                badge=pred["risk_category"],
                badge_type="danger" if pred["prediction"] == 1 else "healthy",
            )
            render_kpi(
                k2,
                "SHAP Base Expected Value",
                f"{shap_output['base_value']:.3f}",
                "Log-odds baseline",
                badge="Baseline",
                badge_type="neutral",
            )
            top_driver_name = shap_output["top_risk_drivers"][0]["feature"] if shap_output["top_risk_drivers"] else "None"
            render_kpi(
                k3,
                "Leading Risk Driver",
                top_driver_name,
                "Primary failure factor",
                badge="Dominant",
                badge_type="warning" if top_driver_name != "None" else "healthy",
            )

            # SHAP Waterfall Chart
            fig_shap = create_shap_waterfall_chart(shap_output["feature_contributions"], shap_output["base_value"])
            st.plotly_chart(fig_shap, use_container_width=True)
            st.markdown(
                """
                <small style='color:#64748B;'>
                    <b>How to read this chart:</b> Red bars indicate sensor readings pushing the model towards <b>FAILURE</b>. 
                    Green bars indicate sensors operating within safe bands that push the prediction towards <b>HEALTHY</b>.
                </small>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Dynamic Factor Cards Breakdown
        col_dr1, col_dr2 = st.columns(2)

        with col_dr1:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("##### 🚨 Top Factors Increasing Failure Risk (Red Drivers)")
            if shap_output["top_risk_drivers"]:
                for d in shap_output["top_risk_drivers"]:
                    st.markdown(
                        f"""
                        <div style='background:#FEF2F2; border-left:3px solid #DC2626; padding:0.4rem 0.8rem; margin-bottom:0.4rem; border-radius:4px;'>
                            <b>{d['feature']}</b>: Actual Value = <code>{d['actual_value']}</code><br>
                            <span style='font-size:0.8rem; color:#991B1B;'>SHAP impact: +{d['shap_value']:.4f} log-odds (Pushes probability toward failure)</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ No significant failure risk drivers detected. All parameters operate safely within nominal tolerances.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_dr2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("##### 🟢 Top Factors Supporting Healthy Operation (Green Factors)")
            if shap_output["top_healthy_factors"]:
                for h in shap_output["top_healthy_factors"]:
                    st.markdown(
                        f"""
                        <div style='background:#ECFDF5; border-left:3px solid #059669; padding:0.4rem 0.8rem; margin-bottom:0.4rem; border-radius:4px;'>
                            <b>{h['feature']}</b>: Actual Value = <code>{h['actual_value']}</code><br>
                            <span style='font-size:0.8rem; color:#065F46;'>SHAP impact: {h['shap_value']:.4f} log-odds (Protective factor preserving healthy state)</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Operating conditions are severe across multiple sensors.")
            st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 2: GLOBAL FEATURE IMPORTANCE
    # -------------------------------------------------------------
    with tab_global:
        st.markdown("#### 🌐 Global Feature Importance Rankings")
        st.caption(
            "Compares global feature importances derived across all 10,000 machines from Random Forest (MDI) "
            "and XGBoost (Feature Gain)."
        )

        global_imp = get_global_feature_importance()

        col_rf, col_xgb = st.columns(2)

        with col_rf:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            fig_rf = create_feature_importance_chart(global_imp["random_forest"], title="Random Forest (Mean Decrease Impurity)")
            st.plotly_chart(fig_rf, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_xgb:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            fig_xgb = create_feature_importance_chart(global_imp["xgboost"], title="XGBoost (Gain Importance)")
            st.plotly_chart(fig_xgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            **Domain Observations:**
            - **Physics-derived features** such as `Tool_Wear_Strain`, `Power_kW`, and `Temp_Difference` rank consistently at the top of feature importances alongside raw `Torque` and `Tool wear`.
            - This validates that injecting physical domain relationships allows tree-based models to capture non-linear thresholds (e.g. overstrain yield points and heat dissipation breakdown) much more cleanly than raw sensor measurements alone.
            """
        )


render_model_explainability_page()
