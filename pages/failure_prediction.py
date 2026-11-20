"""
Module 2: Machine Failure Prediction for MaintenAI.
Provides an interactive operational diagnostics console to input machine parameters,
run real-time inference through the trained ML preprocessing and classification pipelines,
and display failure probabilities, health status badges, and physics indicators.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import (
    PRESET_MACHINES,
    SENSOR_BOUNDS,
    apply_custom_industrial_css,
    render_header,
    render_kpi,
)
from utils.visualizations import create_risk_gauge_chart
from src.failure_prediction import predict_machine_status


def render_failure_prediction_page():
    apply_custom_industrial_css()

    render_header(
        title="MaintenAI – Machine Failure Inference Console",
        subtitle="Real-time multi-sensor inference engine connecting live telemetry to trained machine learning pipelines.",
        module_tag="Module 2: Failure Prediction",
    )

    # 1. Preset Selector & Quick Load
    st.markdown("#### ⚡ Quick-Load Operational Test Scenarios")
    preset_col, model_col = st.columns([1.5, 1])

    with preset_col:
        selected_preset = st.selectbox(
            "Select a pre-configured machine state for instant testing:",
            options=list(PRESET_MACHINES.keys()),
            index=0,
            help="Pre-configured sensor scenarios based on physical failure mechanisms in the AI4I dataset.",
        )
        preset_info = PRESET_MACHINES[selected_preset]
        st.caption(f"ℹ️ **Scenario Description**: {preset_info['description']}")

    with model_col:
        selected_model = st.selectbox(
            "Select Inference Model Architecture:",
            options=["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"],
            index=0,
            help="XGBoost is selected by default as the top-performing model based on F1-score & PR-AUC.",
        )
        st.caption(f"🧠 Selected classifier: **{selected_model}** (Loaded from `models/{selected_model.lower().replace(' ', '_')}.pkl`)")

    st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

    # 2. Interactive Sensor Input Form
    st.markdown("#### 🎛️ Operational Sensor Parameter Controls")

    with st.form("sensor_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🏷️ Machine Variant")
            type_val = st.selectbox(
                "Product Quality Variant (Type)",
                options=["L", "M", "H"],
                index=["L", "M", "H"].index(preset_info["Type"]),
                help="L = Low (60% volume), M = Medium (30% volume), H = High (10% volume)",
            )

            st.markdown("##### 🌡️ Thermal Sensors")
            air_temp_val = st.number_input(
                "Air Temperature [K]",
                min_value=290.0,
                max_value=310.0,
                value=float(preset_info["Air temperature"]),
                step=0.1,
                help=f"Nominal dataset range: {SENSOR_BOUNDS['Air temperature']['min']} - {SENSOR_BOUNDS['Air temperature']['max']} K",
            )

            proc_temp_val = st.number_input(
                "Process Temperature [K]",
                min_value=300.0,
                max_value=320.0,
                value=float(preset_info["Process temperature"]),
                step=0.1,
                help=f"Nominal dataset range: {SENSOR_BOUNDS['Process temperature']['min']} - {SENSOR_BOUNDS['Process temperature']['max']} K",
            )

        with col2:
            st.markdown("##### ⚙️ Kinetic & Dynamic Sensors")
            speed_val = st.number_input(
                "Rotational Speed [rpm]",
                min_value=1000,
                max_value=3000,
                value=int(preset_info["Rotational speed"]),
                step=10,
                help=f"Nominal dataset range: {SENSOR_BOUNDS['Rotational speed']['min']} - {SENSOR_BOUNDS['Rotational speed']['max']} rpm",
            )

            torque_val = st.number_input(
                "Spindle Torque [Nm]",
                min_value=1.0,
                max_value=90.0,
                value=float(preset_info["Torque"]),
                step=0.5,
                help=f"Nominal dataset range: {SENSOR_BOUNDS['Torque']['min']} - {SENSOR_BOUNDS['Torque']['max']} Nm",
            )

        with col3:
            st.markdown("##### ⏱️ Tool Wear & Lifespan")
            tool_wear_val = st.number_input(
                "Cumulative Tool Wear [min]",
                min_value=0,
                max_value=300,
                value=int(preset_info["Tool wear"]),
                step=1,
                help=f"Nominal dataset range: {SENSOR_BOUNDS['Tool wear']['min']} - {SENSOR_BOUNDS['Tool wear']['max']} min. Critical boundary ~ 200 min.",
            )

            # Live preview of physical indicators
            temp_delta = proc_temp_val - air_temp_val
            power_kw = (torque_val * speed_val * 2 * np.pi) / (60 * 1000)
            strain = torque_val * tool_wear_val

            st.markdown("##### 📐 Derived Physics Metrics")
            st.markdown(
                f"""
                <div style='background-color:#F8FAFC; padding:0.6rem 0.8rem; border-radius:6px; font-size:0.83rem; border:1px solid #E2E8F0;'>
                    • <b>Temp Delta:</b> {temp_delta:.1f} K {'⚠️ <8.6K' if temp_delta < 8.6 else '✅'}<br>
                    • <b>Power:</b> {power_kw:.2f} kW {'⚠️ Out of [3.5, 9.0]' if (power_kw < 3.5 or power_kw > 9.0) else '✅'}<br>
                    • <b>Tool Strain:</b> {strain:.0f} Nm*min {'⚠️ High Strain' if strain > 11000 else '✅'}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Input Validation Warning if out of statistical ranges
        validation_warnings = []
        if air_temp_val < 295.0 or air_temp_val > 305.0:
            validation_warnings.append(f"Air temperature {air_temp_val:.1f} K is outside the 1st-99th percentile band [295.0, 305.0 K].")
        if proc_temp_val < 305.0 or proc_temp_val > 315.0:
            validation_warnings.append(f"Process temperature {proc_temp_val:.1f} K is outside the 1st-99th percentile band [305.0, 315.0 K].")
        if speed_val < 1160 or speed_val > 2890:
            validation_warnings.append(f"Rotational speed {speed_val} rpm is outside nominal operating limits [1160, 2890 rpm].")
        if torque_val < 3.8 or torque_val > 77.0:
            validation_warnings.append(f"Torque {torque_val:.1f} Nm exceeds typical baseline distribution [3.8, 77.0 Nm].")
        if tool_wear_val > 255:
            validation_warnings.append(f"Tool wear {tool_wear_val} min exceeds historical maximum in dataset (253 min).")

        predict_btn = st.form_submit_button("🔍 Run Machine Failure Prediction", use_container_width=True)

    if validation_warnings:
        for w in validation_warnings:
            st.warning(f"⚠️ **Range Warning**: {w}")

    # 3. Prediction Execution & Diagnostic Display
    if predict_btn or "last_prediction" in st.session_state:
        # Prepare input dictionary
        input_dict = {
            "Type": type_val,
            "Air temperature": air_temp_val,
            "Process temperature": proc_temp_val,
            "Rotational speed": speed_val,
            "Torque": torque_val,
            "Tool wear": tool_wear_val,
        }

        # Run inference
        result = predict_machine_status(input_dict, model_name=selected_model)
        st.session_state["last_prediction"] = result

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Prediction & Diagnostic Outcome")

        # Result Banner and KPIs
        res_col1, res_col2 = st.columns([1.1, 0.9])

        with res_col1:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            status_is_failure = (result["prediction"] == 1)
            badge_class = "status-danger" if status_is_failure else "status-healthy"
            status_text = "POTENTIAL MACHINE FAILURE" if status_is_failure else "MACHINE NOMINAL / HEALTHY"

            st.markdown(
                f"""
                <div style='margin-bottom: 0.8rem;'>
                    <span style='font-size:0.85rem; color:#64748B; font-weight:600;'>PREDICTED STATUS:</span><br>
                    <span class='status-badge {badge_class}' style='font-size:1.1rem; padding:0.4rem 1rem; margin-top:0.3rem;'>
                        {status_text}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            render_kpi(
                m1,
                title="Failure Probability",
                value=f"{result['failure_probability_pct']:.1f}%",
                subtitle="Calibrated likelihood",
                badge=result["risk_category"],
                badge_type="danger" if result["risk_category"] == "High Risk" else ("warning" if result["risk_category"] == "Moderate Risk" else "healthy"),
            )
            render_kpi(
                m2,
                title="Risk Category",
                value=result["risk_category"],
                subtitle="Threshold tier",
                badge=selected_model,
                badge_type="neutral",
            )
            render_kpi(
                m3,
                title="Confidence",
                value=f"{result['confidence_pct']:.1f}%",
                subtitle="Decision certainty",
                badge="Ensemble",
                badge_type="neutral",
            )

            st.markdown(
                f"""
                <div style='margin-top: 1rem; font-size: 0.86rem; color: #475569; background: #F8FAFC; padding: 0.75rem 1rem; border-radius: 6px; border: 1px solid #E2E8F0;'>
                    <b>Active Classifier:</b> {result['model_name']}<br>
                    <b>Preprocessing Pipeline:</b> Standard Scaler (13 continuous features) + OneHotEncoder (Machine Type: L, M, H)<br>
                    <b>Decision Boundary:</b> Standard 0.5 classification threshold with tiered operational risk bands.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            fig_gauge = create_risk_gauge_chart(
                result["failure_probability_pct"],
                result["risk_category"],
                low_thresh_pct=25.0,
                high_thresh_pct=65.0,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # 4. Detailed Sensor Values & Engineered Physics Table
        st.markdown("#### 🔬 Telemetry Parameter Audit")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("##### Direct Sensor Inputs")
            raw_data = [
                {"Sensor": "Machine Quality Type", "Value": f"Type {type_val}", "Standard Baseline": "L (60%), M (30%), H (10%)"},
                {"Sensor": "Air Temperature", "Value": f"{air_temp_val:.1f} K", "Standard Baseline": "298.0 - 302.0 K (Mean: 300.0 K)"},
                {"Sensor": "Process Temperature", "Value": f"{proc_temp_val:.1f} K", "Standard Baseline": "308.0 - 312.0 K (Mean: 310.0 K)"},
                {"Sensor": "Rotational Speed", "Value": f"{speed_val} rpm", "Standard Baseline": "1300 - 1800 rpm (Mean: 1538 rpm)"},
                {"Sensor": "Spindle Torque", "Value": f"{torque_val:.1f} Nm", "Standard Baseline": "30.0 - 50.0 Nm (Mean: 40.0 Nm)"},
                {"Sensor": "Cumulative Tool Wear", "Value": f"{tool_wear_val} min", "Standard Baseline": "0 - 180 min (Critical: > 200 min)"},
            ]
            st.dataframe(pd.DataFrame(raw_data), use_container_width=True, hide_index=True)

        with col_t2:
            st.markdown("##### Engineered Physics Indicators")
            eng = result["engineered_features"]
            eng_data = [
                {"Engineered Feature": "Temperature Difference", "Value": f"{eng['Temp_Difference']:.2f} K", "Physical Rule": "HDF risk when < 8.6 K and speed < 1380"},
                {"Engineered Feature": "Mechanical Power", "Value": f"{eng['Power_kW']:.2f} kW", "Physical Rule": "PWF risk when < 3.5 kW or > 9.0 kW"},
                {"Engineered Feature": "Tool Wear Strain Index", "Value": f"{eng['Tool_Wear_Strain']:.1f} Nm*min", "Physical Rule": "OSF risk when > 11k (L), 12k (M), 13k (H)"},
                {"Engineered Feature": "Speed / Torque Ratio", "Value": f"{eng['Speed_Torque_Ratio']:.1f}", "Physical Rule": "Kinetic gear regime ratio"},
                {"Engineered Feature": "Heat Dissipation Flag", "Value": f"{'ACTIVE (1)' if eng['HDF_Risk_Flag'] else '0 (Normal)'}", "Physical Rule": "Thermodynamic heat accumulation flag"},
                {"Engineered Feature": "Overstrain Flag", "Value": f"{'ACTIVE (1)' if eng['OSF_Risk_Flag'] else '0 (Normal)'}", "Physical Rule": "Mechanical overload failure trigger"},
            ]
            st.dataframe(pd.DataFrame(eng_data), use_container_width=True, hide_index=True)


render_failure_prediction_page()
