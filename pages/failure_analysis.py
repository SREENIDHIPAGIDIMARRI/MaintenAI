"""
Module 3: Failure Type Analysis for MaintenAI.
Comprehensive analysis of the five AI4I component failure modes (TWF, HDF, PWF, OSF, RNF).
Includes historical distribution, multilabel co-occurrence matrix, sensor patterns,
and live multilabel failure-type prediction using the trained MultiOutputClassifier.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import (
    load_cached_dataset,
    apply_custom_industrial_css,
    render_header,
    render_kpi,
    PRESET_MACHINES,
)
from src.failure_type_prediction import predict_failure_types, FAILURE_TYPE_DESCRIPTIONS
from src.feature_engineering import add_engineered_features, FAILURE_TYPE_COLS


def render_failure_analysis_page():
    apply_custom_industrial_css()

    render_header(
        title="MaintenAI – Component Failure Mode & Multilabel Analysis",
        subtitle="In-depth diagnostic analysis of the 5 physical failure modes and live multilabel inference.",
        module_tag="Module 3: Failure Analysis",
    )

    df = load_cached_dataset()
    df_eng = add_engineered_features(df)

    tab_historical, tab_live_prediction = st.tabs([
        "📊 Historical Failure Mode Analytics",
        "⚡ Live Multilabel Failure Mode Prediction",
    ])

    # -------------------------------------------------------------
    # TAB 1: HISTORICAL FAILURE MODE ANALYTICS
    # -------------------------------------------------------------
    with tab_historical:
        st.markdown("#### 🔍 Historical Fleet Failure Statistics")
        st.info(
            "**Multilabel Architecture Insight:** The five failure types are **non-mutually exclusive**. "
            "In physical industrial machinery, excessive tool wear can trigger overheating, causing simultaneous "
            "thermodynamic and mechanical strain events. Treating these as independent multilabel indicators "
            "prevents critical safety blindspots."
        )

        # Overview metric cards
        twf_cnt = int(df["TWF"].sum())
        hdf_cnt = int(df["HDF"].sum())
        pwf_cnt = int(df["PWF"].sum())
        osf_cnt = int(df["OSF"].sum())
        rnf_cnt = int(df["RNF"].sum())
        multi_cnt = int((df[FAILURE_TYPE_COLS].sum(axis=1) > 1).sum())

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        render_kpi(c1, "Tool Wear (TWF)", f"{twf_cnt}", "Dulling & friction", badge="Wear", badge_type="warning")
        render_kpi(c2, "Heat Dissip. (HDF)", f"{hdf_cnt}", "Thermal accumulation", badge="Thermodynamic", badge_type="danger")
        render_kpi(c3, "Power Fail (PWF)", f"{pwf_cnt}", "Drive out-of-spec", badge="Electrical", badge_type="danger")
        render_kpi(c4, "Overstrain (OSF)", f"{osf_cnt}", "Torque x wear limit", badge="Mechanical", badge_type="danger")
        render_kpi(c5, "Random (RNF)", f"{rnf_cnt}", "Stochastic events", badge="Random", badge_type="neutral")
        render_kpi(c6, "Compound Modes", f"{multi_cnt}", "> 1 failure mode", badge="Multilabel", badge_type="warning")

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1.1, 0.9])

        with col_left:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            # Failure mode distribution
            f_df = pd.DataFrame({
                "Failure Mode": ["Tool Wear (TWF)", "Heat Dissipation (HDF)", "Power Failure (PWF)", "Overstrain (OSF)", "Random (RNF)"],
                "Count": [twf_cnt, hdf_cnt, pwf_cnt, osf_cnt, rnf_cnt],
                "Share of All Failures": [
                    f"{(twf_cnt / 339) * 100:.1f}%",
                    f"{(hdf_cnt / 339) * 100:.1f}%",
                    f"{(pwf_cnt / 339) * 100:.1f}%",
                    f"{(osf_cnt / 339) * 100:.1f}%",
                    f"{(rnf_cnt / 339) * 100:.1f}%",
                ],
            })
            fig_bar = px.bar(
                f_df,
                x="Failure Mode",
                y="Count",
                text="Count",
                color="Failure Mode",
                color_discrete_sequence=["#2563EB", "#D97706", "#DC2626", "#0D9488", "#64748B"],
                title="Historical Incident Frequency by Failure Mode",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=350,
                font=dict(family="Inter", size=12, color="#475569"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            # Multilabel co-occurrence matrix
            co_matrix = np.zeros((5, 5), dtype=int)
            for i, col1 in enumerate(FAILURE_TYPE_COLS):
                for j, col2 in enumerate(FAILURE_TYPE_COLS):
                    co_matrix[i, j] = int(((df[col1] == 1) & (df[col2] == 1)).sum())

            fig_co = go.Figure(
                data=go.Heatmap(
                    z=co_matrix,
                    x=FAILURE_TYPE_COLS,
                    y=FAILURE_TYPE_COLS,
                    colorscale="Blues",
                    text=co_matrix,
                    texttemplate="%{text}",
                    showscale=False,
                )
            )
            fig_co.update_layout(
                title="Failure Mode Co-Occurrence Matrix (Overlap)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=40, r=20, t=50, b=40),
                font=dict(family="Inter", size=12, color="#475569"),
            )
            st.plotly_chart(fig_co, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Sensor patterns associated with each failure type
        st.markdown("#### 🔬 Sensor Signatures Across Failure Modes")
        sensor_choice = st.selectbox(
            "Select Sensor / Derived Parameter to Analyze Across Failure Modes:",
            options=["Temp_Difference", "Power_kW", "Tool_Wear_Strain", "Torque", "Rotational speed", "Tool wear"],
            index=0,
            help="Compare the distribution of this sensor for machines experiencing each failure mode vs healthy machines.",
        )

        # Build boxplot data
        records = []
        # Healthy baseline
        for val in df_eng[df_eng["Machine failure"] == 0][sensor_choice].dropna().sample(min(1500, len(df_eng))):
            records.append({"Group": "Healthy Fleet", sensor_choice: val})

        for f_col in FAILURE_TYPE_COLS:
            sub = df_eng[df_eng[f_col] == 1][sensor_choice].dropna()
            for val in sub:
                records.append({"Group": f_col, sensor_choice: val})

        df_box = pd.DataFrame(records)
        fig_box = px.box(
            df_box,
            x="Group",
            y=sensor_choice,
            color="Group",
            title=f"Comparative Sensor Behavior: {sensor_choice}",
            color_discrete_sequence=["#10B981", "#2563EB", "#D97706", "#DC2626", "#8B5CF6", "#64748B"],
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            showlegend=False,
            font=dict(family="Inter", size=12, color="#475569"),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: LIVE MULTILABEL FAILURE PREDICTION
    # -------------------------------------------------------------
    with tab_live_prediction:
        st.markdown("#### ⚡ Live Multilabel Failure Mode Prediction for New Machine Inputs")
        st.caption(
            "Predicts the presence and estimated probability for each of the 5 failure modes using the trained "
            "MultiOutputClassifier. Evaluates all modes simultaneously."
        )

        default_multi_preset = "Tool Wear Failure (TWF) Scenario"
        if "multi_preset_name" not in st.session_state:
            st.session_state["multi_preset_name"] = default_multi_preset
            p_init = PRESET_MACHINES[default_multi_preset]
            st.session_state["m_type"] = p_init["Type"]
            st.session_state["m_air"] = float(p_init["Air temperature"])
            st.session_state["m_proc"] = float(p_init["Process temperature"])
            st.session_state["m_speed"] = int(p_init["Rotational speed"])
            st.session_state["m_torque"] = float(p_init["Torque"])
            st.session_state["m_tool"] = int(p_init["Tool wear"])

        def on_multi_preset_change():
            chosen = st.session_state["multi_preset_selector"]
            st.session_state["multi_preset_name"] = chosen
            p = PRESET_MACHINES[chosen]
            st.session_state["m_type"] = p["Type"]
            st.session_state["m_air"] = float(p["Air temperature"])
            st.session_state["m_proc"] = float(p["Process temperature"])
            st.session_state["m_speed"] = int(p["Rotational speed"])
            st.session_state["m_torque"] = float(p["Torque"])
            st.session_state["m_tool"] = int(p["Tool wear"])

        st.selectbox(
            "Load operational scenario for failure type testing:",
            options=list(PRESET_MACHINES.keys()),
            key="multi_preset_selector",
            index=list(PRESET_MACHINES.keys()).index(st.session_state["multi_preset_name"]),
            on_change=on_multi_preset_change,
        )
        st.caption(f"ℹ️ **Scenario Focus**: {PRESET_MACHINES[st.session_state['multi_preset_name']]['description']}")

        c_in1, c_in2, c_in3 = st.columns(3)
        with c_in1:
            t_type = st.selectbox("Machine Type", ["L", "M", "H"], key="m_type")
            t_air = st.number_input("Air Temperature [K]", 290.0, 310.0, step=0.1, key="m_air")
        with c_in2:
            t_proc = st.number_input("Process Temperature [K]", 300.0, 320.0, step=0.1, key="m_proc")
            t_speed = st.number_input("Rotational Speed [rpm]", 1000, 3000, step=10, key="m_speed")
        with c_in3:
            t_torque = st.number_input("Torque [Nm]", 1.0, 90.0, step=0.5, key="m_torque")
            t_tool = st.number_input("Tool Wear [min]", 0, 300, step=1, key="m_tool")

        if st.button("🔮 Diagnose Component Failure Modes", use_container_width=True):
            test_dict = {
                "Type": t_type,
                "Air temperature": t_air,
                "Process temperature": t_proc,
                "Rotational speed": t_speed,
                "Torque": t_torque,
                "Tool wear": t_tool,
            }
            multi_res = predict_failure_types(test_dict)

            st.markdown("<br>", unsafe_allow_html=True)
            detected = multi_res["detected_failure_types"]

            if detected:
                st.error(
                    f"⚠️ **FAILURE RISK DETECTED**: Predicted Component Failure Modes: **{', '.join(detected)}** "
                    f"({'Compound Multilabel Event' if multi_res['is_multilabel'] else 'Single Failure Mode'})"
                )
            else:
                st.success("✅ **NOMINAL STATE**: No component failure modes triggered at current operational parameters.")

            # Per-label detailed cards
            st.markdown("##### 🔬 Individual Failure Mode Assessment")
            cols = st.columns(5)
            for i, f_col in enumerate(FAILURE_TYPE_COLS):
                details = multi_res["per_type_details"][f_col]
                is_pos = details["predicted_positive"]
                prob = details["probability_pct"]

                badge = "TRIGGERED" if is_pos else "NORMAL"
                badge_type = "danger" if is_pos else "healthy"

                with cols[i]:
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="border-top: 4px solid {'#DC2626' if is_pos else '#10B981'};">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <b style="font-size:0.9rem;">{f_col}</b>
                                <span class="status-badge status-{badge_type}">{badge}</span>
                            </div>
                            <div style="font-size:1.4rem; font-weight:700; margin:0.3rem 0; color:#0F172A;">{prob:.1f}%</div>
                            <div style="font-size:0.75rem; color:#64748B; font-weight:600;">{details['name']}</div>
                            <div style="font-size:0.72rem; color:#475569; margin-top:0.4rem; line-height:1.2;">
                                {details['description']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


render_failure_analysis_page()
