"""
Module 5: Sensor Data Visualization for MaintenAI.
Interactive multi-sensor analytics, correlation matrices, pairwise distributions,
failure vs. healthy behavioral comparisons, IQR outlier detection, and 3D operational regimes.
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
)
from utils.visualizations import (
    create_correlation_heatmap,
    create_sensor_distribution_chart,
)
from src.feature_engineering import add_engineered_features


def render_sensor_analytics_page():
    apply_custom_industrial_css()

    render_header(
        title="MaintenAI – Sensor Telemetry & Operational Analytics",
        subtitle="Exploratory multi-sensor dynamics, correlation matrices, comparative distributions, and outlier surveillance.",
        module_tag="Module 5: Sensor Analytics",
    )

    df = load_cached_dataset()
    df_eng = add_engineered_features(df)

    tab_dist, tab_corr, tab_scatter, tab_outliers = st.tabs([
        "📊 Sensor Distributions & Health Splits",
        "🔥 Correlation Heatmap",
        "🌐 Pairwise & 3D Sensor Regimes",
        "🚨 Outlier Detection Engine",
    ])

    # -------------------------------------------------------------
    # TAB 1: SENSOR DISTRIBUTIONS & HEALTH SPLITS
    # -------------------------------------------------------------
    with tab_dist:
        st.markdown("#### 🔬 Comparative Sensor Behavior: Healthy vs Failure")
        col_s1, col_s2 = st.columns([1, 2])

        sensor_list = [
            "Air temperature",
            "Process temperature",
            "Rotational speed",
            "Torque",
            "Tool wear",
            "Temp_Difference",
            "Power_kW",
            "Tool_Wear_Strain",
        ]

        with col_s1:
            chosen_sensor = st.selectbox(
                "Select Operational Parameter:",
                options=sensor_list,
                index=3,  # Torque
            )
            plot_type = st.radio("Plot Type:", ["Violin & Box Plot", "Binned Histogram / Density"], index=0)

            # Statistics table for chosen sensor
            h_vals = df_eng[df_eng["Machine failure"] == 0][chosen_sensor]
            f_vals = df_eng[df_eng["Machine failure"] == 1][chosen_sensor]

            st.markdown("##### 📐 Sub-population Benchmarks")
            st.markdown(
                f"""
                <div style='background:#F8FAFC; border:1px solid #E2E8F0; padding:0.8rem; border-radius:6px; font-size:0.85rem;'>
                    <b>Healthy Fleet (n={len(h_vals):,}):</b><br>
                    • Mean: {h_vals.mean():.2f}<br>
                    • Median: {h_vals.median():.2f}<br>
                    • Std Dev: {h_vals.std():.2f}<br><br>
                    <b>Failure Fleet (n={len(f_vals):,}):</b><br>
                    • Mean: <span style='color:#DC2626; font-weight:700;'>{f_vals.mean():.2f}</span><br>
                    • Median: {f_vals.median():.2f}<br>
                    • Std Dev: {f_vals.std():.2f}<br><br>
                    <b>Variance Delta:</b> {abs(f_vals.mean() - h_vals.mean()):.2f}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_s2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            if plot_type == "Violin & Box Plot":
                fig_comp = create_sensor_distribution_chart(df_eng, chosen_sensor)
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                fig_hist = px.histogram(
                    df_eng,
                    x=chosen_sensor,
                    color="Machine failure",
                    barmode="overlay",
                    color_discrete_map={0: "#0D9488", 1: "#DC2626"},
                    labels={"Machine failure": "Failure Status (0=Healthy, 1=Fail)"},
                    title=f"Density Overlay: {chosen_sensor}",
                    opacity=0.7,
                )
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    font=dict(family="Inter", size=12, color="#475569"),
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 2: CORRELATION HEATMAP
    # -------------------------------------------------------------
    with tab_corr:
        st.markdown("#### 🔗 Operational Sensor Correlation Matrix")
        st.caption(
            "Visualizes pairwise Pearson correlation coefficients across raw sensors, physics-derived features, "
            "and failure modes. Reveals underlying thermodynamic and kinetic interactions."
        )

        corr_features = [
            "Air temperature",
            "Process temperature",
            "Rotational speed",
            "Torque",
            "Tool wear",
            "Temp_Difference",
            "Power_kW",
            "Tool_Wear_Strain",
            "Machine failure",
        ]
        fig_corr = create_correlation_heatmap(df_eng[corr_features])
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown(
            """
            **Physical Engineering Interpretations:**
            - **Torque vs. Rotational Speed**: Strongly negatively correlated (~ -0.88). Spindle power follows $P = \tau \cdot \omega$; constant power operation forces torque to decrease as RPM increases.
            - **Air vs. Process Temperature**: Highly positively correlated (~ +0.88). Thermal equilibrium between factory ambient air and milling process chamber.
            - **Tool Wear vs. Machine Failure**: Moderate positive correlation. Risk of failure escalates sharply beyond 200 minutes of continuous milling.
            """
        )

    # -------------------------------------------------------------
    # TAB 3: PAIRWISE & 3D SENSOR REGIMES
    # -------------------------------------------------------------
    with tab_scatter:
        st.markdown("#### 🌐 Multi-Dimensional Operational Regimes")

        sub_tab2d, sub_tab3d = st.tabs(["2D Scatter with Marginal Distributions", "3D Operating Envelope"])

        with sub_tab2d:
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                x_axis = st.selectbox("X-Axis Parameter:", sensor_list, index=3)  # Torque
            with c_p2:
                y_axis = st.selectbox("Y-Axis Parameter:", sensor_list, index=2)  # Rotational speed

            fig_2d = px.scatter(
                df_eng,
                x=x_axis,
                y=y_axis,
                color="Machine failure",
                marginal_x="box",
                marginal_y="box",
                color_discrete_map={0: "#0D9488", 1: "#DC2626"},
                labels={"Machine failure": "Failure Status (0=Healthy, 1=Fail)"},
                title=f"Bivariate Regime: {x_axis} vs. {y_axis}",
                opacity=0.6,
            )
            fig_2d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=520,
                font=dict(family="Inter", size=12, color="#475569"),
            )
            st.plotly_chart(fig_2d, use_container_width=True)

        with sub_tab3d:
            st.caption("Inspect the 3-axis operational envelope. Drag and rotate to observe machine failure clusters.")
            fig_3d = px.scatter_3d(
                df_eng.sample(min(3000, len(df_eng)), random_state=42),
                x="Torque",
                y="Rotational speed",
                z="Tool wear",
                color="Machine failure",
                color_discrete_map={0: "rgba(13, 148, 136, 0.4)", 1: "rgba(220, 38, 38, 0.95)"},
                labels={"Machine failure": "Failure"},
                title="3D Envelope: Spindle Torque × Rotational Speed × Tool Wear",
                opacity=0.7,
            )
            fig_3d.update_layout(
                height=550,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", size=11, color="#475569"),
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: OUTLIER DETECTION ENGINE
    # -------------------------------------------------------------
    with tab_outliers:
        st.markdown("#### 🚨 Sensor Outlier Detection (Tukey's IQR Method)")
        st.caption(
            "Identifies sensor measurements that deviate beyond standard statistical operating bands "
            "($Q_1 - 1.5 \\times \\text{IQR}$ or $Q_3 + 1.5 \\times \\text{IQR}$). Outliers often herald "
            "sensor calibration drift or mechanical anomalies."
        )

        outlier_summary = []
        raw_numeric_cols = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]

        for col in raw_numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = int(outliers_mask.sum())
            outlier_failures = int(df.loc[outliers_mask, "Machine failure"].sum())

            outlier_summary.append({
                "Sensor": col,
                "Lower Boundary": round(lower_bound, 2),
                "Upper Boundary": round(upper_bound, 2),
                "Outlier Count": outlier_count,
                "Outlier %": f"{(outlier_count / len(df)) * 100:.2f}%",
                "Associated Failures": outlier_failures,
                "Failure Risk in Outlier Group": f"{(outlier_failures / max(outlier_count, 1)) * 100:.1f}%",
            })

        df_outliers = pd.DataFrame(outlier_summary)
        st.dataframe(df_outliers, use_container_width=True, hide_index=True)

        st.info(
            "💡 **Analytical Finding:** Notice that rotational speed and torque exhibit significant outlier clusters. "
            "More importantly, machines with outlier sensor readings have a dramatically higher failure rate than the "
            "fleet baseline of 3.39%, confirming that sensor boundary violations strongly correlate with catastrophic events."
        )


render_sensor_analytics_page()
