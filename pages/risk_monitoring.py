"""
Module 4: Failure Probability Monitoring for MaintenAI.
Provides fleet-wide risk monitoring, risk tier segmentation (Low, Moderate, High),
dynamic threshold calibration sliders, machine search & filtering,
and a side-by-side machine risk comparative workbench.
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
from utils.visualizations import create_fleet_risk_donut
from src.risk_assessment import compute_fleet_risk, get_risk_summary


def render_risk_monitoring_page():
    apply_custom_industrial_css()

    render_header(
        title="MaintenAI – Fleet Failure Probability & Risk Tier Monitoring",
        subtitle="Dynamic risk surveillance console with configurable decision thresholds and machine-level diagnostics.",
        module_tag="Module 4: Risk Monitoring",
    )

    df = load_cached_dataset()

    # 1. Configurable Risk Thresholds Sidebar
    st.sidebar.markdown("### ⚙️ Risk Threshold Calibration")
    st.sidebar.caption(
        "Adjust operational decision thresholds. These define risk categorization tiers "
        "and alert boundaries across the fleet."
    )

    low_thresh = st.sidebar.slider(
        "Low-to-Moderate Risk Boundary",
        min_value=0.10,
        max_value=0.50,
        value=0.25,
        step=0.05,
        help="Machines below this probability are marked as Low Risk / Healthy.",
    )

    high_thresh = st.sidebar.slider(
        "Moderate-to-High Risk Boundary",
        min_value=0.50,
        max_value=0.90,
        value=0.65,
        step=0.05,
        help="Machines at or above this probability are marked as High Risk / Imminent Alert.",
    )

    model_choice = st.sidebar.selectbox(
        "Scoring Model:",
        options=["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"],
        index=0,
    )

    # Compute scored fleet
    with st.spinner("Scoring machine fleet failure probabilities..."):
        df_scored = compute_fleet_risk(
            df,
            model_name=model_choice,
            low_thresh=low_thresh,
            high_thresh=high_thresh,
        )
        risk_summary = get_risk_summary(df_scored)

    # 2. Risk Distribution KPI Cards
    st.markdown("#### 🛡️ Fleet Risk Tier Distribution")
    c1, c2, c3, c4, c5 = st.columns(5)

    render_kpi(
        c1,
        "Total Fleet Monitored",
        f"{risk_summary['total_machines']:,}",
        "Active operating units",
        badge="Online",
        badge_type="neutral",
    )
    render_kpi(
        c2,
        "Low Risk Tier",
        f"{risk_summary['low_risk_count']:,}",
        f"{risk_summary['low_risk_pct']:.1f}% of fleet (P < {low_thresh:.2f})",
        badge="Nominal",
        badge_type="healthy",
    )
    render_kpi(
        c3,
        "Moderate Risk Tier",
        f"{risk_summary['mod_risk_count']:,}",
        f"{risk_summary['mod_risk_pct']:.1f}% of fleet ({low_thresh:.2f} ≤ P < {high_thresh:.2f})",
        badge="Monitor",
        badge_type="warning",
    )
    render_kpi(
        c4,
        "High Risk Tier",
        f"{risk_summary['high_risk_count']:,}",
        f"{risk_summary['high_risk_pct']:.1f}% of fleet (P ≥ {high_thresh:.2f})",
        badge="Critical",
        badge_type="danger" if risk_summary["high_risk_count"] > 0 else "healthy",
    )
    render_kpi(
        c5,
        "Mean Failure Prob.",
        f"{risk_summary['average_failure_prob_pct']:.2f}%",
        f"Fleetwide expected risk",
        badge=model_choice,
        badge_type="neutral",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Charts: Risk Donut & Probability Histogram
    ch_col1, ch_col2 = st.columns([1, 1.2])

    with ch_col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig_donut = create_fleet_risk_donut(risk_summary)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ch_col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig_hist = px.histogram(
            df_scored,
            x="Failure_Probability_Pct",
            nbins=50,
            color="Risk_Category",
            color_discrete_map={
                "Low Risk": "#10B981",
                "Moderate Risk": "#F59E0B",
                "High Risk": "#EF4444",
            },
            title="Distribution of Machine Failure Probabilities (%)",
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=40, r=20, t=40, b=30),
            xaxis_title="Failure Probability (%)",
            yaxis_title="Machine Count",
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
            font=dict(family="Inter", size=12, color="#475569"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Interactive Fleet Table with Filters
    st.markdown("#### 📋 Fleet Risk Surveillance Table")
    f_c1, f_c2, f_c3 = st.columns([1, 1, 1.5])

    with f_c1:
        tier_filter = st.multiselect(
            "Filter by Risk Category:",
            options=["High Risk", "Moderate Risk", "Low Risk"],
            default=["High Risk", "Moderate Risk"],
        )
    with f_c2:
        sort_by = st.selectbox(
            "Sort Table By:",
            options=["Failure Probability (Highest First)", "Tool Wear (Highest First)", "Torque (Highest First)"],
            index=0,
        )
    with f_c3:
        search_id = st.text_input("Search by Machine UID or Product ID:", placeholder="e.g. M14860 or 50")

    table_df = df_scored[df_scored["Risk_Category"].isin(tier_filter)]
    if search_id:
        table_df = table_df[
            table_df["Product ID"].astype(str).str.contains(search_id, case=False)
            | table_df["UID"].astype(str).str.contains(search_id, case=False)
        ]

    if sort_by == "Failure Probability (Highest First)":
        table_df = table_df.sort_values(by="Failure_Probability", ascending=False)
    elif sort_by == "Tool Wear (Highest First)":
        table_df = table_df.sort_values(by="Tool wear", ascending=False)
    else:
        table_df = table_df.sort_values(by="Torque", ascending=False)

    display_cols = [
        "UID",
        "Product ID",
        "Type",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
        "Failure_Probability_Pct",
        "Risk_Category",
        "Predicted_Status",
    ]

    st.dataframe(
        table_df[display_cols].head(100).style.format({
            "Air temperature": "{:.1f} K",
            "Process temperature": "{:.1f} K",
            "Rotational speed": "{:.0f} rpm",
            "Torque": "{:.1f} Nm",
            "Tool wear": "{:.0f} min",
            "Failure_Probability_Pct": "{:.2f}%",
        }),
        use_container_width=True,
        height=320,
    )
    st.caption(f"Displaying top {min(100, len(table_df))} machines matching criteria (Total matching: {len(table_df):,}).")

    # 5. Side-by-Side Machine Comparison Tool
    st.markdown("#### ⚖️ Machine-Level Comparative Risk Workbench")
    st.caption("Compare telemetry and probability scores between any two machines in the fleet.")

    comp1, comp2 = st.columns(2)
    high_risk_uids = df_scored[df_scored["Risk_Category"] == "High Risk"]["UID"].tolist()
    low_risk_uids = df_scored[df_scored["Risk_Category"] == "Low Risk"]["UID"].tolist()

    default_m1 = high_risk_uids[0] if high_risk_uids else 1
    default_m2 = low_risk_uids[0] if low_risk_uids else 2

    with comp1:
        m1_uid = st.number_input("Select First Machine UID:", min_value=1, max_value=len(df), value=int(default_m1), step=1)
    with comp2:
        m2_uid = st.number_input("Select Second Machine UID:", min_value=1, max_value=len(df), value=int(default_m2), step=1)

    m1_row = df_scored[df_scored["UID"] == m1_uid].iloc[0]
    m2_row = df_scored[df_scored["UID"] == m2_uid].iloc[0]

    comp_table = pd.DataFrame([
        {"Parameter": "Product ID", f"Machine #{m1_uid}": m1_row["Product ID"], f"Machine #{m2_uid}": m2_row["Product ID"], "Variance / Delta": "-"},
        {"Parameter": "Product Quality Type", f"Machine #{m1_uid}": f"Type {m1_row['Type']}", f"Machine #{m2_uid}": f"Type {m2_row['Type']}", "Variance / Delta": "-"},
        {"Parameter": "Failure Probability", f"Machine #{m1_uid}": f"{m1_row['Failure_Probability_Pct']:.2f}%", f"Machine #{m2_uid}": f"{m2_row['Failure_Probability_Pct']:.2f}%", "Variance / Delta": f"Δ {abs(m1_row['Failure_Probability_Pct'] - m2_row['Failure_Probability_Pct']):.2f}%"},
        {"Parameter": "Risk Tier", f"Machine #{m1_uid}": m1_row["Risk_Category"], f"Machine #{m2_uid}": m2_row["Risk_Category"], "Variance / Delta": "-"},
        {"Parameter": "Air Temperature", f"Machine #{m1_uid}": f"{m1_row['Air temperature']:.1f} K", f"Machine #{m2_uid}": f"{m2_row['Air temperature']:.1f} K", "Variance / Delta": f"{m1_row['Air temperature'] - m2_row['Air temperature']:+.1f} K"},
        {"Parameter": "Process Temperature", f"Machine #{m1_uid}": f"{m1_row['Process temperature']:.1f} K", f"Machine #{m2_uid}": f"{m2_row['Process temperature']:.1f} K", "Variance / Delta": f"{m1_row['Process temperature'] - m2_row['Process temperature']:+.1f} K"},
        {"Parameter": "Rotational Speed", f"Machine #{m1_uid}": f"{m1_row['Rotational speed']:.0f} rpm", f"Machine #{m2_uid}": f"{m2_row['Rotational speed']:.0f} rpm", "Variance / Delta": f"{m1_row['Rotational speed'] - m2_row['Rotational speed']:+.0f} rpm"},
        {"Parameter": "Spindle Torque", f"Machine #{m1_uid}": f"{m1_row['Torque']:.1f} Nm", f"Machine #{m2_uid}": f"{m2_row['Torque']:.1f} Nm", "Variance / Delta": f"{m1_row['Torque'] - m2_row['Torque']:+.1f} Nm"},
        {"Parameter": "Cumulative Tool Wear", f"Machine #{m1_uid}": f"{m1_row['Tool wear']:.0f} min", f"Machine #{m2_uid}": f"{m2_row['Tool wear']:.0f} min", "Variance / Delta": f"{m1_row['Tool wear'] - m2_row['Tool wear']:+.0f} min"},
    ])

    st.table(comp_table)

    # Domain Calibration Notice
    st.markdown(
        """
        <div class="disclaimer-box">
            <b>Safety & Calibration Notice:</b> Machine failure probabilities are generated by statistical ML estimators
            and depend on the selected decision thresholds. They do not constitute guaranteed time-to-failure or
            statutory safety guarantees. Always verify equipment with manufacturer operational limits.
        </div>
        """,
        unsafe_allow_html=True,
    )


render_risk_monitoring_page()
