"""
Module 1: Overview Dashboard for MaintenAI.
Displays high-level fleet KPIs, dataset statistics, failure distributions,
sensor summary cards, and interactive data filters.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import (
    load_cached_dataset,
    load_cached_evaluation_metrics,
    apply_custom_industrial_css,
    render_header,
    render_kpi,
)
from utils.visualizations import (
    create_failure_type_breakdown_chart,
    create_fleet_risk_donut,
)


def render_overview_page():
    apply_custom_industrial_css()

    render_header(
        title="MaintenAI – Executive Fleet Health & Operational Overview",
        subtitle="Holistic predictive maintenance monitoring, machine fleet telemetry, and operational failure analytics.",
        module_tag="Module 1: Overview",
    )

    df = load_cached_dataset()
    eval_metrics = load_cached_evaluation_metrics()

    # Sidebar / Top filters
    st.sidebar.markdown("### 🎛️ Dataset Filters")
    type_filter = st.sidebar.multiselect(
        "Machine Product Type",
        options=["L", "M", "H"],
        default=["L", "M", "H"],
        help="L = Low variant (60%), M = Medium variant (30%), H = High variant (10%)",
    )

    failure_filter = st.sidebar.radio(
        "Failure Status",
        options=["All Machines", "Healthy Only", "Failures Only"],
        index=0,
    )

    filtered_df = df[df["Type"].isin(type_filter)]
    if failure_filter == "Healthy Only":
        filtered_df = filtered_df[filtered_df["Machine failure"] == 0]
    elif failure_filter == "Failures Only":
        filtered_df = filtered_df[filtered_df["Machine failure"] == 1]

    # Compute Core KPIs
    total_machines = len(filtered_df)
    total_failures = int(filtered_df["Machine failure"].sum())
    overall_failure_rate = (total_failures / max(total_machines, 1)) * 100
    healthy_pct = 100.0 - overall_failure_rate

    # Estimate high risk count based on known failures + extreme sensor limits
    high_strain_count = int(
        (
            (filtered_df["Torque"] * filtered_df["Tool wear"]) > 11000
        ).sum()
    )

    # 1. Executive KPI Cards Row
    st.markdown("#### 📊 Fleet Health & Telemetry Metrics")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    render_kpi(
        kpi1,
        title="Total Machines",
        value=f"{total_machines:,}",
        subtitle=f"Filtered from 10,000 baseline",
        badge="Active Fleet",
        badge_type="neutral",
    )
    render_kpi(
        kpi2,
        title="Failure Cases",
        value=f"{total_failures:,}",
        subtitle=f"Historical incident count",
        badge="Critical",
        badge_type="danger" if total_failures > 0 else "healthy",
    )
    render_kpi(
        kpi3,
        title="Failure Rate",
        value=f"{overall_failure_rate:.2f}%",
        subtitle=f"Industry benchmark < 5.0%",
        badge="Imbalanced" if overall_failure_rate < 5 else "High",
        badge_type="warning",
    )
    render_kpi(
        kpi4,
        title="Healthy Fleet",
        value=f"{healthy_pct:.1f}%",
        subtitle=f"{total_machines - total_failures:,} machines nominal",
        badge="Optimal",
        badge_type="healthy",
    )
    render_kpi(
        kpi5,
        title="High Strain Machines",
        value=f"{high_strain_count:,}",
        subtitle="Torque × Tool Wear > 11k",
        badge="Watchlist",
        badge_type="warning",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts Row: Failure Type Breakdown & Product Type Distribution
    col_chart1, col_chart2 = st.columns([1.2, 0.8])

    with col_chart1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig_fail_breakdown = create_failure_type_breakdown_chart(filtered_df)
        st.plotly_chart(fig_fail_breakdown, use_container_width=True)
        st.markdown(
            "<small style='color:#64748B;'>Note: Component failure modes are non-mutually exclusive (multilabel). "
            "A single machine incident may exhibit both tool wear and thermal breakdown.</small>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        # Type breakdown
        type_counts = filtered_df["Type"].value_counts().reset_index()
        type_counts.columns = ["Product Type", "Count"]
        type_counts["Label"] = type_counts["Product Type"].map(
            {"L": "Type L (Low - 60%)", "M": "Type M (Medium - 30%)", "H": "Type H (High - 10%)"}
        )
        import plotly.express as px
        fig_type = px.pie(
            type_counts,
            names="Label",
            values="Count",
            color_discrete_sequence=["#2563EB", "#0D9488", "#D97706"],
            hole=0.5,
            title="Machine Variant Distribution",
        )
        fig_type.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=12, color="#475569"),
        )
        st.plotly_chart(fig_type, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Operational Sensor Baseline Summary
    st.markdown("#### ⚙️ Operational Sensor Statistics & Engineering Tolerances")
    num_cols = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
    stats_df = filtered_df[num_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    stats_df.columns = ["Mean Value", "Std Dev", "Min Recorded", "Median (50%)", "Max Recorded"]
    stats_df["Standard Unit"] = ["Kelvin [K]", "Kelvin [K]", "Revolutions / min [rpm]", "Newton-meters [Nm]", "Minutes [min]"]
    stats_df["Physical Domain Significance"] = [
        "Ambient factory intake air temperature",
        "Internal thermodynamic chamber temperature",
        "Spindle rotational velocity driven by drive motor",
        "Mechanical cutting torque load experienced by spindle",
        "Cumulative milling tool operating duration",
    ]

    st.dataframe(
        stats_df.style.format({
            "Mean Value": "{:.2f}",
            "Std Dev": "{:.2f}",
            "Min Recorded": "{:.1f}",
            "Median (50%)": "{:.1f}",
            "Max Recorded": "{:.1f}",
        }),
        use_container_width=True,
    )

    # 4. Data Quality & Integrity Audit
    with st.expander("🔍 Dataset Quality & Integrity Report (AI4I 2020 Standard)", expanded=False):
        c_q1, c_q2, c_q3, c_q4 = st.columns(4)
        c_q1.metric("Total Records", f"{len(df):,}")
        c_q2.metric("Missing / Null Values", f"{df.isnull().sum().sum()}")
        c_q3.metric("Duplicate Entries", f"{df.duplicated().sum()}")
        c_q4.metric("Engineered Feature Count", "16 Features")

        st.markdown(
            """
            - **Dataset Source**: Official UCI Machine Learning Repository (AI4I 2020 Predictive Maintenance Dataset).
            - **Target Variable**: `Machine failure` (Binary: 0 for Healthy, 1 for Failure).
            - **Class Imbalance**: 339 failure events out of 10,000 samples (3.39% minority class).
            - **Multilabel Indicators**: TWF (46), HDF (115), PWF (95), OSF (98), RNF (19).
            - **Integrity Guarantee**: Identification columns (`UID`, `Product ID`) and failure mode indicators are strictly isolated during predictive modeling to prevent target leakage.
            """
        )

    # Legal / Engineering Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <b>Industrial Domain Notice:</b> The AI4I 2020 dataset is a synthetic industrial dataset designed to reflect
            genuine predictive maintenance physics (heat dissipation equations, cutting strain limits, and spindle power dynamics).
            This application acts as an operational decision-support tool and does not replace certified industrial testing.
        </div>
        """,
        unsafe_allow_html=True,
    )


render_overview_page()
