"""
MaintenAI – ML-Based Industrial Predictive Maintenance and Failure Detection System.
Main Application Entry Point.
"""

import os
import sys
import streamlit as st

# Ensure project root is available in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import apply_custom_industrial_css

# Page Configuration
st.set_page_config(
    page_title="MaintenAI – Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global styling
apply_custom_industrial_css()

# Navigation setup according to prompt modules
pages = {
    "Operational Telemetry": [
        st.Page("pages/overview.py", title="Overview", icon="📊", default=True),
        st.Page("pages/failure_prediction.py", title="Failure Prediction", icon="🔍"),
        st.Page("pages/failure_analysis.py", title="Failure Analysis", icon="🔬"),
        st.Page("pages/risk_monitoring.py", title="Risk Monitoring", icon="🛡️"),
    ],
    "Analytics & Engineering": [
        st.Page("pages/sensor_analytics.py", title="Sensor Analytics", icon="📈"),
        st.Page("pages/model_comparison.py", title="Model Comparison", icon="⚖️"),
        st.Page("pages/maintenance_recommendations.py", title="Maintenance Recommendations", icon="🔧"),
        st.Page("pages/model_explainability.py", title="Model Explainability", icon="💡"),
    ],
}

# Sidebar Branding
st.sidebar.markdown(
    """
    <div style="padding: 0.6rem 0 1rem 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="font-size: 1.6rem; line-height: 1;">⚙️</div>
            <div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">MaintenAI</div>
                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;">Industrial Maintenance AI</div>
            </div>
        </div>
        <div style="margin-top: 0.75rem; display: flex; align-items: center; gap: 8px;">
            <span style="height: 9px; width: 9px; background-color: #10B981; border-radius: 50%; display: inline-block; box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);"></span>
            <span style="font-size: 0.78rem; color: #334155; font-weight: 500;">Telemetry Online • AI4I Fleet (10k)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Run navigation
nav = st.navigation(pages)
nav.run()

# Sidebar Footer
st.sidebar.markdown(
    """
    <div style="padding: 1rem 0 0.5rem 0; border-top: 1px solid #E2E8F0; margin-top: 2rem; font-size: 0.75rem; color: #64748B; line-height: 1.4;">
        <div><b>MaintenAI Enterprise Suite</b></div>
        <div>Model: XGBoost Ensemble (F1: 0.857)</div>
        <div>Dataset: UCI AI4I 2020 Standard</div>
        <div style="margin-top: 0.4rem; color: #94A3B8; font-size: 0.7rem;">Decision-Support System v1.0</div>
    </div>
    """,
    unsafe_allow_html=True,
)
