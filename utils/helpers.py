"""
Application Helpers & State Management for MaintenAI.
Provides cached loaders, preset operational machine profiles, sensor validation ranges,
and custom industrial CSS styles for Streamlit.
"""

import os
import joblib
import pandas as pd
import streamlit as st

# Realistic Operational Presets for Testing & Interactive Demos
PRESET_MACHINES = {
    "Nominal Healthy Machine (Type L)": {
        "Type": "L",
        "Air temperature": 298.2,
        "Process temperature": 308.7,
        "Rotational speed": 1500,
        "Torque": 40.0,
        "Tool wear": 45,
        "description": "Standard steady-state operation within nominal temperature and load limits.",
    },
    "Tool Wear Failure (TWF) Scenario": {
        "Type": "M",
        "Air temperature": 298.5,
        "Process temperature": 309.0,
        "Rotational speed": 1420,
        "Torque": 52.0,
        "Tool wear": 235,
        "description": "Tool wear at 235 min, exceeding the critical 200 min cutter degradation boundary.",
    },
    "Heat Dissipation Failure (HDF) Scenario": {
        "Type": "L",
        "Air temperature": 302.5,
        "Process temperature": 310.2,
        "Rotational speed": 1310,
        "Torque": 48.0,
        "Tool wear": 110,
        "description": "High ambient temperature, low speed (< 1380 rpm), and temperature delta < 8.6 K.",
    },
    "Power Failure (PWF) Scenario": {
        "Type": "L",
        "Air temperature": 299.0,
        "Process temperature": 309.5,
        "Rotational speed": 2850,
        "Torque": 72.0,
        "Tool wear": 85,
        "description": "Extreme rotational speed and maximum torque resulting in abnormal power surge (> 9 kW).",
    },
    "Overstrain Failure (OSF) Scenario": {
        "Type": "L",
        "Air temperature": 300.5,
        "Process temperature": 311.0,
        "Rotational speed": 1360,
        "Torque": 68.5,
        "Tool wear": 215,
        "description": "High torque (68.5 Nm) combined with high tool wear (215 min) exceeding strain threshold.",
    },
}

# Sensor Validation Boundaries
SENSOR_BOUNDS = {
    "Air temperature": {"min": 295.0, "max": 305.0, "default": 300.0, "unit": "K", "step": 0.1},
    "Process temperature": {"min": 305.0, "max": 315.0, "default": 310.0, "unit": "K", "step": 0.1},
    "Rotational speed": {"min": 1100, "max": 2900, "default": 1500, "unit": "rpm", "step": 10},
    "Torque": {"min": 3.0, "max": 80.0, "default": 40.0, "unit": "Nm", "step": 0.5},
    "Tool wear": {"min": 0, "max": 260, "default": 100, "unit": "min", "step": 1},
}


@st.cache_data(show_spinner=False)
def load_cached_dataset(csv_path: str = "data/ai4i2020.csv") -> pd.DataFrame:
    """Loads and caches the AI4I 2020 dataset in memory."""
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at {csv_path}. Please check data setup.")
        st.stop()
    return pd.read_csv(csv_path)


@st.cache_resource(show_spinner=False)
def load_cached_evaluation_metrics(models_dir: str = "models"):
    """Loads evaluation metrics and benchmark summaries."""
    path = os.path.join(models_dir, "evaluation_metrics.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def apply_custom_industrial_css():
    """Injects sleek, clean light-industrial styling for Streamlit."""
    st.markdown(
        """
        <style>
        /* Modern Industrial Typography and Light Slate Theme */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #1E293B;
        }

        /* Top Header Styling */
        .main-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            padding: 1.5rem 2rem;
            border-radius: 8px;
            color: #FFFFFF;
            margin-bottom: 1.5rem;
            border-left: 5px solid #2563EB;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .main-header h1 {
            font-size: 1.85rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
            color: #FFFFFF !important;
        }
        .main-header p {
            color: #94A3B8;
            font-size: 0.95rem;
            margin: 0.35rem 0 0 0;
        }

        /* KPI Card Containers */
        .kpi-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .kpi-card:hover {
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
            border-color: #CBD5E1;
        }
        .kpi-title {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-bottom: 0.4rem;
        }
        .kpi-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.1;
        }
        .kpi-subtitle {
            font-size: 0.8rem;
            color: #64748B;
            margin-top: 0.35rem;
        }

        /* Status Badges */
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .status-healthy {
            background-color: #E6F4EA;
            color: #137333;
            border: 1px solid #CEEAD6;
        }
        .status-warning {
            background-color: #FEF7E0;
            color: #B06000;
            border: 1px solid #FEEFC3;
        }
        .status-danger {
            background-color: #FCE8E6;
            color: #C5221F;
            border: 1px solid #FAD2CF;
        }
        .status-neutral {
            background-color: #F1F5F9;
            color: #475569;
            border: 1px solid #E2E8F0;
        }

        /* Section Card Container */
        .section-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        }

        /* Disclaimer Notice Box */
        .disclaimer-box {
            background-color: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-left: 4px solid #64748B;
            padding: 0.85rem 1.2rem;
            border-radius: 6px;
            font-size: 0.82rem;
            color: #475569;
            margin-top: 1.2rem;
        }

        /* Hide Streamlit default hamburger & footer for clean SaaS feel */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, module_tag: str = ""):
    """Renders the top banner for each application module."""
    tag_html = f"<span class='status-badge status-neutral' style='float:right; margin-top:4px;'>{module_tag}</span>" if module_tag else ""
    st.markdown(
        f"""
        <div class="main-header">
            {tag_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(col, title: str, value: str, subtitle: str = "", badge: str = "", badge_type: str = "neutral"):
    """Renders a standard industrial KPI metric card."""
    badge_html = f"<span class='status-badge status-{badge_type}' style='float:right;'>{badge}</span>" if badge else ""
    col.markdown(
        f"""
        <div class="kpi-card">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div class="kpi-title">{title}</div>
                {badge_html}
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
