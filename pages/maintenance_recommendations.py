"""
Module 7: Maintenance Recommendation System for MaintenAI.
Translates ML predictions and physics-based telemetry rules into transparent,
actionable maintenance work orders, prioritizing technician dispatch and explaining engineering rationales.
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
from src.failure_prediction import predict_machine_status
from src.maintenance_recommendations import evaluate_maintenance_rules, LEGAL_DISCLAIMER
from src.risk_assessment import compute_fleet_risk


def render_maintenance_recommendations_page():
    apply_custom_industrial_css()

    render_header(
        title="MaintenAI – Intelligent Maintenance Recommendation Engine",
        subtitle="Physics-informed rule engine coupled with ML failure probabilities to generate targeted work orders.",
        module_tag="Module 7: Maintenance Recommendations",
    )

    tab_single, tab_fleet = st.tabs([
        "🔧 Single Machine Advisory & Diagnostic Dispatch",
        "🏭 Fleet-Wide Maintenance Dispatch Queue",
    ])

    # -------------------------------------------------------------
    # TAB 1: SINGLE MACHINE DIAGNOSTIC ADVISOR
    # -------------------------------------------------------------
    with tab_single:
        st.markdown("#### 🛠️ Machine Diagnostic & Work Order Generator")

        # Session State Synchronization for Presets and Custom Edits
        default_preset_name = "Heat Dissipation Failure (HDF) Scenario"
        if "rec_preset_name" not in st.session_state:
            st.session_state["rec_preset_name"] = default_preset_name
            init_data = PRESET_MACHINES[default_preset_name]
            st.session_state["rec_type"] = init_data["Type"]
            st.session_state["rec_air"] = float(init_data["Air temperature"])
            st.session_state["rec_proc"] = float(init_data["Process temperature"])
            st.session_state["rec_speed"] = int(init_data["Rotational speed"])
            st.session_state["rec_torque"] = float(init_data["Torque"])
            st.session_state["rec_tool"] = int(init_data["Tool wear"])

        def on_rec_preset_change():
            chosen = st.session_state["rec_preset_selector"]
            st.session_state["rec_preset_name"] = chosen
            p = PRESET_MACHINES[chosen]
            st.session_state["rec_type"] = p["Type"]
            st.session_state["rec_air"] = float(p["Air temperature"])
            st.session_state["rec_proc"] = float(p["Process temperature"])
            st.session_state["rec_speed"] = int(p["Rotational speed"])
            st.session_state["rec_torque"] = float(p["Torque"])
            st.session_state["rec_tool"] = int(p["Tool wear"])

        st.selectbox(
            "⚡ Quick-Load Operational Test Scenario (updates all parameters below):",
            options=list(PRESET_MACHINES.keys()),
            key="rec_preset_selector",
            index=list(PRESET_MACHINES.keys()).index(st.session_state["rec_preset_name"]),
            on_change=on_rec_preset_change,
            help="Select a scenario to pre-fill parameters, or manually edit any sensor below.",
        )
        current_scenario_desc = PRESET_MACHINES[st.session_state["rec_preset_name"]]["description"]
        st.caption(f"ℹ️ **Scenario Focus**: {current_scenario_desc}")

        st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

        # Prominent Parameter Controls
        st.markdown("##### 🎛️ Live Operational Parameters (Tweak any parameter to update recommendations)")

        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            r_type = st.selectbox(
                "Machine Variant (Type)",
                options=["L", "M", "H"],
                key="rec_type",
                help="Type L: Standard (60%), Type M: Medium (30%), Type H: Heavy-duty (10%)",
            )
            r_air = st.number_input(
                "Air Temperature [K]",
                min_value=290.0,
                max_value=310.0,
                step=0.1,
                key="rec_air",
                help="Nominal: 295.0 - 305.0 K",
            )
        with c_p2:
            r_proc = st.number_input(
                "Process Temperature [K]",
                min_value=300.0,
                max_value=320.0,
                step=0.1,
                key="rec_proc",
                help="Nominal: 305.0 - 315.0 K",
            )
            r_speed = st.number_input(
                "Rotational Speed [rpm]",
                min_value=1000,
                max_value=3000,
                step=10,
                key="rec_speed",
                help="Nominal: 1200 - 2000 rpm",
            )
        with c_p3:
            r_torque = st.number_input(
                "Spindle Torque [Nm]",
                min_value=1.0,
                max_value=90.0,
                step=0.5,
                key="rec_torque",
                help="Nominal: 20.0 - 60.0 Nm",
            )
            r_tool = st.number_input(
                "Tool Wear [min]",
                min_value=0,
                max_value=300,
                step=1,
                key="rec_tool",
                help="Critical boundary >= 200 min",
            )

        # Derived Physics Telemetry HUD
        dt_val = r_proc - r_air
        p_kw_val = (r_torque * r_speed * 2 * np.pi) / (60 * 1000)
        strain_val = r_torque * r_tool
        s_limit = {"L": 11000, "M": 12000, "H": 13000}.get(r_type, 11000)

        st.markdown(
            f"""
            <div style='background-color:#F1F5F9; border:1px solid #CBD5E1; border-radius:8px; padding:0.6rem 1rem; margin-bottom:1.2rem; font-size:0.85rem;'>
                <span style='font-weight:700; color:#0F172A;'>⚡ Live Physics Calculations:</span>
                &nbsp;&nbsp;• <b>Delta T:</b> <span style='color:{'#DC2626' if (dt_val < 8.6 and r_speed < 1380) else '#0F172A'};'>{dt_val:.1f} K</span> {'(⚠️ HDF Risk)' if (dt_val < 8.6 and r_speed < 1380) else ''}
                &nbsp;&nbsp;• <b>Power:</b> <span style='color:{'#DC2626' if (p_kw_val < 3.5 or p_kw_val > 9.0) else '#0F172A'};'>{p_kw_val:.2f} kW</span> {'(⚠️ PWF Risk)' if (p_kw_val < 3.5 or p_kw_val > 9.0) else ''}
                &nbsp;&nbsp;• <b>Strain:</b> <span style='color:{'#DC2626' if strain_val > s_limit else '#0F172A'};'>{strain_val:.0f} / {s_limit} Nm*min</span> {'(⚠️ OSF Risk)' if strain_val > s_limit else ''}
                &nbsp;&nbsp;• <b>Tool Wear:</b> <span style='color:{'#DC2626' if r_tool >= 200 else ('#D97706' if r_tool >= 160 else '#0F172A')};'>{r_tool} min</span> {'(⚠️ Critical)' if r_tool >= 200 else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Run inference and evaluation with active parameters
        sensor_dict = {
            "Type": r_type,
            "Air temperature": r_air,
            "Process temperature": r_proc,
            "Rotational speed": r_speed,
            "Torque": r_torque,
            "Tool wear": r_tool,
        }

        pred_res = predict_machine_status(sensor_dict, model_name="XGBoost")
        rec_res = evaluate_maintenance_rules(
            sensor_data=sensor_dict,
            failure_probability=pred_res["failure_probability"],
            predicted_status=pred_res["status_label"],
            risk_category=pred_res["risk_category"],
        )

        # Priority & Status Header Card
        priority = rec_res["priority_level"]
        p_badge_map = {
            "Critical": ("status-danger", "🚨 CRITICAL PRIORITY DISPATCH"),
            "High": ("status-danger", "⚠️ HIGH PRIORITY MAINTENANCE"),
            "Medium": ("status-warning", "🟡 MEDIUM PRIORITY ACTION"),
            "Low": ("status-healthy", "🟢 NORMAL / ROUTINE MONITORING"),
        }
        b_class, b_text = p_badge_map.get(priority, ("status-neutral", priority))

        st.markdown(
            f"""
            <div class="kpi-card" style="border-left: 5px solid {'#DC2626' if priority in ['Critical', 'High'] else ('#F59E0B' if priority == 'Medium' else '#10B981')};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="status-badge {b_class}">{b_text}</span>
                        <h3 style="margin:0.4rem 0 0.1rem 0; color:#0F172A;">Recommended Maintenance Action Plan</h3>
                        <p style="margin:0; font-size:0.85rem; color:#64748B;">
                            Machine Status: <b>{rec_res['predicted_status']}</b> | 
                            Failure Probability: <b>{rec_res['failure_probability_pct']:.1f}%</b> | 
                            Risk Category: <b>{rec_res['risk_category']}</b> | 
                            Machine Variant: <b>Type {r_type}</b>
                        </p>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:1.6rem; font-weight:700; color:{'#DC2626' if priority in ['Critical', 'High'] else '#0F172A'};">
                            {rec_res['failure_probability_pct']:.1f}%
                        </div>
                        <div style="font-size:0.72rem; color:#64748B;">Failure Probability</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Structured Subsystem Diagnostic Cards (Updating Dynamically)
        st.markdown("#### 🔬 Detailed Subsystem Diagnoses & Target Actions")
        subsystems = rec_res.get("subsystem_diagnostics", [])
        if not subsystems:
            # Fallback cards if old module is cached in memory
            for i, act in enumerate(rec_res.get("recommended_actions", [])):
                param_label = rec_res.get("abnormal_parameters", ["Nominal"])[i] if i < len(rec_res.get("abnormal_parameters", [])) else "Parameter"
                reason_label = rec_res.get("engineering_reasons", ["Safe operation"])[i] if i < len(rec_res.get("engineering_reasons", [])) else "Standard operational baseline."
                subsystems.append({
                    "subsystem": f"Operational Subsystem #{i+1}",
                    "parameter": param_label,
                    "status": "ALERT" if priority in ["Critical", "High"] else "NOMINAL",
                    "badge_type": "danger" if priority in ["Critical", "High"] else "healthy",
                    "reason": reason_label,
                    "action": act,
                })

        for diag in subsystems:
            border_col = "#DC2626" if diag["badge_type"] == "danger" else ("#F59E0B" if diag["badge_type"] == "warning" else "#10B981")
            st.markdown(
                f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {border_col}; border-radius:6px; padding:0.9rem 1.2rem; margin-bottom:0.75rem; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                        <span style="font-size:0.92rem; font-weight:700; color:#0F172A;">{diag['subsystem']}</span>
                        <div>
                            <code style="background:#F1F5F9; color:#334155; font-size:0.8rem; padding:0.15rem 0.4rem; border-radius:4px; margin-right:8px;">{diag['parameter']}</code>
                            <span class="status-badge status-{diag['badge_type']}">{diag['status']}</span>
                        </div>
                    </div>
                    <div style="font-size:0.85rem; color:#475569; margin-bottom:0.35rem;">
                        <b>Diagnostic Rationale:</b> {diag['reason']}
                    </div>
                    <div style="font-size:0.85rem; color:#0F172A; background:#F8FAFC; padding:0.5rem 0.8rem; border-radius:4px; border:1px solid #E2E8F0;">
                        <b>🔧 Recommended Action:</b> {diag['action']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Comprehensive Overview of Abnormalities & Master Actions
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("##### 🚨 Active Abnormal Flags")
            for param in rec_res["abnormal_parameters"]:
                st.markdown(f"• **{param}**")
            st.markdown("</div>", unsafe_allow_html=True)

        with c_right:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("##### 📋 Prioritized Work Order Steps")
            for i, act in enumerate(rec_res["recommended_actions"], 1):
                st.markdown(f"**Step {i}:** {act}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="disclaimer-box">
                {rec_res['disclaimer']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # TAB 2: FLEET-WIDE MAINTENANCE DISPATCH QUEUE
    # -------------------------------------------------------------
    with tab_fleet:
        st.markdown("#### 🏭 Fleet Maintenance Dispatch Queue")
        st.caption("Auto-generated queue of machines exhibiting elevated probability or abnormal physical telemetry.")

        df = load_cached_dataset()
        with st.spinner("Auditing entire fleet maintenance queue..."):
            df_scored = compute_fleet_risk(df, model_name="XGBoost", low_thresh=0.25, high_thresh=0.65)

        action_queue = df_scored[df_scored["Risk_Category"].isin(["High Risk", "Moderate Risk"])].copy()
        action_queue = action_queue.sort_values(by="Failure_Probability", ascending=False)

        q1, q2, q3 = st.columns(3)
        render_kpi(q1, "Machines Requiring Attention", f"{len(action_queue):,}", "High & moderate risk", badge="Queue", badge_type="warning")
        render_kpi(q2, "Immediate Shutdown Candidates", f"{(action_queue['Failure_Probability_Pct'] >= 80).sum():,}", "Failure Prob ≥ 80%", badge="Urgent", badge_type="danger")
        render_kpi(q3, "Preventive Inspection Queue", f"{(action_queue['Risk_Category'] == 'Moderate Risk').sum():,}", "Moderate risk watchlist", badge="Preventive", badge_type="warning")

        st.markdown("<br>", unsafe_allow_html=True)

        queue_records = []
        for _, row in action_queue.head(30).iterrows():
            sens = {
                "Type": row["Type"],
                "Air temperature": row["Air temperature"],
                "Process temperature": row["Process temperature"],
                "Rotational speed": row["Rotational speed"],
                "Torque": row["Torque"],
                "Tool wear": row["Tool wear"],
            }
            eval_r = evaluate_maintenance_rules(
                sensor_data=sens,
                failure_probability=row["Failure_Probability"],
                predicted_status=row["Predicted_Status"],
                risk_category=row["Risk_Category"],
            )
            queue_records.append({
                "UID": int(row["UID"]),
                "Product ID": row["Product ID"],
                "Priority": eval_r["priority_level"],
                "Failure Prob": f"{row['Failure_Probability_Pct']:.1f}%",
                "Primary Abnormal Sensor": eval_r["abnormal_parameters"][0] if eval_r["abnormal_parameters"] else "Nominal",
                "Recommended Work Order": eval_r["recommended_actions"][0],
            })

        st.dataframe(pd.DataFrame(queue_records), use_container_width=True, hide_index=True)


render_maintenance_recommendations_page()
