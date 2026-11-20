"""
Module 6: ML Model Comparison for MaintenAI.
Comprehensive benchmark of Logistic Regression, Decision Tree, Random Forest, and XGBoost.
Displays ROC curves, PR curves, Confusion Matrices, Classification Reports,
Radar charts, SMOTE oversampling comparisons, and best model selection rationales.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import (
    load_cached_evaluation_metrics,
    apply_custom_industrial_css,
    render_header,
    render_kpi,
)
from utils.visualizations import (
    create_roc_curves_chart,
    create_pr_curves_chart,
    create_model_comparison_bar,
    create_confusion_matrix_chart,
    create_radar_chart,
)


def render_model_comparison_page():
    apply_custom_industrial_css()

    render_header(
        title="MaintenAI – Machine Learning Model Benchmark & Comparison",
        subtitle="Empirical performance evaluation across 4 classification architectures trained on authentic AI4I data.",
        module_tag="Module 6: Model Comparison",
    )

    metrics_data = load_cached_evaluation_metrics()
    if not metrics_data:
        st.error("Model evaluation artifacts not found. Please verify `models/evaluation_metrics.pkl`.")
        st.stop()

    binary_models = metrics_data["binary_models"]
    smote_models = metrics_data.get("smote_models", {})
    best_model = metrics_data.get("best_model_name", "XGBoost")

    # 1. Best Model Hero Announcement
    best_m = binary_models[best_model]
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left: 5px solid #2563EB; background: #F8FAFC; margin-bottom: 1.5rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="status-badge status-healthy">TOP PERFORMING ARCHITECTURE</span>
                    <h3 style="margin: 0.4rem 0 0.2rem 0; color: #0F172A;">{best_model} Classifier</h3>
                    <p style="margin: 0; font-size: 0.88rem; color: #475569;">
                        Selected as optimal for predictive maintenance: achieves high recall (<b>{best_m['recall'] * 100:.1f}%</b>) 
                        and precision (<b>{best_m['precision'] * 100:.1f}%</b>), leading to superior F1-score (<b>{best_m['f1_score']:.4f}</b>) 
                        and PR-AUC (<b>{best_m['pr_auc']:.4f}</b>).
                    </p>
                </div>
                <div style="text-align:right;">
                    <div style="font-size: 1.8rem; font-weight:700; color:#2563EB;">{best_m['f1_score']:.4f}</div>
                    <div style="font-size: 0.75rem; color:#64748B;">Test Set F1-Score</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Key Metrics Benchmark Table
    st.markdown("#### 📊 Model Performance Benchmark (Held-out Test Set, n=2,000)")

    records = []
    for name, m in binary_models.items():
        records.append({
            "Algorithm": name,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision": f"{m['precision'] * 100:.2f}%",
            "Recall (Sensitivity)": f"{m['recall'] * 100:.2f}%",
            "F1-Score": f"{m['f1_score']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "PR-AUC": f"{m['pr_auc']:.4f}",
        })

    df_comp = pd.DataFrame(records)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="disclaimer-box" style="margin-top:0.5rem; margin-bottom:1.5rem;">
            <b>Why Accuracy is Misleading for Predictive Maintenance:</b>
            Because catastrophic failures represent only 3.39% of historical records, a trivial dummy classifier
            predicting "Healthy" 100% of the time would achieve <b>96.61% accuracy</b>, while missing every single real failure.
            In industrial engineering, <b>Recall</b> (capturing all failures) and <b>Precision</b> (avoiding false alarms)
            are paramount. Therefore, model ranking is driven by <b>F1-Score</b> and <b>PR-AUC</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Visual Comparisons: Grouped Bar & Radar Chart
    st.markdown("#### 📈 Multi-Metric Comparative Visualizations")
    col_v1, col_v2 = st.columns([1.2, 0.8])

    with col_v1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig_bar = create_model_comparison_bar(binary_models)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_v2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig_radar = create_radar_chart(binary_models)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Discrimination Curves: ROC & Precision-Recall
    st.markdown("#### 📉 Discrimination Curves")
    c_roc, c_pr = st.columns(2)

    with c_roc:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig_roc = create_roc_curves_chart(binary_models)
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_pr:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig_pr = create_pr_curves_chart(binary_models)
        st.plotly_chart(fig_pr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 5. Interactive Confusion Matrix & Classification Report Explorer
    st.markdown("#### 🎯 Confusion Matrices & Classification Reports")
    cm_model_choice = st.selectbox(
        "Select Model Architecture for Detailed Diagnostic Inspection:",
        options=list(binary_models.keys()),
        index=list(binary_models.keys()).index(best_model),
    )

    cm_col1, cm_col2 = st.columns([1, 1])
    selected_metrics = binary_models[cm_model_choice]

    with cm_col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        fig_cm = create_confusion_matrix_chart(selected_metrics["confusion_matrix"], model_name=cm_model_choice)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cm_col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown(f"##### Classification Report: {cm_model_choice}")
        rep = selected_metrics["classification_report"]
        rep_rows = []
        for label in ["Healthy (0)", "Failure (1)", "macro avg", "weighted avg"]:
            if label in rep:
                rep_rows.append({
                    "Class": label,
                    "Precision": f"{rep[label]['precision']:.4f}",
                    "Recall": f"{rep[label]['recall']:.4f}",
                    "F1-Score": f"{rep[label]['f1-score']:.4f}",
                    "Support": int(rep[label]['support']),
                })
        st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)
        st.markdown(
            f"""
            <small style='color:#64748B;'>
                • <b>True Positives (TP):</b> Correctly flagged impending failures.<br>
                • <b>False Positives (FP):</b> Unnecessary maintenance alarms.<br>
                • <b>False Negatives (FN):</b> Missed catastrophic failure events (highest industrial cost).
            </small>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # 6. SMOTE vs Class Weighting Benchmark
    if smote_models:
        st.markdown("#### ⚖️ Class Imbalance Handling: Cost-Sensitive Weighting vs. SMOTE")
        st.caption(
            "Empirical comparison of class weighting (cost-sensitive learning) versus Synthetic Minority Over-sampling Technique (SMOTE) "
            "applied strictly on training data."
        )

        smote_rows = []
        for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            base = binary_models[name]
            smote_name = f"{name} (SMOTE)"
            if smote_name in smote_models:
                sm = smote_models[smote_name]
                smote_rows.append({
                    "Algorithm": name,
                    "Base F1": f"{base['f1_score']:.4f}",
                    "SMOTE F1": f"{sm['f1_score']:.4f}",
                    "Base Recall": f"{base['recall'] * 100:.1f}%",
                    "SMOTE Recall": f"{sm['recall'] * 100:.1f}%",
                    "Base Precision": f"{base['precision'] * 100:.1f}%",
                    "SMOTE Precision": f"{sm['precision'] * 100:.1f}%",
                    "Impact Assessment": (
                        "SMOTE preserves recall but slightly increases false positives for tree ensembles"
                        if sm['precision'] < base['precision']
                        else "SMOTE balanced precision and recall effectively"
                    ),
                })
        st.dataframe(pd.DataFrame(smote_rows), use_container_width=True, hide_index=True)


render_model_comparison_page()
