"""
Visualization Utilities for MaintenAI.
Creates sleek, industrial-grade Plotly figures adhering to a light slate/steel aesthetic.
All charts use restrained accents (Slate, Navy, Amber, Crimson, Teal) and responsive layouts.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Industrial Color Palette
COLOR_SLATE_DARK = "#1E293B"
COLOR_SLATE_MED = "#475569"
COLOR_SLATE_LIGHT = "#F8FAFC"
COLOR_BORDER = "#E2E8F0"
COLOR_HEALTHY = "#0D9488"      # Industrial Teal / Green
COLOR_WARNING = "#D97706"      # Industrial Amber
COLOR_DANGER = "#DC2626"       # Industrial Crimson / Red
COLOR_PRIMARY = "#2563EB"      # Industrial Precision Blue
COLOR_ACCENT = "#6366F1"       # Slate Indigo

DEFAULT_FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"


def get_base_layout(title: str = "", height: int = 400) -> Dict[str, Any]:
    """Generates standardized clean, professional chart layout."""
    return dict(
        title=dict(
            text=title,
            font=dict(family=DEFAULT_FONT, size=15, color=COLOR_SLATE_DARK),
            x=0.01,
            y=0.96,
        ),
        font=dict(family=DEFAULT_FONT, color=COLOR_SLATE_MED, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=45, r=30, t=50, b=45),
        height=height,
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family=DEFAULT_FONT,
            font_color="#FFFFFF",
        ),
        xaxis=dict(
            gridcolor="#F1F5F9",
            linecolor=COLOR_BORDER,
            zerolinecolor="#E2E8F0",
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            linecolor=COLOR_BORDER,
            zerolinecolor="#E2E8F0",
        ),
    )


def create_risk_gauge_chart(
    probability_pct: float,
    risk_category: str,
    low_thresh_pct: float = 25.0,
    high_thresh_pct: float = 65.0,
) -> go.Figure:
    """Creates a high-precision industrial risk dial gauge."""
    if risk_category == "Low Risk":
        dial_color = COLOR_HEALTHY
    elif risk_category == "Moderate Risk":
        dial_color = COLOR_WARNING
    else:
        dial_color = COLOR_DANGER

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_pct,
            number=dict(suffix="%", font=dict(family=DEFAULT_FONT, size=32, color=COLOR_SLATE_DARK)),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=COLOR_SLATE_MED),
                bar=dict(color=dial_color, thickness=0.35),
                bgcolor="#F8FAFC",
                borderwidth=1,
                bordercolor=COLOR_BORDER,
                steps=[
                    {"range": [0, low_thresh_pct], "color": "rgba(13, 148, 136, 0.12)"},
                    {"range": [low_thresh_pct, high_thresh_pct], "color": "rgba(217, 119, 6, 0.12)"},
                    {"range": [high_thresh_pct, 100], "color": "rgba(220, 38, 38, 0.12)"},
                ],
                threshold=dict(
                    line=dict(color=COLOR_SLATE_DARK, width=3),
                    thickness=0.8,
                    value=probability_pct,
                ),
            ),
        )
    )
    layout = get_base_layout(title="Failure Risk Assessment", height=260)
    layout["margin"] = dict(l=30, r=30, t=35, b=25)
    fig.update_layout(layout)
    return fig


def create_confusion_matrix_chart(cm: List[List[int]], model_name: str = "Model") -> go.Figure:
    """Creates an annotated confusion matrix heatmap."""
    labels = ["Healthy (0)", "Failure (1)"]
    z = np.array(cm)

    # Annotations with counts and descriptions
    annotations = []
    cell_names = [["True Negative (TN)", "False Positive (FP)"], ["False Negative (FN)", "True Positive (TP)"]]
    for i in range(2):
        for j in range(2):
            count = z[i][j]
            name = cell_names[i][j]
            text_color = "#FFFFFF" if (i == 0 and j == 0) or (i == 1 and j == 1 and count > 20) else COLOR_SLATE_DARK
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"<b>{count}</b><br><span style='font-size:10px;'>{name}</span>",
                    showarrow=False,
                    font=dict(color=text_color, size=13, family=DEFAULT_FONT),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale=[[0, "#F1F5F9"], [0.2, "#CBD5E1"], [1, "#1E293B"]],
            showscale=False,
        )
    )
    layout = get_base_layout(title=f"Confusion Matrix: {model_name}", height=320)
    layout["annotations"] = annotations
    layout["xaxis"]["title"] = "Predicted Label"
    layout["yaxis"]["title"] = "True Label"
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(layout)
    return fig


def create_roc_curves_chart(models_metrics: Dict[str, Any]) -> go.Figure:
    """Plots ROC Curves for all evaluated models on a single chart."""
    fig = go.Figure()
    palette = [COLOR_PRIMARY, COLOR_HEALTHY, COLOR_WARNING, COLOR_SLATE_DARK]

    for i, (name, metrics) in enumerate(models_metrics.items()):
        roc_data = metrics.get("roc_curve", {})
        fpr = roc_data.get("fpr", [0, 1])
        tpr = roc_data.get("tpr", [0, 1])
        auc_val = metrics.get("roc_auc", 0.5)
        color = palette[i % len(palette)]

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC = {auc_val:.3f})",
                line=dict(color=color, width=2.5),
            )
        )

    # Baseline reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess (AUC = 0.500)",
            line=dict(color="#94A3B8", width=1.5, dash="dash"),
        )
    )

    layout = get_base_layout(title="Receiver Operating Characteristic (ROC) Curves", height=380)
    layout["xaxis"]["title"] = "False Positive Rate (1 - Specificity)"
    layout["yaxis"]["title"] = "True Positive Rate (Recall / Sensitivity)"
    layout["xaxis"]["range"] = [-0.01, 1.01]
    layout["yaxis"]["range"] = [-0.01, 1.05]
    layout["legend"] = dict(
        orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5
    )
    fig.update_layout(layout)
    return fig


def create_pr_curves_chart(models_metrics: Dict[str, Any]) -> go.Figure:
    """Plots Precision-Recall Curves for imbalanced failure detection."""
    fig = go.Figure()
    palette = [COLOR_PRIMARY, COLOR_HEALTHY, COLOR_WARNING, COLOR_SLATE_DARK]

    for i, (name, metrics) in enumerate(models_metrics.items()):
        pr_data = metrics.get("pr_curve", {})
        recall = pr_data.get("recall", [0, 1])
        precision = pr_data.get("precision", [1, 0])
        pr_auc = metrics.get("pr_auc", 0.0)
        color = palette[i % len(palette)]

        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{name} (PR-AUC = {pr_auc:.3f})",
                line=dict(color=color, width=2.5),
            )
        )

    layout = get_base_layout(title="Precision-Recall Curves (Critical for Imbalanced Data)", height=380)
    layout["xaxis"]["title"] = "Recall (Detection Coverage)"
    layout["yaxis"]["title"] = "Precision (Accuracy of Failure Flags)"
    layout["xaxis"]["range"] = [-0.01, 1.01]
    layout["yaxis"]["range"] = [-0.01, 1.05]
    layout["legend"] = dict(
        orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5
    )
    fig.update_layout(layout)
    return fig


def create_model_comparison_bar(models_metrics: Dict[str, Any]) -> go.Figure:
    """Grouped bar chart comparing Accuracy, Precision, Recall, F1, ROC-AUC."""
    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    fig = go.Figure()

    palette = [COLOR_PRIMARY, COLOR_HEALTHY, COLOR_WARNING, COLOR_SLATE_DARK]
    for i, (name, m) in enumerate(models_metrics.items()):
        values = [
            m.get("accuracy", 0),
            m.get("precision", 0),
            m.get("recall", 0),
            m.get("f1_score", 0),
            m.get("roc_auc", 0),
            m.get("pr_auc", 0),
        ]
        fig.add_trace(
            go.Bar(
                name=name,
                x=categories,
                y=values,
                marker_color=palette[i % len(palette)],
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
            )
        )

    layout = get_base_layout(title="Cross-Model Metric Benchmark", height=400)
    layout["barmode"] = "group"
    layout["yaxis"]["range"] = [0, 1.15]
    layout["yaxis"]["title"] = "Score (0.0 to 1.0)"
    layout["legend"] = dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
    fig.update_layout(layout)
    return fig


def create_radar_chart(models_metrics: Dict[str, Any]) -> go.Figure:
    """Radar chart comparing multi-dimensional model performance."""
    categories = ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    fig = go.Figure()
    palette = [COLOR_PRIMARY, COLOR_HEALTHY, COLOR_WARNING, COLOR_SLATE_DARK]

    for i, (name, m) in enumerate(models_metrics.items()):
        values = [
            m.get("precision", 0),
            m.get("recall", 0),
            m.get("f1_score", 0),
            m.get("roc_auc", 0),
            m.get("pr_auc", 0),
        ]
        values.append(values[0])  # close loop
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=name,
                line=dict(color=palette[i % len(palette)], width=2),
                fill="toself",
                opacity=0.15,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0], gridcolor="#E2E8F0"),
            angularaxis=dict(gridcolor="#E2E8F0"),
            bgcolor="#FFFFFF",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
        height=380,
        font=dict(family=DEFAULT_FONT, color=COLOR_SLATE_MED, size=11),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    return fig


def create_feature_importance_chart(importance_records: List[Dict[str, Any]], title: str = "Feature Importance") -> go.Figure:
    """Horizontal bar chart for global feature importances."""
    df_imp = pd.DataFrame(importance_records).sort_values(by="importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df_imp["importance"],
            y=df_imp["feature"],
            orientation="h",
            marker=dict(
                color=df_imp["importance"],
                colorscale=[[0, "#93C5FD"], [1, "#1D4ED8"]],
            ),
            text=[f"{v:.4f}" for v in df_imp["importance"]],
            textposition="outside",
        )
    )
    layout = get_base_layout(title=title, height=450)
    layout["xaxis"]["title"] = "Relative Importance"
    layout["yaxis"]["title"] = "Feature"
    layout["margin"] = dict(l=140, r=40, t=45, b=40)
    fig.update_layout(layout)
    return fig


def create_shap_waterfall_chart(shap_records: List[Dict[str, Any]], base_value: float) -> go.Figure:
    """Interactive waterfall/contribution chart for local prediction explainability."""
    # Take top 8 contributors by absolute SHAP value
    df_s = pd.DataFrame(shap_records).head(9).iloc[::-1]  # reverse for bottom-up plot

    colors = [COLOR_DANGER if v > 0 else COLOR_HEALTHY for v in df_s["shap_value"]]

    fig = go.Figure(
        go.Bar(
            x=df_s["shap_value"],
            y=df_s["feature"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:+.3f}" for v in df_s["shap_value"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:+.4f}<extra></extra>",
        )
    )
    layout = get_base_layout(title="Local SHAP Contributions (Failure vs Healthy Drivers)", height=360)
    layout["xaxis"]["title"] = "SHAP Impact on Failure Log-Odds (Red = Risk Driver, Green = Healthy Factor)"
    layout["yaxis"]["title"] = "Feature"
    layout["margin"] = dict(l=150, r=40, t=45, b=40)
    fig.update_layout(layout)
    return fig


def create_correlation_heatmap(df_sensors: pd.DataFrame) -> go.Figure:
    """Diverging correlation matrix heatmap."""
    corr = df_sensors.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[[0, "#1E3A8A"], [0.5, "#F8FAFC"], [1, "#B91C1C"]],
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10, family=DEFAULT_FONT),
            colorbar=dict(title="Pearson r", len=0.8),
        )
    )
    layout = get_base_layout(title="Operational Sensor Correlation Matrix", height=480)
    layout["margin"] = dict(l=120, r=30, t=50, b=120)
    fig.update_layout(layout)
    return fig


def create_sensor_distribution_chart(df: pd.DataFrame, sensor_col: str) -> go.Figure:
    """Violin and box distribution comparing Healthy vs Machine Failure."""
    fig = go.Figure()

    df_healthy = df[df["Machine failure"] == 0][sensor_col].dropna()
    df_failure = df[df["Machine failure"] == 1][sensor_col].dropna()

    fig.add_trace(
        go.Box(
            y=df_healthy,
            name="Healthy (0)",
            marker_color=COLOR_HEALTHY,
            boxmean=True,
        )
    )
    fig.add_trace(
        go.Box(
            y=df_failure,
            name="Machine Failure (1)",
            marker_color=COLOR_DANGER,
            boxmean=True,
        )
    )

    layout = get_base_layout(title=f"Distribution Comparison: {sensor_col}", height=380)
    layout["yaxis"]["title"] = sensor_col
    layout["xaxis"]["title"] = "Machine Health Status"
    fig.update_layout(layout)
    return fig


def create_failure_type_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart for component failure types."""
    cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    names = ["Tool Wear (TWF)", "Heat Dissipation (HDF)", "Power (PWF)", "Overstrain (OSF)", "Random (RNF)"]
    counts = [int(df[c].sum()) for c in cols]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=counts,
            marker_color=[COLOR_PRIMARY, COLOR_WARNING, COLOR_DANGER, COLOR_ACCENT, "#64748B"],
            text=counts,
            textposition="outside",
        )
    )
    layout = get_base_layout(title="Component Failure Mode Occurrences (Multilabel)", height=340)
    layout["xaxis"]["title"] = "Failure Mode"
    layout["yaxis"]["title"] = "Historical Failure Incidents"
    fig.update_layout(layout)
    return fig


def create_fleet_risk_donut(risk_summary: Dict[str, Any]) -> go.Figure:
    """Donut chart for fleet risk segmentation."""
    labels = ["Low Risk", "Moderate Risk", "High Risk"]
    values = [
        risk_summary.get("low_risk_count", 0),
        risk_summary.get("mod_risk_count", 0),
        risk_summary.get("high_risk_count", 0),
    ]
    colors = [COLOR_HEALTHY, COLOR_WARNING, COLOR_DANGER]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hoverinfo="label+value+percent",
        )
    )
    layout = get_base_layout(title="Fleet Risk Distribution", height=320)
    layout["margin"] = dict(l=20, r=20, t=40, b=20)
    fig.update_layout(layout)
    return fig
