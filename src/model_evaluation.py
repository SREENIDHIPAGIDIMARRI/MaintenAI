"""
Model Evaluation Module for MaintenAI.
Computes comprehensive binary and multilabel classification metrics,
ROC curves, Precision-Recall curves, confusion matrices, and benchmark reports.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    hamming_loss,
)


def evaluate_binary_model(
    model: Any,
    X_test_transformed: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Evaluates a binary classification model on a held-out test set.
    Computes threshold metrics, curve coordinates, and confusion matrix.
    """
    y_pred = model.predict(X_test_transformed)

    # Probability estimation
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_transformed)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test_transformed)
        y_prob = 1 / (1 + np.exp(-decision))
    else:
        y_prob = y_pred.astype(float)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_test, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_test))

    cm = confusion_matrix(y_test, y_pred).tolist()

    # ROC curve points (subsampled for lightweight serialization)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    subsample_idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 100), dtype=int)
    roc_curve_data = {
        "fpr": fpr[subsample_idx].tolist(),
        "tpr": tpr[subsample_idx].tolist(),
    }

    # PR curve points
    p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_subsample = np.linspace(0, len(p_curve) - 1, min(len(p_curve), 100), dtype=int)
    pr_curve_data = {
        "precision": p_curve[pr_subsample].tolist(),
        "recall": r_curve[pr_subsample].tolist(),
    }

    report = classification_report(
        y_test, y_pred, target_names=["Healthy (0)", "Failure (1)"], output_dict=True, zero_division=0
    )

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "roc_curve": roc_curve_data,
        "pr_curve": pr_curve_data,
        "classification_report": report,
    }


def evaluate_multilabel_model(
    model: Any,
    X_test_transformed: np.ndarray,
    y_test_multilabel: pd.DataFrame,
    label_cols: List[str],
) -> Dict[str, Any]:
    """
    Evaluates a multilabel classifier for component failure types (TWF, HDF, PWF, OSF, RNF).
    Returns per-label metrics and overall Hamming Loss.
    """
    y_pred_multi = model.predict(X_test_transformed)
    if isinstance(y_pred_multi, list):
        # MultiOutputClassifier can return list of arrays
        y_pred_multi = np.column_stack(y_pred_multi)

    y_true_multi = y_test_multilabel.values
    h_loss = float(hamming_loss(y_true_multi, y_pred_multi))

    per_label_metrics = {}
    for i, col in enumerate(label_cols):
        y_t = y_true_multi[:, i]
        y_p = y_pred_multi[:, i]
        per_label_metrics[col] = {
            "precision": float(precision_score(y_t, y_p, zero_division=0)),
            "recall": float(recall_score(y_t, y_p, zero_division=0)),
            "f1_score": float(f1_score(y_t, y_p, zero_division=0)),
            "support": int(y_t.sum()),
            "predicted_positives": int(y_p.sum()),
        }

    return {
        "hamming_loss": h_loss,
        "per_label": per_label_metrics,
        "labels": label_cols,
    }
