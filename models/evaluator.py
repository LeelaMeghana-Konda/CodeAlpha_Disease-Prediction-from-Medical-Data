"""
evaluator.py
------------
Computes classification metrics and cross-validation scores for each model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.model_selection import cross_validate

from utils.preprocessor import get_cv_splitter, scale_for_cv


# ── Single-split evaluation ───────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """
    Compute all metrics on a held-out test set.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc,
                    confusion_matrix, fpr, tpr, thresholds
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "auc":       round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "report": classification_report(y_test, y_pred),
    }


def evaluate_all(fitted_models, X_test, y_test):
    """
    Evaluate all models and return a dict of {model_name: metrics_dict}.
    """
    results = {}
    for name, model in fitted_models.items():
        results[name] = evaluate_model(model, X_test, y_test)
    return results


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_model(model, X, y, n_splits=5, random_state=42):
    """
    Stratified k-fold CV with per-fold scaling to prevent leakage.

    Returns
    -------
    dict with mean and std for: accuracy, precision, recall, f1, auc
    """
    from sklearn.base import clone

    cv = get_cv_splitter(n_splits=n_splits, random_state=random_state)
    X_arr = np.array(X)
    y_arr = np.array(y)

    fold_metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc"]}

    for train_idx, val_idx in cv.split(X_arr, y_arr):
        X_tr, X_val, y_tr, y_val = scale_for_cv(X_arr, y_arr, train_idx, val_idx)

        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)

        y_pred = fold_model.predict(X_val)
        y_prob = fold_model.predict_proba(X_val)[:, 1]

        fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        fold_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
        fold_metrics["auc"].append(roc_auc_score(y_val, y_prob))

    summary = {}
    for metric, values in fold_metrics.items():
        arr = np.array(values)
        summary[metric] = {
            "mean": round(arr.mean() * 100, 2),
            "std":  round(arr.std()  * 100, 2),
        }
    return summary


def cross_validate_all(models_dict, X, y, n_splits=5):
    """
    Run CV for all models. Returns dict of {model_name: cv_summary}.
    """
    cv_results = {}
    for name, model in models_dict.items():
        print(f"    CV {name}...", end=" ", flush=True)
        cv_results[name] = cross_validate_model(model, X, y, n_splits=n_splits)
        acc = cv_results[name]["accuracy"]
        print(f"✓  acc={acc['mean']:.1f}% ± {acc['std']:.1f}%")
    return cv_results


# ── Summary table ─────────────────────────────────────────────────────────────

def results_to_dataframe(eval_results):
    """Convert evaluate_all() output to a tidy DataFrame."""
    rows = []
    for name, m in eval_results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  m["accuracy"],
            "Precision": m["precision"],
            "Recall":    m["recall"],
            "F1":        m["f1"],
            "AUC":       m["auc"],
        })
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)


def cv_to_dataframe(cv_results):
    """Convert cross_validate_all() output to a tidy DataFrame."""
    rows = []
    for name, metrics in cv_results.items():
        row = {"Model": name}
        for metric, vals in metrics.items():
            row[f"{metric.capitalize()} (mean)"] = vals["mean"]
            row[f"{metric.capitalize()} (std)"]  = vals["std"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Accuracy (mean)", ascending=False).reset_index(drop=True)
