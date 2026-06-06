"""
data_loader.py
--------------
Loads and preprocesses the three UCI medical datasets:
  - Heart Disease (Cleveland)
  - Diabetes (PIMA Indian)
  - Breast Cancer (Wisconsin)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder


# ── Heart Disease ────────────────────────────────────────────────────────────

HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

HEART_FEATURE_NAMES = [
    "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
    "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate",
    "Exercise Angina", "ST Depression", "Slope", "Num Vessels", "Thalassemia"
]


def load_heart_disease():
    """
    UCI Heart Disease (Cleveland).
    Returns X (DataFrame), y (Series), feature_names (list).
    Target: 0 = no disease, 1 = disease (original 0-4 scale binarised).
    """
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=45)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy().squeeze()
        y = (y > 0).astype(int)
        # Ensure consistent column naming
        if len(X.columns) == len(HEART_FEATURE_NAMES):
            X.columns = HEART_FEATURE_NAMES
        feature_names = list(X.columns)
    except Exception:
        # Fallback: fetch from UCI URL directly
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/heart-disease/processed.cleveland.data"
        )
        df = pd.read_csv(url, header=None, names=HEART_COLUMNS, na_values="?")
        df.dropna(inplace=True)
        df["target"] = (df["target"] > 0).astype(int)
        X = df.drop("target", axis=1)
        y = df["target"]
        feature_names = HEART_FEATURE_NAMES

    X = X.fillna(X.median(numeric_only=True))
    return X, y, feature_names


# ── Diabetes ─────────────────────────────────────────────────────────────────

DIABETES_COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome"
]

DIABETES_FEATURE_NAMES = [
    "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
    "Insulin", "BMI", "Diabetes Pedigree", "Age"
]


def load_diabetes():
    """
    PIMA Indian Diabetes dataset.
    Returns X (DataFrame), y (Series), feature_names (list).
    Target: 0 = non-diabetic, 1 = diabetic.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=34)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy().squeeze()
        # Ensure consistent column naming
        if len(X.columns) == len(DIABETES_FEATURE_NAMES):
            X.columns = DIABETES_FEATURE_NAMES
        feature_names = list(X.columns)
    except Exception:
        url = (
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master"
            "/pima-indians-diabetes.data.csv"
        )
        df = pd.read_csv(url, header=None, names=DIABETES_COLUMNS)
        X = df.drop("outcome", axis=1)
        y = df["outcome"]
        feature_names = DIABETES_FEATURE_NAMES

    # Zero values in medical columns are missing data → replace with median
    zero_invalid = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    for col in zero_invalid:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X, y, feature_names


# ── Breast Cancer ─────────────────────────────────────────────────────────────

def load_breast_cancer_data():
    """
    UCI Breast Cancer Wisconsin (Diagnostic).
    Returns X (DataFrame), y (Series), feature_names (list).
    Target: 0 = benign, 1 = malignant.
    """
    bc = load_breast_cancer()
    X = pd.DataFrame(bc.data, columns=bc.feature_names)
    # sklearn encodes: 0=malignant, 1=benign → flip so 1=malignant
    y = pd.Series(1 - bc.target, name="target")
    return X, y, list(bc.feature_names)


# ── Registry ──────────────────────────────────────────────────────────────────

DATASETS = {
    "heart_disease": {
        "loader": load_heart_disease,
        "display_name": "Heart Disease",
        "positive_class": "Disease",
        "negative_class": "No Disease",
    },
    "diabetes": {
        "loader": load_diabetes,
        "display_name": "Diabetes",
        "positive_class": "Diabetic",
        "negative_class": "Non-diabetic",
    },
    "breast_cancer": {
        "loader": load_breast_cancer_data,
        "display_name": "Breast Cancer",
        "positive_class": "Malignant",
        "negative_class": "Benign",
    },
}


def load_all():
    """Load all three datasets. Returns dict of {name: (X, y, feature_names)}."""
    results = {}
    for key, meta in DATASETS.items():
        print(f"  Loading {meta['display_name']}...", end=" ", flush=True)
        X, y, feat_names = meta["loader"]()
        results[key] = (X, y, feat_names)
        print(f"✓  {X.shape[0]} samples, {X.shape[1]} features")
    return results


if __name__ == "__main__":
    data = load_all()
    for name, (X, y, feats) in data.items():
        print(f"\n{name}: shape={X.shape}, class_balance={y.value_counts().to_dict()}")
