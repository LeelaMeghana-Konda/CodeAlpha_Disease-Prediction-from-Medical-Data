"""
classifiers.py
--------------
Defines and trains the four classification models:
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest
  - XGBoost
"""

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ── Model factory ─────────────────────────────────────────────────────────────

def get_models(random_state=42):
    """
    Return a dict of {name: unfitted estimator}.
    All models are configured for binary classification with probability output.
    """
    return {
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,       # needed for ROC/AUC
            random_state=random_state,
        ),
        "Logistic Regression": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=500,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        ),
    }


# ── Training helpers ──────────────────────────────────────────────────────────

def train_model(model, X_train, y_train):
    """Fit a model and return it."""
    model.fit(X_train, y_train)
    return model


def train_all(X_train, y_train, random_state=42):
    """
    Train all four models on the given training data.

    Returns
    -------
    dict of {name: fitted_model}
    """
    models = get_models(random_state=random_state)
    fitted = {}
    for name, model in models.items():
        print(f"    Training {name}...", end=" ", flush=True)
        fitted[name] = train_model(model, X_train, y_train)
        print("✓")
    return fitted


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model, model_name, feature_names):
    """
    Extract feature importances where available.

    - Random Forest / XGBoost  : model.feature_importances_
    - Logistic Regression       : abs(coef_[0])
    - SVM                       : None (not natively available for RBF)

    Returns a list of (feature_name, importance) sorted descending,
    or None if unavailable.
    """
    if model_name in ("Random Forest", "XGBoost"):
        imps = model.feature_importances_
    elif model_name == "Logistic Regression":
        imps = abs(model.coef_[0])
    else:
        return None  # SVM with RBF kernel

    pairs = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)
    return pairs
