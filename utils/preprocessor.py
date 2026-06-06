"""
preprocessor.py
---------------
Handles scaling, encoding, and train/test splitting for all datasets.
"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Stratified train/test split + StandardScaler fit on train only.

    Returns
    -------
    X_train, X_test, y_train, y_test : split arrays
    scaler                            : fitted StandardScaler (for inference)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_cv_splitter(n_splits=5, random_state=42):
    """Return a StratifiedKFold splitter for cross-validation."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def scale_for_cv(X, y, train_idx, val_idx):
    """
    Scale within a single CV fold to avoid data leakage.

    Returns
    -------
    X_tr, X_val : scaled fold arrays
    y_tr, y_val : corresponding labels
    """
    X_arr = np.array(X)
    y_arr = np.array(y)

    X_tr_raw = X_arr[train_idx]
    X_val_raw = X_arr[val_idx]

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr_raw)
    X_val = scaler.transform(X_val_raw)

    return X_tr, X_val, y_arr[train_idx], y_arr[val_idx]
