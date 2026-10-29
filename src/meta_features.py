"""
meta_features.py
Compute ~30 dataset meta-features in three groups:
    Simple      – shape, column-type ratios, cardinality
    Statistical – distributional moments, correlation, missing, class balance
    Landmarkers – quick learners on a 10 % subsample

All features are wrapped in try/except so one failure never crashes the app.

Main entry point:
    compute_meta_features(df, target_col, task) -> dict

The returned dict has keys that match the FEATURE_CATALOG list; missing values
are represented as None.
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.preprocess import build_preprocessor, drop_id_columns

# ---------------------------------------------------------------------------
# Catalog: (feature_key, group, description)
# Used by the UI to render the three grouped tables.
# ---------------------------------------------------------------------------
FEATURE_CATALOG: list[tuple[str, str, str]] = [
    # Simple
    ("n_rows",               "Simple",      "Number of rows"),
    ("n_cols",               "Simple",      "Number of feature columns (excl. target)"),
    ("rows_per_col",         "Simple",      "Rows divided by number of columns"),
    ("pct_categorical",      "Simple",      "Fraction of columns that are categorical (0-1)"),
    ("pct_numeric",          "Simple",      "Fraction of columns that are numeric (0-1)"),
    ("mean_cardinality",     "Simple",      "Mean unique value count across categorical columns"),
    ("max_cardinality",      "Simple",      "Max unique value count across categorical columns"),
    # Statistical
    ("mean_skewness",        "Statistical", "Mean absolute skewness across numeric columns"),
    ("mean_kurtosis",        "Statistical", "Mean excess kurtosis across numeric columns"),
    ("mean_abs_correlation", "Statistical", "Mean of upper triangle of abs pairwise correlation matrix"),
    ("pct_missing",          "Statistical", "Fraction of all cells that are missing (0-1)"),
    ("max_col_missing",      "Statistical", "Highest missing fraction in any single column"),
    ("class_imbalance_ratio","Statistical", "Minority class size / majority class size (classification only)"),
    ("class_entropy",        "Statistical", "Shannon entropy of class distribution (classification only)"),
    ("n_classes",            "Statistical", "Number of distinct target classes (classification only)"),
    ("target_skewness",      "Statistical", "Skewness of the target variable (regression only)"),
    ("target_mean",          "Statistical", "Mean of target variable (regression only)"),
    ("target_std",           "Statistical", "Std dev of target variable (regression only)"),
    # Landmarkers
    ("lm_dummy_score",       "Landmarkers", "Balanced accuracy / R2 of a majority/mean dummy baseline"),
    ("lm_dummy_time",        "Landmarkers", "Wall-clock time (s) for the dummy landmarker"),
    ("lm_stump_score",       "Landmarkers", "Balanced accuracy / R2 of a depth-1 decision tree (decision stump)"),
    ("lm_stump_time",        "Landmarkers", "Wall-clock time (s) for the decision stump landmarker"),
    ("lm_gnb_score",         "Landmarkers", "Balanced accuracy of Gaussian Naive Bayes (classification only)"),
    ("lm_gnb_time",          "Landmarkers", "Wall-clock time (s) for the Gaussian NB landmarker"),
    ("lm_1nn_score",         "Landmarkers", "Balanced accuracy / R2 of 1-Nearest Neighbour"),
    ("lm_1nn_time",          "Landmarkers", "Wall-clock time (s) for the 1-NN landmarker"),
    ("lm_linear_score",      "Landmarkers", "Balanced accuracy / R2 of logistic / linear regression"),
    ("lm_linear_time",       "Landmarkers", "Wall-clock time (s) for the logistic / linear regression landmarker"),
]

FEATURE_KEYS: list[str] = [k for k, _, _ in FEATURE_CATALOG]


def _safe(fn, default=None):
    """Execute fn(); return default on any exception."""
    try:
        return fn()
    except Exception:
        return default


def _subsample(
    X: pd.DataFrame, y: pd.Series, task: str, frac: float = 0.10, min_rows: int = 200
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return a stratified (classification) or random (regression) subsample."""
    n = max(min_rows, int(len(X) * frac))
    n = min(n, len(X))
    if task == "classification":
        # Stratified: keep class ratios
        try:
            from sklearn.model_selection import train_test_split
            if len(X) <= n:
                return X, y
            _, X_sub, _, y_sub = train_test_split(
                X, y, test_size=n / len(X), stratify=y, random_state=42
            )
            return X_sub, y_sub
        except Exception:
            pass
    idx = np.random.default_rng(42).choice(len(X), size=n, replace=False)
    return X.iloc[idx], y.iloc[idx]


def _run_landmarker(
    name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    cv: int = 3,
) -> Tuple[Optional[float], float]:
    """
    Run *model* on preprocessed data with *cv*-fold CV.
    Returns (mean_score, elapsed_seconds).
    Score = balanced_accuracy (classification) or R2 (regression).
    Returns (None, elapsed) on failure.
    """
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task == "classification":
                splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                scores = cross_val_score(
                    model, X, y, cv=splitter, scoring="balanced_accuracy", n_jobs=1
                )
            else:
                splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
                scores = cross_val_score(
                    model, X, y, cv=splitter, scoring="r2", n_jobs=1
                )
        score = float(np.nanmean(scores)) if np.any(~np.isnan(scores)) else None
    except Exception:
        score = None
    elapsed = time.perf_counter() - t0
    return score, round(elapsed, 3)


def compute_meta_features(
    df: pd.DataFrame, target_col: str, task: str
) -> Dict[str, Any]:
    """
    Compute all meta-features for (df, target_col, task).

    Parameters
    ----------
    df         : Full dataset including target column.
    target_col : Name of the target column.
    task       : "classification" or "regression".

    Returns
    -------
    dict mapping feature_key -> value (None for failures).
    """
    result: Dict[str, Any] = {k: None for k in FEATURE_KEYS}

    # ------------------------------------------------------------------ #
    # Basic split                                                          #
    # ------------------------------------------------------------------ #
    y_raw = df[target_col].copy()
    X_raw = df.drop(columns=[target_col])

    # Drop ID-like columns (preprocessing already does this but we want X_raw
    # to reflect the same columns used for training)
    X_raw, _dropped = drop_id_columns(X_raw)

    n_rows, n_cols = X_raw.shape

    # ------------------------------------------------------------------ #
    # Simple features                                                      #
    # ------------------------------------------------------------------ #
    result["n_rows"] = n_rows
    result["n_cols"] = n_cols
    result["rows_per_col"] = _safe(lambda: round(n_rows / max(n_cols, 1), 2))

    cat_cols = X_raw.select_dtypes(exclude="number").columns.tolist()
    num_cols = X_raw.select_dtypes(include="number").columns.tolist()

    result["pct_categorical"] = _safe(
        lambda: round(len(cat_cols) / max(n_cols, 1), 4)
    )
    result["pct_numeric"] = _safe(lambda: round(len(num_cols) / max(n_cols, 1), 4))

    if cat_cols:
        cardinalities = [X_raw[c].nunique(dropna=True) for c in cat_cols]
        result["mean_cardinality"] = _safe(lambda: round(np.mean(cardinalities), 2))
        result["max_cardinality"] = _safe(lambda: int(np.max(cardinalities)))
    else:
        result["mean_cardinality"] = 0
        result["max_cardinality"] = 0

    # ------------------------------------------------------------------ #
    # Statistical features                                                 #
    # ------------------------------------------------------------------ #
    if num_cols:
        result["mean_skewness"] = _safe(
            lambda: round(float(X_raw[num_cols].skew().abs().mean()), 4)
        )
        result["mean_kurtosis"] = _safe(
            lambda: round(float(X_raw[num_cols].kurtosis().mean()), 4)
        )
        def _mean_abs_corr():
            if len(num_cols) < 2:
                return None  # correlation undefined with 1 column
            corr = X_raw[num_cols].corr().abs()
            mask = np.triu(np.ones((len(num_cols), len(num_cols))), k=1).astype(bool)
            vals = corr.where(mask).stack()
            return round(float(vals.mean()), 4) if len(vals) else None
        result["mean_abs_correlation"] = _safe(_mean_abs_corr)
    else:
        result["mean_skewness"] = None
        result["mean_kurtosis"] = None
        result["mean_abs_correlation"] = None

    total_cells = df.shape[0] * df.shape[1]
    result["pct_missing"] = _safe(
        lambda: round(df.isnull().sum().sum() / max(total_cells, 1), 4)
    )
    col_missing_fracs = df.isnull().mean()
    result["max_col_missing"] = _safe(
        lambda: round(float(col_missing_fracs.max()), 4)
    )

    if task == "classification":
        vc = y_raw.value_counts(dropna=True)
        if len(vc) >= 2:
            result["class_imbalance_ratio"] = _safe(
                lambda: round(float(vc.iloc[-1] / vc.iloc[0]), 4)
            )
            probs = vc / vc.sum()
            result["class_entropy"] = _safe(
                lambda: round(float(-np.sum(probs * np.log2(probs + 1e-12))), 4)
            )
        result["n_classes"] = _safe(lambda: int(y_raw.nunique(dropna=True)))
    else:
        result["target_skewness"] = _safe(
            lambda: round(float(y_raw.dropna().skew()), 4)
        )
        result["target_mean"] = _safe(lambda: round(float(y_raw.dropna().mean()), 4))
        result["target_std"] = _safe(lambda: round(float(y_raw.dropna().std()), 4))

    # ------------------------------------------------------------------ #
    # Landmarkers                                                          #
    # ------------------------------------------------------------------ #
    # Encode target for classification
    y_for_lm = y_raw.copy()
    if task == "classification":
        le = LabelEncoder()
        y_for_lm = pd.Series(
            le.fit_transform(y_raw.fillna("__missing__")),
            index=y_raw.index,
        )

    X_sub, y_sub = _subsample(X_raw, y_for_lm, task)

    # Preprocess the subsample
    try:
        preprocessor = build_preprocessor(X_sub)
        X_sub_proc = preprocessor.fit_transform(X_sub)
        y_sub_arr = y_sub.to_numpy()
    except Exception:
        # If preprocessing fails, skip all landmarkers
        return result

    cv_folds = 3

    # Dummy
    if task == "classification":
        dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    else:
        dummy = DummyRegressor(strategy="mean")
    s, t = _run_landmarker("dummy", dummy, X_sub_proc, y_sub_arr, task, cv_folds)
    result["lm_dummy_score"] = s
    result["lm_dummy_time"] = t

    # Decision stump
    if task == "classification":
        stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    else:
        stump = DecisionTreeRegressor(max_depth=1, random_state=42)
    s, t = _run_landmarker("stump", stump, X_sub_proc, y_sub_arr, task, cv_folds)
    result["lm_stump_score"] = s
    result["lm_stump_time"] = t

    # Gaussian NB (classification only)
    if task == "classification":
        gnb = GaussianNB()
        s, t = _run_landmarker("gnb", gnb, X_sub_proc, y_sub_arr, task, cv_folds)
        result["lm_gnb_score"] = s
        result["lm_gnb_time"] = t

    # 1-NN
    if task == "classification":
        nn1 = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    else:
        nn1 = KNeighborsRegressor(n_neighbors=1, n_jobs=1)
    s, t = _run_landmarker("1nn", nn1, X_sub_proc, y_sub_arr, task, cv_folds)
    result["lm_1nn_score"] = s
    result["lm_1nn_time"] = t

    # Logistic / Linear regression
    if task == "classification":
        linear = LogisticRegression(max_iter=300, solver="lbfgs", random_state=42, n_jobs=1)
    else:
        linear = LinearRegression(n_jobs=1)
    s, t = _run_landmarker("linear", linear, X_sub_proc, y_sub_arr, task, cv_folds)
    result["lm_linear_score"] = s
    result["lm_linear_time"] = t

    return result
