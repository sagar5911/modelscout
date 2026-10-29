"""
verify.py
Run real k-fold cross-validation on the top-2 recommended models plus the
dummy baseline, within an overall time budget.

Main entry point:
    run_verification(df, target_col, task, model_names, seed, folds,
                     time_budget_s, progress_cb)
        -> dict[model_name, VerificationResult]

Results are cached with st.cache_data keyed on a dataset hash.
"""

from __future__ import annotations

import hashlib
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.preprocess import build_preprocessor, drop_id_columns


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    model_name: str
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    fit_time_s: Optional[float] = None
    status: str = "pending"        # "ok" | "timed out" | "error" | "skipped"
    error_msg: str = ""
    fold_scores: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------

def _make_model(name: str, task: str, seed: int):
    """Return an unfitted sklearn-compatible estimator for *name*."""
    if name == "Dummy":
        if task == "classification":
            return DummyClassifier(strategy="most_frequent", random_state=seed)
        return DummyRegressor(strategy="mean")

    if name == "Logistic/Linear Regression":
        if task == "classification":
            return LogisticRegression(max_iter=500, solver="lbfgs",
                                      random_state=seed, n_jobs=1)
        return LinearRegression(n_jobs=1)

    if name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if task == "classification":
            return RandomForestClassifier(n_estimators=100, random_state=seed,
                                          n_jobs=1)
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)

    if name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError:
            raise ImportError("LightGBM not installed")
        if task == "classification":
            return LGBMClassifier(n_estimators=100, random_state=seed,
                                  verbose=-1, n_jobs=1)
        return LGBMRegressor(n_estimators=100, random_state=seed,
                             verbose=-1, n_jobs=1)

    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError:
            raise ImportError("XGBoost not installed")
        if task == "classification":
            return XGBClassifier(n_estimators=100, random_state=seed,
                                 verbosity=0, n_jobs=1, eval_metric="logloss")
        return XGBRegressor(n_estimators=100, random_state=seed,
                            verbosity=0, n_jobs=1)

    if name == "k-NN":
        k = 5
        if task == "classification":
            return KNeighborsClassifier(n_neighbors=k, n_jobs=1)
        return KNeighborsRegressor(n_neighbors=k, n_jobs=1)

    if name == "MLP":
        kwargs = dict(hidden_layer_sizes=(64, 32), max_iter=200, random_state=seed,
                      early_stopping=True, n_iter_no_change=10)
        if task == "classification":
            return MLPClassifier(**kwargs)
        return MLPRegressor(**kwargs)

    raise ValueError(f"Unknown model: {name!r}")


# ---------------------------------------------------------------------------
# Dataset fingerprint for cache keying
# ---------------------------------------------------------------------------

def _df_hash(df: pd.DataFrame) -> str:
    h = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    return h


# ---------------------------------------------------------------------------
# Core verification runner (not cached — caching wraps it below)
# ---------------------------------------------------------------------------

def _run_cv_for_model(
    model_name: str,
    task: str,
    X_proc: np.ndarray,
    y_arr: np.ndarray,
    seed: int,
    folds: int,
    time_cap_s: float,
) -> VerificationResult:
    result = VerificationResult(model_name=model_name)
    try:
        estimator = _make_model(model_name, task, seed)
    except ImportError as e:
        result.status = "skipped"
        result.error_msg = str(e)
        return result
    except Exception as e:
        result.status = "error"
        result.error_msg = str(e)
        return result

    if task == "classification":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)

    fold_scores: list[float] = []
    total_fit_time = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, val_idx in splitter.split(X_proc, y_arr):
            X_tr, X_val = X_proc[train_idx], X_proc[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            t0 = time.perf_counter()
            try:
                estimator.fit(X_tr, y_tr)
                elapsed = time.perf_counter() - t0
                total_fit_time += elapsed

                if total_fit_time > time_cap_s:
                    result.status = "timed out"
                    result.fit_time_s = round(total_fit_time, 2)
                    if fold_scores:
                        result.mean_score = round(float(np.mean(fold_scores)), 4)
                        result.std_score = round(float(np.std(fold_scores)), 4)
                        result.fold_scores = fold_scores
                    return result

                y_pred = estimator.predict(X_val)
                if task == "classification":
                    score = balanced_accuracy_score(y_val, y_pred)
                else:
                    score = r2_score(y_val, y_pred)
                fold_scores.append(score)
            except Exception as e:
                result.status = "error"
                result.error_msg = str(e)[:200]
                return result

    result.status = "ok"
    result.mean_score = round(float(np.mean(fold_scores)), 4)
    result.std_score = round(float(np.std(fold_scores)), 4)
    result.fit_time_s = round(total_fit_time, 2)
    result.fold_scores = fold_scores
    return result


# ---------------------------------------------------------------------------
# Public entry point (with Streamlit cache)
# ---------------------------------------------------------------------------

def run_verification(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    model_names: List[str],
    seed: int = 42,
    folds: int = 3,
    time_budget_s: float = 30.0,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, VerificationResult]:
    """
    Run real k-fold CV on *model_names* (typically top-2 + Dummy).

    Parameters
    ----------
    df             : Full dataset including target.
    target_col     : Target column name.
    task           : "classification" or "regression".
    model_names    : Ordered list of model names to evaluate.
    seed           : Random seed.
    folds          : CV folds (3 or 5).
    time_budget_s  : Total wall-clock budget. Shared evenly across models.
    progress_cb    : Optional callable(model_name, fraction_done) for UI updates.

    Returns
    -------
    dict[model_name -> VerificationResult]
    """
    # Ensure Dummy is always included for comparison
    names = list(model_names)
    if "Dummy" not in names:
        names = ["Dummy"] + names

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    X, _ = drop_id_columns(X)

    if task == "classification":
        le = LabelEncoder()
        y_enc = le.fit_transform(y.fillna("__missing__"))
        y_arr = y_enc
    else:
        y_arr = y.to_numpy(dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preprocessor = build_preprocessor(X)
        X_proc = preprocessor.fit_transform(X)

    time_cap_per_model = time_budget_s / max(len(names), 1)

    results: Dict[str, VerificationResult] = {}
    for i, name in enumerate(names):
        if progress_cb:
            progress_cb(name, i / len(names))
        res = _run_cv_for_model(
            model_name=name,
            task=task,
            X_proc=X_proc,
            y_arr=y_arr,
            seed=seed,
            folds=folds,
            time_cap_s=time_cap_per_model,
        )
        results[name] = res

    if progress_cb:
        progress_cb("done", 1.0)

    return results
