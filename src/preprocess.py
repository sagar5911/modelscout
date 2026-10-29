"""
preprocess.py
Shared preprocessing pipeline for ModelScout.

Exports:
    build_preprocessor(X)  -> sklearn ColumnTransformer
    drop_id_columns(df)    -> (df_clean, list[str])
    HIGH_CARDINALITY_THRESHOLD
"""

from __future__ import annotations

import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ID_UNIQUENESS_THRESHOLD = 0.95   # nunique / n_rows above this → ID-like column
HIGH_CARDINALITY_THRESHOLD = 50  # unique values above this → group rare levels


def drop_id_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns that are very likely ID fields (non-numeric AND
    nearly all unique values).  Returns the cleaned DataFrame and a list of
    dropped column names.
    """
    n = len(df)
    if n == 0:
        return df, []

    dropped: List[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            ratio = df[col].nunique(dropna=True) / n
        except Exception:
            continue
        if ratio > ID_UNIQUENESS_THRESHOLD:
            dropped.append(col)

    return df.drop(columns=dropped), dropped


def _group_rare_levels(
    series: pd.Series, threshold: int = HIGH_CARDINALITY_THRESHOLD
) -> pd.Series:
    """
    Replace low-frequency levels in a categorical column with '__other__'
    so that one-hot encoding does not explode dimensionality.
    Only applies when the number of unique values exceeds *threshold*.
    """
    n_unique = series.nunique(dropna=True)
    if n_unique <= threshold:
        return series
    counts = series.value_counts()
    # Keep top-N levels where N == threshold; group the rest
    top_levels = set(counts.iloc[:threshold].index)
    return series.where(series.isin(top_levels), other="__other__")


class _RareLevelGrouper:
    """
    sklearn-compatible transformer that groups rare categorical levels before
    one-hot encoding.  Works column-by-column so it can be embedded in a
    Pipeline.
    """

    def __init__(self, threshold: int = HIGH_CARDINALITY_THRESHOLD) -> None:
        self.threshold = threshold
        self._top_levels: dict[int, set] = {}

    def fit(self, X: np.ndarray, y=None) -> "_RareLevelGrouper":
        self._top_levels = {}
        for i in range(X.shape[1]):
            col = pd.Series(X[:, i].astype(str))
            counts = col.value_counts()
            if col.nunique() > self.threshold:
                self._top_levels[i] = set(counts.iloc[: self.threshold].index)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X.astype(object).copy()
        for i, top in self._top_levels.items():
            col = X_out[:, i].astype(str)
            mask = np.array([v not in top for v in col])
            X_out[mask, i] = "__other__"
        return X_out

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer for the feature matrix X.

    Numeric columns:
        1. Median imputation
        2. StandardScaler

    Categorical columns:
        1. Most-frequent imputation
        2. Rare-level grouping (group to '__other__' if > HIGH_CARDINALITY_THRESHOLD levels)
        3. OneHotEncoder  (handle_unknown='ignore', sparse_output=False)

    Parameters
    ----------
    X : pd.DataFrame
        The raw feature matrix (target column already removed).

    Returns
    -------
    ColumnTransformer
        Not yet fitted.
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("rare_grouper", _RareLevelGrouper(threshold=HIGH_CARDINALITY_THRESHOLD)),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        # Fallback: pass-through (e.g., empty frame)
        warnings.warn("No numeric or categorical columns found; using passthrough.")
        return ColumnTransformer(transformers=[("pass", "passthrough", [])])

    return ColumnTransformer(transformers=transformers, remainder="drop")
