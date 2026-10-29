"""
explain.py
Generate plain-language recommendation bullets and feature importance charts.

Exports:
    generate_bullets(meta, recommendation_df, task) -> list[str]
    compute_importance(model_name, df, target_col, task, seed)
        -> (pd.DataFrame with columns [feature, importance], method_label)
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.preprocess import build_preprocessor, drop_id_columns
from src.verify import _make_model


# ---------------------------------------------------------------------------
# Plain-language bullet generator
# Rules table: (condition, template_string)
# Use {value} as placeholder if needed.
# ---------------------------------------------------------------------------

def generate_bullets(
    meta: dict,
    recommendation_df: pd.DataFrame,
    task: str,
) -> List[str]:
    """
    Generate 3–5 plain-language bullets explaining the recommendation.
    Based purely on meta-feature threshold rules — no LLM calls.
    """
    bullets: list[str] = []
    top_model = recommendation_df.iloc[0]["model"] if len(recommendation_df) else "the top model"

    n_rows = meta.get("n_rows") or 0
    n_cols = meta.get("n_cols") or 0
    pct_cat = meta.get("pct_categorical") or 0
    mean_corr = meta.get("mean_abs_correlation")
    imbalance = meta.get("class_imbalance_ratio")
    lm_linear = meta.get("lm_linear_score")
    lm_stump = meta.get("lm_stump_score")
    lm_1nn = meta.get("lm_1nn_score")
    max_miss = meta.get("max_col_missing") or 0
    rows_per_col = meta.get("rows_per_col") or 0

    # 1. Dataset size bullet
    if n_rows < 500:
        bullets.append(
            f"Your dataset is small ({n_rows:,} rows). "
            f"Regularised linear models and Random Forest are more reliable "
            f"here than deep or complex models."
        )
    elif n_rows < 5_000:
        bullets.append(
            f"With {n_rows:,} rows, you have a medium-sized dataset. "
            f"Tree ensembles like Random Forest work well in this range, "
            f"and boosted models are worth verifying."
        )
    else:
        bullets.append(
            f"Your dataset has {n_rows:,} rows — large enough for boosted tree models "
            f"(LightGBM, XGBoost) to learn complex patterns efficiently."
        )

    # 2. Feature mix bullet
    pct_cat_pct = round(pct_cat * 100)
    if pct_cat_pct > 50:
        bullets.append(
            f"{pct_cat_pct}% of your feature columns are categorical. "
            f"Tree-based models handle this natively, which is one reason "
            f"{top_model} ranks first."
        )
    elif pct_cat_pct < 20:
        bullets.append(
            f"Most features ({100 - pct_cat_pct}%) are numeric. "
            f"Linear and tree models both apply; the recommendation depends on "
            f"whether patterns in the data are linear or non-linear."
        )
    else:
        bullets.append(
            f"Your dataset is a mix of numeric and categorical features "
            f"({100 - pct_cat_pct}% numeric, {pct_cat_pct}% categorical). "
            f"Tree-based models are a robust choice for mixed data."
        )

    # 3. Linearity signal bullet
    if lm_linear is not None and lm_stump is not None:
        diff = lm_linear - lm_stump
        if diff > 0.05:
            bullets.append(
                f"The linear landmarker ({lm_linear:.3f}) outperforms the decision stump "
                f"({lm_stump:.3f}) by {diff:+.3f}, suggesting a strong linear signal. "
                f"Logistic/Linear Regression may perform competitively."
            )
        elif lm_stump - lm_linear > 0.05:
            bullets.append(
                f"The decision stump ({lm_stump:.3f}) outperforms linear regression "
                f"({lm_linear:.3f}) by {lm_stump - lm_linear:+.3f}, indicating "
                f"non-linear patterns that tree models can exploit."
            )
        else:
            bullets.append(
                f"The linear and stump landmarkers score similarly "
                f"({lm_linear:.3f} vs {lm_stump:.3f}), so both linear and "
                f"tree-based models are reasonable candidates."
            )

    # 4. Dimensionality / k-NN bullet
    if n_cols > 50 or rows_per_col < 5:
        bullets.append(
            f"With {n_cols} features and only {n_rows:,} rows (ratio {rows_per_col:.1f} rows/col), "
            f"distance-based methods like k-NN are penalised by the curse of dimensionality."
        )

    # 5. Missing data / imbalance bullet
    if max_miss > 0.2:
        bullets.append(
            f"One or more columns have over {max_miss:.0%} missing values. "
            f"Tree-based models are more robust to missing data than k-NN or neural nets."
        )
    if task == "classification" and imbalance is not None and imbalance < 0.3:
        bullets.append(
            f"Your target is imbalanced (minority class is {imbalance:.0%} of the majority). "
            f"Balanced accuracy is used as the metric. Boosted models and RF can "
            f"partially compensate with class weighting."
        )

    # Trim to 3–5
    return bullets[:5] if len(bullets) > 5 else bullets if bullets else [
        "No specific meta-feature patterns triggered. "
        "Review the landmarker scores in the Meta-features tab for more detail."
    ]


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def compute_importance(
    model_name: str,
    df: pd.DataFrame,
    target_col: str,
    task: str,
    seed: int = 42,
    n_repeats: int = 10,
) -> Tuple[pd.DataFrame, str]:
    """
    Compute feature importance for *model_name* on *df*.

    Tries SHAP if a learned meta-model is present; otherwise falls back to
    permutation importance via sklearn.inspection.permutation_importance.

    Returns
    -------
    (importance_df, method_label)
    importance_df: pd.DataFrame with columns ['feature', 'importance']
    method_label : "Permutation importance" or "SHAP values"
    """
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    X, dropped = drop_id_columns(X)
    feature_names = X.columns.tolist()

    if task == "classification":
        le = LabelEncoder()
        y_enc = le.fit_transform(y.fillna("__missing__"))
    else:
        y_enc = y.to_numpy(dtype=float)

    X_arr = X.values

    # Train/test split for permutation importance
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_arr, y_enc, test_size=0.25, random_state=seed,
            stratify=y_enc if task == "classification" else None
        )
    except Exception:
        X_tr, X_te, y_tr, y_te = (
            X_arr[:int(0.75 * len(X_arr))],
            X_arr[int(0.75 * len(X_arr)):],
            y_enc[:int(0.75 * len(X_arr))],
            y_enc[int(0.75 * len(X_arr)):],
        )

    preprocessor = build_preprocessor(pd.DataFrame(X_tr, columns=feature_names))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            estimator = _make_model(model_name, task, seed)
        except Exception:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            estimator = (
                RandomForestClassifier(n_estimators=50, random_state=seed)
                if task == "classification"
                else RandomForestRegressor(n_estimators=50, random_state=seed)
            )

        pipe = Pipeline([("pre", preprocessor), ("model", estimator)])
        try:
            pipe.fit(pd.DataFrame(X_tr, columns=feature_names), y_tr)
        except Exception as e:
            return pd.DataFrame({"feature": feature_names, "importance": [0.0] * len(feature_names)}), f"Error: {e}"

        # Permutation importance on the raw feature space (before preprocessing)
        # We use the pipeline so that preprocessing is applied internally.
        try:
            perm = permutation_importance(
                pipe,
                pd.DataFrame(X_te, columns=feature_names),
                y_te,
                n_repeats=n_repeats,
                random_state=seed,
                n_jobs=1,
                scoring="balanced_accuracy" if task == "classification" else "r2",
            )
            imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": perm.importances_mean,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            return imp_df, "Permutation importance (held-out split)"
        except Exception as e:
            return pd.DataFrame({"feature": feature_names, "importance": [0.0] * len(feature_names)}), f"Permutation importance failed: {e}"
