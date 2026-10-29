"""
recommender.py
Rule-based model recommender for ModelScout.

If models/meta_model.joblib exists, it is loaded and used instead.

Main entry point:
    recommend(meta, task) -> pd.DataFrame

The returned DataFrame has one row per model with columns:
    model, est_score, score_low, score_high, est_fit_time_s, rank, reason, confidence

Worth-it verdict:
    compare best est_score against max(dummy, linear) landmarker scores.
    Exposed margins can be edited here:
"""

from __future__ import annotations

import math
import os
import pathlib
import warnings
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Worth-it thresholds (edit here; shown in the UI)
# ---------------------------------------------------------------------------
WORTH_IT_MARGINS = {
    "not_worth": 0.02,   # relative improvement < 2 % → not worth it
    "marginal":  0.05,   # 2–5 % → marginal
    # > 5 % → Recommended
}

# ---------------------------------------------------------------------------
# Path to learned meta-model
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).parent.parent
_META_MODEL_PATH = _ROOT / "models" / "meta_model.joblib"

# ---------------------------------------------------------------------------
# Model pool definitions
# ---------------------------------------------------------------------------
_MODEL_POOL = [
    "Dummy",
    "Logistic/Linear Regression",
    "Random Forest",
    "LightGBM",
    "XGBoost",
    "k-NN",
    "MLP",
]

# ---------------------------------------------------------------------------
# Rule table
# Each rule is a dict:
#   condition: callable(meta) -> bool
#   model:     model name (must be in _MODEL_POOL)
#   delta:     adjustment to base score (can be negative)
#   reason:    short explanation used in the 'reason' column
# ---------------------------------------------------------------------------
_RULES: list[dict] = [
    # Boosted trees love large datasets with many categoricals
    {
        "condition": lambda m: (m.get("n_rows", 0) or 0) > 5_000
                               and (m.get("pct_categorical", 0) or 0) > 0.3,
        "model": "LightGBM",
        "delta": +0.06,
        "reason": "large dataset with many categorical features (boosting excels here)",
    },
    {
        "condition": lambda m: (m.get("n_rows", 0) or 0) > 5_000
                               and (m.get("pct_categorical", 0) or 0) > 0.3,
        "model": "XGBoost",
        "delta": +0.05,
        "reason": "large dataset with many categorical features",
    },
    {
        "condition": lambda m: (m.get("n_rows", 0) or 0) > 5_000
                               and (m.get("pct_categorical", 0) or 0) > 0.3,
        "model": "Random Forest",
        "delta": +0.03,
        "reason": "large dataset with many categoricals favors tree ensembles",
    },
    # Very small datasets: penalise MLP, favour linear and RF
    {
        "condition": lambda m: (m.get("n_rows", 0) or 0) < 500,
        "model": "MLP",
        "delta": -0.08,
        "reason": "very small dataset — MLP tends to overfit without sufficient data",
    },
    {
        "condition": lambda m: (m.get("n_rows", 0) or 0) < 500,
        "model": "Logistic/Linear Regression",
        "delta": +0.04,
        "reason": "small dataset favors regularized linear models",
    },
    {
        "condition": lambda m: (m.get("n_rows", 0) or 0) < 500,
        "model": "Random Forest",
        "delta": +0.02,
        "reason": "Random Forest is robust on small datasets",
    },
    # High correlation → linear model gains
    {
        "condition": lambda m: (m.get("mean_abs_correlation", 0) or 0) > 0.5
                               and (m.get("lm_linear_score") is not None)
                               and (m.get("lm_stump_score") is not None)
                               and (m.get("lm_linear_score", 0) or 0)
                                   > (m.get("lm_stump_score", 0) or 0) + 0.05,
        "model": "Logistic/Linear Regression",
        "delta": +0.05,
        "reason": "high feature correlation and linear landmarker outperforms stump — near-linear signal",
    },
    # High dimensionality → penalize k-NN
    {
        "condition": lambda m: (m.get("n_cols", 0) or 0) > 50,
        "model": "k-NN",
        "delta": -0.07,
        "reason": "high dimensionality hurts k-NN due to the curse of dimensionality",
    },
    {
        "condition": lambda m: (m.get("rows_per_col", 0) or 0) < 5,
        "model": "k-NN",
        "delta": -0.05,
        "reason": "low rows-per-column ratio penalises distance-based methods",
    },
    # Stump clearly better than linear → non-linear signal
    {
        "condition": lambda m: (m.get("lm_stump_score") is not None)
                               and (m.get("lm_linear_score") is not None)
                               and (m.get("lm_stump_score", 0) or 0)
                                   > (m.get("lm_linear_score", 0) or 0) + 0.05,
        "model": "Random Forest",
        "delta": +0.04,
        "reason": "decision stump outperforms linear regression — non-linear patterns present",
    },
    {
        "condition": lambda m: (m.get("lm_stump_score") is not None)
                               and (m.get("lm_linear_score") is not None)
                               and (m.get("lm_stump_score", 0) or 0)
                                   > (m.get("lm_linear_score", 0) or 0) + 0.05,
        "model": "LightGBM",
        "delta": +0.03,
        "reason": "non-linear signal detected — boosting likely to capture it",
    },
    # Medium dataset sweet spot for RF
    {
        "condition": lambda m: 500 <= (m.get("n_rows", 0) or 0) <= 20_000,
        "model": "Random Forest",
        "delta": +0.02,
        "reason": "medium-sized dataset — Random Forest is reliable in this range",
    },
    # Large dataset MLP opportunity
    {
        "condition": lambda m: (m.get("n_rows", 0) or 0) > 10_000
                               and (m.get("pct_numeric", 0) or 0) > 0.7,
        "model": "MLP",
        "delta": +0.03,
        "reason": "large dataset with mostly numeric features — MLP can exploit numeric structure",
    },
    # Class imbalance → linear and tree-based benefit, k-NN suffers
    {
        "condition": lambda m: (m.get("class_imbalance_ratio") is not None)
                               and (m.get("class_imbalance_ratio", 1) or 1) < 0.2,
        "model": "k-NN",
        "delta": -0.04,
        "reason": "class imbalance — k-NN vote majority and perform poorly on minority class",
    },
    {
        "condition": lambda m: (m.get("class_imbalance_ratio") is not None)
                               and (m.get("class_imbalance_ratio", 1) or 1) < 0.2,
        "model": "LightGBM",
        "delta": +0.02,
        "reason": "LightGBM handles class imbalance well with class_weight support",
    },
]

# ---------------------------------------------------------------------------
# Rough fit-time formula  (seconds, rough estimate — labelled in UI)
# Coefficients are illustrative; scale is rows × cols.
# ---------------------------------------------------------------------------
_FIT_TIME_COEFF = {
    "Dummy":                    1e-8,
    "Logistic/Linear Regression": 5e-6,
    "Random Forest":            3e-5,
    "LightGBM":                 8e-6,
    "XGBoost":                  1.2e-5,
    "k-NN":                     2e-7,
    "MLP":                      6e-5,
}


def _est_fit_time(model_name: str, n_rows: int, n_cols: int) -> float:
    coeff = _FIT_TIME_COEFF.get(model_name, 1e-5)
    t = coeff * n_rows * max(n_cols, 1)
    return round(max(t, 0.01), 2)


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def _meta_model_mode() -> Optional[object]:
    """Return loaded meta-model if present, else None."""
    if _META_MODEL_PATH.exists():
        try:
            import joblib
            return joblib.load(_META_MODEL_PATH)
        except Exception:
            return None
    return None


def _base_score_from_landmarkers(meta: dict, task: str) -> float:
    """
    Derive a baseline estimated score from the best landmarker score.
    This is the anchor that rule adjustments are applied on top of.
    """
    lm_scores = [
        meta.get("lm_stump_score"),
        meta.get("lm_gnb_score"),
        meta.get("lm_1nn_score"),
        meta.get("lm_linear_score"),
    ]
    valid = [s for s in lm_scores if s is not None]
    if not valid:
        # No landmarkers ran — use a conservative prior
        return 0.55 if task == "classification" else 0.30
    return float(np.mean(sorted(valid)[-2:]))  # mean of top-2 landmarkers


def _worth_it_verdict(best_score: float, meta: dict, task: str) -> tuple[str, str, float]:
    """
    Compare best estimated score against max(dummy, linear) baseline.
    Returns (verdict_key, reasoning, margin).
    verdict_key: 'recommended' | 'marginal' | 'not_worth'
    """
    dummy = meta.get("lm_dummy_score") or 0.0
    linear = meta.get("lm_linear_score") or 0.0
    baseline = max(dummy, linear)
    if baseline <= 0:
        baseline = 0.5 if task == "classification" else 0.0

    margin = best_score - baseline
    threshold = abs(baseline) if baseline != 0 else 1.0
    rel_improvement = margin / max(abs(baseline), 1e-6)

    if rel_improvement < WORTH_IT_MARGINS["not_worth"]:
        verdict = "not_worth"
        reason = (
            f"Best model improves on the baseline by only "
            f"{margin:+.3f} ({rel_improvement:.1%}). "
            f"The simplest approach (dummy or linear) may suffice."
        )
    elif rel_improvement < WORTH_IT_MARGINS["marginal"]:
        verdict = "marginal"
        reason = (
            f"Marginal improvement of {margin:+.3f} ({rel_improvement:.1%}) "
            f"over baseline. Worth verifying before committing to a complex model."
        )
    else:
        verdict = "recommended"
        reason = (
            f"Expected improvement of {margin:+.3f} ({rel_improvement:.1%}) "
            f"over baseline — a meaningful gain that justifies using ML."
        )
    return verdict, reason, round(margin, 4)


def recommend(meta: dict, task: str) -> pd.DataFrame:
    """
    Recommend a ranked list of model families.

    Parameters
    ----------
    meta : dict  returned by compute_meta_features()
    task : "classification" or "regression"

    Returns
    -------
    pd.DataFrame with columns:
        model, est_score, score_low, score_high, est_fit_time_s,
        rank, reason, confidence
    Also attaches attributes:
        .worth_it_verdict  (str: 'recommended' | 'marginal' | 'not_worth')
        .worth_it_reason   (str)
        .worth_it_margin   (float)
        .mode              (str: 'rule-based prototype' | 'learned meta-model')
    """
    meta_model = _meta_model_mode()
    if meta_model is not None:
        return _recommend_learned(meta_model, meta, task)
    return _recommend_rule_based(meta, task)


def _recommend_rule_based(meta: dict, task: str) -> pd.DataFrame:
    n_rows = meta.get("n_rows", 1000) or 1000
    n_cols = meta.get("n_cols", 10) or 10

    # Available models
    pool = list(_MODEL_POOL)
    if not _xgboost_available():
        pool = [m for m in pool if m != "XGBoost"]
    if not _lightgbm_available():
        pool = [m for m in pool if m != "LightGBM"]

    # Base score from landmarkers
    base = _base_score_from_landmarkers(meta, task)

    # Adjustments per model
    adjustments: dict[str, float] = {m: 0.0 for m in pool}
    reasons: dict[str, list[str]] = {m: [] for m in pool}

    for rule in _RULES:
        try:
            if rule["condition"](meta) and rule["model"] in pool:
                adjustments[rule["model"]] += rule["delta"]
                reasons[rule["model"]].append(rule["reason"])
        except Exception:
            pass

    rows = []
    for model in pool:
        adj = adjustments[model]

        if model == "Dummy":
            est = meta.get("lm_dummy_score") or base * 0.7
        elif model == "Logistic/Linear Regression":
            lm = meta.get("lm_linear_score")
            est = (lm + adj) if lm is not None else base + adj
        else:
            est = base + adj

        # Clip to sensible range
        est = float(np.clip(est, 0.0, 1.0 if task == "classification" else 10.0))

        # Score range (uncertainty band)
        uncertainty = 0.05 + max(0.0, 0.08 - adj * 2)  # tighter when rules fire
        score_low = round(max(0.0, est - uncertainty), 3)
        score_high = round(min(1.0, est + uncertainty), 3)

        reason_str = (
            "; ".join(reasons[model]) if reasons[model] else "base estimate from landmarkers"
        )
        est_time = _est_fit_time(model, n_rows, n_cols)

        rows.append(
            {
                "model": model,
                "est_score": round(est, 4),
                "score_low": score_low,
                "score_high": score_high,
                "est_fit_time_s": est_time,
                "reason": reason_str,
            }
        )

    df = pd.DataFrame(rows).sort_values("est_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # Confidence: gap between rank-1 and rank-2
    if len(df) >= 2:
        gap = df.loc[0, "est_score"] - df.loc[1, "est_score"]
        if gap > 0.04:
            confidence = "High"
        elif gap > 0.015:
            confidence = "Medium"
        else:
            confidence = "Low"
    else:
        confidence = "Medium"

    # Check landmarker agreement
    lm_scores_available = [
        v for k, v in meta.items()
        if k.startswith("lm_") and k.endswith("_score") and v is not None
    ]
    if len(lm_scores_available) >= 2:
        lm_range = max(lm_scores_available) - min(lm_scores_available)
        if lm_range > 0.15:
            confidence = "Low"  # Landmarkers disagree heavily

    df["confidence"] = confidence

    # Worth-it verdict
    best_score = df.loc[0, "est_score"]
    verdict, v_reason, margin = _worth_it_verdict(best_score, meta, task)

    # Attach verdict as custom attributes via a subclass trick
    result = _AnnotatedDF(df)
    result.worth_it_verdict = verdict
    result.worth_it_reason = v_reason
    result.worth_it_margin = margin
    result.mode = "rule-based prototype"
    return result


def _recommend_learned(meta_model, meta: dict, task: str) -> pd.DataFrame:
    """
    Placeholder for the learned meta-model path.
    Expected interface:
        meta_model.predict(X: pd.DataFrame) -> np.ndarray of est_scores per model
    """
    warnings.warn(
        "Learned meta-model found but _recommend_learned is a stub. "
        "Falling back to rule-based prototype.",
        UserWarning,
        stacklevel=2,
    )
    result = _recommend_rule_based(meta, task)
    result.mode = "learned meta-model (stub — please implement _recommend_learned)"
    return result


class _AnnotatedDF(pd.DataFrame):
    """pd.DataFrame subclass that carries extra recommendation attributes."""

    _metadata = ["worth_it_verdict", "worth_it_reason", "worth_it_margin", "mode"]

    @property
    def _constructor(self):
        return _AnnotatedDF
