"""
streamlit_app.py
ModelScout — main application entry point.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import io
import os
import pathlib
import sys
import time
import traceback
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap (so src.* imports work regardless of cwd)
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.theme import (
    COLORS,
    apply_theme,
    badge,
    box,
    card,
    plotly_template,
)
from src.meta_features import FEATURE_CATALOG, compute_meta_features
from src.preprocess import drop_id_columns
from src.recommender import WORTH_IT_MARGINS, recommend

# ---------------------------------------------------------------------------
# Page config (MUST be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ModelScout",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_CSV = _ROOT / "data" / "sample_churn.csv"
META_MODEL_PATH = _ROOT / "models" / "meta_model.joblib"
MAX_PREVIEW_ROWS = 10
MIN_ROWS_REQUIRED = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recommender_mode_label() -> str:
    if META_MODEL_PATH.exists():
        return "Recommender: learned meta-model"
    return "Recommender: rule-based prototype (learned meta-model pending)"


def _df_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()[:12]


def _detect_task(y: pd.Series) -> str:
    """Auto-detect classification vs regression."""
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if y.nunique(dropna=True) <= 20:
        return "classification"
    return "regression"


def _fmt_score(v) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def _fmt_val(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _plotly_fig(fig: go.Figure) -> go.Figure:
    tmpl = plotly_template()
    fig.update_layout(**tmpl["layout"])
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render sidebar controls. Returns dict with user choices."""
    st.sidebar.markdown(
        '<div class="ms-sidebar-section">Dataset</div>', unsafe_allow_html=True
    )

    uploaded = st.sidebar.file_uploader(
        "Upload CSV", type=["csv"], label_visibility="collapsed"
    )
    use_sample = st.sidebar.button("Use sample dataset", key="use_sample_btn")

    st.sidebar.markdown(
        '<div class="ms-sidebar-section">Target</div>', unsafe_allow_html=True
    )

    df: Optional[pd.DataFrame] = None
    source_label = ""

    if use_sample:
        st.session_state["use_sample"] = True
        st.session_state["uploaded_file"] = None

    if uploaded is not None:
        

    # Load data
    if st.session_state.get("use_sample"):
        try:
            df = pd.read_csv(SAMPLE_CSV)
            source_label = "Sample dataset (synthetic)"
        except FileNotFoundError:
            st.sidebar.error("Sample CSV not found. Run: python scripts/make_sample.py")
    elif st.session_state.get("uploaded_file") is not None:
        try:
            df = pd.read_csv(st.session_state["uploaded_file"])
            source_label = st.session_state["uploaded_file"].name
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")

    # Target column
    target_col = None
    if df is not None:
        cols = df.columns.tolist()
        default_idx = len(cols) - 1
        target_col = st.sidebar.selectbox(
            "Target column",
            cols,
            index=default_idx,
            key="target_col",
        )

    # Task type
    st.sidebar.markdown(
        '<div class="ms-sidebar-section">Task</div>', unsafe_allow_html=True
    )

    auto_task = None
    if df is not None and target_col is not None:
        auto_task = _detect_task(df[target_col])

    task_options = ["Auto-detect", "Classification", "Regression"]
    task_choice = st.sidebar.radio(
        "Task type",
        task_options,
        index=0,
        key="task_radio",
        help="Auto-detect infers from the target column.",
    )

    if task_choice == "Auto-detect":
        task = auto_task or "classification"
        if auto_task:
            st.sidebar.caption(f"Detected: {auto_task}")
    else:
        task = task_choice.lower()

    # Options
    st.sidebar.markdown(
        '<div class="ms-sidebar-section">Options</div>', unsafe_allow_html=True
    )
    time_budget = st.sidebar.slider(
        "Verification time budget (s)", min_value=10, max_value=120,
        value=30, step=5, key="time_budget"
    )
    cv_folds = st.sidebar.selectbox(
        "CV folds", [3, 5], index=0, key="cv_folds"
    )
    seed = st.sidebar.number_input(
        "Random seed", min_value=0, max_value=99999,
        value=42, step=1, key="seed"
    )

    # Run button
    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button(
        "Run analysis",
        key="run_btn",
        disabled=(df is None or target_col is None),
        type="primary",
    )

    return {
        "df": df,
        "target_col": target_col,
        "task": task,
        "source_label": source_label,
        "time_budget": time_budget,
        "cv_folds": int(cv_folds),
        "seed": int(seed),
        "run_clicked": run_clicked,
    }


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header() -> None:
    mode_label = _recommender_mode_label()
    st.caption("🚀 Autonomous CI/CD Pipeline Active (Version 1.0.1)")
    st.markdown(
        f"""
        <div class="ms-header">
            <div class="ms-header-left">
                <h1>ModelScout</h1>
                <div class="ms-header-desc">
                    Upload a CSV, choose a target column, and get an ML model recommendation
                    — without training every model.
                </div>
            </div>
            <div class="ms-mode-label">{mode_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def render_empty_state() -> None:
    st.markdown(
        """
        <div style="max-width:540px; margin-top:32px;">
            <p style="font-size:15px; color:#1B1F24; margin-bottom:16px;">
                Upload a CSV and choose a target column to begin.
            </p>
            <div style="border-left:3px solid #D9DDE3; padding-left:16px; margin-top:12px;">
                <p style="margin:6px 0; color:#5B6470; font-size:13px;">
                    <strong style="color:#1B1F24;">1. Upload</strong>
                    &nbsp; Use the sidebar to upload a CSV or load the sample dataset.
                </p>
                <p style="margin:6px 0; color:#5B6470; font-size:13px;">
                    <strong style="color:#1B1F24;">2. Analyze</strong>
                    &nbsp; Select the target column and click <em>Run analysis</em>.
                </p>
                <p style="margin:6px 0; color:#5B6470; font-size:13px;">
                    <strong style="color:#1B1F24;">3. Review</strong>
                    &nbsp; Explore recommendations, verification results, and explanations
                    across the five tabs.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------

def render_overview(df: pd.DataFrame, target_col: str, task: str) -> None:
    # Metric row
    n_rows, n_cols = df.shape
    missing_pct = round(df.isnull().mean().mean() * 100, 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{n_rows:,}")
    c2.metric("Columns", str(n_cols))
    c3.metric("Missing", f"{missing_pct}%")
    c4.metric("Task", task.capitalize())

    # Warnings
    warns = []
    if n_rows < 200:
        warns.append(f"Fewer than 200 rows ({n_rows}). Results may be unreliable.")
    if task == "classification":
        vc = df[target_col].value_counts(dropna=True)
        if len(vc) >= 2:
            ratio = vc.iloc[-1] / vc.iloc[0]
            if ratio < 0.05:
                warns.append(
                    f"Severe class imbalance: minority class is {ratio:.1%} of majority."
                )
    col_miss = df.isnull().mean()
    high_miss_cols = col_miss[col_miss > 0.4].index.tolist()
    if high_miss_cols:
        warns.append(
            f"High missing values (>40%) in: {', '.join(high_miss_cols[:5])}."
        )
    cat_cols = df.select_dtypes(exclude="number").columns
    high_card = [c for c in cat_cols if c != target_col and df[c].nunique() > 50]
    if high_card:
        warns.append(
            f"High-cardinality categoricals (>50 levels): {', '.join(high_card[:5])}."
        )

    if warns:
        warn_html = "".join(f"<p style='margin:4px 0'>&#x26A0; {w}</p>" for w in warns)
        st.markdown(
            f'<div class="ms-box ms-box-warn" style="margin-bottom:12px;">'
            f"<strong>Warnings</strong>{warn_html}</div>",
            unsafe_allow_html=True,
        )

    # Data preview
    st.markdown("#### Data preview (first 10 rows)")
    st.dataframe(df.head(MAX_PREVIEW_ROWS), use_container_width=True, height=280)

    # Column type table
    st.markdown("#### Column summary")
    col_info = []
    for col in df.columns:
        dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        col_info.append(
            {
                "Column": col,
                "Type": dtype,
                "Unique values": int(df[col].nunique(dropna=True)),
                "Missing %": f"{df[col].isnull().mean() * 100:.1f}%",
                "Target": "Yes" if col == target_col else "",
            }
        )
    st.dataframe(
        pd.DataFrame(col_info),
        use_container_width=True,
        hide_index=True,
    )

    # Target distribution chart
    st.markdown("#### Target distribution")
    y = df[target_col].dropna()
    tmpl = plotly_template()

    if task == "classification":
        vc = y.value_counts().reset_index()
        vc.columns = [target_col, "Count"]
        fig = go.Figure(
            go.Bar(
                x=vc[target_col].astype(str),
                y=vc["Count"],
                marker_color=COLORS["accent"],
                marker_line_width=0,
            )
        )
        fig.update_layout(
            **tmpl["layout"],
            title="Target class distribution",
            xaxis_title=target_col,
            yaxis_title="Count",
            height=300,
        )
    else:
        fig = go.Figure(
            go.Histogram(
                x=y,
                nbinsx=40,
                marker_color=COLORS["accent"],
                marker_line_width=0,
            )
        )
        fig.update_layout(
            **tmpl["layout"],
            title="Target variable distribution",
            xaxis_title=target_col,
            yaxis_title="Count",
            height=300,
        )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Meta-features
# ---------------------------------------------------------------------------

def render_meta_features(meta: dict) -> None:
    st.markdown(
        '<p style="font-size:12px;color:#5B6470;margin-bottom:12px;">'
        "Landmarkers are trained on a 10% subsample (min 200 rows) for speed.</p>",
        unsafe_allow_html=True,
    )

    groups = {"Simple": [], "Statistical": [], "Landmarkers": []}
    for key, group, desc in FEATURE_CATALOG:
        val = meta.get(key)
        groups[group].append({"Feature": key, "Value": _fmt_val(val), "What it measures": desc})

    for group_name, rows in groups.items():
        st.markdown(f"#### {group_name} features")

        if group_name == "Landmarkers":
            # Add compute-time column for landmarkers
            display_rows = []
            for r in rows:
                display_rows.append(r)
            st.dataframe(
                pd.DataFrame(display_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

    # Download button
    all_rows = []
    for key, group, desc in FEATURE_CATALOG:
        all_rows.append(
            {"group": group, "feature": key, "value": meta.get(key), "description": desc}
        )
    csv_bytes = pd.DataFrame(all_rows).to_csv(index=False).encode()
    st.download_button(
        label="Download meta-features (CSV)",
        data=csv_bytes,
        file_name="meta_features.csv",
        mime="text/csv",
        key="dl_meta",
    )


# ---------------------------------------------------------------------------
# Tab 3: Recommendation
# ---------------------------------------------------------------------------

def render_recommendation(rec_df, task: str, meta: dict) -> None:
    verdict = getattr(rec_df, "worth_it_verdict", "recommended")
    v_reason = getattr(rec_df, "worth_it_reason", "")
    v_margin = getattr(rec_df, "worth_it_margin", 0.0)
    confidence = rec_df.iloc[0]["confidence"] if len(rec_df) else "Low"
    top = rec_df.iloc[0] if len(rec_df) else None

    # Badge config
    verdict_cfg = {
        "recommended": ("Recommended", "positive"),
        "marginal":    ("Marginal",    "caution"),
        "not_worth":   ("Not worth it","negative"),
    }
    verdict_label, verdict_kind = verdict_cfg.get(verdict, ("Unknown", "neutral"))

    conf_cfg = {
        "High":   "positive",
        "Medium": "neutral",
        "Low":    "caution",
    }

    # Three cards
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="ms-card">'
            f'<div class="ms-card-title">Is ML worth it?</div>'
            f'{badge(verdict_label, verdict_kind)}'
            f'<div class="ms-card-sub" style="margin-top:8px;">{v_reason}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c2:
        if top is not None:
            score_range = f"{top['score_low']:.3f} – {top['score_high']:.3f}"
            fit_t = f"~{top['est_fit_time_s']}s (rough estimate)"
            st.markdown(
                f'<div class="ms-card">'
                f'<div class="ms-card-title">Top recommendation</div>'
                f'<div class="ms-card-value">{top["model"]}</div>'
                f'<div class="ms-card-sub">'
                f'Score range: <strong>{score_range}</strong><br>'
                f'<span style="font-size:11px;color:#5B6470;">estimated from quick landmarkers</span><br>'
                f'Est. fit time: {fit_t}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="ms-card"><div class="ms-card-title">Top recommendation</div>'
                        '<div class="ms-card-sub">No models available.</div></div>',
                        unsafe_allow_html=True)

    with c3:
        conf_note = ""
        if confidence == "Low":
            conf_note = "Unsure: verify the top 2 models in the Verification tab."
        elif confidence == "Medium":
            conf_note = "Moderate certainty. Verification is recommended."
        else:
            conf_note = "Strong separation between the top-ranked model and others."
        st.markdown(
            f'<div class="ms-card">'
            f'<div class="ms-card-title">Confidence</div>'
            f'{badge(confidence, conf_cfg.get(confidence, "neutral"))}'
            f'<div class="ms-card-sub" style="margin-top:8px;">{conf_note}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Ranked table
    st.markdown("#### All models — ranked")

    margin_pct_not = int(WORTH_IT_MARGINS["not_worth"] * 100)
    margin_pct_mar = int(WORTH_IT_MARGINS["marginal"] * 100)
    st.caption(
        f"Worth-it gate: <{margin_pct_not}% improvement = Not worth it; "
        f"{margin_pct_not}–{margin_pct_mar}% = Marginal; >{margin_pct_mar}% = Recommended. "
        f"Margin constants are in src/recommender.py."
    )

    display_df = rec_df[["rank", "model", "est_score", "score_low", "score_high",
                          "est_fit_time_s", "reason"]].copy()
    display_df.columns = ["Rank", "Model", "Est. score", "Score low",
                          "Score high", "Est. fit time (s)", "Why"]
    display_df["Est. score"] = display_df["Est. score"].map(lambda x: f"{x:.4f}")
    display_df["Score low"]  = display_df["Score low"].map(lambda x: f"{x:.3f}")
    display_df["Score high"] = display_df["Score high"].map(lambda x: f"{x:.3f}")

    # Highlight rank-1 row with left border via HTML table
    def _row_style(row):
        if row.name == 0:
            return ["border-left: 3px solid #1F4E79; font-weight:600"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart with error bars
    st.markdown("#### Estimated score by model")
    fig = go.Figure()
    tmpl = plotly_template()

    colors = COLORS["series"] + ["#6B7C93"] * 10
    for i, row in rec_df.iterrows():
        err_low  = row["est_score"] - row["score_low"]
        err_high = row["score_high"] - row["est_score"]
        bar_color = COLORS["accent"] if i == 0 else COLORS["series"][min(i, len(COLORS["series"]) - 1)]
        fig.add_trace(
            go.Bar(
                name=row["model"],
                x=[row["model"]],
                y=[row["est_score"]],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[err_high],
                    arrayminus=[err_low],
                    thickness=1.5,
                    width=6,
                    color=COLORS["text_secondary"],
                ),
                marker_color=bar_color,
                marker_line_width=0,
                showlegend=False,
            )
        )

    metric_name = "Balanced Accuracy" if task == "classification" else "R²"
    fig.update_layout(
        **tmpl["layout"],
        title=f"Estimated {metric_name} (from quick landmarkers)",
        yaxis_title=metric_name,
        xaxis_title="Model",
        height=360,
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Verification
# ---------------------------------------------------------------------------

def render_verification(
    df: pd.DataFrame, target_col: str, task: str,
    rec_df, seed: int, cv_folds: int, time_budget: int,
) -> None:
    from src.verify import run_verification, VerificationResult

    # Pick top-2 non-dummy models
    non_dummy = rec_df[rec_df["model"] != "Dummy"]
    top2 = non_dummy["model"].head(2).tolist()
    models_to_run = ["Dummy"] + top2

    st.markdown(
        f"Will verify: **{', '.join(top2)}** against the Dummy baseline. "
        f"Time budget: {time_budget}s | Folds: {cv_folds} | Seed: {seed}"
    )

    if "verification_results" not in st.session_state:
        st.session_state["verification_results"] = None
    if "verification_params" not in st.session_state:
        st.session_state["verification_params"] = None

    current_params = (
        _df_fingerprint(df), target_col, task,
        tuple(models_to_run), seed, cv_folds, time_budget
    )

    run_verify = st.button("Run verification on top 2 models", key="run_verify_btn")

    if run_verify or (
        st.session_state["verification_results"] is not None
        and st.session_state["verification_params"] == current_params
    ):
        if run_verify:
            # Run and stream progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            results: dict[str, VerificationResult] = {}

            def _progress(model_name: str, frac: float):
                if model_name == "done":
                    progress_bar.progress(1.0)
                    status_text.markdown(
                        '<span style="font-size:13px;color:#2E6B4F;">Verification complete.</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    progress_bar.progress(frac)
                    status_text.markdown(
                        f'<span style="font-size:13px;color:#5B6470;">Running: {model_name}...</span>',
                        unsafe_allow_html=True,
                    )

            with st.spinner("Running cross-validation..."):
                results = run_verification(
                    df=df,
                    target_col=target_col,
                    task=task,
                    model_names=models_to_run,
                    seed=seed,
                    folds=cv_folds,
                    time_budget_s=float(time_budget),
                    progress_cb=_progress,
                )

            st.session_state["verification_results"] = results
            st.session_state["verification_params"] = current_params

        results = st.session_state["verification_results"]
        if results is None:
            return

        # ---- Results table ----
        st.markdown("#### Verification results")
        metric_name = "Balanced Accuracy" if task == "classification" else "R²"

        dummy_score = None
        if "Dummy" in results and results["Dummy"].mean_score is not None:
            dummy_score = results["Dummy"].mean_score

        table_rows = []
        for name in models_to_run:
            res = results.get(name)
            if res is None:
                continue
            if res.status == "ok" and res.mean_score is not None:
                score_str = f"{res.mean_score:.4f} ± {res.std_score:.4f}"
                vs_dummy = (
                    f"{res.mean_score - dummy_score:+.4f}"
                    if dummy_score is not None and name != "Dummy"
                    else "—"
                )
                fit_t = f"{res.fit_time_s:.2f}s"
            elif res.status == "timed out":
                score_str = (
                    f"{res.mean_score:.4f} ± {res.std_score:.4f} (partial)"
                    if res.mean_score is not None
                    else "timed out"
                )
                vs_dummy = "—"
                fit_t = f"{res.fit_time_s:.2f}s (capped)"
            else:
                score_str = res.status
                vs_dummy = "—"
                fit_t = "—"

            table_rows.append(
                {
                    "Model": name,
                    f"CV {metric_name} (mean ± std)": score_str,
                    "Fit time (s)": fit_t,
                    "vs Dummy": vs_dummy,
                    "Status": res.status,
                }
            )

        res_df = pd.DataFrame(table_rows)
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # ---- Bar chart ----
        st.markdown("#### CV score comparison")
        chart_names, chart_scores, chart_errs = [], [], []
        for name in models_to_run:
            res = results.get(name)
            if res and res.mean_score is not None:
                chart_names.append(name)
                chart_scores.append(res.mean_score)
                chart_errs.append(res.std_score or 0.0)

        if chart_names:
            tmpl = plotly_template()
            fig = go.Figure()
            for i, (n, s, e) in enumerate(zip(chart_names, chart_scores, chart_errs)):
                bar_color = (
                    COLORS["accent"] if i > 0 else COLORS["text_secondary"]
                )
                fig.add_trace(
                    go.Bar(
                        name=n,
                        x=[n],
                        y=[s],
                        error_y=dict(
                            type="data", array=[e], thickness=1.5, width=6,
                            color=COLORS["text_secondary"]
                        ),
                        marker_color=bar_color,
                        marker_line_width=0,
                        showlegend=False,
                    )
                )
            fig.update_layout(
                **tmpl["layout"],
                title=f"Verified {metric_name} (mean ± std, {cv_folds}-fold CV)",
                yaxis_title=metric_name,
                height=320,
                bargap=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---- Predicted vs actual table ----
        st.markdown("#### Predicted vs. verified score")
        pv_rows = []
        for name in top2:
            est_row = rec_df[rec_df["model"] == name]
            if est_row.empty:
                continue
            est_score = est_row.iloc[0]["est_score"]
            res = results.get(name)
            if res and res.mean_score is not None and res.status in ("ok", "timed out"):
                verified = res.mean_score
                error = abs(verified - est_score)
                pv_rows.append(
                    {
                        "Model": name,
                        "Estimated score": f"{est_score:.4f}",
                        "Verified score": f"{verified:.4f}",
                        "Absolute error": f"{error:.4f}",
                    }
                )
        if pv_rows:
            st.dataframe(pd.DataFrame(pv_rows), use_container_width=True, hide_index=True)

        # ---- Summary sentence ----
        best_name, best_res = None, None
        for name in top2:
            res = results.get(name)
            if res and res.mean_score is not None and res.status == "ok":
                if best_res is None or res.mean_score > best_res.mean_score:
                    best_name, best_res = name, res

        if best_name and best_res:
            est_row = rec_df[rec_df["model"] == best_name]
            est_score = est_row.iloc[0]["est_score"] if not est_row.empty else None
            delta_str = ""
            if est_score is not None:
                delta = best_res.mean_score - est_score
                delta_str = f", which is {delta:+.3f} vs the estimate"
            summary = (
                f"Verified: {best_name} scored "
                f"{best_res.mean_score:.4f} ± {best_res.std_score:.4f}{delta_str}."
            )
            if dummy_score is not None:
                above = best_res.mean_score - dummy_score
                summary += f" That is {above:+.4f} above the dummy baseline."
            st.markdown(
                f'<div class="ms-box ms-box-info" style="margin-top:12px;">'
                f"{summary}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="ms-box ms-box-info">'
            "Click the button above to run real cross-validation on the top 2 models."
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 5: Explanation
# ---------------------------------------------------------------------------

def render_explanation(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    meta: dict,
    rec_df,
    seed: int,
) -> None:
    from src.explain import compute_importance, generate_bullets

    # Why this recommendation
    st.markdown("#### Why this recommendation")
    bullets = generate_bullets(meta, rec_df, task)
    for b in bullets:
        st.markdown(
            f'<p style="margin:4px 0 8px 0; padding-left:12px; '
            f'border-left:2px solid #D9DDE3; font-size:13px;">{b}</p>',
            unsafe_allow_html=True,
        )

    # Feature importance
    st.markdown("#### Feature importance")
    top_model = rec_df.iloc[0]["model"] if len(rec_df) else "Random Forest"

    with st.spinner(f"Computing permutation importance for {top_model}..."):
        try:
            imp_df, method_label = compute_importance(
                model_name=top_model,
                df=df,
                target_col=target_col,
                task=task,
                seed=seed,
            )
        except Exception as e:
            st.markdown(
                f'<div class="ms-box ms-box-error">Could not compute importance: {e}</div>',
                unsafe_allow_html=True,
            )
            imp_df = None
            method_label = "Error"

    if imp_df is not None and not imp_df.empty:
        st.caption(f"Method: {method_label} | Model: {top_model}")
        top_n = imp_df.head(20)
        tmpl = plotly_template()
        fig = go.Figure(
            go.Bar(
                x=top_n["importance"],
                y=top_n["feature"],
                orientation="h",
                marker_color=COLORS["accent"],
                marker_line_width=0,
            )
        )
        fig.update_layout(
            **tmpl["layout"],
            title="Top feature importances (permutation, held-out split)",
            xaxis_title="Mean decrease in score",
            yaxis=dict(autorange="reversed"),
            height=max(300, len(top_n) * 22),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Limitations
    st.markdown("#### Limitations")
    st.markdown(
        '<div class="ms-box ms-box-info">'
        "Recommendations are based on dataset characteristics and quick landmarkers. "
        "Default hyperparameters are used. Results may change with tuning. "
        "Verify before deploying."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Init session state
    for key in ["use_sample", "uploaded_file", "meta", "rec_df", "analysis_done"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "analysis_done" else False

    render_header()

    sidebar = render_sidebar()
    df: Optional[pd.DataFrame] = sidebar["df"]
    target_col: Optional[str] = sidebar["target_col"]
    task: str = sidebar["task"]
    time_budget: int = sidebar["time_budget"]
    cv_folds: int = sidebar["cv_folds"]
    seed: int = sidebar["seed"]
    run_clicked: bool = sidebar["run_clicked"]
    source_label: str = sidebar["source_label"]

    if df is None:
        render_empty_state()
        return

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    if target_col is None:
        box("Please select a target column in the sidebar.", "warn")
        return

    # Drop NaN target rows
    nan_target = df[target_col].isnull().sum()
    if nan_target > 0:
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        box(
            f"{nan_target} row(s) with missing target values were dropped. "
            f"Remaining rows: {len(df)}.",
            "warn",
        )

    if len(df) < MIN_ROWS_REQUIRED:
        box(
            f"Dataset has only {len(df)} rows after cleaning. "
            f"ModelScout requires at least {MIN_ROWS_REQUIRED} rows. "
            f"Please upload a larger dataset.",
            "error",
        )
        return

    # Drop ID columns and warn
    df_clean, dropped_ids = drop_id_columns(df.drop(columns=[target_col]))
    df_clean[target_col] = df[target_col].values
    if dropped_ids:
        box(
            f"Dropped likely ID columns (near-unique non-numeric): "
            f"{', '.join(dropped_ids)}.",
            "warn",
        )
        df = df_clean

    # ------------------------------------------------------------------ #
    # Run analysis
    # ------------------------------------------------------------------ #
    if run_clicked:
        st.session_state["analysis_done"] = False
        with st.spinner("Computing meta-features and landmarkers..."):
            try:
                meta = compute_meta_features(df, target_col, task)
                rec_df = recommend(meta, task)
                st.session_state["meta"] = meta
                st.session_state["rec_df"] = rec_df
                st.session_state["analysis_done"] = True
                # Clear verification cache when re-running
                st.session_state["verification_results"] = None
                st.session_state["verification_params"] = None
            except Exception as e:
                box(
                    f"Analysis failed: {e}. "
                    f"Check your dataset for unsupported column types.",
                    "error",
                )
                st.session_state["analysis_done"] = False

    # ------------------------------------------------------------------ #
    # Tabs
    # ------------------------------------------------------------------ #
    tabs = st.tabs(
        ["Overview", "Meta-features", "Recommendation", "Verification", "Explanation"]
    )

    with tabs[0]:
        render_overview(df, target_col, task)

    if not st.session_state.get("analysis_done"):
        for tab in tabs[1:]:
            with tab:
                box(
                    'Click <strong>Run analysis</strong> in the sidebar to see results here.',
                    "info",
                )
        _render_footer()
        return

    meta = st.session_state["meta"]
    rec_df = st.session_state["rec_df"]

    with tabs[1]:
        render_meta_features(meta)

    with tabs[2]:
        render_recommendation(rec_df, task, meta)

    with tabs[3]:
        render_verification(df, target_col, task, rec_df, seed, cv_folds, time_budget)

    with tabs[4]:
        render_explanation(df, target_col, task, meta, rec_df, seed)

    _render_footer()


def _render_footer() -> None:
    mode = _recommender_mode_label().replace("Recommender: ", "")
    st.markdown(
        f'<div class="ms-footer">ModelScout &nbsp;|&nbsp; DSML project prototype '
        f'&nbsp;|&nbsp; Recommender mode: {mode}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
