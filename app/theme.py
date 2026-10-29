"""
theme.py
Injects the ModelScout design system into Streamlit via CSS and provides a
custom Plotly template.  Call apply_theme() once at the top of streamlit_app.py.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Color tokens (also exported for use in chart code)
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#FFFFFF",
    "surface": "#F6F7F9",
    "border": "#D9DDE3",
    "text_primary": "#1B1F24",
    "text_secondary": "#5B6470",
    "accent": "#1F4E79",
    "accent_hover": "#173B5C",
    "positive": "#2E6B4F",
    "caution": "#8A5A00",
    "negative": "#8C2F2F",
    # Chart series (muted)
    "series": ["#1F4E79", "#6B7C93", "#A9B4C2", "#2E6B4F", "#8A5A00"],
    # Thin gridlines
    "gridline": "#E6E9EE",
}

_CSS = f"""
<style>
/* ── Google Font ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ── Base reset ───────────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
    font-size: 14px;
    color: {COLORS['text_primary']};
    background-color: {COLORS['bg']};
}}

/* ── Hide Streamlit chrome ────────────────────────────────────── */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* ── Remove default top padding ──────────────────────────────── */
.block-container {{
    padding-top: 16px !important;
    padding-bottom: 40px !important;
    max-width: 100% !important;
}}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {COLORS['surface']};
    border-right: 1px solid {COLORS['border']};
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 16px !important;
}}

/* ── Typography ─────────────────────────────────────────────── */
h1 {{
    font-size: 24px !important;
    font-weight: 600 !important;
    color: {COLORS['text_primary']} !important;
    margin: 0 0 2px 0 !important;
    letter-spacing: -0.3px;
}}
h2 {{
    font-size: 16px !important;
    font-weight: 600 !important;
    color: {COLORS['text_primary']} !important;
    margin: 16px 0 8px 0 !important;
}}
h3 {{
    font-size: 14px !important;
    font-weight: 600 !important;
    color: {COLORS['text_primary']} !important;
    margin: 12px 0 6px 0 !important;
}}
p, li {{
    font-size: 14px !important;
    color: {COLORS['text_primary']} !important;
    line-height: 1.55;
}}
caption, .caption {{
    font-size: 12px !important;
    color: {COLORS['text_secondary']} !important;
}}

/* ── Metric cards ────────────────────────────────────────────── */
[data-testid="metric-container"] {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 12px 16px !important;
}}
[data-testid="metric-container"] label {{
    font-size: 12px !important;
    color: {COLORS['text_secondary']} !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 22px !important;
    font-weight: 600 !important;
    color: {COLORS['text_primary']} !important;
    font-variant-numeric: tabular-nums;
}}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {{
    background-color: {COLORS['accent']} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease;
}}
.stButton > button:hover {{
    background-color: {COLORS['accent_hover']} !important;
}}
.stButton > button:active {{
    background-color: {COLORS['accent_hover']} !important;
}}

/* ── Secondary / outline button variant (class="secondary") ──── */
button[kind="secondary"] {{
    background-color: {COLORS['bg']} !important;
    color: {COLORS['accent']} !important;
    border: 1px solid {COLORS['accent']} !important;
}}

/* ── Tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    border-bottom: 1px solid {COLORS['border']} !important;
    gap: 0 !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 14px !important;
    font-weight: 500 !important;
    color: {COLORS['text_secondary']} !important;
    padding: 8px 16px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
    color: {COLORS['accent']} !important;
    border-bottom: 2px solid {COLORS['accent']} !important;
    font-weight: 600 !important;
}}

/* ── DataFrames / Tables ─────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
}}
.dataframe th {{
    background-color: {COLORS['surface']} !important;
    color: {COLORS['text_secondary']} !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.dataframe td {{
    font-size: 13px !important;
    font-variant-numeric: tabular-nums;
}}

/* ── Flat info / warning / error boxes ───────────────────────── */
.ms-box {{
    padding: 12px 16px;
    border-radius: 4px;
    border: 1px solid {COLORS['border']};
    background-color: {COLORS['surface']};
    font-size: 13px;
    line-height: 1.5;
    margin: 8px 0;
}}
.ms-box-warn {{
    background-color: #FFFBEF;
    border-color: #D4A900;
    color: {COLORS['caution']};
}}
.ms-box-error {{
    background-color: #FFF5F5;
    border-color: #C0504D;
    color: {COLORS['negative']};
}}
.ms-box-info {{
    background-color: {COLORS['surface']};
    border-color: {COLORS['border']};
    color: {COLORS['text_primary']};
}}

/* ── Status badges ───────────────────────────────────────────── */
.ms-badge {{
    display: inline-block;
    font-size: 12px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 3px;
    border: 1px solid;
    letter-spacing: 0.03em;
}}
.ms-badge-positive {{
    color: {COLORS['positive']};
    border-color: {COLORS['positive']};
    background: #F0FBF5;
}}
.ms-badge-caution {{
    color: {COLORS['caution']};
    border-color: {COLORS['caution']};
    background: #FFFBEF;
}}
.ms-badge-negative {{
    color: {COLORS['negative']};
    border-color: {COLORS['negative']};
    background: #FFF5F5;
}}
.ms-badge-neutral {{
    color: {COLORS['text_secondary']};
    border-color: {COLORS['border']};
    background: {COLORS['surface']};
}}

/* ── Cards ───────────────────────────────────────────────────── */
.ms-card {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 16px;
    height: 100%;
}}
.ms-card-title {{
    font-size: 12px;
    font-weight: 600;
    color: {COLORS['text_secondary']};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}}
.ms-card-value {{
    font-size: 18px;
    font-weight: 600;
    color: {COLORS['text_primary']};
    font-variant-numeric: tabular-nums;
}}
.ms-card-sub {{
    font-size: 12px;
    color: {COLORS['text_secondary']};
    margin-top: 4px;
    line-height: 1.4;
}}

/* ── Header row ──────────────────────────────────────────────── */
.ms-header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    padding: 0 0 12px 0;
    border-bottom: 1px solid {COLORS['border']};
    margin-bottom: 16px;
}}
.ms-header-left h1 {{
    margin: 0 !important;
}}
.ms-header-desc {{
    font-size: 13px;
    color: {COLORS['text_secondary']};
    margin: 2px 0 0 0;
}}
.ms-mode-label {{
    font-size: 12px;
    color: {COLORS['text_secondary']};
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 3px;
    padding: 4px 10px;
    white-space: nowrap;
    margin-top: 4px;
}}

/* ── Top recommended row highlight ───────────────────────────── */
.ms-top-row {{
    border-left: 3px solid {COLORS['accent']};
    padding-left: 8px;
}}

/* ── Selectbox / slider labels ───────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stRadio"] label {{
    font-size: 13px !important;
    font-weight: 500 !important;
    color: {COLORS['text_primary']} !important;
}}

/* ── Progress bar ────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {{
    background-color: {COLORS['accent']} !important;
}}

/* ── Divider ─────────────────────────────────────────────────── */
hr {{
    border: none;
    border-top: 1px solid {COLORS['border']};
    margin: 16px 0;
}}

/* ── Footer ──────────────────────────────────────────────────── */
.ms-footer {{
    font-size: 12px;
    color: {COLORS['text_secondary']};
    border-top: 1px solid {COLORS['border']};
    padding-top: 12px;
    margin-top: 24px;
}}

/* ── Sidebar section labels ──────────────────────────────────── */
.ms-sidebar-section {{
    font-size: 11px;
    font-weight: 600;
    color: {COLORS['text_secondary']};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 16px 0 6px 0;
}}
</style>
"""


def apply_theme() -> None:
    """Inject ModelScout CSS into the running Streamlit app."""
    st.markdown(_CSS, unsafe_allow_html=True)


def plotly_template() -> dict:
    """Return a minimal Plotly layout template dict."""
    return {
        "layout": {
            "font": {
                "family": "Inter, 'Segoe UI', Roboto, Arial, sans-serif",
                "size": 12,
                "color": COLORS["text_primary"],
            },
            "paper_bgcolor": COLORS["bg"],
            "plot_bgcolor": COLORS["bg"],
            "colorway": COLORS["series"],
            "xaxis": {
                "gridcolor": COLORS["gridline"],
                "gridwidth": 1,
                "showgrid": True,
                "zeroline": False,
                "showline": False,
                "tickfont": {"size": 11, "color": COLORS["text_secondary"]},
                "title": {"font": {"size": 12, "color": COLORS["text_secondary"]}},
            },
            "yaxis": {
                "gridcolor": COLORS["gridline"],
                "gridwidth": 1,
                "showgrid": True,
                "zeroline": False,
                "showline": False,
                "tickfont": {"size": 11, "color": COLORS["text_secondary"]},
                "title": {"font": {"size": 12, "color": COLORS["text_secondary"]}},
            },
            "legend": {
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "font": {"size": 11},
            },
            "margin": {"l": 48, "r": 16, "t": 40, "b": 40},
            "title": {
                "x": 0.0,
                "xanchor": "left",
                "font": {
                    "size": 14,
                    "color": COLORS["text_primary"],
                    "family": "Inter, 'Segoe UI', Roboto, Arial, sans-serif",
                },
            },
            "hoverlabel": {
                "bgcolor": COLORS["bg"],
                "bordercolor": COLORS["border"],
                "font": {"size": 12, "family": "Inter, 'Segoe UI', Roboto, Arial, sans-serif"},
            },
        }
    }


def box(text: str, kind: str = "info") -> None:
    """Render a flat bordered message box. kind: 'info' | 'warn' | 'error'."""
    cls = {"info": "ms-box-info", "warn": "ms-box-warn", "error": "ms-box-error"}.get(
        kind, "ms-box-info"
    )
    st.markdown(f'<div class="ms-box {cls}">{text}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral") -> str:
    """Return HTML for a flat status badge. kind: 'positive'|'caution'|'negative'|'neutral'."""
    cls = f"ms-badge ms-badge-{kind}"
    return f'<span class="{cls}">{text}</span>'


def card(title: str, value: str, sub: str = "") -> str:
    """Return HTML for a flat stat card."""
    sub_html = f'<div class="ms-card-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="ms-card">'
        f'<div class="ms-card-title">{title}</div>'
        f'<div class="ms-card-value">{value}</div>'
        f"{sub_html}"
        f"</div>"
    )
