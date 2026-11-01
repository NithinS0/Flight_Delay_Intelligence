"""Shared Streamlit design system for the Flight Delay Intelligence dashboard.

Every page imports from this module so typography, colours, cards, tables and
navigation look identical across the application. Only presentation lives here;
no statistics or model logic is touched.
"""
from __future__ import annotations

import html
from typing import Iterable, Mapping, Optional, Sequence

import streamlit as st

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
NAVY = "#14305c"
BLUE = "#4a76b0"
TEAL = "#0f8f87"
ORANGE = "#d98a3d"
RED = "#c0392b"
GREY = "#64748b"
INK = "#1f2933"
BORDER = "#e3e8ef"
BG = "#f5f7fa"

_LOGO_SVG = (
    "data:image/svg+xml;utf8,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 36 36'%3E"
    "%3Crect width='36' height='36' rx='9' fill='%2314305c'/%3E"
    "%3Cpath fill='white' transform='translate(6 6)' "
    "d='M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z'/%3E%3C/svg%3E"
)

_CSS = f"""
<style>
/* ---------- Base ---------- */
.stApp {{
  background: {BG};
  color: {INK};
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, Roboto, 'Helvetica Neue', Arial, sans-serif;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDeployButton"], footer {{ display: none !important; }}
.main .block-container {{
  max-width: 1360px;
  padding: 2rem 2.5rem 3.5rem 2.5rem;
}}
@media (max-width: 900px) {{
  .main .block-container {{ padding: 1rem 1rem 2.5rem 1rem; }}
}}
html, body, .stApp {{ overflow-x: hidden; }}
[data-testid="column"] {{ min-width: 0; }}
.stMarkdown p, .stMarkdown li {{ line-height: 1.6; overflow-wrap: anywhere; }}

/* ---------- Static content: never show a text caret ---------- */
.stApp, .stApp * {{ caret-color: transparent; }}
.stApp input[type="text"], .stApp input[type="number"], .stApp input[type="password"],
.stApp textarea {{ caret-color: auto; }}
.stApp div[data-baseweb="select"] input {{ caret-color: transparent !important; }}
.fd-header, .fd-kpi, .fd-section, .fd-flow, .fd-result, .fd-badge,
section[data-testid="stSidebar"] nav, [data-testid="stSidebarNav"] {{
  user-select: none; -webkit-user-select: none; cursor: default;
}}
[contenteditable="true"] {{ caret-color: auto; }}

/* ---------- Typography ---------- */
h1, h2, h3, h4 {{ color: {NAVY}; letter-spacing: -0.01em; }}
h1 {{ font-size: 1.85rem !important; font-weight: 700 !important; }}
h2 {{ font-size: 1.35rem !important; font-weight: 650 !important; }}
h3 {{ font-size: 1.12rem !important; font-weight: 650 !important; }}
h4 {{ font-size: 1.0rem !important; font-weight: 650 !important; }}
[data-testid="stHeadingWithActionElements"] {{ margin-bottom: .25rem; }}

/* ---------- Page header ---------- */
.fd-header {{
  background: #ffffff; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 22px 28px 20px 28px; margin: 0 0 22px 0;
  box-shadow: 0 1px 2px rgba(20,48,92,.05);
}}
.fd-eyebrow {{ color: {TEAL}; font-size: .74rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; margin-bottom: 6px; }}
.fd-title {{ color: {NAVY}; font-size: 1.85rem; font-weight: 700; line-height: 1.2; margin: 0 0 4px 0; overflow-wrap: anywhere; }}
.fd-subtitle {{ color: {GREY}; font-size: 1.0rem; font-style: italic; margin: 0 0 10px 0; }}
.fd-desc {{ color: #3b4a5c; font-size: .93rem; line-height: 1.6; margin: 0; max-width: 980px; }}

/* ---------- Section heading ---------- */
.fd-section {{ margin: 30px 0 12px 0; padding-bottom: 8px; border-bottom: 1px solid {BORDER}; }}
.fd-section-title {{ color: {NAVY}; font-size: 1.15rem; font-weight: 650; margin: 0; }}
.fd-section-desc {{ color: {GREY}; font-size: .88rem; margin: 4px 0 0 0; line-height: 1.5; }}

/* ---------- KPI cards ---------- */
.fd-kpi-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 14px; margin: 6px 0 20px 0;
}}
.fd-kpi {{
  background: #ffffff; border: 1px solid {BORDER}; border-left: 4px solid {NAVY};
  border-radius: 12px; padding: 14px 16px; min-height: 108px; box-sizing: border-box;
  display: flex; flex-direction: column; justify-content: space-between; gap: 6px;
  box-shadow: 0 1px 2px rgba(20,48,92,.05);
}}
.fd-kpi:hover {{ box-shadow: 0 3px 8px rgba(20,48,92,.08); }}
.fd-kpi.teal {{ border-left-color: {TEAL}; }}
.fd-kpi.blue {{ border-left-color: {BLUE}; }}
.fd-kpi.orange {{ border-left-color: {ORANGE}; }}
.fd-kpi.red {{ border-left-color: {RED}; }}
.fd-kpi.grey {{ border-left-color: #94a3b8; }}
.fd-kpi-label {{ color: {GREY}; font-size: .71rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; line-height: 1.3; }}
.fd-kpi-value {{ color: {NAVY}; font-size: 1.55rem; font-weight: 700; line-height: 1.15; overflow-wrap: anywhere; }}
.fd-kpi-note {{ color: {GREY}; font-size: .76rem; line-height: 1.4; overflow-wrap: anywhere; }}

/* ---------- Callouts & notes ---------- */
.fd-callout {{
  background: #eef4fb; border: 1px solid #d5e2f1; border-left: 4px solid {BLUE};
  border-radius: 10px; padding: 12px 16px; margin: 8px 0 16px 0;
  color: #27384d; font-size: .9rem; line-height: 1.6; overflow-wrap: anywhere;
}}
.fd-callout.teal {{ background: #ecf7f6; border-color: #cbe8e5; border-left-color: {TEAL}; }}
.fd-callout.warn {{ background: #fdf4e9; border-color: #f3dcc0; border-left-color: {ORANGE}; }}
.fd-callout.risk {{ background: #fbeeec; border-color: #f0cfca; border-left-color: {RED}; }}
.fd-callout b, .fd-callout strong {{ color: {NAVY}; }}
.fd-note {{ color: {GREY}; font-size: .84rem; line-height: 1.55; margin: 6px 2px 20px 2px; overflow-wrap: anywhere; }}

/* ---------- Result panel (hypothesis / ANOVA) ---------- */
.fd-result {{
  background: #ffffff; border: 1px solid {BORDER}; border-radius: 12px;
  padding: 16px 20px; margin: 0 0 14px 0; box-shadow: 0 1px 2px rgba(20,48,92,.05);
}}
.fd-result-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; }}
.fd-result-title {{ color: {NAVY}; font-size: .98rem; font-weight: 650; line-height: 1.4; flex: 1 1 320px; overflow-wrap: anywhere; }}
.fd-result-id {{ color: {GREY}; font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }}
.fd-badge {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .74rem; font-weight: 650; border: 1px solid {BORDER}; white-space: nowrap; }}
.fd-badge.teal {{ background: #ecf7f6; color: {TEAL}; border-color: #cbe8e5; }}
.fd-badge.grey {{ background: #f1f4f8; color: {GREY}; }}
.fd-badge.orange {{ background: #fdf4e9; color: #a8661f; border-color: #f3dcc0; }}
.fd-badge.navy {{ background: #e8eef7; color: {NAVY}; border-color: #cbd8ea; }}
.fd-hyp {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-bottom: 12px; }}
.fd-hyp > div {{ background: #f8fafc; border: 1px solid {BORDER}; border-radius: 8px; padding: 9px 12px; font-size: .86rem; line-height: 1.5; color: #27384d; overflow-wrap: anywhere; }}
.fd-hyp span {{ display: block; color: {GREY}; font-size: .68rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; margin-bottom: 2px; }}
.fd-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 12px; }}
.fd-stats > div {{ border-left: 3px solid {BLUE}; padding: 2px 10px; }}
.fd-stats span {{ display: block; color: {GREY}; font-size: .68rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }}
.fd-stats b {{ color: {NAVY}; font-size: 1.0rem; overflow-wrap: anywhere; }}
.fd-interp {{ color: #3b4a5c; font-size: .88rem; line-height: 1.6; margin: 0; overflow-wrap: anywhere; }}

/* ---------- Architecture flow ---------- */
.fd-flow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 10px 6px; margin: 10px 0 20px 0; }}
.fd-flow-step {{
  flex: 1 1 130px; min-width: 130px; max-width: 190px; box-sizing: border-box;
  background: #ffffff; border: 1px solid {BORDER}; border-top: 4px solid {NAVY};
  border-radius: 10px; padding: 12px 12px 12px 12px; text-align: left;
  box-shadow: 0 1px 2px rgba(20,48,92,.05);
}}
.fd-flow-step.blue {{ border-top-color: {BLUE}; }}
.fd-flow-step.teal {{ border-top-color: {TEAL}; }}
.fd-flow-step.orange {{ border-top-color: {ORANGE}; }}
.fd-flow-step b {{ display: block; color: {NAVY}; font-size: .86rem; line-height: 1.3; margin-bottom: 4px; }}
.fd-flow-step span {{ display: block; color: {GREY}; font-size: .74rem; line-height: 1.4; overflow-wrap: anywhere; }}
.fd-flow-arrow {{ align-self: center; color: #94a3b8; font-size: 1.35rem; font-weight: 700; flex: 0 0 auto; padding: 0 2px; }}

/* ---------- Native component polish ---------- */
div[data-testid="stMetric"] {{
  background: #ffffff; border: 1px solid {BORDER}; border-left: 4px solid {NAVY};
  border-radius: 12px; padding: 12px 16px; box-shadow: 0 1px 2px rgba(20,48,92,.05);
}}
div[data-testid="stMetricLabel"] p, div[data-testid="stMetricValue"] div, div[data-testid="stMetricDelta"] div {{
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere;
}}
div[data-testid="stMetricLabel"] p {{ color: {GREY}; font-size: .72rem; font-weight: 650; text-transform: uppercase; letter-spacing: .05em; }}
div[data-testid="stMetricValue"] div {{ color: {NAVY}; font-size: 1.4rem; font-weight: 700; }}

div[data-testid="stPlotlyChart"], div[data-testid="stImage"] {{
  background: #ffffff; border: 1px solid {BORDER}; border-radius: 12px; padding: 6px;
  box-shadow: 0 1px 2px rgba(20,48,92,.05); overflow: hidden; box-sizing: border-box;
}}
div[data-testid="stDataFrame"] {{
  border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; background: #ffffff;
}}
div[data-testid="stExpander"] {{ background: #ffffff; border: 1px solid {BORDER}; border-radius: 10px; }}
div[data-testid="stExpander"] summary {{ font-weight: 600; color: {NAVY}; }}
div[data-testid="stAlert"] {{ border-radius: 10px; font-size: .9rem; }}

/* Tabs */
div[data-baseweb="tab-list"] {{ gap: 2px 4px; flex-wrap: wrap; overflow: visible !important; border-bottom: 1px solid {BORDER}; }}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{ display: none !important; }}
button[data-baseweb="tab"] {{
  font-size: .87rem !important; font-weight: 600 !important; color: {GREY} !important;
  padding: 9px 14px !important; border-bottom: 2px solid transparent; white-space: nowrap; height: auto;
}}
button[data-baseweb="tab"]:hover {{ color: {NAVY} !important; background: #eef2f7; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {NAVY} !important; border-bottom-color: {NAVY}; }}
div[data-baseweb="tab-panel"] {{ padding-top: 1.1rem; }}

/* Inputs & buttons */
div[data-testid="stSelectbox"] label p, div[data-testid="stSlider"] label p, div[data-testid="stNumberInput"] label p,
div[data-testid="stRadio"] label p, div[data-testid="stMultiSelect"] label p {{ color: {GREY}; font-size: .8rem; font-weight: 650; }}
div[data-baseweb="select"] > div {{ border-radius: 8px; background: #ffffff; border-color: #d3dbe6; min-height: 40px; }}
div[data-baseweb="select"] span {{ overflow: visible; text-overflow: clip; white-space: normal; }}
div[data-testid="stButton"] button {{ border-radius: 8px; font-weight: 600; padding: .5rem 1rem; }}
div[data-testid="stButton"] button[kind="primary"] {{ background: {NAVY}; border-color: {NAVY}; color: #ffffff; }}
div[data-testid="stButton"] button[kind="primary"]:hover {{ background: #1c3f75; border-color: #1c3f75; }}
:focus-visible {{ outline: 2px solid {BLUE}; outline-offset: 2px; }}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] > div:first-child {{ padding-top: 0; }}
[data-testid="stSidebarNav"]::before {{
  content: "Flight Delay Intelligence"; display: block; box-sizing: border-box;
  margin: 22px 16px 12px 16px; padding: 7px 0 7px 48px; min-height: 38px;
  background: url("{_LOGO_SVG}") no-repeat left center / 38px 38px;
  color: {NAVY}; font-size: .98rem; font-weight: 700; line-height: 1.25;
}}
[data-testid="stSidebarNavItems"] {{ padding: 0 8px 8px 8px; margin: 0; max-height: none; overflow: visible; }}
[data-testid="stSidebarNavItems"] > li:first-child {{ display: none; }}
[data-testid="stSidebarNavLink"] {{
  border-radius: 8px; padding: .52rem .8rem; margin: 2px 0; color: #334155;
  font-size: .9rem; font-weight: 500; text-decoration: none;
}}
[data-testid="stSidebarNavLink"] span {{ white-space: normal; overflow: visible; text-overflow: clip; }}
[data-testid="stSidebarNavLink"]:hover {{ background: #eef2f7; color: {NAVY}; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
  background: #e8eef7; color: {NAVY}; font-weight: 650; box-shadow: inset 3px 0 0 {NAVY};
}}
.fd-side-card {{ background: #f8fafc; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 14px; margin: 6px 0 4px 0; }}
.fd-side-title {{ color: {GREY}; font-size: .68rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; margin-bottom: 8px; }}
.fd-side-row {{ display: flex; justify-content: space-between; gap: 8px; font-size: .8rem; margin-bottom: 4px; color: #475569; }}
.fd-side-row b {{ color: {NAVY}; text-align: right; overflow-wrap: anywhere; }}

/* Legacy class names used by older code paths */
.aviation-badge {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .72rem; font-weight: 650; background: #e8eef7; color: {NAVY}; border: 1px solid #cbd8ea; margin-bottom: 8px; text-transform: uppercase; letter-spacing: .04em; }}
.light-card {{ background: #ffffff; border: 1px solid {BORDER}; border-radius: 12px; padding: 18px 22px; margin-bottom: 16px; }}
</style>
"""


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def _e(value) -> str:
    return html.escape(str(value), quote=False)


def _html(markup: str) -> None:
    """Render raw HTML. Newlines/indentation are stripped so Markdown never treats it as code."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def apply_theme() -> None:
    """Injects the shared stylesheet. Safe to call on every page run."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", description: str = "", eyebrow: str = "") -> None:
    parts = ['<div class="fd-header">']
    if eyebrow:
        parts.append(f'<div class="fd-eyebrow">{_e(eyebrow)}</div>')
    parts.append(f'<h1 class="fd-title">{_e(title)}</h1>')
    if subtitle:
        parts.append(f'<p class="fd-subtitle">{_e(subtitle)}</p>')
    if description:
        parts.append(f'<p class="fd-desc">{_e(description)}</p>')
    parts.append("</div>")
    _html("".join(parts))


def section(title: str, description: str = "") -> None:
    desc = f'<p class="fd-section-desc">{_e(description)}</p>' if description else ""
    _html(f'<div class="fd-section"><p class="fd-section-title">{_e(title)}</p>{desc}</div>')


def kpi_grid(items: Iterable[Mapping[str, str]], min_width: int = 190) -> None:
    """Renders equal-height KPI cards. Each item: label, value, note (optional), tone (optional)."""
    cards = []
    for it in items:
        tone = it.get("tone", "")
        note = f'<div class="fd-kpi-note">{_e(it["note"])}</div>' if it.get("note") else ""
        cards.append(
            f'<div class="fd-kpi {tone}"><div class="fd-kpi-label">{_e(it["label"])}</div>'
            f'<div class="fd-kpi-value">{_e(it["value"])}</div>{note}</div>'
        )
    _html(f'<div class="fd-kpi-grid" style="grid-template-columns:repeat(auto-fit,minmax({min_width}px,1fr))">{"".join(cards)}</div>')


def callout(text: str, tone: str = "info", allow_html: bool = False) -> None:
    cls = {"info": "", "teal": "teal", "warn": "warn", "risk": "risk"}.get(tone, "")
    body = text if allow_html else _e(text)
    _html(f'<div class="fd-callout {cls}">{body}</div>')


def note(text: str) -> None:
    """Short explanatory caption placed under a chart."""
    _html(f'<div class="fd-note">{_e(text)}</div>')


def badge(text: str, tone: str = "navy") -> str:
    return f'<span class="fd-badge {tone}">{_e(text)}</span>'


def result_panel(
    test_id: str,
    title: str,
    h0: str,
    h1: str,
    stats: Sequence[tuple[str, str]],
    decision: str,
    interpretation: str,
    significant: bool = True,
    test_selected: str = "",
) -> None:
    """Hypothesis-test / ANOVA result panel containing all required elements."""
    tone = "teal" if significant else "grey"
    stat_html = "".join(f"<div><span>{_e(k)}</span><b>{_e(v)}</b></div>" for k, v in stats)
    sel = f'<div><span>Selected test</span>{_e(test_selected)}</div>' if test_selected else ""
    _html(
        f'<div class="fd-result"><div class="fd-result-head">'
        f'<div class="fd-result-title"><div class="fd-result-id">{_e(test_id)}</div>{_e(title)}</div>'
        f'{badge(decision, tone)}</div>'
        f'<div class="fd-hyp"><div><span>Null hypothesis (H0)</span>{_e(h0)}</div>'
        f'<div><span>Alternative hypothesis (H1)</span>{_e(h1)}</div>{sel}</div>'
        f'<div class="fd-stats">{stat_html}</div>'
        f'<p class="fd-interp">{_e(interpretation)}</p></div>'
    )


def flow_diagram(steps: Sequence[tuple[str, str, str]]) -> None:
    """Horizontal, wrapping process diagram. steps: (title, caption, tone)."""
    out = []
    for i, (title, caption, tone) in enumerate(steps):
        if i:
            out.append('<div class="fd-flow-arrow">&#10140;</div>')
        out.append(f'<div class="fd-flow-step {tone}"><b>{_e(title)}</b><span>{_e(caption)}</span></div>')
    _html(f'<div class="fd-flow">{"".join(out)}</div>')


def sidebar_status(total: int, delayed: int, dataset_name: str = "flight_data.csv", horizon: str = "") -> None:
    pct = (delayed / total * 100.0) if total else 0.0
    rows = [
        ("Source", dataset_name),
        ("Valid flights", f"{total:,}"),
        ("Delayed (15+ min)", f"{delayed:,} ({pct:.2f}%)"),
    ]
    if horizon:
        rows.append(("Horizon", horizon))
    body = "".join(f'<div class="fd-side-row"><span>{_e(k)}</span><b>{_e(v)}</b></div>' for k, v in rows)
    st.sidebar.markdown(
        "".join(
            line.strip()
            for line in f'<div class="fd-side-card"><div class="fd-side-title">Active dataset</div>{body}</div>'.splitlines()
        ),
        unsafe_allow_html=True,
    )
