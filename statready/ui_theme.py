from __future__ import annotations

import streamlit as st


BRAND_NAVY = "#102A43"
BRAND_BLUE = "#1F5A94"
BRAND_TEAL = "#0E7490"
BRAND_GOLD = "#D9A441"
SURFACE = "#F5F8FC"


def apply_professional_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --sr-navy:#102A43;
          --sr-blue:#1F5A94;
          --sr-teal:#0E7490;
          --sr-gold:#D9A441;
          --sr-surface:#F5F8FC;
          --sr-border:#DCE6F0;
          --sr-text:#243B53;
          --sr-muted:#627D98;
        }
        .stApp { background: linear-gradient(180deg,#F8FAFD 0%,#F3F7FB 100%); }
        .block-container { max-width: 1440px; padding-top: 1.25rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] { background:#0F2942; border-right:1px solid #244766; }
        [data-testid="stSidebar"] * { color:#F5F9FD; }
        [data-testid="stSidebar"] .stRadio label { padding:.34rem .5rem; border-radius:8px; }
        [data-testid="stSidebar"] .stRadio label:hover { background:#183A58; }
        h1,h2,h3,h4 { color:var(--sr-navy); letter-spacing:-.015em; }
        h1 { font-size:2rem !important; margin-bottom:.2rem !important; }
        h2 { font-size:1.48rem !important; margin-top:.3rem !important; }
        h3 { font-size:1.15rem !important; }
        p,li,label { color:var(--sr-text); }
        .sr-hero {
          background:linear-gradient(118deg,#102A43 0%,#164B72 65%,#0E7490 100%);
          color:white; padding:1.25rem 1.45rem; border-radius:16px; margin-bottom:1rem;
          box-shadow:0 12px 32px rgba(16,42,67,.13);
        }
        .sr-hero h1,.sr-hero p { color:white !important; margin:0; }
        .sr-hero p { opacity:.88; margin-top:.35rem; }
        .sr-card {
          background:white; border:1px solid var(--sr-border); border-radius:14px;
          padding:1rem 1.05rem; box-shadow:0 4px 16px rgba(16,42,67,.055); margin-bottom:.85rem;
        }
        .sr-card-title { font-weight:700; color:var(--sr-navy); font-size:1rem; margin-bottom:.35rem; }
        .sr-eyebrow { color:var(--sr-teal); font-size:.75rem; text-transform:uppercase; font-weight:800; letter-spacing:.08em; }
        .sr-step {
          display:flex; gap:.65rem; align-items:flex-start; padding:.55rem .65rem;
          border-radius:10px; margin:.2rem 0; border:1px solid transparent;
        }
        .sr-step.active { background:#E6F3F7; border-color:#9ED2DE; }
        .sr-step.done { background:#EDF7F1; border-color:#B7DEC4; }
        .sr-step.pending { opacity:.72; }
        .sr-dot { width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; flex:none; }
        .active .sr-dot { background:var(--sr-teal); color:white; }
        .done .sr-dot { background:#2E7D55; color:white; }
        .pending .sr-dot { background:#D9E2EC; color:#486581; }
        .sr-badge { display:inline-flex; align-items:center; padding:.22rem .55rem; border-radius:999px; font-size:.75rem; font-weight:700; margin-right:.3rem; }
        .sr-badge.blue { background:#E8F1FB; color:#1F5A94; }
        .sr-badge.teal { background:#E6F5F7; color:#0E7490; }
        .sr-badge.gold { background:#FFF6E0; color:#8A5B00; }
        .sr-badge.green { background:#EAF7EF; color:#246B45; }
        .sr-muted { color:var(--sr-muted); font-size:.9rem; }
        div[data-testid="stMetric"] { background:white; border:1px solid var(--sr-border); border-radius:12px; padding:.65rem .8rem; box-shadow:0 3px 12px rgba(16,42,67,.045); }
        div[data-testid="stDataFrame"] { border:1px solid var(--sr-border); border-radius:10px; overflow:hidden; }
        div[data-testid="stExpander"] { background:white; border:1px solid var(--sr-border); border-radius:12px; }
        .stButton>button { border-radius:9px; font-weight:700; min-height:2.55rem; }
        .stButton>button[kind="primary"] { background:linear-gradient(90deg,#1F5A94,#0E7490); border:none; }
        .stDownloadButton>button { border-radius:9px; font-weight:700; }
        .stTabs [data-baseweb="tab-list"] { gap:.4rem; background:#EAF0F6; border-radius:11px; padding:.3rem; }
        .stTabs [data-baseweb="tab"] { border-radius:8px; padding:.45rem .75rem; }
        .stTabs [aria-selected="true"] { background:white; box-shadow:0 2px 8px rgba(16,42,67,.08); }
        .stAlert { border-radius:11px; }
        footer { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="sr-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str = "", eyebrow: str | None = None) -> None:
    eyebrow_html = f'<div class="sr-eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(
        f'<div class="sr-card">{eyebrow_html}<div class="sr-card-title">{title}</div><div class="sr-muted">{body}</div></div>',
        unsafe_allow_html=True,
    )


def status_badge(text: str, style: str = "blue") -> str:
    return f'<span class="sr-badge {style}">{text}</span>'
