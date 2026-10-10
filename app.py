"""
AI Overreliance Research Framework
----------------------------------
AI-Based Framework for Detecting Generative AI Overreliance and
Providing Adaptive Interventions

Research prototype (M.Tech Artificial Intelligence).
Run with:  py -m streamlit run app.py

The trained pipeline (final_rf_pipeline.pkl), thresholds
(risk_thresholds.pkl) and metadata (model_metadata.pkl) are only LOADED here.
Nothing is retrained inside the application.
"""

import base64
import html as htmllib
import json
import logging
import math
import os
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
try:
    import shap
    SHAP_BACKEND = f"shap {shap.__version__}"
except Exception:  # for example numba blocked by Windows Application Control
    shap = None
    SHAP_BACKEND = "the built-in TreeSHAP implementation (the shap package could not be imported)"
import streamlit as st


# ============================================================
# CONSTANTS
# ============================================================

APP_TITLE = (
    "AI-Based Framework for Detecting Generative AI Overreliance "
    "and Providing Adaptive Interventions"
)
BRAND = "AI Overreliance Research Framework"
BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("overreliance")

C_BG = "#DCEBFA"
C_BG2 = "#E9E7FF"
C_CARD = "#FFFFFF"
C_BORDER = "#C8D9F0"
C_BLUE = "#3478F6"
C_PURPLE = "#8757E8"
C_AQUA = "#DDF7F3"
C_PEACH = "#FFE7D5"
C_HERO = "#E5F1FF"
C_LAV = "#F0E7FF"
C_MINT = "#DDF7EA"
C_NAVY = "#142B55"
C_BODY = "#425B82"
C_CYAN = "#109A8A"
C_TEXT = "#142B55"
C_MUTED = "#425B82"
C_GREEN = "#16A34A"
C_AMBER = "#D97706"
C_RED = "#EF4444"

DEFAULT_NUMERIC = [
    "Own_Ideas", "Cross_Check", "Accept_Without_Verification",
    "Fear_Independent_Thinking", "Recall_Reduction_Awareness",
    "Overreliance_Problem_Solving", "Remorse", "Regret", "Task_Regret",
    "Time_Saving", "Independent_Confidence", "Learning_Efficiency",
    "Learning_Satisfaction", "Collaborative_Learning", "Positive_Emotion",
    "Low_Mental_Effort", "Cognitive_Burden_Reduction", "CGPA",
]
DEFAULT_CATEGORICAL = ["Gender", "Field_of_Study", "AI_Experience", "AI_Usage_Frequency"]

LIKERT = {
    1: "Very Low", 2: "Low", 3: "Slightly Low", 4: "Neutral",
    5: "Slightly High", 6: "High", 7: "Very High",
}

# feature -> (display label, plain-English tooltip)
FEATURE_META = {
    "Gender": ("Gender", "Select the option that applies to you."),
    "Field_of_Study": ("Field of Study", "Your main academic discipline."),
    "AI_Experience": ("AI Experience", "How experienced you consider yourself with AI tools."),
    "AI_Usage_Frequency": ("AI Usage Frequency", "How often you use generative AI tools for study."),
    "CGPA": ("CGPA", "Your current cumulative grade point average. The accepted range (2.31 to 4.00) is the range observed in the cleaned dataset, not a universal CGPA range."),
    "Own_Ideas": ("Own Ideas", "How often do you come up with your own ideas before asking an AI tool?"),
    "Cross_Check": ("Cross Check", "How often do you check AI answers against other reliable sources?"),
    "Accept_Without_Verification": ("Accept Without Verification", "How often do you accept an AI answer without checking whether it is correct?"),
    "Overreliance_Problem_Solving": ("Overreliance in Problem Solving", "How much do you lean on AI for problems you could first try to solve on your own?"),
    "Fear_Independent_Thinking": ("Fear of Independent Thinking", "How uneasy do you feel about working things out without AI help?"),
    "Independent_Confidence": ("Independent Confidence", "How confident are you in your own reasoning when AI is not available?"),
    "Recall_Reduction_Awareness": ("Recall Reduction Awareness", "How aware are you that using AI might reduce how much you remember or recall?"),
    "Time_Saving": ("Time Saving", "How much time does AI save you on academic tasks?"),
    "Learning_Efficiency": ("Learning Efficiency", "How much does AI make your learning more efficient?"),
    "Learning_Satisfaction": ("Learning Satisfaction", "How satisfied are you with your learning when you use AI?"),
    "Collaborative_Learning": ("Collaborative Learning", "How much does AI support learning together with others?"),
    "Positive_Emotion": ("Positive Emotion", "How positive do you feel (interested, pleased) when learning with AI?"),
    "Low_Mental_Effort": ("Low Mental Effort", "How much does AI let you finish tasks with less mental effort?"),
    "Cognitive_Burden_Reduction": ("Cognitive Burden Reduction", "How much does AI reduce the mental load of your academic work?"),
    "Remorse": ("Remorse", "How much remorse do you feel after relying on AI for academic work?"),
    "Regret": ("Regret", "How much do you regret relying on AI instead of your own effort?"),
    "Task_Regret": ("Task Regret", "How much do you regret using AI for specific tasks you could have done yourself?"),
}

ASSESS_STEPS = [
    ("Profile", "Student Profile",
     "Tell us a little about your study context.",
     ["Gender", "Field_of_Study", "AI_Experience", "AI_Usage_Frequency", "CGPA"]),
    ("AI Behaviour", "AI Interaction Behaviour",
     "How do you usually work with AI tools?",
     ["Own_Ideas", "Cross_Check", "Accept_Without_Verification", "Overreliance_Problem_Solving"]),
    ("Independent Thinking", "Independent Thinking",
     "How do you feel about reasoning on your own?",
     ["Fear_Independent_Thinking", "Independent_Confidence", "Recall_Reduction_Awareness"]),
    ("Learning", "Learning Experience",
     "What is your learning experience like with AI?",
     ["Time_Saving", "Learning_Efficiency", "Learning_Satisfaction",
      "Collaborative_Learning", "Positive_Emotion"]),
    ("Cognitive Factors", "Cognitive Factors",
     "How does AI use feel mentally and emotionally?",
     ["Low_Mental_Effort", "Cognitive_Burden_Reduction", "Remorse", "Regret", "Task_Regret"]),
]

# ---- Reported research results (from the project's own experiments) ----
MODEL_RESULTS = [
    ("Ridge Regression", 0.8364, 1.1002, 0.1828),
    ("Random Forest", 0.7694, 1.0509, 0.2544),
    ("Gradient Boosting", 0.7972, 1.0546, 0.2491),
    ("XGBoost", 0.7965, 1.0674, 0.2308),
    ("MLP Regressor", 1.0050, 1.2393, -0.0368),
]
CV_RESULTS = {"MAE": 0.842, "RMSE": 1.142, "R²": 0.266}
CLASS_METRICS = [("Low", 0.55, 0.55), ("Medium", 0.61, 0.86), ("High", 0.57, 0.16)]
GAN_RESULTS = {
    "Real only (Random Forest)": (0.7694, 1.0509, 0.2544),
    "Real + CTGAN synthetic": (0.7703, 1.0853, 0.2048),
}
NOISE_RESULTS = [  # noise %, MAE, RMSE, R²
    (0, 0.7694, 1.0509, 0.2544),
    (5, 0.7734, 1.0532, 0.2512),
    (10, 0.7802, 1.0713, 0.2251),
    (15, 0.7692, 1.0726, 0.2233),
]

CV_FOLDS = None  # optional per-fold results [[MAE, RMSE, R2], ...] from research_results.json


def _apply_results_file():
    """Optionally override the hard-coded research numbers from research_results.json
    (so the values can be checked against the notebook without editing code)."""
    global MODEL_RESULTS, CV_RESULTS, CLASS_METRICS, GAN_RESULTS, NOISE_RESULTS, CV_FOLDS
    path = BASE_DIR / "research_results.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        model_results = [(str(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in data["model_results"]] if data.get("model_results") else MODEL_RESULTS
        cv_results = {k: float(v) for k, v in data["cv_results"].items()} if data.get("cv_results") else CV_RESULTS
        class_metrics = [(str(r[0]), float(r[1]), float(r[2])) for r in data["class_metrics"]] if data.get("class_metrics") else CLASS_METRICS
        gan_results = {str(k): tuple(float(x) for x in v) for k, v in data["gan_results"].items()} if data.get("gan_results") else GAN_RESULTS
        noise_results = [(int(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in data["noise_results"]] if data.get("noise_results") else NOISE_RESULTS
        folds = [[float(x) for x in r] for r in data["cv_folds"]] if data.get("cv_folds") else None
        MODEL_RESULTS, CV_RESULTS, CLASS_METRICS, GAN_RESULTS, NOISE_RESULTS, CV_FOLDS = (
            model_results, cv_results, class_metrics, gan_results, noise_results, folds)
    except Exception:
        logger.exception("research_results.json was ignored because it could not be read")


_apply_results_file()

RESEARCH_NAV = [
    ("Model Insights", "bar_chart", "Models, validation and explainability"),
    ("Research", "science", "Datasets, methods and experiments"),
    ("Testing", "fact_check", "Test cases and requirement coverage"),
    ("About", "info", "Team, contributions and disclaimer"),
]
JOURNEY = [
    ("Assess", "assignment", "Answer five short sections"),
    ("Prediction", "show_chart", "See your AI Reliance Score"),
    ("Why This Result?", "lightbulb", "Understand the influential factors"),
    ("Intervention", "track_changes", "Complete a personalised activity"),
    ("Student Profile", "person", "Review your behavioural summary"),
]

STUDENT_SET = {item[0] for item in JOURNEY}

RISK_META = {
    "Low": {
        "label": "Low Reliance", "level": "Low Intervention Level", "color": C_GREEN,
        "message": (
            "Your predicted AI reliance level is low. The model suggests AI is mostly "
            "working as a supporting tool alongside your own reasoning."
        ),
    },
    "Medium": {
        "label": "Moderate Reliance", "level": "Moderate Intervention Level", "color": C_AMBER,
        "message": (
            "Your predicted AI reliance level is moderate. The model suggests that some "
            "AI-assisted behaviours may benefit from stronger verification and "
            "independent reasoning."
        ),
    },
    "High": {
        "label": "High Reliance", "level": "High Intervention Level", "color": "#DC2626",
        "message": (
            "Your predicted AI reliance level is high relative to the training sample. "
            "The model suggests that stronger verification and independent-reasoning "
            "habits could be especially useful. This is an estimate to guide learning "
            "support, not a judgement."
        ),
    },
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=BRAND,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# STYLING
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea, select {
  font-family: 'Inter','Segoe UI',system-ui,-apple-system,Roboto,sans-serif;
}
.stApp { background: radial-gradient(900px 460px at 8% -10%, rgba(52,120,246,.10), transparent 60%), #DCEBFA; color: #142B55; }
header[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display: none; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 100% !important; padding: 1rem 1.6rem 2rem 1.6rem !important; }
h1, h2, h3, h4, h5 { font-weight: 700; letter-spacing: -0.01em; color: #142B55; }
p, li, label, span { color: inherit; }

/* ---------- brand bar ---------- */
.brandbar { display:flex; align-items:center; justify-content:space-between; gap:12px; margin: 4px 0 14px 0; flex-wrap:wrap; }
.brand { display:flex; align-items:center; gap:12px; }
.logo { width:38px; height:38px; border-radius:11px; background:#3478F6; display:flex; align-items:center; justify-content:center; font-size:20px;  }
.brand-name { font-weight:800; font-size:1.05rem; letter-spacing:-.01em; }
.brand-sub { color:#425B82; font-size:.74rem; margin-top:-2px; }
.badge { display:inline-block; padding:4px 11px; border-radius:999px; font-size:.72rem; font-weight:600; border:1px solid #C8D9F0; background:#EEF4FD; color:#425B82; }
.badge.blue { background:rgba(52,120,246,.14); border-color:rgba(52,120,246,.45); color:#2B63D6; }
.badge.purple { background:rgba(52,120,246,.14); border-color:rgba(52,120,246,.45); color:#2B63D6; }
.badge.cyan { background:rgba(52,120,246,.12); border-color:rgba(52,120,246,.4); color:#109A8A; }

/* ---------- top navbar ---------- */
.st-key-topnav { background:#EEF4FD; border:1px solid #C8D9F0; border-radius:16px; padding:10px 18px; margin-bottom:22px; }
.st-key-topnav [data-testid="stHorizontalBlock"] { align-items:center; gap:.4rem; }
.nav-brand { display:flex; align-items:center; gap:12px; }
.nav-logo { width:38px; height:38px; border-radius:10px; background:#3478F6; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:.9rem; color:#fff; }
.nav-name { font-weight:700; font-size:1.02rem; color:#142B55; letter-spacing:-.01em; }
.st-key-topnav [data-testid="stButton"] { width:100%; }
.st-key-topnav button { width:100%; min-height:42px; border-radius:10px; font-weight:600; }
.st-key-topnav button p { color:inherit; font-weight:600; font-size:.9rem; }
.st-key-topnav button[kind="secondary"], .st-key-topnav button[data-testid="stBaseButton-secondary"] { background:transparent; border:none; color:#425B82; }
.st-key-topnav button[kind="secondary"]:hover, .st-key-topnav button[data-testid="stBaseButton-secondary"]:hover { background:#E3EDFB; color:#142B55; border:none; }
.st-key-topnav button[kind="primary"], .st-key-topnav button[data-testid="stBaseButton-primary"] { background:rgba(52,120,246,.16); color:#2B63D6; border:1px solid rgba(52,120,246,.35); }
.st-key-topnav button[kind="primary"]:hover, .st-key-topnav button[data-testid="stBaseButton-primary"]:hover { background:rgba(52,120,246,.16); color:#2B63D6; filter:none; }
.st-key-nav_cta button { background:#3478F6 !important; color:#fff !important; border:none !important; border-radius:10px !important; min-height:42px; }
.st-key-nav_cta button:hover { background:#5B93F8 !important; }

/* ---------- left panel ---------- */
.panel-label { color:#5A7096; font-size:.7rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin:2px 4px 10px 4px; }
.st-key-leftpanel [data-testid="stButton"] { width:100%; }
.st-key-leftpanel button { width:100%; min-height:48px; padding:.4rem .9rem; border-radius:12px; background:#EEF4FD; border:1px solid #D9E6F7; color:#425B82; }
.st-key-leftpanel button > div { justify-content:flex-start; width:100%; text-align:left; }
.st-key-leftpanel button p { font-weight:600; font-size:.9rem; color:inherit; }
.st-key-leftpanel button:hover { border-color:#B9CDEA; background:#FFFFFF; color:#142B55; }
.st-key-leftpanel button[kind="primary"], .st-key-leftpanel button[data-testid="stBaseButton-primary"] { background:rgba(52,120,246,.14); border:1px solid rgba(52,120,246,.45); color:#142B55; }

/* ---------- right panel ---------- */
.art-card { background:#EEF4FD; border:1px solid #D9E6F7; border-radius:16px; overflow:hidden; }
.art-card svg { width:100%; height:auto; display:block; }
.art-card .cap { padding:14px 16px 16px 16px; }
.art-card .cap b { font-size:.92rem; }
.art-card .cap p { color:#425B82; font-size:.8rem; line-height:1.5; margin:4px 0 0 0; }

/* ---------- cards ---------- */
.card { background:#FFFFFF; border:1px solid #C8D9F0; border-radius:16px; padding:20px 22px;  }
.card.tight { padding:14px 16px; }
.card h4 { margin:0 0 6px 0; font-size:1.02rem; }
.card p { color:#425B82; font-size:.9rem; line-height:1.55; margin:0; }
.card .icon { font-size:1.5rem; margin-bottom:8px; }
.eyebrow { text-transform:uppercase; letter-spacing:.12em; font-size:.7rem; font-weight:700; color:#109A8A; margin-bottom:6px; }
.section-title { font-size:1.35rem; font-weight:800; margin: 26px 0 4px 0; }
.section-sub { color:#425B82; font-size:.92rem; margin-bottom:14px; }
.grid { display:grid; gap:16px; }
.g2 { grid-template-columns:repeat(2,1fr); }
.g3 { grid-template-columns:repeat(3,1fr); }
.g4 { grid-template-columns:repeat(4,1fr); }
.g5 { grid-template-columns:repeat(5,1fr); }
@media (max-width: 980px) { .g4, .g5 { grid-template-columns:repeat(2,1fr); } .g3 { grid-template-columns:repeat(2,1fr); } }
@media (max-width: 640px) { .g2, .g3, .g4, .g5 { grid-template-columns:1fr; } }

/* ---------- hero ---------- */
.hero { display:grid; grid-template-columns:1.1fr .9fr; gap:28px; align-items:center; padding:36px; border-radius:20px; background:#EEF4FD; border:1px solid #C8D9F0; }
@media (max-width: 900px) { .hero { grid-template-columns:1fr; padding:24px; } }
.hero h1 { font-size:2rem; line-height:1.2; margin:12px 0 12px 0; font-weight:800; color:#142B55; }
.hero .sub { color:#425B82; font-size:1.02rem; line-height:1.6; }
.hero .tagline { margin-top:16px; color:#425B82; font-size:.95rem; line-height:1.7; }
.hero-art { width:100%; }
.hero-art svg, .hero-art img { width:100%; height:auto; border-radius:18px; display:block; }

/* ---------- steps / flow ---------- */
.steps { display:flex; align-items:stretch; gap:8px; flex-wrap:wrap; }
.step { flex:1 1 150px; background:#FFFFFF; border:1px solid #C8D9F0; border-radius:14px; padding:16px; text-align:left; min-width:140px; }
.step .num { width:28px; height:28px; border-radius:50%; background:#3478F6; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.82rem; margin-bottom:10px; color:#fff; }
.step h5 { margin:0 0 4px 0; font-size:.95rem; }
.step p { margin:0; color:#425B82; font-size:.8rem; line-height:1.45; }
.arrow { align-self:center; color:#3478F6; font-size:1.3rem; font-weight:700; }
.stat { background:#FFFFFF; border:1px solid #C8D9F0; border-radius:16px; padding:18px 20px; }
.stat .v { font-size:1.75rem; font-weight:800; letter-spacing:-.02em; }
.stat .l { color:#425B82; font-size:.8rem; margin-top:2px; }
.stat .s { color:#5A7096; font-size:.74rem; margin-top:6px; }

/* ---------- assess ---------- */
.stepper { display:flex; gap:8px; flex-wrap:wrap; margin: 6px 0 10px 0; }
.chip { padding:6px 13px; border-radius:999px; font-size:.78rem; font-weight:600; border:1px solid #C8D9F0; background:#EEF4FD; color:#425B82; }
.chip.active { background:#3478F6; color:#fff; border-color:transparent; }
.chip.done { background:rgba(34,197,94,.12); color:#15803D; border-color:rgba(34,197,94,.35); }
div[data-testid="stProgress"] > div > div > div > div { background: #3478F6; }

/* ---------- prediction ---------- */
.gauge-wrap { display:flex; flex-direction:column; align-items:center; justify-content:center; }
.risk-pill { display:inline-block; padding:6px 16px; border-radius:999px; font-weight:700; font-size:.82rem; letter-spacing:.06em; text-transform:uppercase; margin-top:6px; }
.scale { padding: 34px 14px 38px 14px; }
.scale-track { position:relative; display:flex; height:14px; border-radius:999px; }
.scale-track .seg:first-child { border-radius:999px 0 0 999px; }
.scale-track .seg:nth-child(3) { border-radius:0 999px 999px 0; }
.marker { position:absolute; top:-9px; transform:translateX(-50%); width:32px; height:32px; border-radius:50%; background:#142B55; border:4px solid #3478F6;  }
.marker span { position:absolute; top:-26px; left:50%; transform:translateX(-50%); font-size:.78rem; font-weight:700; color:#142B55; white-space:nowrap; }
.ticks { position:relative; height:18px; margin-top:16px; }
.ticks span { position:absolute; transform:translateX(-50%); color:#425B82; font-size:.78rem; }
.zones { display:flex; margin-top:2px; }
.zones span { text-align:center; color:#5A7096; font-size:.74rem; text-transform:uppercase; letter-spacing:.1em; }

/* ---------- XAI ---------- */
.factor { display:flex; gap:10px; align-items:flex-start; padding:10px 0; border-bottom:1px solid #D9E6F7; }
.factor:last-child { border-bottom:none; }
.factor .dot { width:10px; height:10px; border-radius:50%; margin-top:6px; flex:0 0 10px; }
.factor .ft { font-weight:600; font-size:.9rem; }
.factor .fd { color:#425B82; font-size:.82rem; line-height:1.45; }

/* ---------- intervention ---------- */
.flowrow { display:flex; align-items:stretch; gap:10px; flex-wrap:wrap; }
.flowrow .node { flex:1 1 180px; background:#FFFFFF; border:1px solid #C8D9F0; border-radius:14px; padding:14px 16px; }
.flowrow .node .k { color:#109A8A; font-size:.68rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; }
.flowrow .node .t { font-weight:700; margin-top:4px; font-size:.95rem; }
.flowrow .node .d { color:#425B82; font-size:.8rem; margin-top:3px; }
.iv-card { background: linear-gradient(135deg, rgba(52,120,246,.16), rgba(52,120,246,.14)), #FFFFFF; border:1px solid rgba(52,120,246,.45); border-radius:20px; padding:26px 28px;  }
.iv-card h2 { margin:6px 0 8px 0; font-size:1.55rem; text-transform:uppercase; letter-spacing:.02em; }
.iv-card .why { color:#425B82; font-size:.95rem; line-height:1.6; }
.stages { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.stage { padding:10px 16px; border-radius:12px; background:#EEF4FD; border:1px solid #C8D9F0; color:#5A7096; font-weight:600; font-size:.86rem; }
.stage.active { background:#3478F6; color:#fff; border-color:transparent;  }
.stage-arrow { color:#3478F6; font-weight:700; }
.quote { border-left:3px solid #109A8A; padding:12px 18px; background:rgba(52,120,246,.06); border-radius:0 12px 12px 0; color:#1F3A68; font-size:1rem; line-height:1.6; }

/* ---------- tables / bars ---------- */
.rtable { width:100%; border-collapse:separate; border-spacing:0; font-size:.9rem; }
.rtable th { text-align:left; color:#425B82; font-weight:600; font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; padding:10px 14px; border-bottom:1px solid #C8D9F0; }
.rtable td { padding:12px 14px; border-bottom:1px solid #D9E6F7; color:#1F3A68; }
.rtable tr.sel td { background:rgba(52,120,246,.12); font-weight:700; }
.rtable tr.sel td:first-child { border-left:3px solid #3478F6; }
.minibar { height:8px; background:#D9E6F7; border-radius:99px; overflow:hidden; margin-top:6px; }
.minibar > div { height:100%; border-radius:99px; background:linear-gradient(90deg,#3478F6,#109A8A); }
.mb-row { margin-bottom:14px; }
.mb-row .top { display:flex; justify-content:space-between; font-size:.85rem; }
.mb-row .top span:last-child { color:#425B82; font-weight:600; }

/* ---------- architecture ---------- */
.arch-branch { background:#EEF4FD; border:1px solid #C8D9F0; border-radius:16px; padding:16px 18px; margin-bottom:12px; }
.arch-label { color:#109A8A; font-size:.7rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; margin-bottom:10px; }
.flow { display:flex; align-items:center; flex-wrap:wrap; gap:8px; }
.flow .n { background:#FFFFFF; border:1px solid #C8D9F0; border-radius:12px; padding:10px 14px; font-size:.84rem; font-weight:600; }
.flow .n small { display:block; color:#425B82; font-weight:400; font-size:.72rem; margin-top:2px; }
.flow .n.hl { border-color:rgba(52,120,246,.6); background:rgba(52,120,246,.1); }
.flow i { color:#3478F6; font-style:normal; font-weight:700; }
.arch-merge { text-align:center; background:linear-gradient(135deg, rgba(52,120,246,.14), rgba(52,120,246,.14)); border:1px dashed rgba(52,120,246,.6); border-radius:16px; padding:16px; }
.arch-merge b { font-size:1rem; }
.arch-merge div { color:#425B82; font-size:.8rem; margin-top:4px; }

/* ---------- misc ---------- */
.notice { border-radius:14px; padding:14px 18px; font-size:.88rem; line-height:1.55; border:1px solid #C8D9F0; background:#EEF4FD; color:#425B82; }
.notice.info { border-color:rgba(52,120,246,.4); background:rgba(52,120,246,.08); }
.notice.warn { border-color:rgba(245,158,11,.4); background:rgba(245,158,11,.07); }
.notice.ok { border-color:rgba(34,197,94,.4); background:rgba(34,197,94,.07); }
.footer { margin-top:34px; padding:22px 20px; text-align:center; border-top:1px solid #D9E6F7; }
.footer .fb { font-weight:700; font-size:.95rem; color:#1F3A68; }
.footer .ft { color:#425B82; font-size:.84rem; margin:4px 0 8px 0; }
.footer .fs { color:#5A7096; font-size:.76rem; }
.big-quote { font-size:1.15rem; font-weight:600; line-height:1.55; text-align:center; padding:22px; border-radius:18px;
  background:linear-gradient(135deg, rgba(52,120,246,.08), rgba(52,120,246,.10)); border:1px solid #C8D9F0; color:#1F3A68; }

/* ---------- streamlit widgets ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; }
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: #3478F6; border:none; color:#fff; font-weight:700; border-radius:12px; padding:.55rem 1.2rem;
  
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { filter:brightness(1.1); border:none; color:#fff; }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] { border-radius:12px; border:1px solid #C8D9F0; background:#EEF4FD; color:#1F3A68; font-weight:600; }
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover { border-color:#3478F6; color:#2B63D6; }
button[data-baseweb="tab"] { font-weight:600; }
div[data-testid="stExpander"] { border-radius:14px; border:1px solid #C8D9F0; background:#EEF4FD; }
.card, .stat, .step, .art-card, .flowrow .node, .arch-branch, .st-key-topnav, .notice { box-shadow:0 1px 2px rgba(36,52,75,.06), 0 8px 22px rgba(36,52,75,.06); }
.stApp { background:radial-gradient(900px 460px at 8% -10%, rgba(52,120,246,.10), transparent 60%), #DCEBFA; color:#142B55; }
.hero { background:linear-gradient(135deg,#FFFFFF 0%,#E5F1FF 100%); }
.iv-card { background:linear-gradient(135deg,#FFFFFF,#E5F1FF); }
.marker { background:#FFFFFF; }
/* ===== pastel dashboard theme (tokens are replaced from the central palette constants) ===== */
.stApp { background: radial-gradient(900px 520px at 6% -8%, rgba(255,231,213,.75), transparent 60%), radial-gradient(900px 560px at 100% 4%, rgba(233,231,255,.95), transparent 58%), radial-gradient(900px 520px at 50% 112%, rgba(221,247,243,.85), transparent 60%), #DCEBFA; color:#142B55; }
.block-container { padding: 1rem 1.4rem 2rem 1.4rem !important; }
h1, h2, h3, h4, h5 { color:#142B55; }

/* top navigation */
.st-key-topnav { background:linear-gradient(90deg,#E5F1FF 0%,#EDE9FF 52%,#FFF1E3 100%); border:1px solid rgba(255,255,255,.85); border-radius:24px; padding:12px 24px; margin-bottom:18px; box-shadow:0 12px 32px rgba(20,43,85,.10); }
.st-key-topnav [data-testid="stHorizontalBlock"] { align-items:center; gap:.5rem; }
.nav-brand { display:flex; align-items:center; gap:14px; }
.nav-logo { width:46px; height:46px; flex:0 0 46px; filter:drop-shadow(0 6px 12px rgba(36,87,214,.3)); }
.nav-logo svg { width:100%; height:100%; display:block; }
.nav-name { font-weight:800; font-size:1.12rem; color:#142B55; letter-spacing:-.01em; line-height:1.15; }
.nav-sub { font-size:.78rem; color:#425B82; margin-top:1px; }
.st-key-topnav button { width:100%; min-height:46px; border-radius:14px; font-weight:600; }
.st-key-topnav button[kind="secondary"], .st-key-topnav button[data-testid="stBaseButton-secondary"] { background:transparent; border:none; color:#142B55; }
.st-key-topnav button[kind="secondary"]:hover, .st-key-topnav button[data-testid="stBaseButton-secondary"]:hover { background:rgba(255,255,255,.75); color:#142B55; border:none; }
.st-key-topnav button[kind="primary"], .st-key-topnav button[data-testid="stBaseButton-primary"] { background:#CFE0FB; color:#1F4FD1; border:none; }
.st-key-topnav button[kind="primary"]:hover, .st-key-topnav button[data-testid="stBaseButton-primary"]:hover { background:#CFE0FB; color:#1F4FD1; filter:none; }
.st-key-nav_cta button { background:linear-gradient(135deg,#3F7BF2,#2457D6) !important; color:#fff !important; border:none !important; border-radius:14px !important; min-height:48px; box-shadow:0 10px 22px rgba(52,120,246,.35) !important; font-weight:700; }
.st-key-nav_cta button:hover { filter:brightness(1.08); }

/* left sidebar */
.st-key-leftpanel { background:linear-gradient(180deg,#E6F0FD 0%,#ECEAFF 68%,#E5F6F3 100%); border:1px solid rgba(255,255,255,.85); border-radius:26px; padding:18px 14px 6px 14px; box-shadow:0 12px 30px rgba(20,43,85,.08); }
.panel-label { color:#425B82; font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin:6px 6px 10px 6px; }
.panel-divider { height:1px; background:rgba(66,91,130,.18); margin:14px 6px 12px 6px; }
.st-key-leftpanel button { width:100%; min-height:48px; padding:.4rem .9rem; border-radius:14px; background:rgba(255,255,255,.72); border:1px solid rgba(255,255,255,.9); color:#142B55; transition:all .15s ease; }
.st-key-leftpanel button p { color:inherit; font-weight:600; font-size:.93rem; }
.st-key-leftpanel button:hover { background:#fff; border-color:#BCD3F8; transform:translateX(2px); color:#142B55; }
.st-key-leftpanel button[kind="primary"], .st-key-leftpanel button[data-testid="stBaseButton-primary"] { background:#CFE0FB; border:1px solid #BCD3F8; color:#1F4FD1; box-shadow:inset 4px 0 0 #3478F6; }
.side-art { display:flex; align-items:flex-end; gap:4px; margin:10px 0 4px 0; }
.side-art svg { width:56%; height:auto; display:block; flex:0 0 56%; }
.side-art p { font-size:.86rem; color:#425B82; line-height:1.4; font-weight:600; margin:0 0 18px 0; }

/* home hero */
.hero2 { display:grid; grid-template-columns:1.02fr 1fr; gap:6px; align-items:center; padding:30px 28px 22px 34px; border-radius:30px; background:linear-gradient(135deg,#E5F1FF 0%,#F0E7FF 55%,#FFE7D5 125%); border:1px solid rgba(255,255,255,.9); box-shadow:0 18px 44px rgba(20,43,85,.10); }
.hero-badge { display:inline-flex; align-items:center; gap:8px; padding:7px 14px; border-radius:999px; background:#DDF7EA; color:#0B6B5F; font-weight:700; font-size:.8rem; border:1px solid #BFEBD8; }
.hero2 h1 { font-size:2.3rem; line-height:1.14; font-weight:800; margin:16px 0 14px 0; color:#142B55; letter-spacing:-.02em; }
.hero2 h1 .hl { color:#3478F6; }
.hero2 .desc { color:#425B82; font-size:1.02rem; line-height:1.6; max-width:540px; }
.benefits { list-style:none; padding:0; margin:18px 0 0 0; display:grid; gap:10px; }
.benefits li { display:flex; align-items:center; gap:12px; font-weight:600; color:#142B55; font-size:1rem; }
.ck { width:26px; height:26px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; color:#fff; font-size:.8rem; font-weight:800; flex:0 0 26px; }
.ck.b { background:#3478F6; } .ck.p { background:#9B7CF0; } .ck.m { background:#38C18F; }
.hero-art svg, .hero-art img { width:100%; height:auto; display:block; }
.hero-pills { display:flex; gap:12px; justify-content:center; flex-wrap:wrap; margin-top:4px; }
.hero-pills span { padding:8px 22px; border-radius:999px; font-weight:700; font-size:.86rem; }
.pill-b { background:#DCE8FF; color:#2457D6; } .pill-m { background:#D5F3E5; color:#0F7A66; } .pill-l { background:#E8DEFF; color:#6A3FD0; }
.collab { border-radius:30px; overflow:hidden; background:linear-gradient(180deg,#FFE7D5 0%,#DDF7EA 58%,#E5F1FF 100%); border:1px solid rgba(255,255,255,.9); box-shadow:0 18px 44px rgba(20,43,85,.10); }
.collab svg { width:100%; height:auto; display:block; }
.collab .cap { padding:20px 22px 26px 22px; }
.collab h3 { font-size:1.25rem; margin:0 0 8px 0; color:#142B55; font-weight:800; }
.collab p { color:#425B82; font-size:.96rem; line-height:1.55; margin:0; }

/* action buttons */
.st-key-act1 button { width:100%; min-height:58px; border-radius:16px; background:linear-gradient(135deg,#3F7BF2,#2457D6); color:#fff; border:none; font-weight:700; box-shadow:0 12px 26px rgba(52,120,246,.35); }
.st-key-act2 button { width:100%; min-height:58px; border-radius:16px; background:#E1EDFF; color:#142B55; border:1px solid #CFE0FB; font-weight:700; }
.st-key-act3 button { width:100%; min-height:58px; border-radius:16px; background:#DDF7EA; color:#142B55; border:1px solid #BFEBD8; font-weight:700; }
.st-key-act1 button p, .st-key-act2 button p, .st-key-act3 button p { color:inherit; font-weight:700; font-size:1rem; }
.st-key-act1 button:hover, .st-key-act2 button:hover, .st-key-act3 button:hover { transform:translateY(-2px); filter:brightness(1.03); }
.st-key-act2 button:hover { border-color:#3478F6; color:#1F4FD1; } .st-key-act3 button:hover { border-color:#109A8A; color:#0B6B5F; }

/* feature cards */
.feat-ic { width:58px; height:58px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin-bottom:14px; }
.feat-ic svg { width:28px; height:28px; }
.feat-title { font-weight:800; font-size:1.08rem; color:#142B55; margin-bottom:6px; }
.feat-desc { color:#425B82; font-size:.9rem; line-height:1.5; padding-right:46px; }
.st-key-feat0, .st-key-feat1, .st-key-feat2, .st-key-feat3 { border-radius:22px; padding:22px 22px 18px 22px; position:relative; min-height:200px; border:1px solid rgba(255,255,255,.9); box-shadow:0 12px 30px rgba(20,43,85,.07); transition:transform .18s ease, box-shadow .18s ease; }
.st-key-feat0:hover, .st-key-feat1:hover, .st-key-feat2:hover, .st-key-feat3:hover { transform:translateY(-4px); box-shadow:0 18px 38px rgba(20,43,85,.12); }
.st-key-feat0 { background:linear-gradient(135deg,#EDF5FF,#DCEBFF); } .st-key-feat1 { background:linear-gradient(135deg,#F5EFFF,#E8DEFF); }
.st-key-feat2 { background:linear-gradient(135deg,#E9FAF2,#D5F3E5); } .st-key-feat3 { background:linear-gradient(135deg,#FFF3E8,#FFE2CC); }
.st-key-feat0 [data-testid="stButton"], .st-key-feat1 [data-testid="stButton"], .st-key-feat2 [data-testid="stButton"], .st-key-feat3 [data-testid="stButton"] { position:absolute; right:16px; bottom:14px; width:auto; }
.st-key-feat0 button, .st-key-feat1 button, .st-key-feat2 button, .st-key-feat3 button { width:42px; height:42px; min-height:42px; padding:0; border-radius:50%; background:#fff; border:none; box-shadow:0 6px 16px rgba(20,43,85,.12); font-weight:800; font-size:1.05rem; }
.st-key-feat0 button { color:#2457D6; } .st-key-feat1 button { color:#6A3FD0; } .st-key-feat2 button { color:#0F7A66; } .st-key-feat3 button { color:#C2570C; }
.st-key-feat0 button:hover, .st-key-feat1 button:hover, .st-key-feat2 button:hover, .st-key-feat3 button:hover { transform:scale(1.1); }

/* shared pastel components used on every page */
.card { background:rgba(255,255,255,.84); border:1px solid rgba(255,255,255,.95); border-radius:20px; box-shadow:0 10px 28px rgba(20,43,85,.07); transition:transform .18s ease, box-shadow .18s ease; }
.card:hover { transform:translateY(-2px); box-shadow:0 16px 34px rgba(20,43,85,.10); }
.grid > .card:nth-child(4n+1), .grid > .stat:nth-child(4n+1) { background:linear-gradient(135deg,#EDF5FF,#DCEBFF); }
.grid > .card:nth-child(4n+2), .grid > .stat:nth-child(4n+2) { background:linear-gradient(135deg,#F5EFFF,#E8DEFF); }
.grid > .card:nth-child(4n+3), .grid > .stat:nth-child(4n+3) { background:linear-gradient(135deg,#E9FAF2,#D5F3E5); }
.grid > .card:nth-child(4n+4), .grid > .stat:nth-child(4n+4) { background:linear-gradient(135deg,#FFF3E8,#FFE2CC); }
.stat { border:1px solid rgba(255,255,255,.95); border-radius:20px; box-shadow:0 10px 28px rgba(20,43,85,.07); background:rgba(255,255,255,.84); }
.steps > .step:nth-child(1) { background:linear-gradient(135deg,#EDF5FF,#DCEBFF); } .steps > .step:nth-child(3) { background:linear-gradient(135deg,#F5EFFF,#E8DEFF); }
.steps > .step:nth-child(5) { background:linear-gradient(135deg,#E9FAF2,#D5F3E5); } .steps > .step:nth-child(7) { background:linear-gradient(135deg,#FFF3E8,#FFE2CC); }
.steps > .step:nth-child(9) { background:linear-gradient(135deg,#DDF7F3,#C9EFE9); }
.step { border:1px solid rgba(255,255,255,.95); border-radius:18px; }
.stages .stage:nth-child(1) { background:#DCEBFF; color:#2457D6; } .stages .stage:nth-child(3) { background:#E8DEFF; color:#6A3FD0; } .stages .stage:nth-child(5) { background:#D5F3E5; color:#0F7A66; }
.stages .stage:nth-child(7) { background:#FFE2CC; color:#C2570C; } .stages .stage:nth-child(9) { background:#C9EFE9; color:#0B6B5F; }
.stages .stage { border:none; }
.stages .stage.active { background:linear-gradient(135deg,#3F7BF2,#2457D6); color:#fff; box-shadow:0 8px 20px rgba(52,120,246,.35); }
.rtable th { background:#E5F1FF; color:#142B55; border-bottom:1px solid #C8D9F0; }
.rtable td { color:#142B55; border-bottom:1px solid #DCE8F8; }
.rtable tr:hover td { background:rgba(229,241,255,.55); }
.rtable tr.sel td { background:#DCEBFF; }
.chip.active { background:linear-gradient(135deg,#3F7BF2,#2457D6); color:#fff; }
.chip.done { background:#DDF7EA; color:#0F7A66; border-color:#BFEBD8; }
.chip { background:rgba(255,255,255,.75); }
.notice { background:rgba(255,255,255,.8); border-radius:16px; }
.notice.info { background:#E1EDFF; border-color:#BCD3F8; color:#142B55; }
.notice.warn { background:#FFF0E0; border-color:#F6C99A; color:#6B3A0B; }
.notice.ok { background:#DDF7EA; border-color:#BFEBD8; color:#0B5B4E; }
.art-card { background:linear-gradient(180deg,#FFF1E6 0%,#E5F6EF 60%,#E5F1FF 100%); border:1px solid rgba(255,255,255,.95); border-radius:24px; box-shadow:0 12px 30px rgba(20,43,85,.08); }
.footer { background:linear-gradient(135deg,#E5F1FF 0%,#EDE9FF 55%,#FFF1E3 100%); border-radius:24px; border-top:none; box-shadow:0 12px 30px rgba(20,43,85,.08); padding:24px 20px; }
.footer .fb { color:#142B55; } .footer .ft { color:#425B82; } .footer .fs { color:#425B82; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.82); border:1px solid #D4E3F7 !important; border-radius:20px; box-shadow:0 10px 26px rgba(20,43,85,.06); }
div[data-testid="stExpander"] { background:rgba(255,255,255,.75); border:1px solid #D4E3F7; border-radius:16px; }
button[data-baseweb="tab"] { color:#425B82; font-weight:700; }
button[data-baseweb="tab"][aria-selected="true"] { color:#1F4FD1; }
div[data-testid="stProgress"] > div > div > div > div { background:linear-gradient(90deg,#3F7BF2,#8757E8); }
button[kind="primary"], button[data-testid="stBaseButton-primary"] { background:linear-gradient(135deg,#3F7BF2,#2457D6); box-shadow:0 8px 20px rgba(52,120,246,.28); }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] { background:rgba(255,255,255,.85); border:1px solid #CFE0FB; color:#142B55; }

/* responsive: stack panels on tablet and phone */
@media (max-width: 1100px) {
  .hero2 { grid-template-columns:1fr; }
  .st-key-homehero [data-testid="stHorizontalBlock"] { flex-direction:column; }
  .st-key-homehero [data-testid="stColumn"], .st-key-homehero [data-testid="column"] { width:100% !important; flex:1 1 100% !important; min-width:100% !important; }
  .st-key-features [data-testid="stHorizontalBlock"], .st-key-actions [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
  .st-key-features [data-testid="stColumn"], .st-key-features [data-testid="column"] { flex:1 1 calc(50% - 1rem) !important; min-width:calc(50% - 1rem) !important; }
}
@media (max-width: 900px) {
  .st-key-shell > div > [data-testid="stHorizontalBlock"] { flex-direction:column; }
  .st-key-shell > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"], .st-key-shell > div > [data-testid="stHorizontalBlock"] > [data-testid="column"] { width:100% !important; flex:1 1 100% !important; min-width:100% !important; }
  .st-key-actions [data-testid="stColumn"], .st-key-actions [data-testid="column"] { flex:1 1 100% !important; min-width:100% !important; }
  .hero2 h1 { font-size:1.8rem; }
}
@media (max-width: 640px) {
  .st-key-features [data-testid="stColumn"], .st-key-features [data-testid="column"] { flex:1 1 100% !important; min-width:100% !important; }
  .block-container { padding:.6rem .7rem 1.5rem .7rem !important; }
}
button:focus-visible, input:focus-visible, textarea:focus-visible, [role="slider"]:focus-visible { outline:2px solid #2B63D6 !important; outline-offset:2px; }
</style>
"""

HERO_ART = """<svg viewBox="0 0 560 390" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A student using a laptop with an AI assistant, message cards and a behaviour analytics chart">
<defs>
<linearGradient id="hb" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4C8DFF"/><stop offset="1" stop-color="#2457D6"/></linearGradient>
<linearGradient id="hh" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3F7BF2"/><stop offset="1" stop-color="#2B5FDB"/></linearGradient>
<radialGradient id="hg" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#FFFFFF" stop-opacity=".95"/><stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></radialGradient>
</defs>
<circle cx="330" cy="150" r="150" fill="#F3EDFF" opacity=".75"/><circle cx="250" cy="250" r="130" fill="#FFEBDD" opacity=".7"/>
<circle cx="330" cy="130" r="92" fill="url(#hg)"/>
<g fill="none" stroke="#4C8DFF" stroke-width="2.4" stroke-linecap="round" stroke-dasharray="1 8">
<path d="M150 82 C 190 70, 230 80, 276 104"/><path d="M384 108 C 420 84, 440 80, 468 84"/><path d="M372 160 C 400 170, 418 182, 438 200"/><path d="M220 215 C 238 190, 262 168, 292 150"/>
</g>
<g><rect x="80" y="42" width="110" height="72" rx="14" fill="#FFFFFF" stroke="#D7E5FA"/><rect x="96" y="58" width="42" height="32" rx="9" fill="#4C8DFF"/><path d="M106 90 l-6 10 l14 -6z" fill="#4C8DFF"/><rect x="104" y="68" width="26" height="4" rx="2" fill="#fff"/><rect x="104" y="76" width="18" height="4" rx="2" fill="#fff"/><rect x="148" y="62" width="30" height="5" rx="2.5" fill="#C9D9F3"/><rect x="148" y="74" width="24" height="5" rx="2.5" fill="#C9D9F3"/></g>
<g><rect x="440" y="52" width="104" height="70" rx="14" fill="#FFFFFF" stroke="#D7E5FA"/><rect x="456" y="68" width="72" height="6" rx="3" fill="#C9D9F3"/><rect x="456" y="82" width="72" height="6" rx="3" fill="#C9D9F3"/><rect x="456" y="96" width="48" height="6" rx="3" fill="#C9D9F3"/></g>
<circle cx="330" cy="130" r="58" fill="#FFFFFF" stroke="#DCE8FA" stroke-width="3"/>
<circle cx="330" cy="122" r="34" fill="url(#hb)"/>
<g fill="none" stroke="#FFFFFF" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"><path d="M322 112 c-8 0 -10 8 -6 12 c-5 3 -3 12 4 12 h2"/><path d="M338 112 c8 0 10 8 6 12 c5 3 3 12 -4 12 h-2"/><path d="M330 108 v32"/><path d="M322 122 h8 M330 130 h8"/></g>
<text x="330" y="188" text-anchor="middle" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="12.5" font-weight="700" fill="#142B55">AI</text><text x="330" y="203" text-anchor="middle" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="12.5" font-weight="700" fill="#142B55">Assistant</text>
<g><rect x="446" y="188" width="96" height="92" rx="14" fill="#FFFFFF" stroke="#D7E5FA"/>
<rect x="458" y="238" width="14" height="30" rx="3" fill="#8757E8"/><rect x="478" y="226" width="14" height="42" rx="3" fill="#3478F6"/><rect x="498" y="212" width="14" height="56" rx="3" fill="#109A8A"/><rect x="518" y="200" width="14" height="68" rx="3" fill="#109A8A" opacity=".75"/>
<path d="M458 232 L480 214 L498 222 L530 196" fill="none" stroke="#F4A261" stroke-width="2.6" stroke-linecap="round"/><circle cx="530" cy="196" r="4" fill="#F4A261"/></g>
<text x="492" y="304" text-anchor="middle" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" font-weight="700" fill="#142B55">Behaviour Analysis</text>
<rect x="30" y="338" width="360" height="9" rx="4.5" fill="#CFE0F6"/>
<g><rect x="42" y="324" width="104" height="14" rx="3" fill="#109A8A"/><rect x="50" y="311" width="92" height="13" rx="3" fill="#7ED6A8"/><rect x="46" y="298" width="98" height="13" rx="3" fill="#4C8DFF"/><rect x="52" y="318" width="26" height="2.5" fill="#fff" opacity=".7"/></g>
<g><path d="M118 300 C 96 270, 104 246, 116 228 C 128 250, 132 276, 118 300Z" fill="#5BBF8E"/><path d="M118 300 C 138 276, 150 262, 168 256 C 162 280, 142 296, 118 300Z" fill="#3FA574"/><path d="M118 300 C 92 292, 76 276, 70 258 C 94 262, 112 278, 118 300Z" fill="#7CD1A0"/></g>
<path d="M150 342 C 170 300, 188 280, 214 268 L 258 268 C 262 290, 262 320, 266 342Z" fill="url(#hh)"/>
<path d="M196 270 C 204 286, 226 286, 236 270 L 226 262 L 206 262Z" fill="#2457D6"/>
<rect x="206" y="236" width="22" height="30" rx="9" fill="#FFD2B0"/>
<circle cx="214" cy="218" r="27" fill="#FFD7B5"/>
<path d="M186 214 C 184 186, 210 176, 230 188 C 244 196, 246 210, 240 218 C 234 206, 222 200, 208 202 C 198 204, 190 208, 186 214Z" fill="#1B2A55"/>
<path d="M190 200 C 196 178, 226 178, 238 196" fill="none" stroke="#1B2A55" stroke-width="8" stroke-linecap="round"/>
<circle cx="224" cy="222" r="2.6" fill="#1B2A55"/><path d="M222 233 q6 5 12 0" fill="none" stroke="#C97B5A" stroke-width="2" stroke-linecap="round"/>
<g><rect x="262" y="262" width="104" height="70" rx="8" fill="#2A3A6B" transform="rotate(-8 262 332)"/><circle cx="312" cy="296" r="7" fill="#7F93C9" opacity=".8" transform="rotate(-8 262 332)"/>
<path d="M248 336 L388 336 L376 346 L238 346Z" fill="#1B2A55"/></g>
<path d="M244 296 C 256 316, 272 328, 292 334" fill="none" stroke="#3F7BF2" stroke-width="13" stroke-linecap="round"/>
<g><path d="M372 336 q-3 -22 6 -32 q8 10 6 32z" fill="#7CD1A0"/><path d="M380 336 q8 -16 22 -18 q-2 14 -22 18z" fill="#3FA574"/><rect x="368" y="334" width="22" height="14" rx="4" fill="#C9B8F2"/></g>
</svg>"""

COLLAB_ART = """<svg viewBox="0 0 320 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A student collaborating with an AI assistant in a calm landscape">
<defs><linearGradient id="cs" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFE9D2"/><stop offset=".6" stop-color="#E3F6EE"/><stop offset="1" stop-color="#DDEBFF"/></linearGradient>
<linearGradient id="c1" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#7CC7A0"/><stop offset="1" stop-color="#4FA982"/></linearGradient>
<linearGradient id="c2" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#9AD8B8"/><stop offset="1" stop-color="#6CBF97"/></linearGradient>
<radialGradient id="cg"><stop offset="0" stop-color="#FFE29A" stop-opacity=".9"/><stop offset="1" stop-color="#FFE29A" stop-opacity="0"/></radialGradient></defs>
<rect width="320" height="210" fill="url(#cs)"/>
<circle cx="190" cy="62" r="62" fill="url(#cg)"/><circle cx="190" cy="62" r="26" fill="#FFD27A"/>
<path d="M0 120 C 50 70, 96 78, 140 112 C 180 84, 230 70, 320 118 L320 210 L0 210Z" fill="#B7DFC9"/>
<path d="M0 150 C 60 118, 110 126, 170 150 C 220 126, 270 124, 320 146 L320 210 L0 210Z" fill="url(#c2)"/>
<path d="M0 178 C 70 156, 130 162, 200 180 C 250 166, 290 168, 320 178 L320 210 L0 210Z" fill="url(#c1)"/>
<g fill="#3C9A76"><path d="M30 150 C 20 120, 30 100, 42 90 C 52 108, 54 130, 46 152Z"/><path d="M52 154 C 52 130, 64 112, 76 106 C 80 126, 74 144, 62 156Z"/><path d="M270 152 C 262 124, 272 106, 286 98 C 294 118, 292 138, 282 154Z"/></g>
<g fill="#2F8A66"><path d="M18 156 C 10 134, 18 118, 28 112 C 36 126, 38 142, 32 158Z"/><path d="M296 156 C 300 136, 306 124, 316 120 C 318 136, 314 148, 306 158Z"/></g>
<g><path d="M96 200 C 98 170, 124 160, 148 164 C 168 168, 176 186, 176 200Z" fill="#3F7BF2"/><circle cx="136" cy="146" r="17" fill="#FFD7B5"/><path d="M119 144 C 118 126, 138 120, 150 130 C 156 136, 154 146, 152 150 C 146 140, 130 138, 119 144Z" fill="#1B2A55"/><rect x="120" y="178" width="58" height="34" rx="6" fill="#2A3A6B"/></g>
<g><circle cx="226" cy="126" r="26" fill="#FFFFFF" stroke="#DCE8FA" stroke-width="3"/><circle cx="226" cy="126" r="16" fill="#3478F6"/><path d="M219 122 q4 -6 8 0 q4 -4 7 2 M219 130 q4 6 8 0 q4 4 7 -2" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round"/></g>
<path d="M156 140 C 180 120, 196 118, 202 124" fill="none" stroke="#3478F6" stroke-width="2.4" stroke-linecap="round" stroke-dasharray="1 7"/>
<rect x="244" y="86" width="40" height="26" rx="8" fill="#FFFFFF" opacity=".95"/><rect x="252" y="94" width="24" height="4" rx="2" fill="#C9D9F3"/><rect x="252" y="102" width="16" height="4" rx="2" fill="#C9D9F3"/>
</svg>"""

SIDE_ART = """<svg viewBox="0 0 240 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A plant growing from a pot beside stacked books">
<defs><linearGradient id="sp" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#D9C9FA"/><stop offset="1" stop-color="#A98CF0"/></linearGradient></defs>
<ellipse cx="112" cy="144" rx="100" ry="5" fill="#BFD3EE" opacity=".6"/>
<g><path d="M60 150 L 66 112 L 112 112 L 118 150Z" fill="url(#sp)"/><rect x="58" y="104" width="62" height="12" rx="6" fill="#C7B3F7"/></g>
<g><path d="M89 108 C 66 98, 58 72, 66 50 C 86 62, 94 86, 89 108Z" fill="#4FB68A"/><path d="M89 108 C 100 82, 116 62, 138 54 C 138 82, 118 102, 89 108Z" fill="#3AA175"/><path d="M89 108 C 74 108, 54 98, 44 82 C 66 80, 84 90, 89 108Z" fill="#6FCB9E"/><path d="M89 108 C 90 80, 92 56, 100 34 C 112 58, 108 88, 89 108Z" fill="#5BC092"/></g>
<g><rect x="132" y="130" width="92" height="16" rx="3" fill="#2F6BF0"/><rect x="140" y="114" width="80" height="16" rx="3" fill="#3F86F5"/><rect x="136" y="98" width="86" height="16" rx="3" fill="#1F3F9E"/><rect x="140" y="104" width="30" height="3" fill="#fff" opacity=".6"/><rect x="146" y="136" width="30" height="3" fill="#fff" opacity=".6"/></g>
</svg>"""

LOGO_SVG = """<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3F7BF2"/><stop offset="1" stop-color="#17358F"/></linearGradient></defs><rect width="48" height="48" rx="13" fill="url(#lg)"/><path d="M24 11 L41 19 L24 27 L7 19Z" fill="#FFFFFF"/><path d="M14 24 V31 C 18 35, 30 35, 34 31 V24 L24 29Z" fill="#DCE8FF"/><path d="M41 19 V30" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round"/></svg>"""

HERO_SVG = """
<svg viewBox="0 0 520 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Student interacting with an AI assistant while a dashboard analyses learning behaviour">
<defs>
<linearGradient id="g1" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3478F6"/><stop offset="1" stop-color="#8757E8"/></linearGradient>
<radialGradient id="glow"><stop offset="0" stop-color="#109A8A" stop-opacity=".45"/><stop offset="1" stop-color="#109A8A" stop-opacity="0"/></radialGradient>
</defs>
<rect width="520" height="340" rx="22" fill="#EAF2FD" stroke="#C8D9F0"/>
<circle cx="260" cy="160" r="86" fill="url(#glow)"/>
<path d="M122 160 L 212 160" stroke="#3478F6" stroke-width="2.2" stroke-dasharray="5 6" fill="none"><animate attributeName="stroke-dashoffset" from="22" to="0" dur="1.6s" repeatCount="indefinite"/></path>
<path d="M308 160 L 372 160" stroke="#8757E8" stroke-width="2.2" stroke-dasharray="5 6" fill="none"><animate attributeName="stroke-dashoffset" from="22" to="0" dur="1.6s" repeatCount="indefinite"/></path>
<circle cx="78" cy="118" r="22" fill="#142B55" opacity=".92"/>
<path d="M40 196 Q78 138 116 196 Z" fill="url(#g1)"/>
<rect x="46" y="204" width="64" height="38" rx="7" fill="#E3EDFB" stroke="#3478F6" stroke-width="1.5"/>
<rect x="54" y="212" width="48" height="4" rx="2" fill="#3478F6" opacity=".7"/>
<rect x="54" y="222" width="34" height="4" rx="2" fill="#8757E8" opacity=".7"/>
<polygon points="260,112 300,135 300,185 260,208 220,185 220,135" fill="#FFFFFF" stroke="url(#g1)" stroke-width="3"/>
<text x="260" y="168" text-anchor="middle" font-family="Inter,Segoe UI,sans-serif" font-size="26" font-weight="800" fill="#109A8A">AI</text>
<circle cx="260" cy="92" r="5" fill="#3478F6"/><circle cx="326" cy="205" r="4" fill="#8757E8"/><circle cx="198" cy="214" r="4" fill="#109A8A"/>
<rect x="372" y="86" width="132" height="150" rx="12" fill="#FFFFFF" stroke="#C8D9F0"/>
<rect x="386" y="100" width="56" height="6" rx="3" fill="#425B82" opacity=".6"/>
<circle cx="480" cy="108" r="12" fill="none" stroke="#D9E6F7" stroke-width="5"/>
<circle cx="480" cy="108" r="12" fill="none" stroke="#109A8A" stroke-width="5" stroke-dasharray="46 76" transform="rotate(-90 480 108)"/>
<rect x="388" y="190" width="16" height="30" rx="3" fill="#3478F6" opacity=".85"/>
<rect x="412" y="168" width="16" height="52" rx="3" fill="#8757E8" opacity=".85"/>
<rect x="436" y="146" width="16" height="74" rx="3" fill="#109A8A" opacity=".85"/>
<rect x="460" y="176" width="16" height="44" rx="3" fill="#3478F6" opacity=".6"/>
<path d="M388 150 L 412 138 L 436 122 L 476 134" stroke="#142B55" stroke-width="1.6" fill="none" opacity=".55"/>
<g font-family="Inter,Segoe UI,sans-serif" font-size="11" fill="#425B82" text-anchor="middle">
<text x="78" y="266">Student</text><text x="260" y="238">AI assistant</text><text x="438" y="256">Behaviour analysis</text>
</g>
<g font-family="Inter,Segoe UI,sans-serif" font-size="11" font-weight="600" text-anchor="middle">
<rect x="96" y="292" width="76" height="24" rx="12" fill="#E3EDFB"/><text x="134" y="308" fill="#2B63D6">Predict</text>
<rect x="222" y="292" width="76" height="24" rx="12" fill="#E3EDFB"/><text x="260" y="308" fill="#109A8A">Explain</text>
<rect x="348" y="292" width="76" height="24" rx="12" fill="#E3EDFB"/><text x="386" y="308" fill="#2B63D6">Intervene</text>
</g>
</svg>
"""


def md_html(markup: str) -> None:
    """Render raw HTML safely through Streamlit markdown (indentation stripped
    so that Markdown never mistakes it for a code block)."""
    flat = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def esc(value) -> str:
    return htmllib.escape(str(value))


md_html(CSS)


# ============================================================
# LOAD MODEL ARTIFACTS  (never retrain here)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_obj = joblib.load(BASE_DIR / "final_rf_pipeline.pkl")
    notes = []
    try:
        thresholds_obj = joblib.load(BASE_DIR / "risk_thresholds.pkl")
    except Exception:
        thresholds_obj = {"low_threshold": 3.5, "high_threshold": 5.0}
        notes.append("risk_thresholds.pkl could not be loaded; the documented defaults (3.5 / 5.0) are in use.")
    try:
        metadata_obj = joblib.load(BASE_DIR / "model_metadata.pkl")
    except Exception:
        metadata_obj = {
            "numeric_features": DEFAULT_NUMERIC,
            "categorical_features": DEFAULT_CATEGORICAL,
            "target": "AI_Reliance_Score",
        }
        notes.append("model_metadata.pkl could not be loaded; the documented feature lists are in use.")
    return model_obj, thresholds_obj, metadata_obj, notes


try:
    model, thresholds, metadata, load_notes = load_artifacts()
    preprocessor = model.named_steps["preprocessor"]
    rf_model = model.named_steps["model"]
except Exception:
    md_html(
        """
        <div class="notice warn"><b>The model could not be loaded.</b><br>
        Please check that <code>final_rf_pipeline.pkl</code>, <code>risk_thresholds.pkl</code> and
        <code>model_metadata.pkl</code> are in the same folder as <code>app.py</code>, and that the
        environment uses <code>scikit-learn==1.6.1</code> (the version used to train the model).</div>
        """
    )
    st.stop()

numeric_features = list(metadata.get("numeric_features", DEFAULT_NUMERIC))
categorical_features = list(metadata.get("categorical_features", DEFAULT_CATEGORICAL))
low_threshold = float(thresholds.get("low_threshold", 3.5))
high_threshold = float(thresholds.get("high_threshold", 5.0))

# Categorical options come straight from the trained encoder
category_options = {}
try:
    _encoder = preprocessor.named_transformers_["categorical"]
    for _feat, _cats in zip(categorical_features, _encoder.categories_):
        category_options[_feat] = list(_cats)
except Exception:
    pass
for _feat in categorical_features:
    if not category_options.get(_feat):
        category_options[_feat] = ["Not specified"]


# ============================================================
# CORE LOGIC  (prediction rules preserved from the original app)
# ============================================================

def classify_risk(score):
    if score < low_threshold:
        return "Low"
    elif score < high_threshold:
        return "Medium"
    else:
        return "High"


def get_intervention_with_trigger(row, predicted_score):
    """Same rules as the original get_intervention(); also returns the
    behaviour trigger text so the UI can explain *why* it was selected."""
    risk = classify_risk(predicted_score)

    if risk == "High":
        if row["Accept_Without_Verification"] >= 5:
            return ("Verification Challenge",
                    f"Accept Without Verification = {row['Accept_Without_Verification']:.0f} (rule: ≥ 5)")
        elif row["Overreliance_Problem_Solving"] >= 5:
            return ("Independent Problem-Solving Task",
                    f"Overreliance in Problem Solving = {row['Overreliance_Problem_Solving']:.0f} (rule: ≥ 5)")
        elif row["Independent_Confidence"] <= 3:
            return ("Guided Independent Reasoning",
                    f"Independent Confidence = {row['Independent_Confidence']:.0f} (rule: ≤ 3)")
        else:
            return ("Critical Analysis Task", "No single behaviour trigger met; default high-level task")

    elif risk == "Medium":
        if row["Cross_Check"] <= 3:
            return ("AI Output Verification Task", f"Cross Check = {row['Cross_Check']:.0f} (rule: ≤ 3)")
        elif row["Independent_Confidence"] <= 3:
            return ("Guided Reasoning Question",
                    f"Independent Confidence = {row['Independent_Confidence']:.0f} (rule: ≤ 3)")
        elif row["Own_Ideas"] <= 3:
            return ("Independent Idea Generation", f"Own Ideas = {row['Own_Ideas']:.0f} (rule: ≤ 3)")
        else:
            return ("Reflective AI-Use Question", "No single behaviour trigger met; default moderate-level task")

    else:
        return ("Responsible AI Use Guidance", "Low predicted reliance; general guidance")


def get_intervention(row, predicted_score):
    return get_intervention_with_trigger(row, predicted_score)[0]


def get_intervention_description(intervention):
    descriptions = {
        "Verification Challenge":
            "Verify the AI-generated response using independent evidence before accepting it.",
        "Independent Problem-Solving Task":
            "Attempt the problem independently before consulting or relying on an AI-generated solution.",
        "Guided Independent Reasoning":
            "Develop your own reasoning step-by-step before comparing it with AI output.",
        "Critical Analysis Task":
            "Critically analyse the AI response and identify assumptions, errors, or missing evidence.",
        "AI Output Verification Task":
            "Cross-check the AI response against reliable sources before using the result.",
        "Guided Reasoning Question":
            "Answer a reasoning question independently before reviewing AI assistance.",
        "Independent Idea Generation":
            "Generate your own ideas first and use AI only for comparison or refinement.",
        "Reflective AI-Use Question":
            "Reflect on whether AI is supporting your learning or replacing your own thinking.",
        "Responsible AI Use Guidance":
            "Continue using AI as a supporting tool while maintaining independent thinking and verification.",
    }
    return descriptions.get(
        intervention,
        "Use AI as a supporting tool while maintaining independent reasoning.",
    )


# Interactive intervention content: stage (in Hint→Question→Verification→Analysis→Independent Answer),
# ordered steps, and the skills practised.
STAGES = ["Hint", "Question", "Verification", "Analysis", "Independent Answer"]

INTERVENTION_LIBRARY = {
    "Verification Challenge": {
        "stage": "Verification",
        "skills": ["Verification", "Independent reasoning", "Critical thinking", "Metacognition"],
        "steps": [
            "Think independently about the question for 60 seconds.",
            "Write your own answer before looking at any AI output.",
            "Compare your answer with the AI-generated answer.",
            "Identify at least one difference between the two answers.",
            "Explain which answer you trust more, and why.",
        ],
    },
    "Independent Problem-Solving Task": {
        "stage": "Independent Answer",
        "skills": ["Independent reasoning", "Persistence", "Metacognition"],
        "steps": [
            "Pick one problem you would normally hand straight to an AI tool.",
            "Spend five minutes attempting it yourself and write down each step you try.",
            "Note exactly where you got stuck.",
            "If needed, ask AI for a hint about that step only, not the full solution.",
            "Finish the solution yourself and write one sentence about what you learned.",
        ],
    },
    "Guided Independent Reasoning": {
        "stage": "Hint",
        "skills": ["Reasoning", "Confidence in own thinking", "Metacognition"],
        "steps": [
            "Restate the question in your own words.",
            "List what you already know that could be relevant.",
            "Write the first step of your reasoning, using notes or a textbook if you need a hint.",
            "Continue step by step until you have a complete answer of your own.",
            "Compare with AI reasoning and mark where your paths diverged.",
        ],
    },
    "Critical Analysis Task": {
        "stage": "Analysis",
        "skills": ["Critical thinking", "Evidence evaluation", "Verification"],
        "steps": [
            "Take an AI-generated response you recently used.",
            "Underline the main claims it makes.",
            "Identify one assumption and one claim that lacks supporting evidence.",
            "Check one claim against a reliable source.",
            "Rewrite the weakest part in your own words.",
        ],
    },
    "AI Output Verification Task": {
        "stage": "Verification",
        "skills": ["Cross-checking", "Source evaluation", "Verification"],
        "steps": [
            "Choose one AI answer you plan to use.",
            "Find two reliable sources that cover the same point.",
            "Mark what matches and what differs.",
            "Note any correction the AI answer needs.",
            "Decide whether to keep, revise or discard the answer, and say why.",
        ],
    },
    "Guided Reasoning Question": {
        "stage": "Question",
        "skills": ["Reasoning", "Self-explanation", "Metacognition"],
        "steps": [
            "Choose one question from your current coursework.",
            "Write your reasoning in two or three sentences before consulting AI.",
            "Ask AI for feedback on your reasoning rather than for the answer.",
            "Note one thing you would change in your own reasoning.",
        ],
    },
    "Independent Idea Generation": {
        "stage": "Question",
        "skills": ["Creativity", "Independent thinking", "Evaluation"],
        "steps": [
            "Set a three-minute timer and write at least three ideas on your own.",
            "Pick the strongest idea and note why.",
            "Ask AI for one alternative angle or a critique of your idea.",
            "Write the final idea in your own words and note what you kept from each source.",
        ],
    },
    "Reflective AI-Use Question": {
        "stage": "Analysis",
        "skills": ["Metacognition", "Self-regulation"],
        "steps": [
            "Recall the last time you used AI for coursework.",
            "Write down which part you could have tried yourself first.",
            "Write down what AI helped you understand that you could not on your own.",
            "Choose one task next week that you will attempt on your own before using AI.",
        ],
    },
    "Responsible AI Use Guidance": {
        "stage": "Independent Answer",
        "skills": ["Responsible AI use", "Verification", "Independent reasoning"],
        "steps": [
            "Where practical, draft your own first attempt before using AI.",
            "Keep cross-checking important AI claims against reliable sources.",
            "Use AI to explain or give feedback, not to replace your reasoning.",
            "Note which tasks AI genuinely helped with, and why.",
        ],
    },
}


INTERVENTION_EXTRAS = {
    "Verification Challenge": ("Practise checking an AI answer against your own reasoning before accepting it.",
                               "Why does a larger sample size reduce the standard error of the mean? Write your own answer first, then compare it with an AI answer."),
    "Independent Problem-Solving Task": ("Build the habit of attempting a problem yourself before asking for a solution.",
                                         "Pick a problem from your coursework and work on it unaided for five minutes, noting each step you try."),
    "Guided Independent Reasoning": ("Practise building reasoning step by step so that your own thinking leads.",
                                     "Explain why a model can perform well on training data but poorly on new data."),
    "Critical Analysis Task": ("Practise evaluating an AI response instead of accepting it as it is.",
                               "Ask an AI to summarise the advantages of a method you are studying, then list its assumptions and any claims without evidence."),
    "AI Output Verification Task": ("Practise confirming AI claims with reliable sources.",
                                    "Take one factual claim from an AI answer (a definition, a number or a date) and find two reliable sources for it."),
    "Guided Reasoning Question": ("Practise explaining your reasoning before you see any AI feedback.",
                                  "Why might a high correlation between two variables not mean that one causes the other?"),
    "Independent Idea Generation": ("Practise generating your own ideas before seeking AI input.",
                                    "Suggest three ways to make a study group more effective."),
    "Reflective AI-Use Question": ("Notice how AI is supporting, or replacing, your own thinking.",
                                   "Think of your last AI-assisted assignment. Which parts could you explain today without AI?"),
    "Responsible AI Use Guidance": ("Keep using AI as a supporting tool while protecting your own reasoning.",
                                    "Choose one upcoming task and decide in advance which part you will do first without AI."),
}


# ============================================================
# SHAP EXPLANATION  (logic preserved; baseline + aggregation added)
# ============================================================

class TreeShapFallback:
    """Exact path-dependent TreeSHAP for scikit-learn forests (the algorithm behind shap.TreeExplainer),
    used only when the shap package cannot be imported (for example when Windows Application Control blocks numba)."""

    def __init__(self, forest):
        self.trees = []
        for estimator in forest.estimators_:
            t = estimator.tree_
            self.trees.append((t.children_left.tolist(), t.children_right.tolist(), t.feature.tolist(),
                               t.threshold.tolist(), t.value[:, 0, 0].tolist(), t.weighted_n_node_samples.tolist()))
        self.expected_value = float(np.mean([tree[4][0] for tree in self.trees]))

    @staticmethod
    def _extend(f, z, o, w, d, pz, po, pi):
        f.append(pi)
        z.append(pz)
        o.append(po)
        w.append(1.0 if d == 0 else 0.0)
        for i in range(d - 1, -1, -1):
            w[i + 1] += po * w[i] * (i + 1) / (d + 1)
            w[i] = pz * w[i] * (d - i) / (d + 1)

    @staticmethod
    def _unwind(f, z, o, w, d, idx):
        one, zero = o[idx], z[idx]
        nxt = w[d]
        for i in range(d - 1, -1, -1):
            if one != 0:
                tmp = w[i]
                w[i] = nxt * (d + 1) / ((i + 1) * one)
                nxt = tmp - w[i] * zero * (d - i) / (d + 1)
            else:
                w[i] = w[i] * (d + 1) / (zero * (d - i))
        for i in range(idx, d):
            f[i], z[i], o[i] = f[i + 1], z[i + 1], o[i + 1]
        f.pop()
        z.pop()
        o.pop()
        w.pop()

    @staticmethod
    def _unwound_sum(z, o, w, d, idx):
        one, zero = o[idx], z[idx]
        nxt = w[d]
        total = 0.0
        for i in range(d - 1, -1, -1):
            if one != 0:
                tmp = nxt * (d + 1) / ((i + 1) * one)
                total += tmp
                nxt = w[i] - tmp * zero * ((d - i) / (d + 1))
            elif zero != 0:
                total += (w[i] / zero) / ((d - i) / (d + 1))
        return total

    def _tree_phi(self, tree, x, phi):
        cl, cr, feat, thr, val, cov = tree
        extend, unwind, unwound_sum = self._extend, self._unwind, self._unwound_sum

        def recurse(node, d, f, z, o, w, pz, po, pi):
            f, z, o, w = f[:], z[:], o[:], w[:]
            extend(f, z, o, w, d, pz, po, pi)
            if cl[node] < 0:
                v = val[node]
                for i in range(1, d + 1):
                    phi[f[i]] += unwound_sum(z, o, w, d, i) * (o[i] - z[i]) * v
                return
            split = feat[node]
            hot = cl[node] if x[split] <= thr[node] else cr[node]
            cold = cr[node] if hot == cl[node] else cl[node]
            total = cov[node]
            hot_zero, cold_zero = cov[hot] / total, cov[cold] / total
            in_zero, in_one = 1.0, 1.0
            idx = None
            for k in range(d + 1):
                if f[k] == split:
                    idx = k
                    break
            if idx is not None:
                in_zero, in_one = z[idx], o[idx]
                unwind(f, z, o, w, d, idx)
                d -= 1
            recurse(hot, d + 1, f, z, o, w, hot_zero * in_zero, in_one, split)
            recurse(cold, d + 1, f, z, o, w, cold_zero * in_zero, 0.0, split)

        recurse(0, 0, [], [], [], [], 1.0, 1.0, -1)

    def shap_values(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.zeros(X.shape, dtype=float)
        for r in range(X.shape[0]):
            x = X[r].astype(float).tolist()
            phi = [0.0] * X.shape[1]
            for tree in self.trees:
                self._tree_phi(tree, x, phi)
            out[r] = np.asarray(phi) / len(self.trees)
        return out


@st.cache_resource(show_spinner=False)
def get_explainer():
    if shap is not None:
        return shap.TreeExplainer(rf_model)
    return TreeShapFallback(rf_model)


def generate_shap_explanation(input_df):
    if os.environ.get("OVERRELIANCE_SIMULATE_SHAP_FAILURE"):  # test hook used by test_app.py
        raise RuntimeError("Simulated SHAP failure")
    processed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    explainer = get_explainer()
    shap_values = explainer.shap_values(processed)

    values = np.asarray(shap_values).reshape(-1)
    if len(values) != len(feature_names):
        raise ValueError("Unexpected SHAP output shape")

    explanation_df = pd.DataFrame({"Feature": feature_names, "SHAP_Value": values})
    explanation_df["Absolute_SHAP"] = explanation_df["SHAP_Value"].abs()
    explanation_df = explanation_df.sort_values("Absolute_SHAP", ascending=False)
    return explanation_df


def get_baseline():
    try:
        return float(np.ravel(get_explainer().expected_value)[0])
    except Exception:
        return None


def base_feature(encoded_name: str) -> str:
    """Map an encoded feature name (e.g. categorical__Gender_Male) back to its source column."""
    name = re.sub(r"^(numeric|categorical|num|cat)__", "", str(encoded_name))
    if name in numeric_features:
        return name
    for feat in categorical_features:
        if name == feat or name.startswith(feat + "_"):
            return feat
    return name


def feature_label(feat: str) -> str:
    return FEATURE_META.get(feat, (feat.replace("_", " "), ""))[0]


def aggregate_shap(explanation_df, input_row):
    """Sum one-hot columns back into their original question (SHAP is additive)."""
    tmp = explanation_df.copy()
    tmp["Source"] = tmp["Feature"].map(base_feature)
    agg = tmp.groupby("Source", as_index=False)["SHAP_Value"].sum()
    agg["Label"] = agg["Source"].map(feature_label)
    agg["Value"] = agg["Source"].map(lambda f: input_row[f] if f in input_row.index else "")
    return agg.sort_values("SHAP_Value", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


# (feature, direction, value condition, sentence) -> plain-language phrases
EXPLAIN_RULES = [
    ("Accept_Without_Verification", 1, lambda v: v >= 4, "Accepting AI answers without verification contributed to a higher predicted reliance level."),
    ("Accept_Without_Verification", -1, lambda v: v <= 4, "Checking AI answers before accepting them contributed to a lower predicted reliance level."),
    ("Cross_Check", -1, lambda v: v >= 4, "Regularly cross-checking AI output contributed to a lower predicted reliance level."),
    ("Cross_Check", 1, lambda v: v <= 4, "Less frequent cross-checking of AI output contributed to a higher predicted reliance level."),
    ("Own_Ideas", -1, lambda v: v >= 4, "Generating your own ideas contributed to a lower predicted reliance level."),
    ("Own_Ideas", 1, lambda v: v <= 4, "Generating fewer ideas of your own before using AI contributed to a higher predicted reliance level."),
    ("Independent_Confidence", -1, lambda v: v >= 4, "Your independent confidence contributed in the opposite direction, toward a lower predicted reliance level."),
    ("Independent_Confidence", 1, lambda v: v <= 4, "Lower confidence in independent reasoning contributed to a higher predicted reliance level."),
    ("Overreliance_Problem_Solving", 1, lambda v: v >= 4, "Leaning on AI for problem solving contributed to a higher predicted reliance level."),
    ("Overreliance_Problem_Solving", -1, lambda v: v <= 4, "Solving problems more independently contributed to a lower predicted reliance level."),
]


def plain_sentence(feat, shap_value, value):
    direction = 1 if shap_value > 0 else -1
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        for f, d, cond, text in EXPLAIN_RULES:
            if f == feat and d == direction:
                try:
                    if cond(float(value)):
                        return text
                except Exception:
                    pass
    word = "higher" if direction > 0 else "lower"
    shown = f"{value:.2f}" if feat == "CGPA" and isinstance(value, (int, float, np.floating)) else esc(value)
    return f"Your answer for “{feature_label(feat)}” ({shown}) pushed the predicted score {word} relative to the model's baseline."


# ============================================================
# SESSION STATE HELPERS
# ============================================================

def init_state():
    st.session_state.setdefault("nav", "Home")
    st.session_state.setdefault("step", 0)
    st.session_state.setdefault("confirmed", set())
    st.session_state.setdefault("_sticky", {})
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("assess_error", None)
    st.session_state.setdefault("iv_started", False)


init_state()


def sticky(key, default):
    """Make a widget value survive page changes (Streamlit drops state of unrendered widgets)."""
    store = st.session_state["_sticky"]
    store.setdefault(key, default)
    if key not in st.session_state:
        st.session_state[key] = store[key]


def remember(key):
    st.session_state["_sticky"][key] = st.session_state.get(key)


def goto(page):
    st.session_state["nav"] = page


def change_step(delta):
    current = st.session_state["step"]
    if delta > 0 and current < len(ASSESS_STEPS):
        st.session_state["confirmed"].add(current)
    st.session_state["step"] = int(min(max(current + delta, 0), len(ASSESS_STEPS)))


def start_intervention():
    st.session_state["iv_started"] = True


def reset_intervention():
    store = st.session_state["_sticky"]
    for k in [k for k in store if k.startswith("iv_")]:
        store.pop(k, None)
        st.session_state.pop(k, None)
    st.session_state["iv_started"] = False


def collect_inputs():
    store = st.session_state["_sticky"]
    data = {}
    for feat in categorical_features:
        options = category_options[feat]
        value = store.get(f"in_{feat}", options[0])
        data[feat] = value if value in options else options[0]
    for feat in numeric_features:
        default = 3.30 if feat == "CGPA" else 4
        raw = store.get(f"in_{feat}", default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float(default)
        if not math.isfinite(value):
            value = float(default)
        if feat == "CGPA":
            value = min(max(value, 2.31), 4.00)
        else:
            value = min(max(value, 1.0), 7.0)
        data[feat] = value
    return data


def reset_assessment():
    store = st.session_state["_sticky"]
    for k in [k for k in store if k.startswith("in_")]:
        store.pop(k, None)
        st.session_state.pop(k, None)
    st.session_state["confirmed"] = set()
    st.session_state["step"] = 0
    st.session_state["result"] = None
    st.session_state["assess_error"] = None
    reset_intervention()


def jump_step(index):
    st.session_state["step"] = int(index)


def run_assessment():
    if len(st.session_state["confirmed"]) < len(ASSESS_STEPS):
        st.session_state["assess_error"] = (
            "Please confirm all five sections before submitting, so that no answer is a silent default."
        )
        return
    try:
        input_df = pd.DataFrame([collect_inputs()])
        score = float(model.predict(input_df)[0])
        if not math.isfinite(score):
            raise ValueError("Non-finite prediction")
        risk = classify_risk(score)
        name, trigger = get_intervention_with_trigger(input_df.iloc[0], score)
        reset_intervention()
        st.session_state["result"] = {
            "input_df": input_df,
            "score": score,
            "risk": risk,
            "intervention": name,
            "trigger": trigger,
            "shap": None,
        }
        st.session_state["assess_error"] = None
        goto("Prediction")
    except Exception:
        logger.exception("Prediction failed")
        st.session_state["assess_error"] = (
            "Something went wrong while generating the prediction. "
            "Please review your answers and try again."
        )


# ============================================================
# UI HELPERS
# ============================================================

def card(title, body, icon=""):
    icon_html = ""
    return f'<div class="card">{icon_html}<h4>{title}</h4><p>{body}</p></div>'


def stat(value, label, sub="", color=None):
    style = f' style="color:{color}"' if color else ""
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    return f'<div class="stat"><div class="v"{style}>{value}</div><div class="l">{label}</div>{sub_html}</div>'


def section_title(title, sub=""):
    md_html(f'<div class="section-title">{title}</div>' + (f'<div class="section-sub">{sub}</div>' if sub else ""))


def notice(text, kind="info"):
    md_html(f'<div class="notice {kind}">{text}</div>')


def themed_fig(width=7.5, height=4.0):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(C_CARD)
    ax.set_facecolor(C_CARD)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=9, length=0)
    ax.grid(color="#D3E2F5", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)
    ax.title.set_color(C_TEXT)
    return fig, ax


def show_fig(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def chart_card(question):
    """Bordered container with the question the chart answers."""
    box = st.container(border=True)
    box.markdown(f"**{question}**")
    return box


def gauge_svg(score, color):
    radius = 88
    circumference = 2 * math.pi * radius
    frac = min(max((score - 1) / 6, 0), 1)
    dash = circumference * frac
    return f"""
    <svg width="250" height="250" viewBox="0 0 240 240" role="img" aria-label="AI reliance score {score:.2f} out of 7">
    <defs><linearGradient id="gg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3478F6"/><stop offset="1" stop-color="{color}"/></linearGradient></defs>
    <circle cx="120" cy="120" r="{radius}" fill="none" stroke="#D9E6F7" stroke-width="16"/>
    <circle cx="120" cy="120" r="{radius}" fill="none" stroke="url(#gg)" stroke-width="16" stroke-linecap="round" stroke-dasharray="{dash:.1f} {circumference:.1f}" transform="rotate(-90 120 120)"/>
    <text x="120" y="76" text-anchor="middle" font-family="Inter,Segoe UI,sans-serif" font-size="11" font-weight="700" letter-spacing="2" fill="#425B82">ESTIMATED SCORE</text>
    <text x="120" y="130" text-anchor="middle" font-family="Inter,Segoe UI,sans-serif" font-size="52" font-weight="800" fill="#142B55">{score:.2f}</text>
    <text x="120" y="158" text-anchor="middle" font-family="Inter,Segoe UI,sans-serif" font-size="16" fill="#425B82">out of 7</text>
    </svg>
    """


def scale_html(score):
    pct = min(max((score - 1) / 6 * 100, 0), 100)
    p_low = (low_threshold - 1) / 6 * 100
    p_high = (high_threshold - 1) / 6 * 100
    return f"""
    <div class="scale">
    <div class="scale-track">
    <div class="seg" style="width:{p_low:.2f}%;background:linear-gradient(90deg,rgba(52,120,246,.45),rgba(34,197,94,.45))"></div>
    <div class="seg" style="width:{p_high - p_low:.2f}%;background:linear-gradient(90deg,rgba(52,120,246,.45),rgba(52,120,246,.5))"></div>
    <div class="seg" style="width:{100 - p_high:.2f}%;background:linear-gradient(90deg,rgba(52,120,246,.5),rgba(245,158,11,.55))"></div>
    <div class="marker" style="left:{pct:.2f}%"><span>{score:.2f}</span></div>
    </div>
    <div class="ticks">
    <span style="left:0%">1</span><span style="left:{p_low:.2f}%">{low_threshold:g}</span>
    <span style="left:{p_high:.2f}%">{high_threshold:g}</span><span style="left:100%">7</span>
    </div>
    <div class="zones">
    <span style="width:{p_low:.2f}%">Low</span><span style="width:{p_high - p_low:.2f}%">Moderate</span><span style="width:{100 - p_high:.2f}%">High</span>
    </div>
    </div>
    """


def stage_row(active):
    parts = []
    for i, stage_name in enumerate(STAGES):
        cls = "stage active" if stage_name == active else "stage"
        parts.append(f'<div class="{cls}">{stage_name}</div>')
        if i < len(STAGES) - 1:
            parts.append('<div class="stage-arrow">→</div>')
    return f'<div class="stages">{"".join(parts)}</div>'


def need_result_guard():
    if st.session_state.get("result") is None:
        md_html(
            """
            <div class="card"><div class="icon">🧪</div><h4>No assessment yet</h4>
            <p>Complete the short assessment first. This page will then show your personalised result.</p></div>
            """
        )
        st.button("Go to Assessment", type="primary", on_click=goto, args=("Assess",), key=f"guard_{st.session_state['nav']}")
        return True
    return False


@st.cache_data(show_spinner=False)
def load_optional_csv(filename):
    path = BASE_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def missing_data_card(title, filename, needed):
    notice(
        f"<b>{title}</b><br>This view needs real held-out results, and none are bundled with the app, "
        f"so nothing is shown rather than inventing numbers. Export <code>{filename}</code> "
        f"from your notebook ({needed}) into the app folder to enable it.",
        "info",
    )


# ============================================================
# PAGES
# ============================================================

_ART_DEFS = '<defs><linearGradient id="ag" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3478F6"/><stop offset="1" stop-color="#8757E8"/></linearGradient><linearGradient id="abg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#E5F1FF"/><stop offset="1" stop-color="#F0E7FF"/></linearGradient></defs>'
_ART_HEAD = '<svg viewBox="0 0 300 190" xmlns="http://www.w3.org/2000/svg" role="img" aria-hidden="true">' + _ART_DEFS + '<rect width="300" height="190" fill="url(#abg)"/>'

ART = {
    "collab": _ART_HEAD + '''
<circle cx="150" cy="100" r="76" fill="#109A8A" opacity=".07"/>
<circle cx="76" cy="84" r="21" fill="#142B55" opacity=".92"/><path d="M40 156 Q76 102 112 156 Z" fill="url(#ag)"/>
<polygon points="226,64 254,80 254,112 226,128 198,112 198,80" fill="#FFFFFF" stroke="url(#ag)" stroke-width="3"/>
<text x="226" y="102" text-anchor="middle" font-family="Inter,Segoe UI,sans-serif" font-size="18" font-weight="800" fill="#109A8A">AI</text>
<path d="M112 104 Q154 70 198 96" stroke="#109A8A" stroke-width="2" stroke-dasharray="4 5" fill="none"><animate attributeName="stroke-dashoffset" from="18" to="0" dur="1.5s" repeatCount="indefinite"/></path>
<circle cx="154" cy="86" r="5" fill="#3478F6"/><circle cx="154" cy="38" r="12" fill="none" stroke="#2B63D6" stroke-width="2.5"/><path d="M149 52 h10" stroke="#2B63D6" stroke-width="3"/>
</svg>''',
    "think": _ART_HEAD + '''
<circle cx="150" cy="86" r="62" fill="#8757E8" opacity=".08"/>
<circle cx="150" cy="80" r="30" fill="none" stroke="#2B63D6" stroke-width="3"/><path d="M138 116 h24 M141 124 h18" stroke="#2B63D6" stroke-width="3" stroke-linecap="round"/>
<path d="M150 28 v-10 M108 44 l-8 -7 M192 44 l8 -7" stroke="#2B63D6" stroke-width="2.5" stroke-linecap="round"/>
<g stroke="#3478F6" stroke-width="1.6" opacity=".8"><line x1="150" y1="80" x2="70" y2="140"/><line x1="150" y1="80" x2="230" y2="140"/><line x1="70" y1="140" x2="150" y2="162"/><line x1="230" y1="140" x2="150" y2="162"/></g>
<circle cx="70" cy="140" r="9" fill="#3478F6"/><circle cx="230" cy="140" r="9" fill="#8757E8"/><circle cx="150" cy="162" r="9" fill="#109A8A"/>
</svg>''',
    "gauge": _ART_HEAD + '''
<path d="M70 150 A80 80 0 0 1 129.3 72.7" stroke="#16A34A" stroke-width="15" fill="none" opacity=".85"/>
<path d="M129.3 72.7 A80 80 0 0 1 190 80.7" stroke="#3478F6" stroke-width="15" fill="none" opacity=".9"/>
<path d="M190 80.7 A80 80 0 0 1 230 150" stroke="#D97706" stroke-width="15" fill="none" opacity=".85"/>
<line x1="150" y1="150" x2="158" y2="88" stroke="#142B55" stroke-width="3" stroke-linecap="round"/><circle cx="150" cy="150" r="8" fill="#142B55"/>
<g font-family="Inter,Segoe UI,sans-serif" font-size="11" fill="#425B82" text-anchor="middle"><text x="70" y="172">1</text><text x="150" y="176">Low · Moderate · High</text><text x="230" y="172">7</text></g>
</svg>''',
    "xai": _ART_HEAD + '''
<line x1="150" y1="22" x2="150" y2="170" stroke="#425B82" stroke-width="1.5"/>
<rect x="150" y="34" width="86" height="18" rx="5" fill="#D97706"/><rect x="150" y="62" width="58" height="18" rx="5" fill="#D97706" opacity=".85"/>
<rect x="150" y="90" width="34" height="18" rx="5" fill="#D97706" opacity=".7"/>
<rect x="76" y="118" width="74" height="18" rx="5" fill="#109A8A"/><rect x="104" y="146" width="46" height="18" rx="5" fill="#109A8A" opacity=".8"/>
<g font-family="Inter,Segoe UI,sans-serif" font-size="10" fill="#425B82"><text x="30" y="30">lower</text><text x="238" y="30">higher</text></g>
</svg>''',
    "path": _ART_HEAD + '''
<rect x="30" y="132" width="40" height="30" rx="5" fill="#B9C9E6"/><rect x="74" y="112" width="40" height="50" rx="5" fill="#B9C9E6"/>
<rect x="118" y="92" width="40" height="70" rx="5" fill="#3478F6" opacity=".8"/><rect x="162" y="72" width="40" height="90" rx="5" fill="#8757E8" opacity=".85"/>
<rect x="206" y="52" width="40" height="110" rx="5" fill="url(#ag)"/>
<circle cx="226" cy="36" r="6" fill="#109A8A"/><path d="M226 30 v-14 l14 5 l-14 5" fill="#2B63D6"/>
<path d="M50 120 L94 100 L138 80 L182 60 L226 42" stroke="#109A8A" stroke-width="2" stroke-dasharray="4 5" fill="none"/>
</svg>''',
    "radar": _ART_HEAD + '''
<polygon points="150,30 216.6,78.4 191.1,156.6 108.9,156.6 83.4,78.4" fill="none" stroke="#C8D9F0" stroke-width="1.5"/>
<polygon points="150,65 183.3,89.2 170.5,128.3 129.5,128.3 116.7,89.2" fill="none" stroke="#C8D9F0" stroke-width="1.2"/>
<polygon points="150,44 186.6,88.1 178.8,139.6 113,151 110.1,87" fill="#109A8A" fill-opacity=".2" stroke="#109A8A" stroke-width="2.2"/>
<g fill="#109A8A"><circle cx="150" cy="44" r="4"/><circle cx="186.6" cy="88.1" r="4"/><circle cx="178.8" cy="139.6" r="4"/><circle cx="113" cy="151" r="4"/><circle cx="110.1" cy="87" r="4"/></g>
</svg>''',
    "lab": _ART_HEAD + '''
<path d="M40 30 V160 H262" stroke="#425B82" stroke-width="1.6" fill="none"/>
<g fill="#3478F6" opacity=".9"><circle cx="70" cy="132" r="4"/><circle cx="95" cy="118" r="4"/><circle cx="118" cy="124" r="4"/><circle cx="142" cy="98" r="4"/><circle cx="170" cy="92" r="4"/><circle cx="198" cy="70" r="4"/><circle cx="226" cy="64" r="4"/></g>
<path d="M56 140 L240 56" stroke="#109A8A" stroke-width="2.2" stroke-dasharray="5 5" fill="none"/>
<rect x="196" y="116" width="14" height="44" fill="#8757E8" opacity=".8"/><rect x="216" y="100" width="14" height="60" fill="#8757E8" opacity=".8"/><rect x="236" y="130" width="14" height="30" fill="#8757E8" opacity=".8"/>
</svg>''',
}

ART_FOR_PAGE = {
    "Home": ("collab", "Human–AI collaboration", "AI as a learning assistant, with the student still doing the thinking."),
    "Assess": ("think", "Independent thinking", "Reflect on how you work with AI. Your answers stay in this session."),
    "Prediction": ("gauge", "From behaviour to estimate", "A 1–7 AI Reliance Score and a relative intervention level."),
    "Why This Result?": ("xai", "Explainable AI", "SHAP shows which behaviours moved the prediction up or down."),
    "Intervention": ("path", "Adaptive learning path", "From hints toward independent answers, one small step at a time."),
    "Student Profile": ("radar", "Behavioural insights", "Your self-reported habits across key learning dimensions."),
    "Model Insights": ("lab", "Research laboratory", "Model comparison, validation and explainability for researchers."),
    "Research": ("lab", "Research laboratory", "Datasets, methods, experiments and honest limitations."),
    "Testing": ("lab", "Testing and validation", "Documented test cases with expected and observed results."),
    "About": ("collab", "Human–AI collaboration", "Promoting responsible AI-assisted learning."),
}


def render_navbar(current):
    mode = "Home" if current == "Home" else ("Student" if current in STUDENT_SET else "Research")
    with st.container(key="topnav"):
        cols = st.columns([2.9, 0.9, 1.75, 2.0, 1.4], gap="small")
        with cols[0]:
            md_html(f'<div class="nav-brand"><div class="nav-logo">{LOGO_SVG}</div><div><div class="nav-name">AI Overreliance Framework</div>'
                    '<div class="nav-sub">Detect · Understand · Support · Learn</div></div></div>')
        with cols[1]:
            st.button("Home", icon=":material/home:", key="top_home", type="primary" if mode == "Home" else "secondary",
                      on_click=goto, args=("Home",))
        with cols[2]:
            st.button("Student Experience", icon=":material/school:", key="top_student", type="primary" if mode == "Student" else "secondary",
                      on_click=goto, args=("Assess",))
        with cols[3]:
            st.button("Research & Evaluation", icon=":material/bar_chart:", key="top_research", type="primary" if mode == "Research" else "secondary",
                      on_click=goto, args=("Model Insights",))
        with cols[4]:
            with st.container(key="nav_cta"):
                st.button("Get Started →", key="top_cta", on_click=goto, args=("Assess",))


def render_left_panel(current):
    with st.container(key="leftpanel"):
        md_html('<div class="panel-label">Student experience</div>')
        assessed = st.session_state.get("result") is not None
        for i, (page, icon, hint) in enumerate(JOURNEY):
            shown = "check_circle" if (page == "Assess" and assessed) else icon
            st.button(page, icon=f":material/{shown}:", key=f"side_{i}", help=hint,
                      type="primary" if current == page else "secondary",
                      on_click=goto, args=(page,))
        md_html('<div class="panel-divider"></div><div class="panel-label">Research &amp; evaluation</div>')
        for i, (page, icon, hint) in enumerate(RESEARCH_NAV):
            st.button(page, icon=f":material/{icon}:", key=f"rside_{i}", help=hint,
                      type="primary" if current == page else "secondary",
                      on_click=goto, args=(page,))
        md_html(f'<div class="side-art">{SIDE_ART}<p>Responsible AI for Brighter Learning</p></div>')


def render_right_panel(current):
    key, title, caption = ART_FOR_PAGE.get(current, ART_FOR_PAGE["Home"])
    with st.container(key="rightpanel"):
        md_html(f'<div class="art-card">{ART[key]}<div class="cap"><b>{title}</b><p>{caption}</p></div></div>')


# ---------------- HOME ----------------

def arch_html():
    return (
        '<div class="arch-branch"><div class="arch-label">Branch A · Exploratory behavioural profiling</div>'
        '<div class="flow"><div class="n">AIBT-50K<small>50,000 × 56</small></div><i>→</i>'
        '<div class="n">Behavioural Feature Construction<small>9 construct scores</small></div><i>→</i>'
        '<div class="n">K-Means (K = 2)<small>exploratory</small></div><i>→</i>'
        '<div class="n hl">Behavioural Profiles<small>weak separation</small></div></div></div>'
        '<div class="arch-branch"><div class="arch-label">Branch B · Supervised risk prediction</div>'
        '<div class="flow"><div class="n">Student Awareness Dataset<small>394 records</small></div><i>→</i>'
        '<div class="n">Preprocessing</div><i>→</i><div class="n">Feature Engineering</div><i>→</i>'
        '<div class="n hl">Random Forest Regression</div><i>→</i><div class="n">AI Reliance Score</div><i>→</i>'
        '<div class="n">Risk Level</div><i>→</i><div class="n">SHAP Explanation</div><i>→</i>'
        '<div class="n hl">Adaptive Intervention</div><i>→</i><div class="n">Student Reflection</div></div></div>'
        '<div class="arch-merge"><b>Behavioural Profile + Risk Prediction</b><div>↓</div>'
        '<b>Decision / Intervention Layer</b><div>The branches meet only at the conceptual decision layer. '
        'The same students are not linked across datasets, and there is no row-level merging.</div></div>'
    )


def hero_visual():
    path = BASE_DIR / "assets" / "hero_ai_learning.png"
    if path.exists():
        try:
            encoded = base64.b64encode(path.read_bytes()).decode()
            return f'<img src="data:image/png;base64,{encoded}" alt="Student learning with an AI assistant">'
        except Exception:
            pass
    return HERO_ART


FEATURES = [
    ("AI Reliance Prediction", "Estimate your AI reliance using an explainable machine learning model.", "Assess", "#CFE0FB",
     '<svg viewBox="0 0 24 24"><rect x="3" y="12" width="4" height="9" rx="1.2" fill="#3478F6"/><rect x="10" y="7" width="4" height="14" rx="1.2" fill="#3478F6"/><rect x="17" y="3" width="4" height="18" rx="1.2" fill="#3478F6"/></svg>'),
    ("Explainable AI", "See which factors influence your result with SHAP explanations.", "Why This Result?", "#E0D2FF",
     '<svg viewBox="0 0 24 24"><path d="M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z" fill="#8757E8"/><rect x="9" y="19" width="6" height="2.4" rx="1.2" fill="#8757E8"/></svg>'),
    ("Behavioural Insights", "Understand your AI usage patterns and key behavioural drivers.", "Student Profile", "#BFEBD8",
     '<svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3.4" fill="#109A8A"/><circle cx="17" cy="9" r="2.6" fill="#109A8A" opacity=".7"/><path d="M2.5 20c0-3.6 2.9-6 6.5-6s6.5 2.4 6.5 6z" fill="#109A8A"/><path d="M16 14.4c2.8.2 5.5 2 5.5 5.6h-4.2c0-2-.5-3.9-1.3-5.6z" fill="#109A8A" opacity=".7"/></svg>'),
    ("Adaptive Intervention", "Receive personalised, research-informed learning interventions.", "Intervention", "#FFD9BE",
     '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="none" stroke="#E8731F" stroke-width="2.4"/><circle cx="12" cy="12" r="5" fill="none" stroke="#E8731F" stroke-width="2.4"/><circle cx="12" cy="12" r="1.8" fill="#E8731F"/></svg>'),
]


def page_home():
    with st.container(key="homehero"):
        h_col, c_col = st.columns([3.1, 1.0], gap="medium")
        with h_col:
            md_html(
                f"""
                <div class="hero2">
                <div>
                <span class="hero-badge">⚗ Research Prototype · Decision-support framework</span>
                <h1>AI-Based Framework for Detecting <span class="hl">Generative AI</span> Overreliance and Providing Adaptive Interventions</h1>
                <div class="desc">An explainable machine learning framework for estimating AI reliance, understanding behavioural drivers,
                and delivering adaptive learning interventions.</div>
                <ul class="benefits">
                <li><span class="ck b">✓</span>Understand your AI reliance.</li>
                <li><span class="ck p">✓</span>Understand what influences it.</li>
                <li><span class="ck m">✓</span>Build stronger independent reasoning habits.</li>
                </ul>
                </div>
                <div><div class="hero-art">{hero_visual()}</div>
                <div class="hero-pills"><span class="pill-b">Predict</span><span class="pill-m">Explain</span><span class="pill-l">Intervene</span></div></div>
                </div>
                """
            )
        with c_col:
            md_html(
                f"""
                <div class="collab">{COLLAB_ART}
                <div class="cap"><h3>Human–AI Collaboration</h3>
                <p>AI as a learning assistant, with the student still doing the thinking.</p></div></div>
                """
            )
    st.write("")
    with st.container(key="actions"):
        a1, a2, a3 = st.columns(3, gap="medium")
        with a1:
            with st.container(key="act1"):
                st.button("Try the assessment →", icon=":material/assignment:", on_click=goto, args=("Assess",), key="home_start")
        with a2:
            with st.container(key="act2"):
                st.button("Explore model results", icon=":material/bar_chart:", on_click=goto, args=("Model Insights",), key="home_models")
        with a3:
            with st.container(key="act3"):
                st.button("Understand the research", icon=":material/menu_book:", on_click=goto, args=("Research",), key="home_research")

    section_title("What the framework offers",
                  "A complete research-based framework to understand, explain and address AI overreliance. "
                  "Insights come from self-reported questionnaire answers, not from analysis of your actual AI prompts.")
    with st.container(key="features"):
        cols = st.columns(4, gap="medium")
        for i, (col, (title, desc, page, bg, icon)) in enumerate(zip(cols, FEATURES)):
            with col:
                with st.container(key=f"feat{i}"):
                    md_html(f'<div class="feat-ic" style="background:{bg}">{icon}</div><div class="feat-title">{title}</div><div class="feat-desc">{desc}</div>')
                    st.button("→", key=f"feat_btn_{i}", help=f"Open: {page}", on_click=goto, args=(page,))

    section_title("Student experience", "Five steps from self-report to a personalised learning activity.")
    steps = [
        ("Student Assessment", "Answer short, grouped questions about your AI use."),
        ("Feature Processing", "Answers are scaled and encoded by the saved pipeline."),
        ("ML Prediction", "A Random Forest estimates your AI Reliance Score."),
        ("Explainable Analysis", "SHAP highlights the behaviours behind the estimate."),
        ("Adaptive Intervention", "A rule-based activity encourages independent reasoning."),
    ]
    parts = []
    for i, (title, text) in enumerate(steps, 1):
        parts.append(f'<div class="step"><div class="num">{i}</div><h5>{title}</h5><p>{text}</p></div>')
        if i < len(steps):
            parts.append('<div class="arrow">→</div>')
    md_html(f'<div class="steps">{"".join(parts)}</div>')

    section_title("Research architecture", "Two analytically separate datasets that meet only at the conceptual decision layer.")
    md_html(arch_html())

    section_title("Research evaluation", "Reported from the project's own experiments on the Student Awareness Dataset.")
    md_html(
        '<div class="grid g4">'
        + stat("394", "Cleaned student records", "after removing duplicates")
        + stat("22", "Input features", "18 numeric · 4 categorical")
        + stat("0.254", "Held-out R²", "Random Forest explains ≈ 25% of variance", C_CYAN)
        + stat("0.769", "Held-out MAE", "on a 1–7 target scale", C_BLUE)
        + "</div>"
    )


# ---------------- ASSESS ----------------

def render_field(feat):
    key = f"in_{feat}"
    label, tip = FEATURE_META.get(feat, (feat.replace("_", " "), ""))
    if feat in categorical_features:
        options = category_options[feat]
        sticky(key, options[0])
        if st.session_state.get(key) not in options:
            st.session_state[key] = options[0]
        st.selectbox(label, options, key=key, help=tip, format_func=str, on_change=remember, args=(key,))
    elif feat == "CGPA":
        sticky(key, 3.30)
        st.number_input(label, min_value=2.31, max_value=4.00, step=0.01, format="%.2f",
                        key=key, help=tip, on_change=remember, args=(key,))
    else:
        sticky(key, 4)
        st.select_slider(label, options=[1, 2, 3, 4, 5, 6, 7], key=key, help=tip,
                         format_func=lambda v: f"{v} · {LIKERT[v]}", on_change=remember, args=(key,))
    remember(key)


def legend_html():
    return '<div class="stepper">' + "".join(f'<div class="chip">{k} · {v}</div>' for k, v in LIKERT.items()) + "</div>"


def display_answer(feat):
    store = st.session_state["_sticky"]
    key = f"in_{feat}"
    if feat in categorical_features:
        return esc(store.get(key, category_options[feat][0]))
    value = store.get(key, 3.30 if feat == "CGPA" else 4)
    try:
        if feat == "CGPA":
            return f"{float(value):.2f}"
        v = int(round(float(value)))
        return f"{v} · {LIKERT.get(v, '')}"
    except Exception:
        return esc(value)


def render_review():
    confirmed = st.session_state["confirmed"]
    notice(
        "<b>How your answers are used.</b> They are used only to compute this estimate in your current browser session. "
        "This application does not write them to a file or database, and refreshing the page clears them.",
        "info",
    )
    st.write("")
    for i, (short, name, _, feats) in enumerate(ASSESS_STEPS):
        if i in confirmed:
            badge = '<span class="badge" style="color:#15803D;border-color:rgba(34,197,94,.45)">Confirmed</span>'
        else:
            badge = '<span class="badge" style="color:#B45309;border-color:rgba(245,158,11,.55)">Not confirmed · neutral defaults</span>'
        rows = "".join(f"<tr><td>{esc(feature_label(f))}</td><td>{display_answer(f)}</td></tr>" for f in feats)
        md_html(
            f'<div class="card" style="margin-bottom:12px"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
            f'<h4 style="margin:0">{i + 1}. {name}</h4>{badge}</div>'
            f'<table class="rtable"><tr><th>Question</th><th>Your answer</th></tr>{rows}</table></div>'
        )
    st.caption("Edit a section:")
    edit_cols = st.columns(len(ASSESS_STEPS))
    for i, col in enumerate(edit_cols):
        with col:
            st.button(f"Edit {i + 1}", on_click=jump_step, args=(i,), key=f"review_edit_{i}")

    all_ok = len(confirmed) == len(ASSESS_STEPS)
    if not all_ok:
        missing = ", ".join(str(i + 1) for i in range(len(ASSESS_STEPS)) if i not in confirmed)
        notice(f"Section(s) {missing} have not been confirmed, so they still hold the neutral default. "
               "Open each one and press Next before submitting.", "warn")
    if st.session_state.get("assess_error"):
        notice(st.session_state["assess_error"], "warn")
    st.write("")
    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        st.button("← Back", on_click=change_step, args=(-1,), key="assess_back")
    with b2:
        st.button("Start over", on_click=reset_assessment, key="assess_reset")
    with b3:
        st.button("Assess My AI Reliance", type="primary", on_click=run_assessment,
                  key="assess_go", disabled=not all_ok)


def page_assess():
    n = len(ASSESS_STEPS)
    step = min(st.session_state["step"], n)
    confirmed = st.session_state["confirmed"]
    review = step >= n
    chips = []
    for i, (short, _, _, _) in enumerate(ASSESS_STEPS):
        done = i in confirmed
        cls = "chip active" if i == step else ("chip done" if done else "chip")
        chips.append(f'<div class="{cls}">{"✓ " if done else ""}{i + 1}. {short}</div>')
    chips.append(f'<div class="{"chip active" if review else "chip"}">Review</div>')

    md_html(
        f"""
        <div class="eyebrow">Assessment</div>
        <div class="section-title" style="margin-top:0">Tell us how you work with AI</div>
        <div class="section-sub">Rate each statement from 1 (very low) to 7 (very high). Hover over the ? icons for plain-English explanations.
        There are no right or wrong answers. Each section must be confirmed with Next before you can submit.</div>
        <div class="stepper">{"".join(chips)}</div>
        """
    )
    if review:
        st.progress(1.0, text="Final review · confirm your answers before submitting")
        render_review()
        return

    progress = int(round((step + 1) / n * 100))
    st.progress((step + 1) / n, text=f"Step {step + 1} of {n} · Assessment progress: {progress}%")

    _, section_name, section_sub, features = ASSESS_STEPS[step]
    md_html(f'<div class="card tight"><h4>{section_name}</h4><p>{section_sub}</p></div>')
    if step > 0:
        md_html(f'<div style="margin-top:10px"><span class="eyebrow">Scale</span>{legend_html()}</div>')
    st.write("")

    cols = st.columns(2, gap="large")
    for i, feat in enumerate(features):
        with cols[i % 2]:
            render_field(feat)

    st.write("")
    back_col, _, next_col = st.columns([1, 2, 1.3])
    with back_col:
        if step > 0:
            st.button("← Back", on_click=change_step, args=(-1,), key="assess_back")
    with next_col:
        st.button("Review answers →" if step == n - 1 else "Next →", type="primary",
                  on_click=change_step, args=(1,), key="assess_next")


# ---------------- PREDICTION ----------------

def page_prediction():
    if need_result_guard():
        return
    res = st.session_state["result"]
    score, risk = res["score"], res["risk"]
    meta = RISK_META[risk]

    section_title("Your Estimated AI Reliance Score", "A decision-support estimate from self-reported behaviour.")
    notice("Low, Moderate and High are <b>relative intervention levels</b> derived from training-set thresholds. They are not validated diagnoses or definitive labels of a student, and the score is a regression estimate on a 1–7 scale, not a probability.", "info")
    st.write("")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        md_html(
            f"""
            <div class="card gauge-wrap">
            {gauge_svg(score, meta["color"])}
            <div class="risk-pill" style="background:{meta['color']}22;color:{meta['color']};border:1px solid {meta['color']}66">{meta['label']}</div>
            </div>
            """
        )
    with right:
        md_html(
            f"""
            <div class="card">
            <div class="eyebrow">Interpretation</div>
            <h4 style="font-size:1.2rem">{meta['level']}</h4>
            <p style="color:#425B82;font-size:.95rem">{meta['message']}</p>
            </div>
            """
        )

    section_title("Where you sit on the scale")
    md_html(f'<div class="card">{scale_html(score)}'
            f'<p>Thresholds ({low_threshold:g} and {high_threshold:g}) come from training-set tertiles. '
            f'They describe <b>relative intervention levels</b>, not clinical or diagnostic categories.</p></div>')

    section_title("How to read this result")
    md_html(
        '<div class="grid g2">'
        + card("Prediction confidence and limitations",
               "Model performance is evaluated at the population level. This individual prediction should be interpreted "
               "as an estimate, not a diagnosis. The score is not a probability or a calibrated confidence value, and no individual confidence interval is produced.", "⚖️")
        + card("What the model can and cannot do",
               "It is trained on self-reported data and has moderate predictive performance. The Low / Moderate / High "
               "labels are a post-hoc layer on a regression output, and recall for the High level is limited.", "🔬")
        + "</div>"
    )
    st.write("")
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        st.button("Why this result?", type="primary", on_click=goto, args=("Why This Result?",), key="pred_why")
    with c2:
        st.button("View my intervention", on_click=goto, args=("Intervention",), key="pred_iv")


# ---------------- WHY THIS RESULT ----------------

def shap_bar_chart(agg, n=10):
    top = agg.head(n).sort_values("SHAP_Value")
    fig, ax = themed_fig(8, 0.46 * len(top) + 1.4)
    colors = [C_AMBER if v > 0 else C_CYAN for v in top["SHAP_Value"]]
    ax.barh(top["Label"], top["SHAP_Value"], color=colors, height=0.62)
    ax.axvline(0, color=C_MUTED, linewidth=1)
    ax.grid(axis="y", visible=False)
    span = max(top["SHAP_Value"].abs().max(), 1e-6)
    for y, v in enumerate(top["SHAP_Value"]):
        ax.text(v + (0.03 * span if v >= 0 else -0.03 * span), y, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", color=C_TEXT, fontsize=8.5)
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_xlabel("Contribution to predicted score  (← lower reliance | higher reliance →)")
    ax.set_title("Which behaviours moved your prediction?", fontsize=11, loc="left", fontweight="bold")
    ax.legend(handles=[Patch(color=C_AMBER, label="Pushes prediction higher"), Patch(color=C_CYAN, label="Pushes prediction lower")],
              loc="lower right", facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT, fontsize=8)
    show_fig(fig)


def page_why():
    if need_result_guard():
        return
    res = st.session_state["result"]
    section_title("Why did the model predict this?", "Each bar shows how much a behaviour moved your score away from the model's baseline.")

    if res["shap"] is None:
        try:
            with st.spinner("Generating explanation…"):
                raw = generate_shap_explanation(res["input_df"])
                agg = aggregate_shap(raw, res["input_df"].iloc[0])
                res["shap"] = {"raw": raw, "agg": agg, "baseline": get_baseline()}
        except Exception:
            logger.exception("SHAP explanation failed")
            res["shap"] = False
    shap_res = res["shap"]

    if shap_res is False:
        notice("Something went wrong while generating the explanation. The prediction is still available.", "warn")
        st.button("Back to prediction", on_click=goto, args=("Prediction",), key="why_back")
        return

    raw, agg, baseline = shap_res["raw"], shap_res["agg"], shap_res["baseline"]
    row = res["input_df"].iloc[0]

    metrics = [stat(f"{res['score']:.2f}", "Predicted score", "your result", C_CYAN)]
    if baseline is not None:
        metrics.append(stat(f"{baseline:.2f}", "Baseline (expected) score", "average model output", C_BLUE))
        metrics.append(stat(f"{res['score'] - baseline:+.2f}", "Difference from baseline", "sum of all contributions", C_PURPLE))
    md_html(f'<div class="grid g3">{"".join(metrics)}</div>')
    if baseline is not None:
        recon = baseline + float(agg["SHAP_Value"].sum())
        if abs(recon - res["score"]) <= 1e-3:
            st.caption(f"Consistency check passed: baseline {baseline:.3f} + summed contributions "
                       f"{agg['SHAP_Value'].sum():+.3f} = {recon:.3f}, matching the model output {res['score']:.3f}.")
        else:
            notice("The explanation does not reconstruct the model output within tolerance, so treat it with caution.", "warn")
    st.write("")

    inc = agg[agg["SHAP_Value"] > 0].head(3)
    dec = agg[agg["SHAP_Value"] < 0].head(3)

    def factor_html(frame, color):
        if frame.empty:
            return '<p>No factors in this direction.</p>'
        out = []
        for _, r in frame.iterrows():
            out.append(
                f'<div class="factor"><div class="dot" style="background:{color}"></div><div>'
                f'<div class="ft">{esc(r["Label"])} <span style="color:{color}">({r["SHAP_Value"]:+.2f})</span></div>'
                f'<div class="fd">{plain_sentence(r["Source"], r["SHAP_Value"], r["Value"])}</div></div></div>'
            )
        return "".join(out)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        md_html(f'<div class="card"><h4 style="color:{C_AMBER}">▲ Factors increasing predicted reliance</h4>{factor_html(inc, C_AMBER)}</div>')
    with c2:
        md_html(f'<div class="card"><h4 style="color:{C_CYAN}">▼ Factors reducing predicted reliance</h4>{factor_html(dec, C_CYAN)}</div>')

    st.write("")
    with chart_card("Which factors influenced this prediction most?"):
        shap_bar_chart(agg)
        st.caption("Amber bars push the prediction toward higher reliance; cyan bars push it toward lower reliance. "
                   "One-hot categories (e.g. field of study) are combined into one bar.")

    notice(
        "These are <b>influential predictors</b> within the model, not causes. A factor that pushed the estimate "
        "up does not mean it causes AI dependence.", "info")

    with st.expander("Technical SHAP explanation (for researchers)"):
        st.markdown(
            f"SHAP values are computed with {SHAP_BACKEND} (path-dependent TreeSHAP) on the saved Random Forest and are additive: "
            "baseline + Σ SHAP ≈ predicted score."
        )
        if baseline is not None:
            total = baseline + float(raw["SHAP_Value"].sum())
            st.markdown(f"Baseline **{baseline:.4f}** + Σ SHAP **{raw['SHAP_Value'].sum():+.4f}** = **{total:.4f}** "
                        f"(model prediction **{res['score']:.4f}**).")
        view = raw.head(15).copy()
        view["Feature"] = view["Feature"].map(lambda f: re.sub(r"^(numeric|categorical)__", "", str(f)))
        view = view[["Feature", "SHAP_Value", "Absolute_SHAP"]].round(4)
        st.markdown(
            "<table class='rtable'><tr><th>Encoded feature</th><th>SHAP</th><th>|SHAP|</th></tr>"
            + "".join(f"<tr><td>{esc(r.Feature)}</td><td>{r.SHAP_Value:+.4f}</td><td>{r.Absolute_SHAP:.4f}</td></tr>"
                      for r in view.itertuples())
            + "</table>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.button("Continue to my intervention →", type="primary", on_click=goto, args=("Intervention",), key="why_iv")


# ---------------- INTERVENTION ----------------

def page_intervention():
    if need_result_guard():
        return
    res = st.session_state["result"]
    name = res["intervention"]
    risk = res["risk"]
    meta = RISK_META[risk]
    lib = INTERVENTION_LIBRARY.get(name, {"stage": "Question", "skills": [], "steps": []})

    section_title("Your Adaptive Learning Intervention", "A rule-based activity matched to your level and your behaviour pattern.")
    md_html(
        f"""
        <div class="flowrow">
        <div class="node"><div class="k">Risk level</div><div class="t" style="color:{meta['color']}">{meta['label']}</div><div class="d">{meta['level']}</div></div>
        <div class="arrow" style="align-self:center">→</div>
        <div class="node"><div class="k">Behaviour trigger</div><div class="t">{esc(res['trigger'])}</div></div>
        <div class="arrow" style="align-self:center">→</div>
        <div class="node"><div class="k">Recommended intervention</div><div class="t">{esc(name)}</div></div>
        </div>
        """
    )
    st.write("")

    skills = "".join(f'<span class="badge cyan" style="margin-right:6px">{esc(s)}</span>' for s in lib["skills"])
    md_html(
        f"""
        <div class="iv-card">
        <div class="eyebrow">Recommended activity</div>
        <h2>{esc(name)}</h2>
        <div class="why">{esc(get_intervention_description(name))}</div>
        <div style="margin-top:14px">{skills}</div>
        </div>
        """
    )
    st.write("")

    steps = lib["steps"]
    objective, example = INTERVENTION_EXTRAS.get(name, ("Practise independent reasoning with AI as a support.", ""))
    md_html(
        f'<div class="card"><h4>Objective</h4><p style="color:#425B82">{esc(objective)}</p>'
        + (f'<p style="margin-top:10px"><b style="color:#1F3A68">Example task.</b> {esc(example)}</p>' if example else "")
        + "</div>"
    )
    st.write("")
    if not st.session_state.get("iv_started"):
        md_html(
            '<div class="card"><h4>Steps in this activity</h4><p>'
            + "<br>".join(f"<b>Step {i}.</b> {esc(s)}" for i, s in enumerate(steps, 1))
            + "</p></div>"
        )
        st.write("")
        st.button("Start Intervention", type="primary", on_click=start_intervention, key="iv_start")
    else:
        st.markdown("#### Work through the activity")
        done = 0
        for i, text in enumerate(steps, 1):
            key = f"iv_step_{i}"
            sticky(key, False)
            checked = st.checkbox(f"Step {i}: {text}", key=key, on_change=remember, args=(key,))
            remember(key)
            done += int(bool(checked))
        answer_key, reflect_key = "iv_answer", "iv_reflection"
        sticky(answer_key, "")
        sticky(reflect_key, "")
        st.text_area("Your independent answer (written before comparing with any AI output)", key=answer_key,
                     height=120, on_change=remember, args=(answer_key,),
                     placeholder="Write your own answer or reasoning here…")
        remember(answer_key)
        st.text_area("Final reflection: what did you notice about your own thinking and the AI output?", key=reflect_key,
                     height=100, on_change=remember, args=(reflect_key,),
                     placeholder="For example: where did your answer differ, and what will you do differently next time?")
        remember(reflect_key)
        has_answer = bool(str(st.session_state.get(answer_key, "")).strip())
        has_reflection = bool(str(st.session_state.get(reflect_key, "")).strip())
        total_parts = len(steps) + 2
        completed = done + int(has_answer) + int(has_reflection)
        st.progress(completed / total_parts,
                    text=f"Activity progress: {done} of {len(steps)} steps · answer {'written' if has_answer else 'pending'} · reflection {'written' if has_reflection else 'pending'}")
        if completed == total_parts:
            notice("✅ Activity completed. This only records that you finished the task. "
                   "This prototype does not measure whether the activity improves learning or reduces overreliance.", "ok")
        st.caption("Your text stays in this browser session only and is cleared when you refresh the page.")
        st.button("Reset activity", on_click=reset_intervention, key="iv_reset")

    section_title("Intervention progression", "The activity above sits at the highlighted stage of the framework.")
    md_html(f'<div class="card">{stage_row(lib["stage"])}</div>')
    st.write("")
    md_html(
        """
        <div class="quote">The goal is not to remove AI from learning. The goal is to help students use AI
        without replacing independent reasoning.</div>
        """
    )
    st.caption("Interventions are rule-based prototype recommendations. The stage mapping is a design choice of this prototype; "
               "longitudinal experiments are needed to test whether they reduce overreliance.")
    st.button("View my profile →", on_click=goto, args=("Student Profile",), key="iv_profile")


# ---------------- STUDENT PROFILE ----------------

def radar_chart(values):
    labels = list(values.keys())
    vals = list(values.values())
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals += vals[:1]
    angles += angles[:1]
    fig = plt.figure(figsize=(5.6, 5.0))
    ax = fig.add_subplot(111, polar=True)
    fig.patch.set_facecolor(C_CARD)
    ax.set_facecolor(C_CARD)
    ax.set_ylim(0, 7)
    ax.set_yticks([1, 3, 5, 7])
    ax.set_yticklabels(["1", "3", "5", "7"], color=C_MUTED, fontsize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=C_TEXT, fontsize=8.5)
    ax.grid(color="#D3E2F5")
    ax.spines["polar"].set_color("#D3E2F5")
    ax.plot(angles, vals, color=C_CYAN, linewidth=2)
    ax.fill(angles, vals, color=C_CYAN, alpha=0.18)
    show_fig(fig)


def page_profile():
    if need_result_guard():
        return
    res = st.session_state["result"]
    row = res["input_df"].iloc[0]
    meta = RISK_META[res["risk"]]

    section_title("Your Behavioural Summary", "Based only on the answers you entered in this session.")
    md_html(
        '<div class="grid g5">'
        + stat(f"{res['score']:.2f}", "AI Reliance", meta["label"], meta["color"])
        + stat(f"{row['Cross_Check']:.0f}/7", "AI Verification", "Cross Check rating", C_BLUE)
        + stat(f"{row['Independent_Confidence']:.0f}/7", "Independent Confidence", "self-rated", C_CYAN)
        + stat(f"{row['Learning_Satisfaction']:.0f}/7", "Learning Satisfaction", "self-rated", C_PURPLE)
        + stat(esc(row["AI_Usage_Frequency"]), "AI Usage Frequency", "as selected")
        + "</div>"
    )
    st.write("")

    dims = {
        "Own Ideas": row["Own_Ideas"],
        "Cross Check": row["Cross_Check"],
        "Accept w/o Verif.": row["Accept_Without_Verification"],
        "Independent Confidence": row["Independent_Confidence"],
        "Learning Satisfaction": row["Learning_Satisfaction"],
        "Learning Efficiency": row["Learning_Efficiency"],
        "Collaborative Learning": row["Collaborative_Learning"],
    }
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        with chart_card("What does my behavioural pattern look like?"):
            radar_chart({k: float(v) for k, v in dims.items()})
            st.caption("Self-reported ratings on a 1–7 scale.")
    with c2:
        bars = []
        for k, v in dims.items():
            bars.append(
                f'<div class="mb-row"><div class="top"><span>{k}</span><span>{float(v):.0f} / 7</span></div>'
                f'<div class="minibar"><div style="width:{float(v) / 7 * 100:.0f}%"></div></div></div>'
            )
        md_html(f'<div class="card"><h4>Dimension scores</h4><p style="margin-bottom:14px">Higher is not always better; these describe your own habits.</p>{"".join(bars)}</div>')

    section_title("Research dataset behavioural profiles")
    md_html(
        """
        <div class="card"><h4>Exploratory K-Means profiling (AIBT-50K)</h4>
        <p>In the research dataset, K-Means (K = 2) was used for <b>exploratory behavioural profiling</b> on nine constructs:
        Access, Understand, Apply, Analyze, Evaluate, Co-Create, Govern, AI Trust and Metacognition. Cluster separation was weak,
        so these profiles are not an overreliance classifier. Your current profile is <b>not</b> assigned to a cluster because this
        assessment does not collect the AIBT questionnaire inputs.</p></div>
        """
    )
    st.write("")
    st.button("Explore model insights →", on_click=goto, args=("Model Insights",), key="prof_mi")


# ---------------- MODEL INSIGHTS ----------------

def importance_df():
    names = preprocessor.get_feature_names_out()
    imp = np.asarray(rf_model.feature_importances_)
    df = pd.DataFrame({"Source": [base_feature(n) for n in names], "Importance": imp})
    df = df.groupby("Source", as_index=False)["Importance"].sum()
    df["Label"] = df["Source"].map(feature_label)
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def reference_analysis():
    """Optional: SHAP summary + risk/intervention distributions from reference_data.csv."""
    ref = load_optional_csv("reference_data.csv")
    needed = numeric_features + categorical_features
    if ref is None or any(c not in ref.columns for c in needed):
        return None
    ref = ref[needed].dropna().head(400).reset_index(drop=True)
    if ref.empty:
        return None
    preds = model.predict(ref)
    risks = [classify_risk(p) for p in preds]
    ivs = [get_intervention(ref.iloc[i], preds[i]) for i in range(len(ref))]
    mean_abs = None
    if shap is not None:  # the all-student SHAP summary is only computed when the shap package is available
        processed = preprocessor.transform(ref)
        vals = np.asarray(get_explainer().shap_values(processed))
        if vals.ndim == 3:
            vals = vals[..., 0]
        names = preprocessor.get_feature_names_out()
        mean_abs = pd.DataFrame({"Source": [base_feature(n) for n in names], "v": np.abs(vals).mean(axis=0)})
        mean_abs = mean_abs.groupby("Source", as_index=False)["v"].sum()
        mean_abs["Label"] = mean_abs["Source"].map(feature_label)
        mean_abs = mean_abs.sort_values("v", ascending=False)
    return {
        "shap": mean_abs,
        "risks": pd.Series(risks).value_counts(),
        "ivs": pd.Series(ivs).value_counts(),
        "n": len(ref),
    }


def hbar(labels, values, colors, xlabel, title, fmt="{:.3f}"):
    fig, ax = themed_fig(7.5, 0.45 * len(labels) + 1.3)
    ax.barh(labels, values, color=colors, height=0.6)
    ax.grid(axis="y", visible=False)
    span = max(max(values), 1e-6)
    for y, v in enumerate(values):
        ax.text(v + 0.012 * span, y, fmt.format(v), va="center", color=C_TEXT, fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    ax.invert_yaxis()
    show_fig(fig)


def page_model_insights():
    section_title("Model Insights", "For faculty and researchers: how the deployed model was selected and how it behaves.")
    notice("<b>Key limitation.</b> The post-hoc High level has recall of only about 0.16, so most actual High cases are not flagged. The Low / Moderate / High levels are not a reliable three-class classifier; the regression score is the primary output.", "warn")
    st.write("")
    tabs = st.tabs(["Model comparison", "Cross-validation", "Feature importance", "SHAP summary",
                    "Predictions & residuals", "Distributions", "Decision layer", "Deep-learning comparison"])

    with tabs[0]:
        rows = []
        for name, mae, rmse, r2 in MODEL_RESULTS:
            sel = name == "Random Forest"
            badge = ' <span class="badge blue">Selected</span>' if sel else ""
            rows.append(f'<tr class="{"sel" if sel else ""}"><td>{name}{badge}</td><td>{mae:.4f}</td><td>{rmse:.4f}</td><td>{r2:.4f}</td></tr>')
        md_html(f'<div class="card"><table class="rtable"><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr>{"".join(rows)}</table></div>')
        st.write("")
        with chart_card("Which model performs best on held-out data?"):
            ordered = sorted(MODEL_RESULTS, key=lambda r: r[3], reverse=True)
            fig, ax = themed_fig(7.5, 3.4)
            names = [r[0] for r in ordered]
            vals = [r[3] for r in ordered]
            ax.barh(names, vals, color=[C_BLUE if n == "Random Forest" else "#B9C9E6" for n in names], height=0.6)
            ax.axvline(0, color=C_MUTED, linewidth=1)
            ax.grid(axis="y", visible=False)
            ax.invert_yaxis()
            for y, v in enumerate(vals):
                ax.text(v + (0.006 if v >= 0 else -0.006), y, f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", color=C_TEXT, fontsize=8.5)
            ax.set_xlabel("R² on held-out test set (higher is better)")
            show_fig(fig)
        notice("R² is the share of variance explained, <b>not accuracy</b>. Random Forest explains approximately 25% of "
               "variance on the held-out test set.", "info")

        st.write("")
        rf = next((r for r in MODEL_RESULTS if r[0] == "Random Forest"), None)
        others = [r for r in MODEL_RESULTS if r[0] != "Random Forest"]
        if rf and others:
            runner = max(others, key=lambda r: r[3])
            won = [n for n, ok in (("MAE", rf[1] <= min(r[1] for r in others)),
                                   ("RMSE", rf[2] <= min(r[2] for r in others)),
                                   ("R²", rf[3] >= max(r[3] for r in others))) if ok]
            md_html(card("Why Random Forest?",
                         f"Random Forest was best on {', '.join(won) if won else 'none of the held-out metrics'} among the five candidates. "
                         f"The margin over {runner[0]} is small (R² {rf[3]:.3f} vs {runner[3]:.3f}) and absolute performance is moderate. "
                         "It was selected on these experiments, not because its performance is perfect."))

    with tabs[1]:
        if CV_FOLDS and len(CV_FOLDS) >= 2:
            arr = np.array(CV_FOLDS, dtype=float)
            mean, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
            md_html('<div class="grid g3">'
                    + stat(f"{mean[0]:.3f} ± {sd[0]:.3f}", f"{len(arr)}-fold CV MAE (mean ± SD)", "Random Forest", C_BLUE)
                    + stat(f"{mean[1]:.3f} ± {sd[1]:.3f}", f"{len(arr)}-fold CV RMSE (mean ± SD)", "Random Forest", C_PURPLE)
                    + stat(f"{mean[2]:.3f} ± {sd[2]:.3f}", f"{len(arr)}-fold CV R² (mean ± SD)", "Random Forest", C_CYAN)
                    + "</div>")
            rows = "".join(f"<tr><td>Fold {i}</td><td>{r[0]:.3f}</td><td>{r[1]:.3f}</td><td>{r[2]:.3f}</td></tr>" for i, r in enumerate(arr, 1))
            st.write("")
            md_html(f'<div class="card"><table class="rtable"><tr><th>Fold</th><th>MAE</th><th>RMSE</th><th>R²</th></tr>{rows}</table></div>')
        else:
            md_html('<div class="grid g3">'
                    + stat(f"{CV_RESULTS['MAE']:.3f}", "5-fold CV MAE (mean)", "Random Forest", C_BLUE)
                    + stat(f"{CV_RESULTS['RMSE']:.3f}", "5-fold CV RMSE (mean)", "Random Forest", C_PURPLE)
                    + stat(f"{CV_RESULTS['R²']:.3f}", "5-fold CV R² (mean)", "Random Forest", C_CYAN)
                    + "</div>")
            st.write("")
            notice("Only the fold means are bundled with the app, so fold-to-fold variability (standard deviation) is not shown. "
                   "Add the per-fold results as <code>cv_folds</code> in <code>research_results.json</code> to display mean ± SD.", "info")
        st.write("")
        notice("These cross-validation results are for the selected Random Forest model. They are consistent with the held-out "
               "result: the model has useful predictive signal but is far from perfect.", "info")

    with tabs[2]:
        try:
            imp = importance_df().head(12)
            with chart_card("Which factors are the most influential predictors?"):
                hbar(list(imp["Label"]), list(imp["Importance"]), [C_BLUE] * len(imp),
                     "Random Forest feature importance (summed over encoded columns)",
                     "Influential predictors in the deployed model")
                st.caption("Computed from the loaded model. Importance describes predictive influence, not causality.")
        except Exception:
            notice("Feature importance could not be computed from the loaded model.", "warn")

    with tabs[3]:
        ref = None
        try:
            ref = reference_analysis()
        except Exception:
            ref = None
        if ref is None:
            missing_data_card("SHAP summary", "reference_data.csv",
                              "the 22 feature columns of your held-out or training rows")
        elif ref["shap"] is None:
            notice("The all-student SHAP summary needs the shap package, which could not be imported on this machine. "
                   "Individual explanations on the Why This Result? page still work using the built-in TreeSHAP implementation.", "info")
        else:
            top = ref["shap"].head(12)
            with chart_card("Across students, which behaviours move predictions the most?"):
                hbar(list(top["Label"]), list(top["v"]), [C_PURPLE] * len(top),
                     "Mean |SHAP| (summed over encoded columns)", f"SHAP summary on {ref['n']} reference rows")

    with tabs[4]:
        ev = load_optional_csv("evaluation_predictions.csv")
        cols = {c.lower(): c for c in ev.columns} if ev is not None else {}
        if ev is None or "actual" not in cols or "predicted" not in cols:
            missing_data_card("Actual vs predicted and residual analysis", "evaluation_predictions.csv",
                              "columns Actual and Predicted for the held-out test set")
        else:
            actual = ev[cols["actual"]].astype(float).to_numpy()
            pred = ev[cols["predicted"]].astype(float).to_numpy()
            resid = actual - pred
            c1, c2 = st.columns(2)
            with c1:
                with chart_card("How close are predictions to actual scores?"):
                    fig, ax = themed_fig(5, 4)
                    ax.scatter(actual, pred, s=22, color=C_BLUE, alpha=0.7)
                    ax.plot([1, 7], [1, 7], color=C_CYAN, linestyle="--", linewidth=1.2)
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    show_fig(fig)
            with c2:
                with chart_card("Are errors balanced across the range?"):
                    fig, ax = themed_fig(5, 4)
                    ax.scatter(pred, resid, s=22, color=C_PURPLE, alpha=0.7)
                    ax.axhline(0, color=C_CYAN, linestyle="--", linewidth=1.2)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Residual (actual − predicted)")
                    show_fig(fig)

    with tabs[5]:
        ref = None
        try:
            ref = reference_analysis()
        except Exception:
            ref = None
        ev = load_optional_csv("evaluation_predictions.csv")
        cols = {c.lower(): c for c in ev.columns} if ev is not None else {}
        if ref is None and (ev is None or "predicted" not in cols):
            missing_data_card("Risk and intervention distributions", "reference_data.csv",
                              "the 22 feature columns; optionally evaluation_predictions.csv")
        else:
            c1, c2 = st.columns(2)
            if ref is not None:
                risks = ref["risks"]
                src = f"{ref['n']} reference rows"
            else:
                risks = pd.Series([classify_risk(p) for p in ev[cols["predicted"]].astype(float)]).value_counts()
                src = "evaluation predictions"
            with c1:
                with chart_card("How are students spread across intervention levels?"):
                    order = [r for r in ["Low", "Medium", "High"] if r in risks.index]
                    hbar([RISK_META[r]["label"] for r in order], [int(risks[r]) for r in order],
                         [RISK_META[r]["color"] for r in order], "Count", f"Predicted levels ({src})", fmt="{:.0f}")
            with c2:
                if ref is not None:
                    ivs = ref["ivs"]
                    with chart_card("Which interventions would be assigned most often?"):
                        hbar(list(ivs.index), [int(v) for v in ivs.values], [C_BLUE] * len(ivs),
                             "Count", "Intervention distribution", fmt="{:.0f}")
                else:
                    missing_data_card("Intervention distribution", "reference_data.csv", "the 22 feature columns")

    with tabs[6]:
        rows = "".join(f"<tr><td>{n}</td><td>{p:.2f}</td><td>{r:.2f}</td></tr>" for n, p, r in CLASS_METRICS)
        md_html(f'<div class="card"><h4>Low / Moderate / High labels are a post-hoc layer</h4>'
                f'<p style="margin-bottom:10px">Categories are derived from the regression output. Reported precision and recall:</p>'
                f'<table class="rtable"><tr><th>Level</th><th>Precision</th><th>Recall</th></tr>{rows}</table></div>')
        st.write("")
        notice("Recall for the High level is limited (≈ 0.16), so the system is <b>not</b> presented as a strong three-class "
               "classifier. The main task is regression; use the score alongside the level.", "warn")


    with tabs[7]:
        dl = load_optional_csv("deep_learning_comparison.csv")
        needed = {"Member", "Dataset", "Model", "MAE", "RMSE", "R2"}
        if dl is not None and len(dl) and needed <= set(dl.columns):
            rows = "".join(
                f"<tr><td>{esc(r.Member)}</td><td>{esc(r.Dataset)}</td><td>{esc(r.Model)}</td>"
                f"<td>{float(r.MAE):.4f}</td><td>{float(r.RMSE):.4f}</td><td>{float(r.R2):.4f}</td></tr>"
                for r in dl.itertuples())
            md_html(f'<div class="card"><table class="rtable"><tr><th>Member</th><th>Dataset</th><th>Deep-learning model</th>'
                    f'<th>MAE</th><th>RMSE</th><th>R²</th></tr>{rows}</table></div>')
            st.write("")
            with chart_card("How do the deep-learning models compare across members and datasets?"):
                labels = [f"{r.Member} · {r.Model} · {r.Dataset}" for r in dl.itertuples()]
                hbar(labels, [float(v) for v in dl["R2"]], [C_PURPLE] * len(dl), "R² (higher is better)",
                     "Deep-learning comparison (from deep_learning_comparison.csv)")
        else:
            notice("<b>Faculty requirement still open.</b> Each team member is expected to train a deep-learning model and compare it "
                   "across datasets. Verified results are not available in this app yet, so none are shown. The MLP Regressor in the "
                   "model comparison is a baseline neural network on the Student Awareness Dataset only; it is not presented as "
                   "evidence for this requirement.", "warn")
            st.write("")
            rows = "".join(f"<tr><td>{esc(m['role'])}</td><td>{esc(m['name'] or 'To be added')}</td><td>Pending</td></tr>" for m in TEAM)
            md_html(f'<div class="card"><table class="rtable"><tr><th>Member</th><th>Name</th><th>Deep-learning comparison</th></tr>{rows}</table></div>')
            st.caption("To display results, add deep_learning_comparison.csv with columns Member, Dataset, Model, MAE, RMSE, R2 "
                       "(only after the experiments are complete and verified).")


# ---------------- RESEARCH ----------------

def line_chart_noise():
    xs = [r[0] for r in NOISE_RESULTS]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    fig.patch.set_facecolor(C_CARD)
    for ax in (a1, a2):
        ax.set_facecolor(C_CARD)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=C_MUTED, labelsize=9, length=0)
        ax.grid(color="#D3E2F5", linewidth=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x}%" for x in xs])
        ax.set_xlabel("Noise level", color=C_MUTED)
    a1.plot(xs, [r[1] for r in NOISE_RESULTS], marker="o", color=C_BLUE, label="MAE")
    a1.plot(xs, [r[2] for r in NOISE_RESULTS], marker="o", color=C_PURPLE, label="RMSE")
    a1.set_title("Error vs noise (lower is better)", color=C_TEXT, fontsize=10, loc="left")
    a1.legend(facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT, fontsize=8)
    a2.plot(xs, [r[3] for r in NOISE_RESULTS], marker="o", color=C_CYAN, label="R²")
    a2.set_title("R² vs noise (higher is better)", color=C_TEXT, fontsize=10, loc="left")
    show_fig(fig)


def gan_chart():
    metrics = ["MAE", "RMSE", "R²"]
    fig, ax = themed_fig(7.5, 3.6)
    x = np.arange(3)
    real = GAN_RESULTS["Real only (Random Forest)"]
    syn = GAN_RESULTS["Real + CTGAN synthetic"]
    ax.bar(x - 0.18, real, width=0.34, color=C_BLUE, label="Real only (final model)")
    ax.bar(x + 0.18, syn, width=0.34, color="#8FA6C9", label="Real + CTGAN")
    for i in range(3):
        ax.text(x[i] - 0.18, real[i] + 0.02, f"{real[i]:.3f}", ha="center", color=C_TEXT, fontsize=8)
        ax.text(x[i] + 0.18, syn[i] + 0.02, f"{syn[i]:.3f}", ha="center", color=C_TEXT, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(axis="x", visible=False)
    ax.legend(facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT, fontsize=8)
    show_fig(fig)


def page_research():
    section_title("Research", "Methodology, experiments and honest limitations of the framework.")
    tabs = st.tabs(["Dataset", "Methodology", "Machine Learning", "Explainable AI",
                    "GAN Experiment", "Noise Robustness", "Limitations", "Future Work", "SDG 4 Relevance"])

    with tabs[0]:
        md_html(
            '<div class="grid g2">'
            + card("AIBT-50K (behavioural profiling)",
                   "50,000 rows × 56 columns: 55 questionnaire items plus Student_ID. Eleven constructs with five items each "
                   "(Access, Understand, Apply, Analyze, Evaluate, Co-Create, Govern, AI Trust, AI Dependency, Metacognition, "
                   "Learning Performance). Used only for exploratory K-Means profiling.", "🗂️")
            + card("Student Awareness Dataset (supervised ML)",
                   "557 × 26 raw rows; 394 unique records after cleaning (duplicates removed; empty Score and mostly-missing Comments "
                   "dropped; 5 missing Remorse values median-imputed; malformed CGPA “3:08” read as 3.08 with the raw value preserved). "
                   "Mostly 1–7 questionnaire scales.", "🗂️")
            + "</div>"
        )
        st.write("")
        md_html(
            '<div class="grid g3">'
            + stat("4.204", "Target mean", "AI_Reliance_Score")
            + stat("1.362", "Target SD", "range 1 to 7")
            + stat("0.715", "Cronbach α", "two target items")
            + "</div>"
        )
        st.write("")
        notice("The two datasets are analytically separate and are <b>never merged row by row</b>.", "info")

    with tabs[1]:
        md_html(arch_html())

    with tabs[2]:
        md_html(
            '<div class="grid g2">'
            + card("Target and leakage control",
                   "AI_Reliance_Score = (Decision_Reliance + Independent_Reliance) / 2. Both component items are excluded from the "
                   "inputs because they define the target (target leakage).", "🎯")
            + card("Pipeline",
                   "Numeric features: StandardScaler. Categorical features: OneHotEncoder (unknown categories ignored). "
                   "RandomForestRegressor (n_estimators = 300, random_state = 42). The saved pipeline is loaded, never retrained.", "⚙️")
            + card("Features",
                   "18 numeric (including CGPA) and 4 categorical: Gender, Field of Study, AI Experience, AI Usage Frequency.", "🧩")
            + card("Intervention levels",
                   f"Training-set tertiles: Low &lt; {low_threshold:g}; Moderate {low_threshold:g} to &lt; {high_threshold:g}; High ≥ {high_threshold:g}. "
                   "Relative intervention levels, not clinical labels.", "🎚️")
            + "</div>"
        )

    with tabs[3]:
        md_html(
            '<div class="grid g2">'
            + card("SHAP for individual explanations",
                   "TreeExplainer shows which behaviours pushed a prediction above or below the model's baseline, "
                   "and by how much.", "🔎")
            + card("Interpretation",
                   "Feature importance and SHAP describe predictive influence. They do not prove causality, so we call them "
                   "<i>influential predictors</i>.", "⚠️")
            + "</div>"
        )

    with tabs[4]:
        md_html(
            '<div class="card"><h4>Generative Data Augmentation Experiment</h4><p>CTGAN generated 315 synthetic rows to accompany 315 real '
            'training rows (630 total). Raw synthetic values fell outside questionnaire ranges, so domain constraints were applied '
            '(Likert items and target 1–7; CGPA 2.31–4.00). Real + synthetic data did <b>not</b> outperform the real-only model, so the '
            'real-data-only Random Forest remains the production model.</p></div>'
        )
        st.write("")
        with chart_card("How does synthetic augmentation compare with real-only training?"):
            gan_chart()
        notice("Reported honestly: GAN augmentation did not improve performance (R² 0.2544 → 0.2048).", "warn")

    with tabs[5]:
        with chart_card("How stable is the model under noisy questionnaire inputs?"):
            line_chart_noise()
        rows = "".join(f"<tr><td>{n}%</td><td>{a:.4f}</td><td>{b:.4f}</td><td>{c:.4f}</td></tr>" for n, a, b, c in NOISE_RESULTS)
        md_html(f'<div class="card"><table class="rtable"><tr><th>Noise</th><th>MAE</th><th>RMSE</th><th>R²</th></tr>{rows}</table></div>')
        st.write("")
        notice("The model shows reasonable stability under small perturbations (±1 on the 1–7 scale), while performance gradually "
               "degrades at higher noise. It is not noise-proof.", "info")

    with tabs[6]:
        limits = [
            "Trained on self-reported behavioural data.",
            "Predictions estimate AI reliance; they are not a diagnosis.",
            "Predictive performance is moderate (R² ≈ 0.25).",
            "Recall for the High level is limited (about 0.16): most actual High cases are not flagged by the decision layer.",
            "K-Means profiling is exploratory, with weak cluster separation.",
            "GAN augmentation did not improve the final model.",
            "Performance degrades at higher input-noise levels.",
            "Interventions are rule-based prototype recommendations.",
            "Longitudinal experiments are needed to show whether interventions reduce overreliance.",
            "Feature importance and SHAP do not establish causality.",
        ]
        notice("<b>Most important:</b> the High level has recall of about 0.16, so this decision layer must not be read as a reliable "
               "three-class classifier.", "warn")
        st.write("")
        md_html('<div class="card"><h4>Limitations</h4><p>' + "<br>".join(f"{i}. {t}" for i, t in enumerate(limits, 1)) + "</p></div>")

    with tabs[7]:
        md_html(
            '<div class="card"><h4>Proposed future work</h4><p>'
            "1. Longitudinal study to test whether adaptive interventions change verification and independent-reasoning behaviour.<br>"
            "2. Larger, more diverse samples and external validation of the regression model.<br>"
            "3. Improving recall for the High level and testing alternative decision thresholds.<br>"
            "4. Collecting a shared instrument so behavioural profiling and risk prediction can be linked at the individual level."
            "</p></div>"
        )


    with tabs[8]:
        md_html(
            '<div class="grid g2">'
            + card("SDG 4: Quality Education",
                   "The project relates to Sustainable Development Goal 4, which aims for inclusive and equitable quality education and "
                   "lifelong learning opportunities. Generative AI can support learning, but uncritical dependence can weaken independent "
                   "reasoning and verification.")
            + card("Intended contribution",
                   "A decision-support prototype that estimates AI reliance from self-reported behaviour, explains the estimate, and suggests "
                   "activities that encourage verification, critical analysis and independent reasoning.")
            + card("How the prototype aligns",
                   "Responsible AI-assisted learning (verification and independent-answer activities), personalised learning support "
                   "(rule-based adaptive interventions) and transparency (SHAP explanations and stated limitations).")
            + card("What is not claimed",
                   "This is the project's intended contribution, not proven impact. Longitudinal studies are needed to show whether the "
                   "interventions improve learning or reduce overreliance.")
            + "</div>"
        )


# ---------------- ABOUT ----------------

TEAM = [
    {"role": "Member 1", "focus": "AI Dependency, Cognitive Behaviour and Overreliance Analysis", "name": "", "id": ""},
    {"role": "Member 2", "focus": "AI Trust, Learning Behaviour and Psychological Factors", "name": "", "id": ""},
    {"role": "Member 3", "focus": "Machine Learning, Risk Prediction and Adaptive Intervention", "name": "C Arun", "id": "CB.SC.P2AIE26008"},
]


def page_about():
    section_title("About")
    md_html(
        f"""
        <div class="card"><div class="eyebrow">Research Methodology Project · M.Tech Artificial Intelligence</div>
        <h4 style="font-size:1.25rem">{APP_TITLE}</h4>
        <p>The framework combines behavioural profiling, AI reliance prediction, explainable AI and adaptive intervention to support
        responsible AI-assisted learning.</p></div>
        """
    )
    section_title("Team contributions")
    cards = []
    for m in TEAM:
        who = esc(m["name"]) if m["name"] else "Name to be added"
        ident = esc(m["id"]) if m["id"] else "ID to be added"
        cards.append(card(f'{who}', f'<b>{esc(m["role"])}</b> · {ident}<br>{esc(m["focus"])}'))
    md_html(f'<div class="grid g3">{"".join(cards)}</div>')
    st.write("")
    rows = "".join(f"<tr><td>{esc(m['name'] or 'To be added')}</td><td>{esc(m['id'] or 'To be added')}</td><td>{esc(m['focus'])}</td></tr>" for m in TEAM)
    md_html(f'<div class="card"><h4>Contribution table</h4><table class="rtable"><tr><th>Name</th><th>ID</th><th>Contribution</th></tr>{rows}</table></div>')
    st.write("")
    md_html(
        """
        <div class="card"><h4>Individual contribution: Machine Learning, Risk Prediction and Adaptive Intervention</h4>
        <p>Supervised learning, AI Reliance Score prediction, risk-level categorisation, model comparison, Random Forest, SHAP explanations,
        the adaptive intervention engine, a CTGAN augmentation experiment, noise-robustness analysis and Streamlit integration.</p></div>
        """
    )
    st.write("")
    notice(
        "<b>Research prototype disclaimer.</b> This system is a research prototype designed to estimate AI reliance from self-reported "
        "behavioural indicators. Predictions should not be interpreted as clinical, psychological, or diagnostic assessments.",
        "warn",
    )


def load_test_results():
    path = BASE_DIR / "test_results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("test_results.json could not be read")
        return None


def evidence_inventory():
    ev = load_optional_csv("evaluation_predictions.csv")
    ev_ok = ev is not None and {"actual", "predicted"} <= {c.lower() for c in ev.columns}
    try:
        ref_data = reference_analysis()
        ref_ok = ref_data is not None
        shap_sum_ok = ref_ok and ref_data["shap"] is not None
    except Exception:
        ref_ok = shap_sum_ok = False
    dl = load_optional_csv("deep_learning_comparison.csv")
    dl_ok = dl is not None and len(dl) > 0 and {"Member", "Dataset", "Model", "MAE", "RMSE", "R2"} <= set(dl.columns)
    tests_ok = load_test_results() is not None
    graphs = [
        ("Held-out R² by model", "Model Insights", True, ""),
        ("Feature importance (deployed model)", "Model Insights", True, ""),
        ("GAN augmentation comparison", "Research", True, ""),
        ("Noise: MAE and RMSE vs noise level", "Research", True, ""),
        ("Noise: R² vs noise level", "Research", True, ""),
        ("SHAP contributions (per prediction)", "Why This Result?", True, ""),
        ("Behaviour radar (per student)", "Student Profile", True, ""),
        ("Actual vs predicted", "Model Insights", ev_ok, "evaluation_predictions.csv"),
        ("Residual plot", "Model Insights", ev_ok, "evaluation_predictions.csv"),
        ("SHAP summary (all students)", "Model Insights", shap_sum_ok, "reference_data.csv and the shap package"),
        ("Risk-level distribution", "Model Insights", ev_ok or ref_ok, "evaluation_predictions.csv or reference_data.csv"),
        ("Intervention distribution", "Model Insights", ref_ok, "reference_data.csv"),
    ]
    tables = [
        ("Held-out model comparison", "Model Insights", True, ""),
        ("Cross-validation summary", "Model Insights", True, ""),
        ("Post-hoc level precision and recall", "Model Insights", True, ""),
        ("GAN augmentation comparison", "Research", True, ""),
        ("Noise robustness", "Research", True, ""),
        ("Team contribution table", "About", True, ""),
        ("Test cases and results", "Testing", tests_ok, "test_results.json (run test_app.py)"),
        ("Deep-learning comparison", "Model Insights", dl_ok, "deep_learning_comparison.csv"),
    ]
    return graphs, tables


def status_badge(ok, pending="Needs data"):
    if ok:
        return '<span class="badge" style="color:#15803D;border-color:rgba(22,163,74,.5)">Available</span>'
    return f'<span class="badge" style="color:#B45309;border-color:rgba(217,119,6,.55)">{pending}</span>'


def page_testing():
    section_title("Testing and Requirement Coverage", "Documented test cases, faculty-requirement status and the evidence available in this application.")
    tabs = st.tabs(["Test cases", "Requirement coverage", "Evidence inventory"])

    with tabs[0]:
        data = load_test_results()
        if not data:
            notice("No test results file was found. Run <code>py test_app.py</code> in the project folder to execute the tests and "
                   "create <code>test_results.json</code>. Nothing is shown as passed until the tests have actually been run.", "warn")
        else:
            tests = data.get("tests", [])
            passed = sum(t["status"] == "Pass" for t in tests)
            failed = sum(t["status"] == "Fail" for t in tests)
            other = len(tests) - passed - failed
            env = data.get("environment", {})
            md_html('<div class="grid g4">'
                    + stat(str(len(tests)), "Test cases", "documented below")
                    + stat(str(passed), "Passed", "observed result matched", C_GREEN)
                    + stat(str(failed), "Failed", "needs attention", C_RED if failed else None)
                    + stat(str(other), "Not covered", "no matching input found")
                    + "</div>")
            st.write("")
            notice(f"Generated {esc(data.get('generated', 'unknown'))} by <code>test_app.py</code> · Python {esc(env.get('python', '?'))} · "
                   f"Streamlit {esc(env.get('streamlit', '?'))} · scikit-learn {esc(env.get('scikit-learn', '?'))} · SHAP {esc(env.get('shap', '?'))}. "
                   "Re-run the script on your own machine before submission, and add screenshots to your report.", "info")
            st.write("")
            colors = {"Pass": "#15803D", "Fail": "#DC2626"}
            rows = "".join(
                f"<tr><td>{esc(t['id'])}</td><td>{esc(t['category'])}</td><td>{esc(t['test'])}</td><td>{esc(t['expected'])}</td>"
                f"<td>{esc(t['observed'])}</td><td style='font-weight:700;color:{colors.get(t['status'], '#B45309')}'>{esc(t['status'])}</td></tr>"
                for t in tests)
            md_html(f'<div class="card" style="overflow-x:auto"><table class="rtable"><tr><th>ID</th><th>Category</th><th>Test</th>'
                    f'<th>Expected</th><th>Observed</th><th>Status</th></tr>{rows}</table></div>')

    with tabs[1]:
        graphs, tables = evidence_inventory()
        names_filled = all(m["name"] and m["id"] for m in TEAM)
        dl_ok = any(t[0] == "Deep-learning comparison" and t[2] for t in tables)
        tests_ok = load_test_results() is not None
        items = [
            ("Complete two-dataset research architecture", "Present", "Home and Research pages (no row-level merging)."),
            ("Noise component", "Present", "Research page: 0, 5, 10 and 15% noise with MAE, RMSE and R²."),
            ("GAN experiment", "Present", "Research page: real-only vs real + CTGAN (no improvement reported)."),
            ("Explainable AI", "Present", "Why This Result? page with SHAP, baseline and a reconstruction check."),
            ("Deep-learning model per member, compared across datasets", "Present" if dl_ok else "Pending",
             "Needs verified results in deep_learning_comparison.csv." if not dl_ok else "Model Insights tab."),
            ("At least 8 graphs and 5 tables (target noted by the team)", "Present" if (sum(g[2] for g in graphs) >= 8 and sum(t[2] for t in tables) >= 5) else "Partial",
             f"{sum(g[2] for g in graphs)} graphs and {sum(t[2] for t in tables)} tables currently available; see Evidence inventory."),
            ("Testing and test design", "Present" if tests_ok else "Pending", "This page, from test_results.json." if tests_ok else "Run test_app.py to create test_results.json."),
            ("SDG relevance", "Present", "Research page, SDG 4 Relevance tab (intended contribution, not proven impact)."),
            ("Team names, IDs and contributions", "Present" if names_filled else "Partial",
             "About page." if names_filled else "Add the remaining names and IDs in the TEAM list in app.py."),
            ("Journal-format report, literature review and five-page discussion", "Report", "Written report requirement; not satisfied by the website."),
        ]
        colors = {"Present": "#15803D", "Partial": "#B45309", "Pending": "#B45309", "Report": "#425B82"}
        rows = "".join(
            f"<tr><td>{esc(a)}</td><td style='font-weight:700;color:{colors[b]}'>{b}</td><td>{esc(c)}</td></tr>" for a, b, c in items)
        md_html(f'<div class="card" style="overflow-x:auto"><table class="rtable"><tr><th>Requirement</th><th>Status</th><th>Where / what is needed</th></tr>{rows}</table></div>')

    with tabs[2]:
        graphs, tables = evidence_inventory()
        g_ok, t_ok = sum(g[2] for g in graphs), sum(t[2] for t in tables)
        md_html('<div class="grid g2">'
                + stat(f"{g_ok} / {len(graphs)}", "Graphs available", "team target: at least 8")
                + stat(f"{t_ok} / {len(tables)}", "Tables available", "team target: at least 5")
                + "</div>")
        st.write("")
        def inv_rows(entries):
            return "".join(f"<tr><td>{esc(n)}</td><td>{esc(w)}</td><td>{status_badge(ok)}</td><td>{esc(f)}</td></tr>" for n, w, ok, f in entries)
        md_html(f'<div class="card" style="overflow-x:auto"><h4>Graphs</h4><table class="rtable"><tr><th>Graph</th><th>Page</th><th>Status</th><th>Needs file</th></tr>{inv_rows(graphs)}</table></div>')
        st.write("")
        md_html(f'<div class="card" style="overflow-x:auto"><h4>Tables</h4><table class="rtable"><tr><th>Table</th><th>Page</th><th>Status</th><th>Needs file</th></tr>{inv_rows(tables)}</table></div>')
        st.caption("Optional views use only genuine files exported from your notebooks. Nothing is generated or simulated.")


def render_footer():
    md_html(
        """
        <div class="footer"><div class="fb">AI Overreliance Research Framework</div>
        <div class="ft">Promoting responsible AI-assisted learning through explainable machine learning and adaptive interventions.</div>
        <div class="fs">Research Prototype · M.Tech Artificial Intelligence</div></div>
        """
    )


# ============================================================
# ROUTER
# ============================================================

PAGES = {
    "Home": page_home,
    "Assess": page_assess,
    "Prediction": page_prediction,
    "Why This Result?": page_why,
    "Intervention": page_intervention,
    "Student Profile": page_profile,
    "Model Insights": page_model_insights,
    "Research": page_research,
    "Testing": page_testing,
    "About": page_about,
}

_current = st.session_state.get("nav")
if _current not in PAGES:
    _current = "Home"

render_navbar(_current)

for _note in load_notes:
    notice(_note, "warn")

show_right = _current in STUDENT_SET
with st.container(key="shell"):
    shell_cols = st.columns([0.95, 4.2, 1.2] if show_right else [0.95, 5.4], gap="medium")
    with shell_cols[0]:
        render_left_panel(_current)
    with shell_cols[1]:
        try:
            PAGES[_current]()
        except Exception:
            logger.exception("Page %s failed", _current)
            notice("Something went wrong while displaying this page. Please try another section or reload the app.", "warn")
    if show_right:
        with shell_cols[2]:
            render_right_panel(_current)

render_footer()