"""
CipherBench Streamlit Application.

Demonstrates the full project end-to-end: dataset exploration, live cipher
identification (all 6 models, including MLP/CNN), model comparison,
verified experiment results, the 330-run experiment matrix, methodology,
and reproducibility notes.

Design goals (see docs/PROJECT_REPORT.md):
- Works without a database or any network service. All data comes from the
  CSV/JSON files already in the repository (data/, experiments/results/).
- Fails gracefully: a missing results file or an unavailable optional
  dependency degrades a single section instead of crashing the app.
- No hard-coded absolute/Windows paths; everything is resolved relative to
  the project root.
- Caches expensive work (dataset loads, live model training) so navigating
  the UI does not repeatedly retrain models.
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import (  # noqa: E402
    load_binary_dataset,
    load_multiclass_dataset,
    load_hknnrf_split,
    load_binary_hknnrf_split,
    get_all_binary_pairs,
    get_feature_names,
    load_config,
)
from src.utils.metrics import compute_metrics, compute_confusion_matrix  # noqa: E402
from src.models.baseline import get_svm_model, get_knn_model, get_random_forest_model  # noqa: E402
from src.models.hknnrf import HKNNRFClassifier  # noqa: E402
from src.models.mlp import MLPClassifier  # noqa: E402
from src.models.cnn import CNN1DClassifier  # noqa: E402

st.set_page_config(
    page_title="CipherBench",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

SIZES = ["1KB", "8KB", "64KB", "256KB", "512KB"]
ALGORITHMS = ["AES", "3DES", "Blowfish", "CAST", "RC2"]
ALL_MODELS = ["SVM", "KNN", "RF", "HKNNRF", "MLP", "CNN"]
DEEP_MODELS = {"MLP", "CNN"}

MODEL_COLORS = {
    "SVM": "#6366F1",
    "KNN": "#06B6D4",
    "RF": "#10B981",
    "HKNNRF": "#F59E0B",
    "MLP": "#EC4899",
    "CNN": "#8B5CF6",
}
ALGO_COLORS = {
    "AES": "#6366F1",
    "3DES": "#06B6D4",
    "Blowfish": "#10B981",
    "CAST": "#F59E0B",
    "RC2": "#EC4899",
}
PLOTLY_TEMPLATE = "plotly_white"
FONT_FAMILY = "Inter, -apple-system, sans-serif"
PAPER_MODELS = ["SVM", "KNN", "RF", "HKNNRF"]  # the only 4 models the base paper implements

# --------------------------------------------------------------------------
# Reference numbers transcribed verbatim from Yuan et al. (2022), PeerJ
# Computer Science, DOI 10.7717/peerj-cs.1110 — Tables 3, 4 and 5.
# The paper implements SVM, KNN, RF and HKNNRF only; it has no neural
# network baseline, so there is nothing here for MLP/CNN to be compared
# against — those two are this project's own extension (see the badges on
# the Paper Comparison page).
# --------------------------------------------------------------------------

# Table 3 — binary identification, AES vs 3DES, by ciphertext size (KB).
PAPER_BINARY_AES_3DES = {
    "accuracy": {
        512: {"SVM": 0.600, "KNN": 0.600, "RF": 0.600, "HKNNRF": 0.725},
        256: {"SVM": 0.575, "KNN": 0.575, "RF": 0.625, "HKNNRF": 0.650},
        64: {"SVM": 0.625, "KNN": 0.600, "RF": 0.650, "HKNNRF": 0.675},
        8: {"SVM": 0.525, "KNN": 0.525, "RF": 0.575, "HKNNRF": 0.700},
        1: {"SVM": 0.500, "KNN": 0.550, "RF": 0.525, "HKNNRF": 0.725},
    },
    "precision": {
        512: {"SVM": 0.580, "KNN": 0.601, "RF": 0.600, "HKNNRF": 0.725},
        256: {"SVM": 0.580, "KNN": 0.583, "RF": 0.628, "HKNNRF": 0.700},
        64: {"SVM": 0.620, "KNN": 0.594, "RF": 0.650, "HKNNRF": 0.650},
        8: {"SVM": 0.530, "KNN": 0.532, "RF": 0.600, "HKNNRF": 0.675},
        1: {"SVM": 0.420, "KNN": 0.530, "RF": 0.615, "HKNNRF": 0.700},
    },
    "recall": {
        512: {"SVM": 0.580, "KNN": 0.600, "RF": 0.600, "HKNNRF": 0.700},
        256: {"SVM": 0.580, "KNN": 0.575, "RF": 0.625, "HKNNRF": 0.650},
        64: {"SVM": 0.660, "KNN": 0.600, "RF": 0.650, "HKNNRF": 0.700},
        8: {"SVM": 0.520, "KNN": 0.525, "RF": 0.575, "HKNNRF": 0.650},
        1: {"SVM": 0.420, "KNN": 0.550, "RF": 0.525, "HKNNRF": 0.625},
    },
}

# Table 5 — five-class identification, by ciphertext size (KB).
PAPER_MULTICLASS = {
    "accuracy": {
        512: {"SVM": 0.190, "KNN": 0.200, "RF": 0.170, "HKNNRF": 0.240},
        256: {"SVM": 0.210, "KNN": 0.200, "RF": 0.200, "HKNNRF": 0.330},
        64: {"SVM": 0.100, "KNN": 0.160, "RF": 0.220, "HKNNRF": 0.270},
        8: {"SVM": 0.170, "KNN": 0.190, "RF": 0.240, "HKNNRF": 0.300},
        1: {"SVM": 0.230, "KNN": 0.210, "RF": 0.220, "HKNNRF": 0.340},
    },
    "precision": {
        512: {"SVM": 0.230, "KNN": 0.219, "RF": 0.210, "HKNNRF": 0.298},
        256: {"SVM": 0.233, "KNN": 0.225, "RF": 0.208, "HKNNRF": 0.330},
        64: {"SVM": 0.115, "KNN": 0.099, "RF": 0.231, "HKNNRF": 0.227},
        8: {"SVM": 0.204, "KNN": 0.207, "RF": 0.233, "HKNNRF": 0.305},
        1: {"SVM": 0.214, "KNN": 0.222, "RF": 0.223, "HKNNRF": 0.371},
    },
    "recall": {
        512: {"SVM": 0.190, "KNN": 0.200, "RF": 0.170, "HKNNRF": 0.240},
        256: {"SVM": 0.210, "KNN": 0.200, "RF": 0.200, "HKNNRF": 0.330},
        64: {"SVM": 0.100, "KNN": 0.160, "RF": 0.220, "HKNNRF": 0.270},
        8: {"SVM": 0.170, "KNN": 0.190, "RF": 0.240, "HKNNRF": 0.300},
        1: {"SVM": 0.230, "KNN": 0.210, "RF": 0.220, "HKNNRF": 0.340},
    },
}

# Table 4 — binary accuracy of HKNNRF specifically, across all 10 pairs.
# Keys use this project's "<A> and <B>" pair naming (see get_all_binary_pairs).
PAPER_HKNNRF_BINARY_PAIRS = {
    "AES and 3DES": {1: 0.725, 8: 0.700, 64: 0.675, 256: 0.650, 512: 0.725},
    "AES and Blowfish": {1: 0.600, 8: 0.700, 64: 0.675, 256: 0.650, 512: 0.675},
    "AES and CAST": {1: 0.650, 8: 0.675, 64: 0.650, 256: 0.700, 512: 0.700},
    "AES and RC2": {1: 0.625, 8: 0.600, 64: 0.700, 256: 0.675, 512: 0.650},
    "3DES and Blowfish": {1: 0.625, 8: 0.625, 64: 0.650, 256: 0.625, 512: 0.650},
    "3DES and CAST": {1: 0.650, 8: 0.625, 64: 0.725, 256: 0.675, 512: 0.625},
    "3DES and RC2": {1: 0.650, 8: 0.675, 64: 0.700, 256: 0.625, 512: 0.650},
    "Blowfish and CAST": {1: 0.650, 8: 0.650, 64: 0.650, 256: 0.625, 512: 0.625},
    "Blowfish and RC2": {1: 0.700, 8: 0.600, 64: 0.675, 256: 0.700, 512: 0.700},
    "CAST and RC2": {1: 0.725, 8: 0.650, 64: 0.625, 256: 0.675, 512: 0.725},
}
PAPER_CITATION = (
    "Yuan, K., Yu, D., Feng, J., Yang, L., Jia, C., Huang, Y. (2022). "
    "A block cipher algorithm identification scheme based on hybrid k-nearest "
    "neighbor and random forest algorithm. PeerJ Computer Science. "
    "DOI 10.7717/peerj-cs.1110 — Tables 3, 4, 5."
)


def our_value(df, model, task, size_label, pair, metric):
    """Look up one metric for one (model, task, size, pair) row in the
    verified results file. Returns None if that experiment isn't present."""
    if df is None:
        return None
    row = df[
        (df["model_name"] == model)
        & (df["task_type"] == task)
        & (df["ciphertext_size"] == size_label)
        & (df["algorithm_pair"] == pair)
    ]
    return float(row.iloc[0][metric]) if len(row) else None


# --------------------------------------------------------------------------
# Visual theme — injected once, purely presentational
# --------------------------------------------------------------------------

def inject_theme():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --cb-primary: #6366F1;
    --cb-secondary: #EC4899;
    --cb-accent: #06B6D4;
    --cb-success: #10B981;
    --cb-warning: #F59E0B;
    --cb-bg-soft: #F8FAFC;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Poppins', sans-serif !important; letter-spacing: -0.02em; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes floatY {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-6px); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.35); }
    50%      { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
}

/* Page content fades/slides in on every render */
[data-testid="stAppViewContainer"] .main .block-container {
    animation: fadeInUp 0.5s ease-out;
    padding-top: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%);
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stRadio label { transition: transform 0.15s ease; }
[data-testid="stSidebar"] .stRadio label:hover { transform: translateX(4px); }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

/* Hero header */
.hero-header {
    padding: 1.6rem 2rem;
    border-radius: 20px;
    background: linear-gradient(120deg, #6366F1, #8B5CF6, #EC4899, #6366F1);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite, fadeInUp 0.6s ease-out;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.5);
}
.hero-icon { font-size: 2.4rem; animation: floatY 3s ease-in-out infinite; display: inline-block; }
.hero-title {
    color: white !important; font-weight: 800 !important; font-size: 2.1rem !important;
    margin: 0.2rem 0 0.2rem 0 !important; -webkit-font-smoothing: antialiased;
}
.hero-subtitle { color: rgba(255,255,255,0.92); font-size: 1.02rem; margin: 0; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 1rem 1.1rem 0.8rem 1.1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    border-top: 3px solid var(--cb-primary);
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px -8px rgba(99, 102, 241, 0.35);
    border-color: var(--cb-primary);
}
div[data-testid="stMetricValue"] {
    font-family: 'Poppins', sans-serif; font-weight: 700 !important;
    background: linear-gradient(90deg, #6366F1, #EC4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    border: none !important;
    background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 4px 14px -4px rgba(99, 102, 241, 0.55) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.015);
    box-shadow: 0 10px 22px -6px rgba(99, 102, 241, 0.65) !important;
}
.stDownloadButton > button {
    border-radius: 10px !important;
    transition: transform 0.15s ease !important;
}
.stDownloadButton > button:hover { transform: translateY(-2px); }

/* Cards for generic content blocks */
.cb-card {
    background: white; border: 1px solid #E5E7EB; border-radius: 16px;
    padding: 1.1rem 1.3rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    animation: fadeInUp 0.5s ease-out;
}

/* Badges */
.cb-badge {
    display: inline-block; padding: 0.18rem 0.65rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.02em;
}
.cb-badge-correct { background: #D1FAE5; color: #047857; }
.cb-badge-wrong { background: #FEE2E2; color: #B91C1C; }
.cb-badge-model {
    background: linear-gradient(90deg, #6366F1, #8B5CF6); color: white;
    animation: pulseGlow 2.4s infinite;
}

/* Dataframes */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Section dividers */
hr { border-top: 1px solid #E5E7EB; }

/* Selectbox / radio focus glow */
[data-baseweb="select"] > div:focus-within {
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title, subtitle="", icon="🔐"):
    st.markdown(
        f"""
<div class="hero-header">
    <span class="hero-icon">{icon}</span>
    <h1 class="hero-title">{title}</h1>
    <p class="hero-subtitle">{subtitle}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


inject_theme()


# --------------------------------------------------------------------------
# Cached data access
# --------------------------------------------------------------------------

@st.cache_data
def find_latest_results_file():
    pattern = str(PROJECT_ROOT / "experiments" / "results" / "results_consolidated_*.csv")
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


@st.cache_data
def load_results_df():
    path = find_latest_results_file()
    if path is None:
        return None, None
    df = pd.read_csv(path)
    return df, os.path.basename(path)


@st.cache_data
def load_dataset_cached(task, size, pair=None):
    if task == "multiclass":
        X_train, X_test, y_train, y_test = load_multiclass_dataset(size)
    else:
        size_folder = size.lower()
        X_train, X_test, y_train, y_test = load_binary_dataset(pair, size_folder)
    return X_train, X_test, y_train, y_test


@st.cache_resource(show_spinner=False)
def run_live_model(model_name, task, size, pair=None, seed=42):
    """
    Train any of the 6 models live for demonstration purposes and evaluate
    on the held-out test split.

    Classical models (SVM/KNN/RF) train directly on the 80% training split.
    HKNNRF uses its own RF/KNN sub-split. MLP/CNN carve an additional
    stratified validation split out of the *training* portion only (never
    the test set) for early stopping, exactly matching the protocol used
    to produce the verified results in experiments/results/ — see
    experiments/run_complete_experiments.py::_fit_deep_model.
    """
    config = load_config()
    num_classes = 5 if task == "multiclass" else 2

    if model_name == "HKNNRF":
        if task == "multiclass":
            X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test = load_hknnrf_split(
                size, test_size=0.2, random_state=seed
            )
        else:
            X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test = load_binary_hknnrf_split(
                pair, size.lower(), test_size=0.2, random_state=seed
            )
        model = HKNNRFClassifier(n_estimators=85, max_depth=5, n_neighbors=7, random_state=seed)
        model.fit(X_train_rf, y_train_rf, X_train_knn, y_train_knn)
        X_test_out, y_test_out = X_test, y_test

    elif model_name in DEEP_MODELS:
        X_train, X_test, y_train, y_test = load_dataset_cached(task, size, pair)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
        )
        if model_name == "MLP":
            model = MLPClassifier(
                num_classes=num_classes, verbose=0, random_state=seed,
                **config["hyperparameters"]["mlp"],
            )
        else:
            model = CNN1DClassifier(
                num_classes=num_classes, verbose=0, random_state=seed,
                **config["hyperparameters"]["cnn"],
            )
        model.fit(X_fit, y_fit, X_val, y_val)
        X_test_out, y_test_out = X_test, y_test

    else:
        X_train, X_test, y_train, y_test = load_dataset_cached(task, size, pair)
        if model_name == "SVM":
            model = get_svm_model(**config["hyperparameters"]["svm"])
        elif model_name == "KNN":
            model = get_knn_model(**config["hyperparameters"]["knn"])
        elif model_name == "RF":
            model = get_random_forest_model(**config["hyperparameters"]["random_forest"])
        else:
            raise ValueError(f"Unknown model: {model_name}")
        model.fit(X_train, y_train)
        X_test_out, y_test_out = X_test, y_test

    y_pred = model.predict(X_test_out)
    y_proba = model.predict_proba(X_test_out) if hasattr(model, "predict_proba") else None
    task_type = "multiclass" if task == "multiclass" else "binary"
    metrics = compute_metrics(y_test_out, y_pred, task_type)
    cm = compute_confusion_matrix(y_test_out, y_pred)
    return model, X_test_out, y_test_out, y_pred, y_proba, metrics, cm


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------

st.sidebar.markdown(
    """
<div style="text-align:center; padding: 0.4rem 0 1rem 0;">
    <div style="font-size:2.2rem;">🔐</div>
    <div style="font-family:'Poppins',sans-serif; font-weight:800; font-size:1.35rem;
                background: linear-gradient(90deg, #A5B4FC, #F0ABFC);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        CipherBench
    </div>
    <div style="font-size:0.8rem; opacity:0.75;">Block cipher ID via machine learning</div>
</div>
""",
    unsafe_allow_html=True,
)

SECTIONS = [
    "🏠 Home / Overview",
    "🔍 Dataset Explorer",
    "🧪 Cipher Identification",
    "📊 Model Comparison",
    "📖 Paper Comparison",
    "📈 Results",
    "🗂️ Experiment Matrix",
    "📚 Research / Methodology",
    "⚙️ About / Reproducibility",
]
section = st.sidebar.radio("Navigate", SECTIONS, label_visibility="collapsed")
section = section.split(" ", 1)[1]  # strip the emoji back off for comparisons below

results_df, results_filename = load_results_df()
st.sidebar.markdown("---")
if results_df is not None:
    st.sidebar.success(f"✅ {len(results_df)} verified results loaded\n\n`{results_filename}`")
else:
    st.sidebar.warning("⚠️ No experiment results file found yet.")


# --------------------------------------------------------------------------
# Home / Overview
# --------------------------------------------------------------------------

if section == "Home / Overview":
    page_header(
        "CipherBench",
        "Identifying block cipher algorithms from ciphertext, using statistics — not brute force.",
        icon="🔐",
    )

    st.markdown(
        """
CipherBench determines **which block cipher algorithm** produced a ciphertext
(AES, 3DES, Blowfish, CAST or RC2) using only 10 NIST-randomness-derived
statistical features of the ciphertext — no key or plaintext required.

The project reproduces the **HKNNRF** (Hybrid K-Nearest-Neighbours +
Random Forest) method of Yuan et al. (2022) and extends it with two deep
learning baselines (MLP, 1D-CNN), comparing all six models fairly across
**two tasks** (binary pairwise, five-class) and **five ciphertext sizes**
(1KB – 512KB).
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧬 Ciphers", len(ALGORITHMS))
    col2.metric("📏 Ciphertext sizes", len(SIZES))
    col3.metric("🤖 Models", len(ALL_MODELS))
    col4.metric("✅ Verified experiments", len(results_df) if results_df is not None else "N/A")

    st.markdown("#### Supported ciphers")
    chip_html = "".join(
        f'<span style="display:inline-block; margin:4px 6px 4px 0; padding:6px 14px; '
        f'border-radius:999px; color:white; font-weight:600; font-size:0.85rem; '
        f'background:{ALGO_COLORS[a]};">{a}</span>'
        for a in ALGORITHMS
    )
    st.markdown(chip_html, unsafe_allow_html=True)

    st.markdown("#### Model lineup")
    chip_html = "".join(
        f'<span style="display:inline-block; margin:4px 6px 4px 0; padding:6px 14px; '
        f'border-radius:999px; color:white; font-weight:600; font-size:0.85rem; '
        f'background:{MODEL_COLORS[m]};">{m}</span>'
        for m in ALL_MODELS
    )
    st.markdown(chip_html, unsafe_allow_html=True)

    st.markdown("#### Architecture overview")
    st.markdown(
        """
```
data/ (55 CSVs)  →  src/utils/data_loader  →  train/test split (stratified, seeded)
                                                    │
                        ┌───────────────────────────┼───────────────────────────┐
                        ▼                            ▼                           ▼
                 SVM / KNN / RF                   HKNNRF                    MLP / CNN
                 (src/models/baseline.py)   (src/models/hknnrf.py)   (src/models/mlp.py, cnn.py)
                        │                            │                           │
                        └───────────────────────────┼───────────────────────────┘
                                                      ▼
                                    src/utils/metrics.py → experiments/results/*.csv, *.json
                                                      │
                                                      ▼
                                         this Streamlit app (app/streamlit_app.py)
```
        """
    )
    st.info(
        "💡 This app reads directly from the CSV/JSON files in `experiments/results/` and "
        "`data/`. No database connection is required to explore the project."
    )


# --------------------------------------------------------------------------
# Dataset Explorer
# --------------------------------------------------------------------------

elif section == "Dataset Explorer":
    page_header("Dataset Explorer", "Inspect the 55 datasets that feed every model.", icon="🔍")

    task = st.radio("Task", ["multiclass", "binary"], horizontal=True)
    size = st.selectbox("Ciphertext size", SIZES, index=4)

    pair = None
    if task == "binary":
        pair = st.selectbox("Algorithm pair", get_all_binary_pairs())

    try:
        X_train, X_test, y_train, y_test = load_dataset_cached(task, size, pair)
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.stop()

    n_total = len(X_train) + len(X_test)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total samples", n_total)
    c2.metric("Train / Test", f"{len(X_train)} / {len(X_test)}")
    c3.metric("Features", X_train.shape[1])

    st.markdown("#### Class distribution (full dataset)")
    y_all = np.concatenate([y_train, y_test])
    dist = pd.Series(y_all).value_counts().sort_index()
    if task == "multiclass":
        label_map = load_config()["algorithms"]["label_mapping"]
        dist.index = [label_map.get(int(i), str(i)) for i in dist.index]
        colors = [ALGO_COLORS.get(str(i), "#6366F1") for i in dist.index]
    else:
        dist.index = [str(i) for i in dist.index]
        colors = [ALGO_COLORS.get(str(i), MODEL_COLORS["SVM"]) for i in dist.index]

    fig = go.Figure(
        go.Bar(
            x=dist.index, y=dist.values, marker_color=colors,
            text=dist.values, textposition="outside",
        )
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=340,
        margin=dict(t=20, l=10, r=10, b=10), yaxis_title="Samples",
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Feature preview (first 15 rows of training data)")
    feat_names = get_feature_names()
    preview = pd.DataFrame(X_train[:15], columns=feat_names)
    preview["label"] = y_train[:15]
    st.dataframe(preview, use_container_width=True)

    st.markdown("#### Feature summary statistics")
    st.dataframe(
        pd.DataFrame(X_train, columns=feat_names).describe().T.style.background_gradient(
            cmap="Purples", subset=["mean", "std"]
        ),
        use_container_width=True,
    )


# --------------------------------------------------------------------------
# Cipher Identification (live demo)
# --------------------------------------------------------------------------

elif section == "Cipher Identification":
    page_header(
        "Cipher Identification",
        "Train any of the 6 models live and watch it identify ciphers on held-out data.",
        icon="🧪",
    )
    st.caption(
        "Trains a fresh model on the selected task/size and evaluates it on the held-out "
        "test split, using the same seeded 80/20 split used for the verified experiments. "
        "Classical models finish in well under a second; MLP/CNN take roughly 10–20 seconds."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        task = st.radio("Task", ["multiclass", "binary"], horizontal=True)
    with col2:
        size = st.selectbox("Ciphertext size", SIZES, index=4, key="ci_size")
    with col3:
        model_name = st.selectbox(
            "Model", ALL_MODELS,
            help="All 6 models are trainable live. MLP/CNN take longer (~10–20s) "
                 "since they train a small neural network with early stopping.",
        )

    pair = None
    if task == "binary":
        pair = st.selectbox("Algorithm pair", get_all_binary_pairs(), key="ci_pair")

    is_deep = model_name in DEEP_MODELS
    badge = (
        '<span class="cb-badge cb-badge-model">🧠 Neural network — trains in ~10–20s</span>'
        if is_deep else
        '<span class="cb-badge" style="background:#E0E7FF; color:#3730A3;">⚡ Classical model — trains instantly</span>'
    )
    st.markdown(badge, unsafe_allow_html=True)
    st.write("")

    if st.button("🚀 Train & Evaluate", type="primary"):
        steps = (
            ["Loading dataset…", "Carving out a validation split…", "Training the network…", "Evaluating on the test set…"]
            if is_deep else
            ["Loading dataset…", "Fitting the model…", "Evaluating on the test set…"]
        )
        with st.status(f"Training {model_name}…", expanded=True) as status:
            for s in steps[:-1]:
                st.write(s)
            model, X_test, y_test, y_pred, y_proba, metrics, cm = run_live_model(
                model_name, task, size, pair
            )
            st.write(steps[-1])
            status.update(label=f"{model_name} trained ✅", state="complete", expanded=False)

        st.session_state["ci_result"] = (
            model_name, task, size, pair, X_test, y_test, y_pred, y_proba, metrics, cm
        )
        st.balloons()

    if "ci_result" in st.session_state:
        model_name, task, size, pair, X_test, y_test, y_pred, y_proba, metrics, cm = st.session_state["ci_result"]

        st.markdown(f"### Results — {model_name} on {'5-class' if task == 'multiclass' else pair} ({size})")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        m2.metric("Precision", f"{metrics['precision']:.3f}")
        m3.metric("Recall", f"{metrics['recall']:.3f}")
        m4.metric("F1-score", f"{metrics['f1_score']:.3f}")

        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.markdown("#### Confusion matrix (test set)")
            labels = sorted(set(np.unique(y_test)) | set(np.unique(y_pred)))
            if task == "multiclass":
                label_map = load_config()["algorithms"]["label_mapping"]
                display_labels = [label_map.get(int(i), str(i)) for i in labels]
            else:
                display_labels = [str(i) for i in labels]

            fig = px.imshow(
                cm, x=display_labels, y=display_labels, text_auto=True,
                color_continuous_scale="Purples", aspect="auto",
                labels=dict(x="Predicted", y="True", color="Count"),
            )
            fig.update_layout(
                template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=380,
                margin=dict(t=10, l=10, r=10, b=10),
                transition=dict(duration=400, easing="cubic-in-out"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("#### Confidence on one test sample")
            sample_idx = st.slider("Sample index", 0, len(X_test) - 1, 0)
            true_val = y_test[sample_idx]
            pred_val = y_pred[sample_idx]
            correct = bool(true_val == pred_val)
            badge_cls = "cb-badge-correct" if correct else "cb-badge-wrong"
            badge_txt = "✅ Correct" if correct else "❌ Incorrect"
            st.markdown(
                f'<span class="cb-badge {badge_cls}">{badge_txt}</span>&nbsp;&nbsp;'
                f'True: <b>{true_val}</b> &nbsp;·&nbsp; Predicted: <b>{pred_val}</b>',
                unsafe_allow_html=True,
            )
            if y_proba is not None:
                proba_row = y_proba[sample_idx]
                classes = getattr(model, "classes_", np.unique(np.concatenate([y_test, y_pred])))
                if task == "multiclass":
                    label_map = load_config()["algorithms"]["label_mapping"]
                    class_labels = [label_map.get(int(c), str(c)) for c in classes]
                else:
                    class_labels = [str(c) for c in classes]
                order = np.argsort(proba_row)[::-1]
                fig2 = go.Figure(
                    go.Bar(
                        x=[proba_row[i] for i in order],
                        y=[class_labels[i] for i in order],
                        orientation="h",
                        marker_color=MODEL_COLORS.get(model_name, "#6366F1"),
                        text=[f"{proba_row[i]:.1%}" for i in order],
                        textposition="outside",
                    )
                )
                fig2.update_layout(
                    template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=380,
                    margin=dict(t=10, l=10, r=30, b=10), xaxis_range=[0, 1],
                    xaxis_title="Predicted probability",
                    transition=dict(duration=400, easing="cubic-in-out"),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("This model does not expose class probabilities.")

        st.markdown("#### Sample predictions")
        feat_names = get_feature_names()
        n_show = min(10, len(X_test))
        sample_df = pd.DataFrame(X_test[:n_show], columns=feat_names)
        sample_df["true_label"] = y_test[:n_show]
        sample_df["predicted_label"] = y_pred[:n_show]
        sample_df["correct"] = sample_df["true_label"] == sample_df["predicted_label"]
        st.dataframe(
            sample_df.style.apply(
                lambda row: ["background-color: #D1FAE5" if row["correct"] else "background-color: #FEE2E2"] * len(row),
                axis=1,
            ),
            use_container_width=True,
        )


# --------------------------------------------------------------------------
# Model Comparison
# --------------------------------------------------------------------------

elif section == "Model Comparison":
    page_header("Model Comparison", "See how all 6 models stack up, side by side.", icon="📊")

    if results_df is None:
        st.warning("No results file found. Run `python experiments/run_complete_experiments.py` first.")
        st.stop()

    task = st.radio("Task", sorted(results_df["task_type"].unique()), horizontal=True)
    size_options = sorted(
        results_df.loc[results_df["task_type"] == task, "ciphertext_size"].unique()
    )
    size = st.selectbox("Ciphertext size", size_options)

    subset = results_df[(results_df["task_type"] == task) & (results_df["ciphertext_size"] == size)]
    agg = subset.groupby("model_name")[["accuracy", "precision", "recall", "f1_score"]].mean()
    agg = agg.reindex([m for m in ALL_MODELS if m in agg.index])

    st.markdown(f"#### Mean metrics — {task}, {size}")
    st.dataframe(agg.style.format("{:.3f}").background_gradient(cmap="YlGn", subset=["accuracy"]),
                 use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Accuracy by model")
        fig = go.Figure(
            go.Bar(
                x=agg.index, y=agg["accuracy"],
                marker_color=[MODEL_COLORS.get(m, "#6366F1") for m in agg.index],
                text=[f"{v:.1%}" for v in agg["accuracy"]], textposition="outside",
            )
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=380,
            margin=dict(t=10, l=10, r=10, b=10), yaxis_title="Accuracy",
            transition=dict(duration=500, easing="elastic"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### All 4 metrics at once")
        metrics_cols = ["accuracy", "precision", "recall", "f1_score"]
        fig = go.Figure()
        for m in agg.index:
            values = agg.loc[m, metrics_cols].tolist()
            fig.add_trace(
                go.Scatterpolar(
                    r=values + values[:1],
                    theta=metrics_cols + metrics_cols[:1],
                    fill="toself", name=m, opacity=0.55,
                    line_color=MODEL_COLORS.get(m, "#6366F1"),
                )
            )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=380,
            polar=dict(radialaxis=dict(visible=True, range=[0, max(0.05, agg[metrics_cols].values.max() * 1.15)])),
            margin=dict(t=20, l=30, r=30, b=10), showlegend=True,
            transition=dict(duration=500, easing="cubic-in-out"),
        )
        st.plotly_chart(fig, use_container_width=True)

    if subset["training_time"].notna().any():
        st.markdown("#### Mean training time (s)")
        tt = subset.groupby("model_name")["training_time"].mean().reindex(
            [m for m in ALL_MODELS if m in agg.index]
        )
        fig = go.Figure(
            go.Bar(
                x=tt.index, y=tt.values,
                marker_color=[MODEL_COLORS.get(m, "#6366F1") for m in tt.index],
            )
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=320,
            margin=dict(t=10, l=10, r=10, b=10), yaxis_title="Seconds",
        )
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# Paper Comparison — designed to be shown directly to evaluators/teachers
# --------------------------------------------------------------------------

elif section == "Paper Comparison":
    page_header(
        "CipherBench vs. the Base Paper",
        "Every reported metric, side-by-side with Yuan et al. (2022) — model by model, size by size.",
        icon="📖",
    )

    st.markdown(f'<div class="cb-card" style="font-size:0.85rem; color:#475569;">📄 {PAPER_CITATION}</div>', unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
<div class="cb-card" style="border-left:4px solid #F59E0B;">
<b>📘 In the base paper</b><br>
<span style="color:#475569;">SVM, KNN, Random Forest, HKNNRF — all 4 evaluated on binary
(AES vs 3DES, plus HKNNRF on all 10 pairs) and five-class identification.
Numbers below are transcribed directly from the paper's Tables 3–5.</span>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
<div class="cb-card" style="border-left:4px solid #EC4899;">
<b>🧠 Our extension — not in the paper</b><br>
<span style="color:#475569;">MLP and 1D-CNN. The paper has <u>no neural-network baseline</u>,
so there is no paper number to place next to them — they are shown on their
own, using the identical evaluation protocol, to see whether deep learning
helps on this feature set.</span>
</div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")

    if results_df is None:
        st.warning("No results file found — run `python experiments/run_complete_experiments.py` first.")
        st.stop()

    view = st.radio(
        "Comparison view",
        ["Five-class identification (Table 5)", "Binary: AES vs 3DES (Table 3)", "Binary: HKNNRF, all 10 pairs (Table 4)"],
        horizontal=True,
    )
    metric = st.selectbox("Metric", ["accuracy", "precision", "recall"])

    def grouped_paper_vs_ours(paper_table, task, pair, size_labels_by_kb):
        """Build a tidy long-form dataframe: rows = model × size, columns =
        Paper / Ours for the given metric, for the 4 paper models."""
        records = []
        for kb, size_label in size_labels_by_kb.items():
            for model in PAPER_MODELS:
                paper_val = paper_table[metric][kb][model]
                ours_val = our_value(results_df, model, task, size_label, pair, metric)
                records.append({"Size": f"{kb}KB", "Model": model, "Source": "Paper", "Value": paper_val})
                records.append({"Size": f"{kb}KB", "Model": model, "Source": "Ours", "Value": ours_val})
        return pd.DataFrame(records)

    # ----------------------------------------------------------------
    if view.startswith("Five-class"):
        size_map = {1: "1KB", 8: "8KB", 64: "64KB", 256: "256KB", 512: "512KB"}
        long_df = grouped_paper_vs_ours(PAPER_MULTICLASS, "multiclass", "5-class", size_map)

        st.markdown("#### Paper vs. Ours — averaged across all 5 ciphertext sizes")
        avg = long_df.dropna(subset=["Value"]).groupby(["Model", "Source"])["Value"].mean().reset_index()
        fig = px.bar(
            avg, x="Model", y="Value", color="Source", barmode="group",
            color_discrete_map={"Paper": "#94A3B8", "Ours": "#6366F1"},
            category_orders={"Model": PAPER_MODELS}, text_auto=".2f",
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=380,
            yaxis_title=metric.capitalize(), margin=dict(t=10, l=10, r=10, b=10),
            transition=dict(duration=500, easing="cubic-in-out"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Per-size breakdown for one model")
        pick_model = st.selectbox("Model", PAPER_MODELS, key="mc_pick")
        detail = long_df[long_df["Model"] == pick_model]
        fig2 = px.line(
            detail, x="Size", y="Value", color="Source", markers=True,
            color_discrete_map={"Paper": "#94A3B8", "Ours": MODEL_COLORS[pick_model]},
            category_orders={"Size": ["1KB", "8KB", "64KB", "256KB", "512KB"]},
        )
        fig2.update_traces(line=dict(width=3), marker=dict(size=9))
        fig2.update_layout(
            template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=340,
            yaxis_title=metric.capitalize(), margin=dict(t=10, l=10, r=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Full comparison table (with gap)")
        table = long_df.pivot_table(index=["Size", "Model"], columns="Source", values="Value").reset_index()
        table["Gap (Ours − Paper)"] = table["Ours"] - table["Paper"]
        st.dataframe(
            table.style.format({"Paper": "{:.3f}", "Ours": "{:.3f}", "Gap (Ours − Paper)": "{:+.3f}"})
            .background_gradient(cmap="RdYlGn", subset=["Gap (Ours − Paper)"], vmin=-0.15, vmax=0.15),
            use_container_width=True, height=420,
        )

        st.markdown("#### 🧠 Our extension beyond the paper: MLP & 1D-CNN (five-class)")
        ext_records = []
        for kb, size_label in size_map.items():
            for model in ["MLP", "CNN"]:
                v = our_value(results_df, model, "multiclass", size_label, "5-class", metric)
                ext_records.append({"Size": f"{kb}KB", "Model": model, "Value": v})
        ext_df = pd.DataFrame(ext_records)
        fig3 = px.bar(
            ext_df, x="Size", y="Value", color="Model", barmode="group",
            color_discrete_map={"MLP": MODEL_COLORS["MLP"], "CNN": MODEL_COLORS["CNN"]},
            category_orders={"Size": ["1KB", "8KB", "64KB", "256KB", "512KB"]},
        )
        fig3.update_layout(
            template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=340,
            yaxis_title=metric.capitalize(), margin=dict(t=10, l=10, r=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("No paper bar here by design — the paper never tested a neural network on this task.")

    # ----------------------------------------------------------------
    elif view.startswith("Binary: AES vs 3DES"):
        size_map_upper = {1: "1KB", 8: "8KB", 64: "64KB", 256: "256KB", 512: "512KB"}
        size_map_lower = {kb: lbl.lower() for kb, lbl in size_map_upper.items()}
        long_df = grouped_paper_vs_ours(PAPER_BINARY_AES_3DES, "binary", "AES and 3DES", size_map_lower)

        st.markdown("#### Paper vs. Ours — averaged across all 5 ciphertext sizes (AES vs 3DES)")
        avg = long_df.dropna(subset=["Value"]).groupby(["Model", "Source"])["Value"].mean().reset_index()
        fig = px.bar(
            avg, x="Model", y="Value", color="Source", barmode="group",
            color_discrete_map={"Paper": "#94A3B8", "Ours": "#6366F1"},
            category_orders={"Model": PAPER_MODELS}, text_auto=".2f",
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=380,
            yaxis_title=metric.capitalize(), margin=dict(t=10, l=10, r=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Full comparison table (with gap)")
        table = long_df.pivot_table(index=["Size", "Model"], columns="Source", values="Value").reset_index()
        table["Gap (Ours − Paper)"] = table["Ours"] - table["Paper"]
        st.dataframe(
            table.style.format({"Paper": "{:.3f}", "Ours": "{:.3f}", "Gap (Ours − Paper)": "{:+.3f}"})
            .background_gradient(cmap="RdYlGn", subset=["Gap (Ours − Paper)"], vmin=-0.3, vmax=0.3),
            use_container_width=True, height=420,
        )

        st.markdown("#### 🧠 Our extension beyond the paper: MLP & 1D-CNN (AES vs 3DES)")
        ext_records = []
        for kb, size_label in size_map_lower.items():
            for model in ["MLP", "CNN"]:
                v = our_value(results_df, model, "binary", size_label, "AES and 3DES", metric)
                ext_records.append({"Size": f"{kb}KB", "Model": model, "Value": v})
        ext_df = pd.DataFrame(ext_records)
        fig3 = px.bar(
            ext_df, x="Size", y="Value", color="Model", barmode="group",
            color_discrete_map={"MLP": MODEL_COLORS["MLP"], "CNN": MODEL_COLORS["CNN"]},
            category_orders={"Size": ["1KB", "8KB", "64KB", "256KB", "512KB"]},
        )
        fig3.update_layout(
            template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=340,
            yaxis_title=metric.capitalize(), margin=dict(t=10, l=10, r=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("No paper bar here by design — the paper never tested a neural network on this task.")

    # ----------------------------------------------------------------
    else:  # Binary: HKNNRF, all 10 pairs
        pairs = list(PAPER_HKNNRF_BINARY_PAIRS.keys())
        size_order = ["1KB", "8KB", "64KB", "256KB", "512KB"]

        paper_matrix = pd.DataFrame(
            {f"{kb}KB": [PAPER_HKNNRF_BINARY_PAIRS[p][kb] for p in pairs] for kb in [1, 8, 64, 256, 512]},
            index=pairs,
        )[size_order]
        ours_matrix = pd.DataFrame(
            {
                f"{kb}KB": [
                    our_value(results_df, "HKNNRF", "binary", f"{kb}kb", p, "accuracy") for p in pairs
                ]
                for kb in [1, 8, 64, 256, 512]
            },
            index=pairs,
        )[size_order]

        st.markdown("#### HKNNRF binary accuracy — Paper (Table 4) vs. Ours, all 10 pairs")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Paper")
            fig = px.imshow(paper_matrix, text_auto=".2f", color_continuous_scale="Oranges", zmin=0, zmax=1, aspect="auto")
            fig.update_layout(template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=420, margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.caption("Ours")
            fig = px.imshow(ours_matrix, text_auto=".2f", color_continuous_scale="Purples", zmin=0, zmax=1, aspect="auto")
            fig.update_layout(template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=420, margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Gap (Ours − Paper)")
        delta = ours_matrix - paper_matrix
        fig = px.imshow(
            delta, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=-0.3, zmax=0.3, aspect="auto",
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=420, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🧠 Our extension beyond the paper: MLP & CNN, all 10 pairs (no paper baseline)")
        ext_model = st.selectbox("Model", ["MLP", "CNN"], key="hknnrf_ext_model")
        ext_matrix = pd.DataFrame(
            {
                f"{kb}KB": [
                    our_value(results_df, ext_model, "binary", f"{kb}kb", p, "accuracy") for p in pairs
                ]
                for kb in [1, 8, 64, 256, 512]
            },
            index=pairs,
        )[size_order]
        fig = px.imshow(
            ext_matrix, text_auto=".2f", color_continuous_scale="Purples",
            zmin=0, zmax=1, aspect="auto",
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=420, margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **Why the gap?** An ANOVA test on these 10 NIST p-value features found none "
        "statistically significant for distinguishing algorithms — the feature set carries "
        "less signal than whatever the paper's authors used, and every model (including "
        "HKNNRF, MLP and CNN) shows the same ceiling. Full discussion in "
        "`docs/PROJECT_REPORT.md` §7."
    )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

elif section == "Results":
    page_header("Verified Experiment Results", "Every one of the 330 experiments, filterable.", icon="📈")

    if results_df is None:
        st.warning("No results file found.")
        st.stop()

    st.caption(
        f"Source: `experiments/results/{results_filename}` — {len(results_df)} experiments. "
        "See docs/PROJECT_REPORT.md and docs/RESULTS.md for methodology and headline numbers."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        model_filter = st.multiselect("Model", ALL_MODELS, default=ALL_MODELS)
    with c2:
        task_filter = st.multiselect(
            "Task", sorted(results_df["task_type"].unique()),
            default=sorted(results_df["task_type"].unique()),
        )
    with c3:
        size_filter = st.multiselect(
            "Size", sorted(results_df["ciphertext_size"].unique()),
            default=sorted(results_df["ciphertext_size"].unique()),
        )

    filtered = results_df[
        results_df["model_name"].isin(model_filter)
        & results_df["task_type"].isin(task_filter)
        & results_df["ciphertext_size"].isin(size_filter)
    ]

    sort_col = st.selectbox("Sort by", ["accuracy", "f1_score", "precision", "recall", "training_time"])
    filtered = filtered.sort_values(sort_col, ascending=False)

    st.markdown("#### Accuracy distribution by model")
    fig = px.violin(
        filtered, x="model_name", y="accuracy", color="model_name",
        color_discrete_map=MODEL_COLORS, box=True, points="all",
        category_orders={"model_name": [m for m in ALL_MODELS if m in filtered["model_name"].unique()]},
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=380, showlegend=False,
        margin=dict(t=10, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Filtered results table")
    st.dataframe(filtered, use_container_width=True, height=420)
    st.download_button(
        "📥 Download filtered results as CSV",
        data=filtered.to_csv(index=False),
        file_name="cipherbench_filtered_results.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------
# Experiment Matrix
# --------------------------------------------------------------------------

elif section == "Experiment Matrix":
    page_header("Experiment Matrix Coverage", "330 experiments: 6 models × 2 tasks × 5 sizes.", icon="🗂️")

    if results_df is None:
        st.warning("No results file found.")
        st.stop()

    expected_multiclass = len(SIZES) * len(ALL_MODELS)
    expected_binary = len(get_all_binary_pairs()) * len(SIZES) * len(ALL_MODELS)
    expected_total = expected_multiclass + expected_binary

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected total", expected_total)
    c2.metric("Present in results file", len(results_df))
    c3.metric("Coverage", f"{len(results_df) / expected_total * 100:.1f}%")

    st.markdown("#### Coverage heatmap — count of runs per (model, size)")
    pivot = results_df.pivot_table(
        index="model_name", columns="ciphertext_size", values="accuracy", aggfunc="count"
    ).reindex(index=[m for m in ALL_MODELS if m in results_df["model_name"].unique()])
    fig = px.imshow(
        pivot, text_auto=True, color_continuous_scale="Blues", aspect="auto",
        labels=dict(x="Ciphertext size", y="Model", color="Runs"),
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, font_family=FONT_FAMILY, height=340,
        margin=dict(t=10, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Data quality check")
    zero_rows = results_df[results_df["accuracy"] == 0.0]
    if len(zero_rows) == 0:
        st.success("✅ All rows have a non-zero accuracy value.")
    else:
        st.warning(f"{len(zero_rows)} rows have accuracy == 0.0:")
        st.dataframe(
            zero_rows[["model_name", "task_type", "algorithm_pair", "ciphertext_size", "accuracy"]],
            use_container_width=True,
        )


# --------------------------------------------------------------------------
# Research / Methodology
# --------------------------------------------------------------------------

elif section == "Research / Methodology":
    page_header("Research / Methodology", "The paper, the pipeline, and how evaluation stays honest.", icon="📚")
    st.markdown(
        """
### Base paper
Yuan, K., Yu, D., Feng, J., Yang, L., Jia, C., Huang, Y. (2022).
*A block cipher algorithm identification scheme based on hybrid k-nearest
neighbor and random forest algorithm.* PeerJ Computer Science.

### Feature extraction
Each ciphertext sample is represented by **10 p-values from NIST
randomness tests**: Approximate Entropy, Cumulative Sums, Discrete Fourier
Transform, Frequency-within-Block, Linear Complexity, Monobit, Random
Excursions, Random Excursions Variant, Runs, and Serial. These 10 columns
are consumed as-is by every model — no additional scaling or feature
selection is applied, matching the paper's methodology.

### HKNNRF pipeline (`src/models/hknnrf.py`)
1. Split the training portion 50/50 into an RF-training set and a
   KNN-training set (both stratified, seeded).
2. Train a Random Forest on the RF-training set.
3. Extract each sample's leaf index per tree (`RandomForestClassifier.apply`)
   and one-hot encode them — these are the RF-derived features.
4. **Concatenate** the RF-derived features with the original 10 NIST
   features (paper Steps 9–10).
5. Train a KNN classifier on the combined feature vector.

### Deep learning extension (not in the original paper)
- **MLP**: 2 hidden dense layers [64, 32] with dropout 0.3, trained with
  early stopping on a validation split carved out of the training data only.
- **1D-CNN**: Conv1D layers [32, 64] with batch normalization and
  max-pooling, same validation discipline as the MLP.
- Framework: **TensorFlow/Keras** (`tf-nightly`), not PyTorch. The synopsis
  lists PyTorch; PyTorch does not currently ship wheels for the Python
  version used in this environment, so Keras was substituted — see
  `docs/PROJECT_REPORT.md` §6.

### Evaluation protocol
- 80/20 train/test split, stratified, fixed `random_state` per experiment.
- For MLP/CNN, an additional 80/20 split of the *training* portion produces
  the validation set used for early stopping. **The test set is never seen
  until the single final `predict` call.** This app's live demo (Cipher
  Identification tab) follows the identical protocol for every model.
- Metrics: accuracy, weighted precision/recall/F1, full confusion matrix.
        """
    )


# --------------------------------------------------------------------------
# About / Reproducibility
# --------------------------------------------------------------------------

else:
    page_header("About / Reproducibility", "Repository layout, commands, and honest limitations.", icon="⚙️")

    st.markdown("#### Repository layout")
    st.code(
        """
CipherBench/
├── src/            # models, training script, data/metrics utilities
├── data/           # 55 CSV datasets (5 multiclass + 50 binary)
├── experiments/    # experiment runners + results/
├── app/            # this Streamlit app
├── tests/          # test suite
├── database/       # optional MySQL integration
├── api/            # optional Flask REST API
└── docs/           # PROJECT_REPORT.md, RESULTS.md
        """,
        language="text",
    )

    st.markdown("#### Commands")
    st.code(
        """# Install dependencies
pip install -r requirements.txt

# Run the test suite
python tests/test_suite.py

# Train one model on one task/size
python src/train.py --model hknnrf --task multiclass --size 512KB

# Run the full 330-experiment matrix (long-running)
python experiments/run_complete_experiments.py --no-db

# Launch this app
streamlit run app/streamlit_app.py
        """,
        language="bash",
    )

    st.markdown("#### Reproducibility")
    st.markdown(
        """
- All train/test splits use `sklearn.model_selection.train_test_split` with
  a fixed `random_state` and `stratify=y`.
- MLP/CNN additionally call `keras.utils.set_random_seed(seed)` before
  building the model so Keras/TensorFlow/NumPy RNGs are seeded consistently.
- Full determinism is **not guaranteed** for MLP/CNN across different
  hardware/CPU instruction sets (float non-associativity), so re-running the
  deep-learning experiments may shift accuracy by a few percentage points
  even with a fixed seed. Classical models (SVM/KNN/RF/HKNNRF) are exactly
  reproducible.
        """
    )

    st.markdown("#### Known limitations")
    st.markdown(
        """
- Deep-learning framework is TensorFlow/Keras, not PyTorch as listed in the
  synopsis (Python-version compatibility constraint).
- No ciphertext-generation pipeline is included; the project consumes the
  pre-extracted NIST-feature CSV datasets already provided.
- Multi-seed stability analysis and cross-ciphertext-size generalization
  (train on one size, test on another) are not part of the verified 330-run
  matrix — see `docs/PROJECT_REPORT.md` §8.
        """
    )
