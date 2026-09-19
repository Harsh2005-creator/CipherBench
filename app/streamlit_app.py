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

Presentation lives in app/style.css, app/effects.js and .streamlit/config.toml; everything
above the "Presentation" section below is data/model logic and does not
depend on either.
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
from matplotlib.colors import LinearSegmentedColormap
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
    page_title="CipherBench · Block Cipher Identification",
    page_icon=":material/lock:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SIZES = ["1KB", "8KB", "64KB", "256KB", "512KB"]
ALGORITHMS = ["AES", "3DES", "Blowfish", "CAST", "RC2"]
ALL_MODELS = ["SVM", "KNN", "RF", "HKNNRF", "MLP", "CNN"]
DEEP_MODELS = {"MLP", "CNN"}

# --------------------------------------------------------------------------
# Presentation — palette and helpers (purely visual; no data logic)
# --------------------------------------------------------------------------

ACCENT = "#7c6af7"
TEXT_COLOR = "#e8e8f0"
GRID_COLOR = "#2a2a38"
PAPER_COLOR = "#8b8ba0"
FONT_FAMILY = "Inter, sans-serif"

MODEL_COLORS = {
    "SVM": "#7c6af7",
    "KNN": "#38bdf8",
    "RF": "#22d3a5",
    "HKNNRF": "#f97316",
    "MLP": "#f43f5e",
    "CNN": "#facc15",
}
ALGO_COLORS = {
    "AES": "#7c6af7",
    "3DES": "#38bdf8",
    "Blowfish": "#22d3a5",
    "CAST": "#f97316",
    "RC2": "#f43f5e",
}

SCALE_ACCENT = ["#1c1c27", "#5a4fcf", "#7c6af7"]
SCALE_PAPER = ["#1c1c27", "#7a4a1e", "#f97316"]
SCALE_GAP = ["#f43f5e", "#1c1c27", "#22d3a5"]
CMAP_ACCENT = LinearSegmentedColormap.from_list("cb_accent", SCALE_ACCENT)
CMAP_GAP = LinearSegmentedColormap.from_list("cb_gap", ["#8f1d34", "#1c1c27", "#0f6b55"])

CSS_FILE = Path(__file__).resolve().parent / "style.css"


def load_css():
    """Inject app/style.css. If the file is missing the app still works,
    just with Streamlit's own dark theme from .streamlit/config.toml."""
    try:
        css = CSS_FILE.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


JS_FILE = Path(__file__).resolve().parent / "effects.js"


def load_effects():
    """Inject app/effects.js (scroll reveals, nav indicator, ambient field).
    It runs once per browser session; on Streamlit versions without
    st.html(unsafe_allow_javascript=...) the app simply stays static."""
    try:
        js = JS_FILE.read_text(encoding="utf-8")
        with st.container(key="fx"):
            st.html(f"<script>{js}</script>", unsafe_allow_javascript=True)
    except (OSError, TypeError):
        return


def style_fig(fig, height=360):
    """Dark, transparent Plotly figure that sits on the app's card surface."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=13),
        height=height,
        margin=dict(t=24, b=10, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1c1c27", font_color=TEXT_COLOR, bordercolor=GRID_COLOR),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    return fig


def page_title(title, sub=""):
    n = SECTIONS.index(section) + 1
    st.markdown(
        f'<div class="page-head"><div class="eyebrow">{n:02d} · {NAV_LABELS[section]}</div>'
        f'<div class="page-title">{title}</div><div class="page-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def insight(text, tone=""):
    st.markdown(f'<div class="insight-box {tone}">{text}</div>', unsafe_allow_html=True)


def chip(label, color):
    return (
        f'<span class="chip" style="background:{color}26;color:{color};'
        f'border:1px solid {color}55">{label}</span>'
    )


def class_names(values):
    """Turn numeric class labels (0-4) into cipher names for display only."""
    label_map = load_config()["algorithms"]["label_mapping"]
    return [label_map.get(int(v), str(v)) for v in values]


load_css()
load_effects()

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
# Header navigation
# --------------------------------------------------------------------------

SECTIONS = [
    "Home / Overview",
    "Dataset Explorer",
    "Cipher Identification",
    "Model Comparison",
    "Paper Comparison",
    "Results",
    "Experiment Matrix",
    "Research / Methodology",
    "About / Reproducibility",
]
# Short labels for the header bar; the page logic keys off the full names above.
NAV_LABELS = {
    "Home / Overview": "Home",
    "Dataset Explorer": "Data",
    "Cipher Identification": "Identify",
    "Model Comparison": "Models",
    "Paper Comparison": "Paper",
    "Results": "Results",
    "Experiment Matrix": "Matrix",
    "Research / Methodology": "Method",
    "About / Reproducibility": "About",
}

results_df, results_filename = load_results_df()

LOCK_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="10" rx="2.5"/>'
    '<path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>'
)

with st.container(key="navbar"):
    nav_brand, nav_links, nav_status = st.columns([1.75, 8.4, 1.15], vertical_alignment="center")
    nav_brand.markdown(
        f'<div class="brand"><span class="brand-mark">{LOCK_SVG}</span>'
        '<span class="brand-word">Cipher<b>Bench</b></span></div>',
        unsafe_allow_html=True,
    )
    section = nav_links.radio(
        "Navigate", SECTIONS, key="nav", horizontal=True,
        label_visibility="collapsed", format_func=NAV_LABELS.get,
    )
    if results_df is not None:
        _pill = f'<span class="status-pill"><span class="dot"></span><b>{len(results_df)}</b> verified</span>'
    else:
        _pill = '<span class="status-pill">No results file yet</span>'
    nav_status.markdown(
        f'<div class="nav-right">{_pill}<span class="nav-toggle" role="button" tabindex="0" '
        'aria-label="Menu"><i></i><i></i><i></i></span></div>',
        unsafe_allow_html=True,
    )


def go_to(page):
    """Button callback: switch the header navigation to another page."""
    st.session_state["nav"] = page


# --------------------------------------------------------------------------
# Home / Overview
# --------------------------------------------------------------------------

if section == "Home / Overview":
    n_exp = len(results_df) if results_df is not None else None

    # ---- 01 · Hero ------------------------------------------------------
    hero_l, hero_r = st.columns([1.12, 1], gap="large", vertical_alignment="center")
    with hero_l:
        st.markdown(
            '<div class="hero-tag"><i></i>Block cipher identification</div>'
            '<div class="hero-title" role="heading" aria-level="1"><span class="ln"><span>Identify the cipher.</span></span>'
            '<span class="ln"><span class="grad">From the ciphertext alone.</span></span></div>'
            '<div class="hero-sub">Recognise <b>AES, 3DES, Blowfish, CAST and RC2</b> from the statistical '
            "fingerprint a ciphertext leaves behind, using ten randomness measurements and machine "
            "learning. No key, no plaintext.</div>",
            unsafe_allow_html=True,
        )
        cta1, cta2, _ = st.columns([1.0, 1.15, 0.5])
        cta1.button("Try live demo", key="cta_demo", type="primary", on_click=go_to, args=("Cipher Identification",))
        cta2.button("Compare with the paper", key="cta_paper", on_click=go_to, args=("Paper Comparison",))
    with hero_r:
        _bars = [0.42, 0.78, 0.31, 0.9, 0.55, 0.2, 0.68, 0.47, 0.83, 0.36]
        st.markdown(
            '<div class="hero-vis"><div class="terminal"><div class="term-bar"><i></i><i></i><i></i>'
            "<span>cipherbench · pipeline</span></div>"
            '<div class="term-body">'
            '<span class="ln l1"><span class="p">$</span>extract <b>10</b> NIST p-values</span>'
            '<span class="ln l2"><span class="p">$</span>train <b>6</b> models · <b>2</b> tasks · <b>5</b> sizes</span>'
            f'<span class="ln l3"><span class="p">$</span>compare <b>{n_exp if n_exp is not None else "330"}</b> runs vs. <em>Yuan et al.</em></span>'
            '<span class="ln l4"><span class="p">$</span><span class="cursor"></span></span>'
            "</div></div>"
            '<div class="float f-fp"><div class="f-cap">Statistical fingerprint</div><div class="fp-bars">'
            + "".join(f'<i style="--h:{h};--k:{k}"></i>' for k, h in enumerate(_bars))
            + '</div><div class="f-foot">10 features · one ciphertext</div></div>'
            '<div class="float f-cand"><div class="f-cap">Candidates</div><div class="cand-row">'
            + "".join(
                f'<span style="color:{ALGO_COLORS[a]};background:{ALGO_COLORS[a]}1f;border:1px solid {ALGO_COLORS[a]}55">{a}</span>'
                for a in ALGORITHMS
            )
            + "</div></div></div>",
            unsafe_allow_html=True,
        )

    stat_items = [
        (len(ALGORITHMS), "Block ciphers"),
        (len(ALL_MODELS), "Models"),
        (len(SIZES), "Ciphertext sizes"),
        (n_exp, "Verified experiments"),
    ]
    st.markdown(
        '<div class="stats">'
        + "".join(
            f'<div class="stat"><div class="num" style="--to:{v}"></div><div class="lbl">{lbl}</div></div>'
            if v is not None
            else f'<div class="stat"><div class="num na">N/A</div><div class="lbl">{lbl}</div></div>'
            for v, lbl in stat_items
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # ---- 02 · The problem -----------------------------------------------
    def story(no, title, sub):
        st.markdown(
            f'<div class="story-head"><div class="s-no">{no}</div><div><h2>{title}</h2><p>{sub}</p></div></div>',
            unsafe_allow_html=True,
        )

    story("02", "A ciphertext carries no label.",
          "Before an analyst can attack an encryption, they need to know which algorithm produced it.")
    hexes = (
        "9f 3a c1 7e 04 b8 5d 22 e6 91 0c 47 ad 68 fb 13 "
        "5e c9 70 2b 8a d4 16 f0 39 bc 62 07 e5 9d 4a 81 "
        "c3 1f 76 a8 0e 5b d2 94 3d e7 28 b1 6c f5 0a 4e "
        "97 d0 33 8c 5f a2 19 e4 7b 06 c8 3e 91 fa 55 2d"
    ).split()
    p_l, p_r = st.columns([1.05, 1], gap="large", vertical_alignment="center")
    with p_l:
        st.markdown(
            '<div class="cipher-field"><div class="hexes">'
            + " ".join(f"<b>{h}</b>" if i % 7 == 0 else h for i, h in enumerate(hexes))
            + '</div><div class="qmark">?</div><div class="cand-row">'
            + "".join(
                f'<span style="color:{ALGO_COLORS[a]};background:{ALGO_COLORS[a]}1f;border:1px solid {ALGO_COLORS[a]}55">{a}</span>'
                for a in ALGORITHMS
            )
            + "</div></div>",
            unsafe_allow_html=True,
        )
    with p_r:
        st.markdown(
            '<p class="lead">Good ciphers are built to look random. The question is whether '
            "<em>how</em> they look random is still a signature.</p>"
            '<p class="muted-p">CipherBench tests that idea: measure ten NIST randomness statistics on '
            "each ciphertext and let a classifier decide which of five block ciphers it came from.</p>",
            unsafe_allow_html=True,
        )

    # ---- 03 · The fingerprint -------------------------------------------
    story("03", "Ten measurements. One fingerprint.",
          "Each ciphertext is reduced to ten NIST p-values. Those ten numbers are all a model ever sees.")
    _ic = lambda d: (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{d}</svg>'
    )
    flow = [
        (_ic('<rect x="4" y="11" width="16" height="10" rx="2.5"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>'),
         "Ciphertext", "Raw encrypted sample"),
        (_ic('<path d="M3 12h3l2-6 4 12 3-9 2 3h4"/>'), "Fingerprint", "NIST randomness tests"),
        (_ic('<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'), "10 features", "One p-value per test"),
        (_ic('<circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/>'
             '<path d="M8 7l3 9M16 7l-3 9M8.5 6h7"/>'), "ML models", "Six classifiers"),
        (_ic('<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2.6"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>'),
         "Identification", "Which cipher?"),
    ]
    st.markdown(
        '<div class="flow">'
        + "".join(f'<div class="flow-node"><div class="ic">{i}</div><b>{t}</b><span>{d}</span></div>' for i, t, d in flow)
        + "</div>",
        unsafe_allow_html=True,
    )
    feats = [
        ("Approximate Entropy", "aetPValue", "Regularity of overlapping bit patterns"),
        ("Cumulative Sums", "custPValue", "Drift of the running sum"),
        ("Discrete Fourier", "dtfPValue", "Periodic structure in the spectrum"),
        ("Block Frequency", "fwbtPValue", "Balance of ones inside blocks"),
        ("Linear Complexity", "lrobPValue", "Shift-register style structure"),
        ("Monobit", "mtPValue", "Overall balance of ones and zeros"),
        ("Random Excursions", "retPValue", "Walk visits to fixed states"),
        ("Excursions Variant", "revtPValue", "Total visits across many states"),
        ("Runs", "runsPValue", "Length of uninterrupted bit runs"),
        ("Serial", "stPValue", "Frequency of m-bit patterns"),
    ]
    st.markdown(
        '<div class="feats">'
        + "".join(
            f'<div class="feat"><div class="f-n">{i:02d}<i></i></div><div class="f-t">{t}</div>'
            f'<div class="f-d">{d}</div><code>{c}</code></div>'
            for i, (t, c, d) in enumerate(feats, 1)
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # ---- 04 · The models ------------------------------------------------
    story("04", "Six models, one question.",
          "Four classifiers reproduce the base paper. Two neural networks are this project's own extension.")
    m_ic = {
        "SVM": '<path d="M4 20 20 4M4 12l8-8M12 20l8-8"/>',
        "KNN": '<circle cx="12" cy="12" r="2"/><circle cx="5" cy="7" r="1.6"/><circle cx="19" cy="8" r="1.6"/><circle cx="7" cy="18" r="1.6"/><circle cx="18" cy="17" r="1.6"/><path d="M12 12 5 7M12 12l7-4M12 12l-5 6M12 12l6 5"/>',
        "RF": '<path d="M12 21V11M12 11 6 5M12 11l6-6M12 16 8 12M12 16l4-4"/><circle cx="6" cy="5" r="1.4"/><circle cx="18" cy="5" r="1.4"/>',
        "HKNNRF": '<circle cx="8" cy="12" r="5"/><circle cx="16" cy="12" r="5"/>',
        "MLP": '<circle cx="5" cy="6" r="1.7"/><circle cx="5" cy="18" r="1.7"/><circle cx="12" cy="4" r="1.7"/><circle cx="12" cy="12" r="1.7"/><circle cx="12" cy="20" r="1.7"/><circle cx="19" cy="12" r="1.7"/><path d="M6.5 6.5 10.5 4.5M6.5 6.5l4 5M6.5 18l4-5.5M6.5 18l4 2M13.5 4.5l4 6.5M13.5 12h4M13.5 19.5l4-6.5"/>',
        "CNN": '<rect x="3" y="4" width="7" height="7" rx="1"/><rect x="7" y="8" width="7" height="7" rx="1"/><rect x="11" y="12" width="7" height="7" rx="1"/>',
    }
    m_info = [
        ("SVM", "Support vector machine", "Separates classes with a maximum-margin boundary.", False),
        ("KNN", "k-nearest neighbours", "Labels a sample by its closest neighbours in feature space.", False),
        ("RF", "Random forest", "Votes across many randomised decision trees.", False),
        ("HKNNRF", "Hybrid KNN + RF", "The paper's method: forest leaves feed a nearest-neighbour vote.", False),
        ("MLP", "Multi-layer perceptron", "A small feed-forward neural network on the ten features.", True),
        ("CNN", "1-D convolutional net", "Convolutions slide across the ordered feature vector.", True),
    ]
    st.markdown(
        '<div class="eco-wrap"><span class="eco-node"><i></i>10 NIST features</span>'
        '<div class="eco-stem"></div><div class="eco-bus"></div><div class="eco-stem"></div></div>'
        '<div class="models">'
        + "".join(
            f'<div class="model-card{" ext" if ext else ""}" style="--mc:{MODEL_COLORS[k]}">'
            f'<div class="m-ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{m_ic[k]}</svg></div>'
            f'<h5>{k}{"<span class=badge-ext>Extension</span>" if ext else "<span class=badge-paper>Paper</span>"}</h5>'
            f'<div class="m-type">{full}</div><p>{txt}</p></div>'
            for k, full, txt, ext in m_info
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # ---- 05 · The experiment --------------------------------------------
    story("05", "Every result, verified.",
          "One seeded run per model, task, ciphertext size and cipher pair. Each square below is one of them.")
    n_show = n_exp if n_exp is not None else 330
    st.markdown(
        f'<div class="big330"><div class="num" style="--to:{n_show}"></div>'
        '<div class="cap"><b>Verified experiments</b>5 block ciphers · 6 models<br>'
        "1 five-class task + 10 pairwise tasks · 5 sizes</div></div>"
        '<div class="facts">'
        + "".join(
            f'<div class="fact"><div class="num" style="--to:{v}"></div><div class="lbl">{lbl}</div></div>'
            for v, lbl in [(6, "Models"), (2, "Task types"), (5, "Ciphertext sizes"), (10, "Cipher pairs")]
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    if results_df is not None and {"model_name", "task_type"} <= set(results_df.columns):
        _rows = results_df.sort_values(["task_type", "ciphertext_size", "model_name"]).reset_index(drop=True)
        cells = "".join(
            f'<i class="{"bin" if r.task_type != "multiclass" else ""}" '
            f'style="--c:{MODEL_COLORS.get(r.model_name, ACCENT)};--d:{min(i * 4, 1400)}" '
            f'title="{r.model_name} · {r.algorithm_pair} · {r.ciphertext_size}"></i>'
            for i, r in enumerate(_rows.itertuples())
        )
        st.markdown(
            f'<div class="exp-grid">{cells}</div><div class="legend">'
            + "".join(f'<span style="--c:{c}"><i></i>{m}</span>' for m, c in MODEL_COLORS.items())
            + "<span>Brighter squares: five-class runs. Dimmer: pairwise runs.</span></div>",
            unsafe_allow_html=True,
        )

    # ---- 06 · The evidence ----------------------------------------------
    story("06", "What the models achieve.",
          "Mean accuracy per model across every ciphertext size. Pairwise tasks are easier than five-class.")
    if results_df is not None:
        _agg = results_df.groupby(["model_name", "task_type"], as_index=False)["accuracy"].mean()
        _agg["Task"] = _agg["task_type"].map({"multiclass": "Five-class", "binary": "Pairwise"}).fillna(_agg["task_type"])
        fig_ev = px.bar(
            _agg, x="model_name", y="accuracy", color="Task", barmode="group",
            category_orders={"model_name": ALL_MODELS},
            color_discrete_map={"Five-class": "#f97316", "Pairwise": "#7c6af7"},
            labels={"model_name": "Model", "accuracy": "Mean accuracy"},
        )
        fig_ev.update_yaxes(tickformat=".0%", range=[0, 1])
        fig_ev.update_traces(marker_line_width=0)
        fig_ev.update_layout(legend=dict(orientation="h", y=1.1, x=0, title_text=""))
        st.plotly_chart(style_fig(fig_ev, 380), use_container_width=True, key="home_evidence")

    # ---- 07 · Research comparison ---------------------------------------
    story("07", "Measured against the paper.",
          "HKNNRF is the base paper's headline method, so it is the fairest head-to-head. Mean accuracy over all five sizes.")

    def _bar_rows(paper_tbl, task, pair):
        pv = np.mean([paper_tbl["accuracy"][int(s[:-2])]["HKNNRF"] for s in SIZES])
        ov = [our_value(results_df, "HKNNRF", task, s, pair, "accuracy") for s in SIZES]
        ov = [v for v in ov if v is not None]
        rows = [("Paper", pv, "paper")]
        if ov:
            rows.append(("Ours", float(np.mean(ov)), "ours"))
        return "".join(
            f'<div class="vs-row"><span>{n}</span><div class="vs-track"><i class="{c}" style="--w:{v * 100:.1f}%"></i></div>'
            f"<b>{v * 100:.0f}%</b></div>"
            for n, v, c in rows
        )

    st.markdown(
        '<div class="vs"><div class="vs-block"><h5>Five-class identification</h5>'
        '<div class="vs-sub">HKNNRF · all five ciphers at once</div>'
        + _bar_rows(PAPER_MULTICLASS, "multiclass", "5-class")
        + '</div><div class="vs-block"><h5>Binary identification</h5>'
        '<div class="vs-sub">HKNNRF · AES versus 3DES</div>'
        + _bar_rows(PAPER_BINARY_AES_3DES, "binary", "AES and 3DES")
        + "</div></div>",
        unsafe_allow_html=True,
    )

    # ---- 08 · Explore ---------------------------------------------------
    story("08", "Explore it yourself.", "Every page reads from the same verified result set.")
    _go = lambda d: (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{d}</svg>'
    )
    cards = [
        ("Identify", "Run a live model", "Train any of six models and identify a cipher.",
         '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2.6"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>'),
        ("Paper", "Paper comparison", "Side-by-side with Yuan et al. (2022).",
         '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>'),
        ("Matrix", "Experiment matrix", "All 330 runs and their coverage.",
         '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>'),
        ("Data", "Dataset explorer", "Browse the 55 CSV datasets.",
         '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>'),
    ]
    st.markdown(
        '<div class="go-grid">'
        + "".join(
            f'<a class="go-card" data-go="{k}" role="link" tabindex="0"><div class="g-ic">{_go(ic)}</div>'
            f"<b>{t}</b><span>{d}</span></a>"
            for k, t, d, ic in cards
        )
        + "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Dataset Explorer
# --------------------------------------------------------------------------

elif section == "Dataset Explorer":
    page_title("Dataset Explorer", "Browse the 55 CSV datasets that feed every model.")

    ctl1, ctl2, ctl3 = st.columns([1.2, 1, 1.6])
    with ctl1:
        task = st.radio("Task", ["multiclass", "binary"], horizontal=True)
    with ctl2:
        size = st.selectbox("Ciphertext size", SIZES, index=4)
    pair = None
    with ctl3:
        if task == "binary":
            pair = st.selectbox("Algorithm pair", get_all_binary_pairs())

    try:
        X_train, X_test, y_train, y_test = load_dataset_cached(task, size, pair)
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.stop()

    feat_names = get_feature_names()
    tab_over, tab_prev, tab_stats = st.tabs(["Overview", "Feature preview", "Statistics"])

    with tab_over:
        n_total = len(X_train) + len(X_test)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total samples", n_total)
        c2.metric("Train / test", f"{len(X_train)} / {len(X_test)}")
        c3.metric("Features", X_train.shape[1])

        st.markdown("#### Class distribution")
        y_all = np.concatenate([y_train, y_test])
        dist = pd.Series(y_all).value_counts().sort_index()
        dist.index = class_names(dist.index)
        colors = [ALGO_COLORS.get(name, ACCENT) for name in dist.index]

        fig = go.Figure(
            go.Bar(
                x=dist.index, y=dist.values, marker_color=colors,
                text=dist.values, textposition="outside",
            )
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(yaxis_title="Samples")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    with tab_prev:
        st.markdown("#### First 15 training rows")
        preview = pd.DataFrame(X_train[:15], columns=feat_names)
        preview["label"] = y_train[:15]
        st.dataframe(preview, use_container_width=True, hide_index=True)

    with tab_stats:
        st.markdown("#### Feature summary statistics")
        st.dataframe(
            pd.DataFrame(X_train, columns=feat_names).describe().T.style.background_gradient(
                cmap=CMAP_ACCENT, subset=["mean", "std"]
            ),
            use_container_width=True,
        )


# --------------------------------------------------------------------------
# Cipher Identification (live demo)
# --------------------------------------------------------------------------

elif section == "Cipher Identification":
    page_title(
        "Cipher Identification",
        "Configure a model, run it on held-out ciphertext samples, and inspect what it concludes.",
    )

    with st.container(key="cfg"):
        st.markdown('<div class="panel-label">Model configuration</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1.2, 1, 1])
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
        st.markdown(
            chip("Neural network · about 10–20 s", MODEL_COLORS[model_name]) if is_deep
            else chip("Classical model · near-instant", MODEL_COLORS[model_name]),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hint">Uses the same seeded 80/20 split as the verified experiments; '
            "the test set is only touched for the final prediction.</p>",
            unsafe_allow_html=True,
        )
        btn_train, btn_clear, _ = st.columns([1.5, 1, 3])
        run_clicked = btn_train.button("Run identification", key="run_id", type="primary")
        if btn_clear.button("Clear result", key="reset_demo", icon=":material/restart_alt:"):
            st.session_state.pop("ci_result", None)
            st.rerun()

    if run_clicked:
        analysis = st.empty()
        analysis.markdown(
            '<div class="analysis"><div class="a-head"><span class="a-ring"></span>Analyzing ciphertext</div>'
            '<ul class="a-steps">'
            '<li style="--s:0">Extracting statistical fingerprint</li>'
            f'<li style="--s:1">Evaluating {model_name}</li>'
            '<li style="--s:2">Calculating confidence</li>'
            '<li style="--s:3">Generating prediction</li></ul>'
            '<div class="a-bar"><i></i></div>'
            f'<div class="a-note">{"Training a neural network with early stopping. This can take 10 to 20 seconds." if is_deep else "Fitting the model on the training split."}</div></div>',
            unsafe_allow_html=True,
        )
        model, X_test, y_test, y_pred, y_proba, metrics, cm = run_live_model(
            model_name, task, size, pair
        )
        analysis.empty()

        st.session_state["ci_result"] = (
            model_name, task, size, pair, X_test, y_test, y_pred, y_proba, metrics, cm, model
        )
        st.session_state["ci_sample"] = 0

    if "ci_result" in st.session_state:
        model_name, task, size, pair, X_test, y_test, y_pred, y_proba, metrics, cm, model = st.session_state["ci_result"]

        st.markdown(
            f'<div class="panel-label" style="margin-top:2rem">Result · {model_name} · '
            f'{"five-class" if task == "multiclass" else pair} · {size}</div>',
            unsafe_allow_html=True,
        )
        sample_idx = st.slider("Test sample to identify", 0, len(X_test) - 1, 0, key="ci_sample")
        true_val = y_test[sample_idx]
        pred_val = y_pred[sample_idx]
        correct = bool(true_val == pred_val)
        pred_name, true_name = class_names([pred_val])[0], class_names([true_val])[0]
        vcol = ALGO_COLORS.get(pred_name, ACCENT)

        conf_html = ""
        proba_row, class_labels = None, None
        if y_proba is not None:
            proba_row = y_proba[sample_idx]
            classes = getattr(model, "classes_", np.unique(np.concatenate([y_test, y_pred])))
            class_labels = class_names(classes)
            conf = float(np.max(proba_row))
            conf_html = (
                f'<div class="v-conf-top"><span>Model confidence</span><b>{conf:.1%}</b></div>'
                f'<div class="v-bar"><i style="--w:{conf * 100:.1f}%"></i></div>'
            )
        st.markdown(
            f'<div class="verdict" style="--c:{vcol}"><div class="v-label">Identified cipher</div>'
            f'<div class="v-name">{pred_name}</div>'
            f'<div class="v-meta">Sample <b>#{sample_idx}</b> · true label <b>{true_name}</b> · '
            f'<span class="{"normal-badge" if correct else "anomaly-badge"}">{"Correct" if correct else "Incorrect"}</span></div>'
            f"{conf_html}</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        m2.metric("Precision", f"{metrics['precision']:.3f}")
        m3.metric("Recall", f"{metrics['recall']:.3f}")
        m4.metric("F1-score", f"{metrics['f1_score']:.3f}")
        st.write("")

        tab_cm, tab_conf, tab_pred = st.tabs(["Confusion matrix", "Class probabilities", "Predictions"])

        with tab_cm:
            labels = sorted(set(np.unique(y_test)) | set(np.unique(y_pred)))
            display_labels = class_names(labels)
            fig = px.imshow(
                cm, x=display_labels, y=display_labels, text_auto=True,
                color_continuous_scale=SCALE_ACCENT, aspect="auto",
                labels=dict(x="Predicted", y="True", color="Count"),
            )
            st.plotly_chart(style_fig(fig, 400), use_container_width=True)

        with tab_conf:
            if proba_row is not None:
                order = np.argsort(proba_row)[::-1]
                fig2 = go.Figure(
                    go.Bar(
                        x=[proba_row[i] for i in order],
                        y=[class_labels[i] for i in order],
                        orientation="h",
                        marker_color=[ALGO_COLORS.get(class_labels[i], ACCENT) for i in order],
                        text=[f"{proba_row[i]:.1%}" for i in order],
                        textposition="outside",
                    )
                )
                fig2.update_layout(xaxis_range=[0, 1.1], xaxis_title=f"Predicted probability · sample #{sample_idx}")
                fig2.update_yaxes(autorange="reversed")
                st.plotly_chart(style_fig(fig2, 340), use_container_width=True)
            else:
                insight("This model does not expose class probabilities.")

        with tab_pred:
            feat_names = get_feature_names()
            n_show = min(10, len(X_test))
            sample_df = pd.DataFrame(X_test[:n_show], columns=feat_names)
            sample_df["true_label"] = y_test[:n_show]
            sample_df["predicted_label"] = y_pred[:n_show]
            sample_df["correct"] = sample_df["true_label"] == sample_df["predicted_label"]
            st.dataframe(
                sample_df.style.apply(
                    lambda row: ["background-color: #12372f" if row["correct"] else "background-color: #3d1620"] * len(row),
                    axis=1,
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown('<p class="hint">First 10 test samples. Green rows were classified correctly.</p>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Model Comparison
# --------------------------------------------------------------------------

elif section == "Model Comparison":
    page_title("Model Comparison", "How all six models stack up on the same task and ciphertext size.")

    if results_df is None:
        st.warning("No results file found. Run `python experiments/run_complete_experiments.py` first.")
        st.stop()

    ctl1, ctl2 = st.columns([1, 1])
    with ctl1:
        task = st.radio("Task", sorted(results_df["task_type"].unique()), horizontal=True)
    size_options = sorted(
        results_df.loc[results_df["task_type"] == task, "ciphertext_size"].unique(),
        key=lambda s: int(s[:-2]),
    )
    with ctl2:
        size = st.selectbox("Ciphertext size", size_options, format_func=str.upper)

    subset = results_df[(results_df["task_type"] == task) & (results_df["ciphertext_size"] == size)]
    agg = subset.groupby("model_name")[["accuracy", "precision", "recall", "f1_score"]].mean()
    agg = agg.reindex([m for m in ALL_MODELS if m in agg.index])

    tab_tbl, tab_acc, tab_radar, tab_time = st.tabs(
        ["Summary table", "Accuracy", "All four metrics", "Training time"]
    )

    with tab_tbl:
        st.markdown(f"#### Mean metrics · {task} · {size.upper()}")
        st.dataframe(
            agg.style.format("{:.3f}").background_gradient(cmap=CMAP_ACCENT, subset=["accuracy"]),
            use_container_width=True,
        )

    with tab_acc:
        fig = go.Figure(
            go.Bar(
                x=agg.index, y=agg["accuracy"],
                marker_color=[MODEL_COLORS.get(m, ACCENT) for m in agg.index],
                text=[f"{v:.1%}" for v in agg["accuracy"]], textposition="outside",
            )
        )
        fig.update_layout(yaxis_title="Accuracy")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    with tab_radar:
        metrics_cols = ["accuracy", "precision", "recall", "f1_score"]
        fig = go.Figure()
        for m in agg.index:
            values = agg.loc[m, metrics_cols].tolist()
            fig.add_trace(
                go.Scatterpolar(
                    r=values + values[:1],
                    theta=metrics_cols + metrics_cols[:1],
                    fill="toself", name=m, opacity=0.55,
                    line_color=MODEL_COLORS.get(m, ACCENT),
                )
            )
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, max(0.05, agg[metrics_cols].values.max() * 1.15)],
                                gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
                angularaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
            ),
            showlegend=True,
        )
        st.plotly_chart(style_fig(fig, 440), use_container_width=True)

    with tab_time:
        if subset["training_time"].notna().any():
            tt = subset.groupby("model_name")["training_time"].mean().reindex(
                [m for m in ALL_MODELS if m in agg.index]
            )
            fig = go.Figure(
                go.Bar(
                    x=tt.index, y=tt.values,
                    marker_color=[MODEL_COLORS.get(m, ACCENT) for m in tt.index],
                )
            )
            fig.update_layout(yaxis_title="Seconds")
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        else:
            insight("No training-time data in the results file.")


# --------------------------------------------------------------------------
# Paper Comparison — designed to be shown directly to evaluators/teachers
# --------------------------------------------------------------------------

elif section == "Paper Comparison":
    page_title(
        "CipherBench vs. the base paper",
        "Every reported metric, side by side with Yuan et al. (2022), model by model and size by size.",
    )

    st.markdown(
        f'<div class="card"><div class="eyebrow">Reference</div><p>{PAPER_CITATION}</p></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    col1.markdown(
        '<div class="card card-accent2"><h5>In the base paper</h5>'
        "<p>SVM, KNN, Random Forest and HKNNRF, evaluated on binary identification "
        "(AES vs 3DES, plus HKNNRF on all 10 pairs) and five-class identification. "
        "Paper numbers are transcribed directly from its Tables 3 to 5.</p></div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        '<div class="card card-accent"><h5>Our extension, not in the paper</h5>'
        "<p>MLP and 1D-CNN. The paper has no neural-network baseline, so there is no paper "
        "number to place next to them. They are shown on their own under the identical "
        "evaluation protocol.</p></div>",
        unsafe_allow_html=True,
    )

    if results_df is None:
        st.warning("No results file found — run `python experiments/run_complete_experiments.py` first.")
        st.stop()

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

    def gap_table(long_df, limit):
        table = long_df.pivot_table(index=["Size", "Model"], columns="Source", values="Value").reset_index()
        table["Gap (Ours − Paper)"] = table["Ours"] - table["Paper"]
        return table.style.format(
            {"Paper": "{:.3f}", "Ours": "{:.3f}", "Gap (Ours − Paper)": "{:+.3f}"}
        ).background_gradient(cmap=CMAP_GAP, subset=["Gap (Ours − Paper)"], vmin=-limit, vmax=limit)

    def source_bar(long_df):
        avg = long_df.dropna(subset=["Value"]).groupby(["Model", "Source"])["Value"].mean().reset_index()
        fig = px.bar(
            avg, x="Model", y="Value", color="Source", barmode="group",
            color_discrete_map={"Paper": PAPER_COLOR, "Ours": ACCENT},
            category_orders={"Model": PAPER_MODELS}, text_auto=".2f",
        )
        fig.update_layout(yaxis_title=metric.capitalize())
        return style_fig(fig, 380)

    def extension_bar(task, size_labels_by_kb, pair):
        ext_records = []
        for kb, size_label in size_labels_by_kb.items():
            for model in ["MLP", "CNN"]:
                ext_records.append(
                    {"Size": f"{kb}KB", "Model": model,
                     "Value": our_value(results_df, model, task, size_label, pair, metric)}
                )
        fig = px.bar(
            pd.DataFrame(ext_records), x="Size", y="Value", color="Model", barmode="group",
            color_discrete_map={"MLP": MODEL_COLORS["MLP"], "CNN": MODEL_COLORS["CNN"]},
            category_orders={"Size": ["1KB", "8KB", "64KB", "256KB", "512KB"]},
        )
        fig.update_layout(yaxis_title=metric.capitalize())
        return style_fig(fig, 340)

    size_map_upper = {1: "1KB", 8: "8KB", 64: "64KB", 256: "256KB", 512: "512KB"}
    size_map_lower = {kb: lbl.lower() for kb, lbl in size_map_upper.items()}

    tab_mc, tab_bin, tab_pairs = st.tabs(
        ["Five-class (Table 5)", "Binary: AES vs 3DES (Table 3)", "HKNNRF, all 10 pairs (Table 4)"]
    )

    # ----------------------------------------------------------------
    with tab_mc:
        long_df = grouped_paper_vs_ours(PAPER_MULTICLASS, "multiclass", "5-class", size_map_upper)

        st.markdown("#### Paper vs. ours, averaged across all five sizes")
        st.plotly_chart(source_bar(long_df), use_container_width=True, key="pc_mc_avg")

        st.markdown("#### Per-size breakdown for one model")
        pick_model = st.selectbox("Model", PAPER_MODELS, key="mc_pick")
        detail = long_df[long_df["Model"] == pick_model]
        fig2 = px.line(
            detail, x="Size", y="Value", color="Source", markers=True,
            color_discrete_map={"Paper": PAPER_COLOR, "Ours": MODEL_COLORS[pick_model]},
            category_orders={"Size": ["1KB", "8KB", "64KB", "256KB", "512KB"]},
        )
        fig2.update_traces(line=dict(width=3), marker=dict(size=9))
        fig2.update_layout(yaxis_title=metric.capitalize())
        st.plotly_chart(style_fig(fig2, 340), use_container_width=True, key="pc_mc_line")

        st.markdown("#### Full comparison table")
        st.dataframe(gap_table(long_df, 0.15), use_container_width=True, height=420, hide_index=True)

        st.markdown("#### Our extension beyond the paper: MLP and 1D-CNN")
        st.plotly_chart(extension_bar("multiclass", size_map_upper, "5-class"),
                        use_container_width=True, key="pc_mc_ext")
        st.markdown(
            '<p class="hint">No paper bar here by design: the paper never tested a neural network on this task.</p>',
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------------------
    with tab_bin:
        long_df = grouped_paper_vs_ours(PAPER_BINARY_AES_3DES, "binary", "AES and 3DES", size_map_lower)

        st.markdown("#### Paper vs. ours, averaged across all five sizes (AES vs 3DES)")
        st.plotly_chart(source_bar(long_df), use_container_width=True, key="pc_bin_avg")

        st.markdown("#### Full comparison table")
        st.dataframe(gap_table(long_df, 0.3), use_container_width=True, height=420, hide_index=True)

        st.markdown("#### Our extension beyond the paper: MLP and 1D-CNN")
        st.plotly_chart(extension_bar("binary", size_map_lower, "AES and 3DES"),
                        use_container_width=True, key="pc_bin_ext")
        st.markdown(
            '<p class="hint">No paper bar here by design: the paper never tested a neural network on this task.</p>',
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------------------
    with tab_pairs:
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

        st.markdown("#### HKNNRF binary accuracy: paper (Table 4) vs. ours")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Paper")
            fig = px.imshow(paper_matrix, text_auto=".2f", color_continuous_scale=SCALE_PAPER,
                            zmin=0, zmax=1, aspect="auto")
            st.plotly_chart(style_fig(fig, 420), use_container_width=True, key="pc_pairs_paper")
        with c2:
            st.caption("Ours")
            fig = px.imshow(ours_matrix, text_auto=".2f", color_continuous_scale=SCALE_ACCENT,
                            zmin=0, zmax=1, aspect="auto")
            st.plotly_chart(style_fig(fig, 420), use_container_width=True, key="pc_pairs_ours")

        st.markdown("#### Gap (ours − paper)")
        delta = ours_matrix - paper_matrix
        fig = px.imshow(delta, text_auto=".2f", color_continuous_scale=SCALE_GAP,
                        zmin=-0.3, zmax=0.3, aspect="auto")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True, key="pc_pairs_gap")

        st.markdown("#### Our extension beyond the paper: MLP and CNN, all 10 pairs")
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
        fig = px.imshow(ext_matrix, text_auto=".2f", color_continuous_scale=SCALE_ACCENT,
                        zmin=0, zmax=1, aspect="auto")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True, key="pc_pairs_ext")

    insight(
        "<b>Why the gap?</b> An ANOVA test on these 10 NIST p-value features found none "
        "statistically significant for telling the ciphers apart. The feature set carries less "
        "signal than whatever the paper's authors used, and every model (HKNNRF, MLP and CNN "
        "included) shows the same ceiling. Full discussion in <code>docs/PROJECT_REPORT.md</code> §7.",
        "tip",
    )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

elif section == "Results":
    page_title("Verified results", "Every one of the 330 experiments, filterable and downloadable.")

    if results_df is None:
        st.warning("No results file found.")
        st.stop()

    st.markdown(
        f'<p class="hint">Source: <code>experiments/results/{results_filename}</code> · '
        f"{len(results_df)} experiments · see docs/PROJECT_REPORT.md and docs/RESULTS.md "
        "for methodology and headline numbers.</p>",
        unsafe_allow_html=True,
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
        # multiclass rows store sizes as "512KB", binary rows as "512kb": show one list
        size_choices = sorted(results_df["ciphertext_size"].str.upper().unique(), key=lambda s: int(s[:-2]))
        size_filter = st.multiselect("Size", size_choices, default=size_choices)

    filtered = results_df[
        results_df["model_name"].isin(model_filter)
        & results_df["task_type"].isin(task_filter)
        & results_df["ciphertext_size"].str.upper().isin(size_filter)
    ]

    tab_dist, tab_tbl = st.tabs(["Accuracy distribution", "Results table"])

    with tab_dist:
        if len(filtered):
            fig = px.violin(
                filtered, x="model_name", y="accuracy", color="model_name",
                color_discrete_map=MODEL_COLORS, box=True, points="all",
                category_orders={"model_name": [m for m in ALL_MODELS if m in filtered["model_name"].unique()]},
            )
            fig.update_traces(marker_size=4)
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Accuracy")
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        else:
            insight("No experiments match the current filters.")

    with tab_tbl:
        sort_col = st.selectbox("Sort by", ["accuracy", "f1_score", "precision", "recall", "training_time"])
        filtered = filtered.sort_values(sort_col, ascending=False)
        st.dataframe(filtered, use_container_width=True, height=420, hide_index=True)
        st.download_button(
            "Download filtered results (CSV)",
            data=filtered.to_csv(index=False),
            file_name="cipherbench_filtered_results.csv",
            mime="text/csv",
            icon=":material/download:",
        )


# --------------------------------------------------------------------------
# Experiment Matrix
# --------------------------------------------------------------------------

elif section == "Experiment Matrix":
    page_title("Experiment matrix", "330 experiments: 6 models across 2 tasks and 5 ciphertext sizes.")

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
    st.write("")

    tab_map, tab_cov, tab_q = st.tabs(["Accuracy map", "Coverage", "Data quality"])

    with tab_map:
        st.markdown("#### Mean accuracy per model and ciphertext size")
        map_task = st.radio("Task", ["multiclass", "binary"], horizontal=True, key="mx_task")
        sub = results_df[results_df["task_type"] == map_task]
        acc_pivot = sub.pivot_table(
            index="model_name", columns="ciphertext_size", values="accuracy", aggfunc="mean"
        ).reindex(index=[m for m in ALL_MODELS if m in sub["model_name"].unique()])
        acc_pivot = acc_pivot[[c for c in SIZES if c in acc_pivot.columns]]
        fig_map = px.imshow(
            acc_pivot, text_auto=".2f", color_continuous_scale=SCALE_ACCENT, aspect="auto",
            labels=dict(x="Ciphertext size", y="Model", color="Accuracy"),
        )
        st.plotly_chart(style_fig(fig_map, 380), use_container_width=True, key="mx_map")

    with tab_cov:
        st.markdown("#### Runs per model and ciphertext size")
        pivot = results_df.pivot_table(
            index="model_name", columns="ciphertext_size", values="accuracy", aggfunc="count"
        ).reindex(index=[m for m in ALL_MODELS if m in results_df["model_name"].unique()])
        fig = px.imshow(
            pivot, text_auto=True, color_continuous_scale=SCALE_ACCENT, aspect="auto",
            labels=dict(x="Ciphertext size", y="Model", color="Runs"),
        )
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    with tab_q:
        zero_rows = results_df[results_df["accuracy"] == 0.0]
        if len(zero_rows) == 0:
            insight("All rows have a non-zero accuracy value.", "good")
        else:
            insight(f"{len(zero_rows)} rows have accuracy == 0.0:", "warn")
            st.dataframe(
                zero_rows[["model_name", "task_type", "algorithm_pair", "ciphertext_size", "accuracy"]],
                use_container_width=True, hide_index=True,
            )


# --------------------------------------------------------------------------
# Research / Methodology
# --------------------------------------------------------------------------

elif section == "Research / Methodology":
    page_title("Research and methodology", "The paper, the pipeline, and how evaluation stays honest.")

    tab_paper, tab_feat, tab_hk, tab_dl, tab_eval = st.tabs(
        ["Base paper", "Features", "HKNNRF", "Deep learning", "Evaluation"]
    )

    with tab_paper:
        st.markdown(
            """
Yuan, K., Yu, D., Feng, J., Yang, L., Jia, C., Huang, Y. (2022).
*A block cipher algorithm identification scheme based on hybrid k-nearest
neighbor and random forest algorithm.* PeerJ Computer Science.

The paper proposes HKNNRF and compares it against SVM, KNN and Random
Forest on binary and five-class identification of AES, 3DES, Blowfish,
CAST and RC2, at ciphertext sizes from 1KB to 512KB.
            """
        )

    with tab_feat:
        st.markdown(
            """
Each ciphertext sample is represented by **10 p-values from NIST
randomness tests**: Approximate Entropy, Cumulative Sums, Discrete Fourier
Transform, Frequency-within-Block, Linear Complexity, Monobit, Random
Excursions, Random Excursions Variant, Runs, and Serial. These 10 columns
are consumed as-is by every model. No additional scaling or feature
selection is applied, matching the paper's methodology.
            """
        )

    with tab_hk:
        st.markdown(
            """
Implemented in `src/models/hknnrf.py`:

1. Split the training portion 50/50 into an RF-training set and a
   KNN-training set (both stratified, seeded).
2. Train a Random Forest on the RF-training set.
3. Extract each sample's leaf index per tree (`RandomForestClassifier.apply`)
   and one-hot encode them. These are the RF-derived features.
4. **Concatenate** the RF-derived features with the original 10 NIST
   features (paper Steps 9–10).
5. Train a KNN classifier on the combined feature vector.
            """
        )

    with tab_dl:
        st.markdown(
            """
Not in the original paper:

- **MLP**: 2 hidden dense layers [64, 32] with dropout 0.3, trained with
  early stopping on a validation split carved out of the training data only.
- **1D-CNN**: Conv1D layers [32, 64] with batch normalization and
  max-pooling, same validation discipline as the MLP.
- Framework: **TensorFlow/Keras** (`tf-nightly`), not PyTorch. The synopsis
  lists PyTorch; PyTorch does not currently ship wheels for the Python
  version used in this environment, so Keras was substituted. See
  `docs/PROJECT_REPORT.md` §6.
            """
        )

    with tab_eval:
        st.markdown(
            """
- 80/20 train/test split, stratified, fixed `random_state` per experiment.
- For MLP/CNN, an additional 80/20 split of the *training* portion produces
  the validation set used for early stopping. **The test set is never seen
  until the single final `predict` call.** The live demo in the Cipher
  Identification page follows the identical protocol for every model.
- Metrics: accuracy, weighted precision/recall/F1, full confusion matrix.
            """
        )


# --------------------------------------------------------------------------
# About / Reproducibility
# --------------------------------------------------------------------------

else:
    page_title("About and reproducibility", "Repository layout, commands, and honest limitations.")

    tab_layout, tab_cmd, tab_repro, tab_lim = st.tabs(
        ["Repository layout", "Commands", "Reproducibility", "Known limitations"]
    )

    with tab_layout:
        st.code(
            """
CipherBench/
├── src/            # models, training script, data/metrics utilities
├── data/           # 55 CSV datasets (5 multiclass + 50 binary)
├── experiments/    # experiment runners + results/
├── app/            # this Streamlit app (streamlit_app.py + style.css + effects.js)
├── .streamlit/     # dark theme config
├── tests/          # test suite
├── database/       # optional MySQL integration
├── api/            # optional Flask REST API
└── docs/           # PROJECT_REPORT.md, RESULTS.md
            """,
            language="text",
        )

    with tab_cmd:
        st.code(
            """# Install dependencies
pip install -r requirements.txt

# Run the test suite
python tests/test_suite.py

# Train one model on one task/size
python src/train.py --model hknnrf --task multiclass --size 512KB

# Run the full 330-experiment matrix (long-running)
python experiments/run_complete_experiments.py --no-db

# Launch this app (run from the project root so the theme in .streamlit/ is picked up)
streamlit run app/streamlit_app.py
            """,
            language="bash",
        )

    with tab_repro:
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

    with tab_lim:
        st.markdown(
            """
- Deep-learning framework is TensorFlow/Keras, not PyTorch as listed in the
  synopsis (Python-version compatibility constraint).
- No ciphertext-generation pipeline is included; the project consumes the
  pre-extracted NIST-feature CSV datasets already provided.
- Multi-seed stability analysis and cross-ciphertext-size generalization
  (train on one size, test on another) are not part of the verified 330-run
  matrix. See `docs/PROJECT_REPORT.md` §8.
            """
        )


def _footer_links(items):
    return "".join(f'<a class="fl" data-go="{k}" role="link" tabindex="0">{t}</a>' for t, k in items)


st.markdown(
    '<div class="site-footer"><div><div class="f-brand">'
    f'<span class="brand-mark" style="width:34px;height:34px;border-radius:11px">{LOCK_SVG}</span>'
    "<span>Cipher<b>Bench</b></span></div>"
    "<p>Identifying block ciphers from the statistical fingerprint of their ciphertext.</p></div>"
    "<div><h6>Research</h6>"
    + _footer_links([("Overview", "Home"), ("Results", "Results"), ("Paper comparison", "Paper")])
    + "</div><div><h6>Explore</h6>"
    + _footer_links([("Models", "Models"), ("Experiments", "Matrix"), ("Methodology", "Method")])
    + '</div><div><h6>Reference</h6><div class="f-fact"><b>Yuan et al. (2022)</b></div>'
    '<div class="f-fact"><b>330</b> verified experiments</div>'
    '<div class="f-fact"><b>5</b> block ciphers · <b>6</b> models</div></div>'
    '<div class="foot-base"><span>Minor Project · Dept. of Information Technology, MSIT New Delhi</span>'
    "<span>Built with Streamlit and scikit-learn</span></div></div>",
    unsafe_allow_html=True,
)
