"""
ArtGuard AI — Real vs AI-Generated Image Detector
Single-model (V4) with Streamlit frontend.

Threshold logic:
  score >= 0.5  →  REAL
  score <  0.5  →  AI GENERATED
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
dir               = os.path.dirname(os.path.abspath(__file__))
MODEL_V4          = os.path.join(dir, "fake_test_modelv4.keras")
THRESHOLD         = 0.5
MAX_BATCH         = 20
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]
APP_VERSION       = "v1.0"
IMG_SIZE: tuple[int, int] = (128, 128)


# ─────────────────────────────────────────────
# MODEL LOADING  (single model, cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load Model V4 once and cache it for the session."""
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_V4)


# ─────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    assert arr.shape == (128, 128, 3), \
        f"Shape error: expected (128,128,3) got {arr.shape}"
    return np.expand_dims(arr, axis=0)


def predict_single(image: Image.Image, model) -> dict:
    arr   = preprocess(image)
    score = float(model.predict(arr, verbose=0)[0][0])

    label = "Real" if score >= THRESHOLD else "AI Generated"
    conf  = score if label == "Real" else 1.0 - score

    return {
        "individual": [{
            "model":      "Model V4",
            "label":      label,
            "confidence": conf,
            "raw_score":  score,
        }],
        "avg_score":        score,
        "final_label":      label,
        "final_confidence": conf,
        "ai_votes":   0 if label == "Real" else 1,
        "real_votes": 1 if label == "Real" else 0,
    }


def predict_batch(
    files: list,
    model,
    progress_placeholder,
    status_placeholder,
) -> list[dict]:
    batch_results: list[dict] = []
    total = len(files)

    for idx, f in enumerate(files):
        status_placeholder.markdown(
            f'<div class="batch-status">Analysing image {idx + 1} of {total} — '
            f'<span style="opacity:.7">{f.name}</span></div>',
            unsafe_allow_html=True,
        )
        progress_placeholder.progress(idx / total)
        try:
            img  = Image.open(f).convert("RGB")
            pred = predict_single(img, model)
            batch_results.append({"filename": f.name, "image": img,
                                   "result": pred, "error": None})
        except Exception as exc:
            batch_results.append({"filename": f.name, "image": None,
                                   "result": None, "error": str(exc)})

    progress_placeholder.progress(1.0)
    status_placeholder.empty()
    return batch_results


# ─────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────
def inject_css(theme: str) -> None:
    is_dark = theme == "dark"

    bg         = "#0A0A0F" if is_dark else "#F0F0FF"
    card_bg    = "#1A1A2E" if is_dark else "#FFFFFF"
    text_main  = "#E0E0FF" if is_dark else "#1A1A2E"
    text_muted = "#8888AA" if is_dark else "#6666AA"
    border_col = "#2A2A4E" if is_dark else "#D0D0F0"
    sidebar_bg = "#12121E" if is_dark else "#F8F8FF"
    tab_bg     = "#1A1A2E" if is_dark else "#EEEEFF"
    tab_active = "#7F77DD"
    hero_grad  = (
        "linear-gradient(135deg, #0A0A0F 0%, #1A1A2E 60%, #0D0D2B 100%)"
        if is_dark else
        "linear-gradient(135deg, #E0E0FF 0%, #F5F5FF 60%, #EBEBFF 100%)"
    )


    css_html = f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [data-testid="stAppViewContainer"] {{
    background-color: {bg} !important;
    color: {text_main} !important;
    font-family: 'Inter', sans-serif !important;
    transition: background-color 0.3s ease, color 0.3s ease;
}}

[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    border-right: 1px solid {border_col};
}}

.block-container {{ padding-top: 1rem !important; max-width: 1200px; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

/* Restore the sidebar expand button that gets hidden by header  visibility: hidden  */
[data-testid="stExpandSidebarButton"] {{
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: all !important;
    z-index: 999999 !important;
}}

h1, h2, h3, h4 {{
    font-family: 'Inter', sans-serif !important;
    color: {text_main} !important;
}}

/* FIX: scope text colour to the main content area only, NOT bare global selectors.
   This prevents Streamlit widget internals (file uploader label, toggle text, etc.)
   from having their hidden/internal text nodes made visible by a global rule. */
.block-container p,
.block-container .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown {{
    font-family: 'Inter', sans-serif !important;
    color: {text_main};
}}

/* FIX: explicitly hide the file uploader label node that stays in the DOM
   when label_visibility="collapsed", so it never overlaps the Browse button */
[data-testid="stFileUploader"] label {{
    display: none !important;
}}

[data-testid="stFileUploader"] {{
    border: 2px dashed #7F77DD;
    border-radius: 16px;
    padding: 1.5rem;
    background: {'rgba(127,119,221,0.06)' if is_dark else 'rgba(127,119,221,0.04)'};
    transition: border-color 0.3s ease;
}}
[data-testid="stFileUploader"]:hover {{ border-color: #A09AEE; }}

/* File uploader inner dropzone button — force light/dark background */
[data-testid="stFileUploaderDropzone"] {{
    background-color: {card_bg} !important;
    color: {text_main} !important;
}}
[data-testid="stFileUploaderDropzone"] button {{
    background-color: {'#2A2A4E' if is_dark else '#E8E8F8'} !important;
    color: {text_main} !important;
    border: none !important;
}}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {{
    color: {text_muted} !important;
}}

/* Expander header background */
[data-testid="stExpander"] summary {{
    background: {'#1E1E38' if is_dark else '#E8E8F8'} !important;
    color: {text_main} !important;
    border-radius: 10px;
}}
[data-testid="stExpander"] summary:hover {{
    background: {'#252545' if is_dark else '#DDDDF5'} !important;
}}

/* Markdown code blocks and table cells inside sidebar */
[data-testid="stSidebar"] code,
[data-testid="stSidebar"] td,
[data-testid="stSidebar"] th {{
    background: {'#2A2A4E' if is_dark else '#E8E8F8'} !important;
    color: {text_main} !important;
}}
[data-testid="stSidebar"] table {{
    border-color: {border_col} !important;
}}
[data-testid="stSidebar"] tr {{
    background: {card_bg} !important;
    color: {text_main} !important;
}}

[data-testid="stTabs"] [role="tablist"] {{
    gap: 0.5rem;
    border-bottom: 1px solid {border_col};
    padding-bottom: 0;
}}
[data-testid="stTabs"] [role="tab"] {{
    background: {tab_bg};
    color: {text_muted};
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
    font-size: 0.9rem;
    border: 1px solid {border_col};
    border-bottom: none;
    transition: all 0.2s ease;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    background: {tab_active};
    color: #FFFFFF;
    border-color: {tab_active};
}}

[data-testid="stProgress"] > div > div {{
    background: #2A2A4E;
    border-radius: 99px;
}}
[data-testid="stProgress"] > div > div > div {{
    border-radius: 99px;
    transition: width 0.4s ease;
}}
[data-testid="stProgress"] [role="progressbar"] {{
    background: #7F77DD;
    border-radius: 99px;
}}

[data-testid="stSidebar"] * {{ color: {text_main} !important; }}
[data-testid="stSidebar"] .stToggle label {{ color: {text_main} !important; }}

[data-testid="stExpander"] {{
    background: {'#1E1E38' if is_dark else '#F0F0FF'};
    border: 1px solid {border_col};
    border-radius: 10px;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {border_col};
    border-radius: 12px;
    overflow: hidden;
}}

.hero-banner {{
    background: {hero_grad};
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    border: 1px solid {border_col};
    position: relative;
    overflow: hidden;
}}
.hero-banner::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(127,119,221,0.15), transparent 70%);
    pointer-events: none;
}}
.hero-emoji {{ font-size: 3.5rem; line-height: 1; margin-bottom: 0.5rem; }}
.hero-title {{
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0.25rem 0;
    background: linear-gradient(90deg, #A09AEE, #7F77DD, #5C55CC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero-sub {{
    font-size: 0.95rem;
    color: {text_muted};
    margin: 0.4rem auto 1rem;
    max-width: 480px;
    font-weight: 400;
    line-height: 1.5;
}}
.badge-row {{ display: flex; justify-content: center; gap: 0.6rem; flex-wrap: wrap; }}
.badge-pill {{
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    border: 1.5px solid #7F77DD;
    color: #A09AEE;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    background: rgba(127,119,221,0.08);
    transition: all 0.2s ease;
    display: inline-block;
}}
.badge-pill:hover {{
    background: rgba(127,119,221,0.2);
    box-shadow: 0 0 12px rgba(127,119,221,0.4);
}}

.card-model {{
    background: {card_bg};
    border: 1px solid {border_col};
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}}
.card-model .model-name {{ font-weight: 700; font-size: 0.95rem; margin-bottom: 0.3rem; color: {text_main}; }}
.card-model .verdict-badge {{
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}}
.ai-badge   {{ background: rgba(255,68,68,0.18);  color: #FF6666; border: 1px solid #FF4444; }}
.real-badge {{ background: rgba(0,255,136,0.14);  color: #00DD77; border: 1px solid #00FF88; }}
.model-conf {{ font-size: 0.82rem; color: {text_muted}; margin-top: 0.25rem; }}
.model-raw  {{ font-size: 0.75rem; color: {text_muted}; font-family: 'Courier New', monospace; }}

.vote-section {{
    background: {card_bg};
    border: 1px solid {border_col};
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.8rem;
}}
.vote-title {{
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {text_muted};
    margin-bottom: 0.75rem;
}}
.vote-row   {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; }}
.vote-label {{ font-size: 0.88rem; font-weight: 600; width: 150px; flex-shrink: 0; }}
.vote-bar-wrap {{
    flex: 1;
    background: {'#2A2A4E' if is_dark else '#E0E0F5'};
    border-radius: 99px;
    height: 10px;
}}
.vote-bar-ai   {{ height: 10px; border-radius: 99px; background: #FF4444; transition: width 0.5s ease; }}
.vote-bar-real {{ height: 10px; border-radius: 99px; background: #00FF88; transition: width 0.5s ease; }}
.vote-count    {{ font-size: 0.82rem; color: {text_muted}; width: 30px; text-align: right; }}

.verdict-card {{
    border-radius: 18px;
    padding: 2rem 1.5rem;
    text-align: center;
    animation: fadeInUp 0.5s ease both;
    margin-bottom: 1rem;
}}
.verdict-card.ai   {{ background: rgba(255,68,68,0.06);  border: 1.5px solid #FF4444; box-shadow: 0 0 40px rgba(255,68,68,0.2); }}
.verdict-card.real {{ background: rgba(0,255,136,0.05);  border: 1.5px solid #00FF88; box-shadow: 0 0 40px rgba(0,255,136,0.18); }}
.verdict-emoji  {{ font-size: 3.5rem; line-height: 1; margin-bottom: 0.6rem; display: block; }}
.verdict-label  {{ font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 0.3rem; }}
.verdict-label.ai   {{ color: #FF4444; }}
.verdict-label.real {{ color: #00FF88; }}
.verdict-conf {{ font-size: 1rem; font-weight: 500; color: {text_muted}; margin-bottom: 0.8rem; }}
.verdict-raw  {{ font-size: 0.8rem; color: {text_muted}; margin-top: 0.6rem; font-family: monospace; }}

.img-container {{ position: relative; border-radius: 14px; overflow: hidden; border: 1px solid {border_col}; }}
.img-overlay-badge {{
    position: absolute;
    top: 12px; left: 12px;
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    backdrop-filter: blur(6px);
}}
.img-overlay-badge.ai   {{ background: rgba(255,68,68,0.85);  color: #fff; }}
.img-overlay-badge.real {{ background: rgba(0,200,100,0.85);  color: #fff; }}

.batch-status {{ font-size: 0.85rem; color: {text_muted}; padding: 0.4rem 0; font-style: italic; }}
.section-divider {{ height: 1px; background: {border_col}; margin: 1.2rem 0; }}
.sidebar-version {{ font-size: 0.72rem; color: {text_muted}; margin-top: 0.2rem; }}
.sidebar-footer {{
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid {border_col};
    font-size: 0.75rem;
    color: {text_muted};
    text-align: center;
}}

.stats-row {{ display: flex; gap: 0.75rem; margin: 1rem 0; flex-wrap: wrap; }}
.stat-chip {{
    padding: 0.4rem 0.9rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid {border_col};
    background: {card_bg};
}}
.stat-chip.ai   {{ border-color: #FF4444; color: #FF6666; background: rgba(255,68,68,0.08); }}
.stat-chip.real {{ border-color: #00FF88; color: #00CC77; background: rgba(0,255,136,0.07); }}

.upload-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(127,119,221,0.12);
    border: 1px solid #7F77DD;
    color: #A09AEE;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.5rem 0 1rem;
}}
.info-box {{
    background: rgba(127,119,221,0.08);
    border: 1px solid rgba(127,119,221,0.3);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: {text_muted};
    margin: 0.5rem 0;
}}
.err-box {{
    background: rgba(255,68,68,0.08);
    border: 1px solid rgba(255,68,68,0.3);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #FF8888;
    margin: 0.5rem 0;
}}

.stButton > button {{
    background: linear-gradient(135deg, #7F77DD, #5C55CC) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.5rem !important;
    width: 100%;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 18px rgba(127,119,221,0.35) !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(127,119,221,0.5) !important;
}}
.stButton > button:active {{ transform: translateY(0) !important; }}

[data-testid="stDownloadButton"] > button {{
    background: rgba(127,119,221,0.12) !important;
    color: #A09AEE !important;
    border: 1px solid #7F77DD !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    width: auto !important;
    box-shadow: none !important;
}}

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
</style>
"""
    st.html(css_html)


# ─────────────────────────────────────────────
# RENDER FUNCTIONS
# ─────────────────────────────────────────────

def render_header() -> None:
    st.markdown("""
<div class="hero-banner">
    <div class="hero-emoji">🎨</div>
    <div class="hero-title">ArtGuard AI</div>
    <div class="hero-sub">Powered by a 60K image trained deep learning model — V4</div>
    <div class="badge-row">
        <span class="badge-pill">⚡ Model V4</span>
        <span class="badge-pill">🧠 60K Training Images</span>
        <span class="badge-pill">🎯 Binary Classification</span>
    </div>
</div>
""", unsafe_allow_html=True)


def render_model_card(result: dict, theme: str) -> None:
    label      = result["label"]
    conf       = result["confidence"]
    is_ai      = label == "AI Generated"
    badge_cls  = "ai-badge"       if is_ai else "real-badge"
    badge_text = "🤖 AI Generated" if is_ai else "✅ Real"
    bar_colour = "#FF4444"        if is_ai else "#00FF88"
    bar_pct    = int(conf * 100)

    st.markdown(f"""
<div class="card-model">
    <div class="model-name">🧠 {result['model']}</div>
    <div class="verdict-badge {badge_cls}">{badge_text}</div>
    <div class="model-conf">Confidence: <strong>{conf * 100:.1f}%</strong></div>
    <div class="model-raw">raw score: {result['raw_score']:.4f}</div>
</div>
<div style="background: {'#2A2A4E' if theme=='dark' else '#E0E0F0'};
            border-radius: 99px; height: 8px; margin: -0.4rem 0 0.8rem; overflow: hidden;">
    <div style="width: {bar_pct}%; height: 100%; background: {bar_colour};
                border-radius: 99px; transition: width 0.5s ease;"></div>
</div>
""", unsafe_allow_html=True)


def render_verdict_summary(pred: dict, theme: str) -> None:
    is_ai      = pred["final_label"] == "AI Generated"
    card_cls   = "ai"           if is_ai else "real"
    emoji      = "🤖"           if is_ai else "✅"
    lbl_cls    = "ai"           if is_ai else "real"
    label_text = "AI GENERATED" if is_ai else "REAL IMAGE"
    conf       = pred["final_confidence"]
    bar_colour = "#FF4444"      if is_ai else "#00FF88"
    bar_pct    = int(conf * 100)
    score      = pred["avg_score"]

    st.markdown(f"""
<div class="vote-section">
    <div class="vote-title">📊 Score Breakdown</div>
    <div class="vote-row">
        <span class="vote-label">✅ Real probability</span>
        <div class="vote-bar-wrap">
            <div class="vote-bar-real" style="width: {int(score*100)}%;"></div>
        </div>
        <span class="vote-count">{score*100:.0f}%</span>
    </div>
    <div class="vote-row">
        <span class="vote-label">🤖 AI probability</span>
        <div class="vote-bar-wrap">
            <div class="vote-bar-ai" style="width: {int((1-score)*100)}%;"></div>
        </div>
        <span class="vote-count">{(1-score)*100:.0f}%</span>
    </div>
</div>
<div class="verdict-card {card_cls}">
    <span class="verdict-emoji">{emoji}</span>
    <div class="verdict-label {lbl_cls}">{label_text}</div>
    <div class="verdict-conf">Confidence: <strong>{conf * 100:.1f}%</strong></div>
    <div style="background: #2A2A4E; border-radius: 99px; height: 10px;
                margin: 0.4rem auto 0; max-width: 260px; overflow: hidden;">
        <div style="width: {bar_pct}%; height: 100%; background: {bar_colour};
                    border-radius: 99px; transition: width 0.6s ease;"></div>
    </div>
    <div class="verdict-raw">
        raw score: {score:.4f} &nbsp;|&nbsp;
        threshold: {THRESHOLD} &nbsp;|&nbsp;
        ≥ {THRESHOLD} → Real
    </div>
</div>
""", unsafe_allow_html=True)


def render_result(image: Image.Image, pred: dict, theme: str) -> None:
    is_ai     = pred["final_label"] == "AI Generated"
    overlay   = "ai"             if is_ai else "real"
    overlay_t = "🤖 AI GENERATED" if is_ai else "✅ REAL"

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.4], gap="large")

    with left_col:
        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(
            f'<div class="img-overlay-badge {overlay}">{overlay_t}</div></div>',
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            '<div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:1px;color:#8888AA;margin-bottom:0.6rem;">Model Analysis</div>',
            unsafe_allow_html=True,
        )
        render_model_card(pred["individual"][0], theme)
        render_verdict_summary(pred, theme)


def render_single_tab(model, theme: str) -> None:
    uploaded = st.file_uploader(
        "Upload image",
        type=SUPPORTED_FORMATS,
        key="single_uploader",
        label_visibility="hidden",
    )

    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)

            if st.button("🔍  Analyse Image", key="single_analyse"):
                with st.spinner("Running analysis…"):
                    pred = predict_single(image, model)
                st.session_state["single_result"] = pred
                st.session_state["single_image"]  = image

        except Exception as exc:
            st.markdown(
                f'<div class="err-box">⚠️ Could not open image: {exc}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="info-box">🎨 Upload a JPG, PNG, or WebP image to get started.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.get("single_result"):
        render_result(
            st.session_state["single_image"],
            st.session_state["single_result"],
            theme,
        )


def render_batch_tab(model, theme: str) -> None:
    uploaded_files = st.file_uploader(
        "Upload images",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True,
        key="batch_uploader",
        label_visibility="hidden",
    )

    if uploaded_files:
        if len(uploaded_files) > MAX_BATCH:
            st.markdown(
                f'<div class="err-box">⚠️ Maximum {MAX_BATCH} images. '
                f'Only the first {MAX_BATCH} will be processed.</div>',
                unsafe_allow_html=True,
            )
            uploaded_files = uploaded_files[:MAX_BATCH]

        st.markdown(
            f'<div class="upload-badge">📦 {len(uploaded_files)} image(s) ready</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:0.8rem;color:#8888AA;font-weight:600;'
            'text-transform:uppercase;letter-spacing:1px;margin-bottom:0.4rem;">Preview</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(min(4, len(uploaded_files)))
        for idx, f in enumerate(uploaded_files):
            col_idx = idx % 4
            if col_idx == 0 and idx != 0:
                cols = st.columns(min(4, len(uploaded_files) - idx))
            try:
                img = Image.open(f)
                cols[col_idx].image(img, use_container_width=True, caption=f.name[:20])
            except Exception:
                cols[col_idx].markdown("❌ Bad file")

        if st.button("🔍  Analyse All Images", key="batch_analyse"):
            progress_ph = st.empty()
            status_ph   = st.empty()
            results = predict_batch(uploaded_files, model, progress_ph, status_ph)
            st.session_state["batch_results"] = results

    if st.session_state.get("batch_results"):
        batch_results = st.session_state["batch_results"]

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.85rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:1px;color:#8888AA;margin-bottom:0.75rem;">Batch Results</div>',
            unsafe_allow_html=True,
        )

        rows = []
        for idx, item in enumerate(batch_results):
            if item["error"]:
                rows.append({
                    "#": idx + 1, "Filename": item["filename"],
                    "Model V4": "ERROR", "Verdict": "ERROR", "Confidence": "—",
                })
            else:
                r   = item["result"]
                ind = r["individual"][0]
                rows.append({
                    "#":          idx + 1,
                    "Filename":   item["filename"],
                    "Model V4":   f"{ind['label']} ({ind['confidence']*100:.0f}%)",
                    "Verdict":    r["final_label"],
                    "Confidence": f"{r['final_confidence']*100:.1f}%",
                })

        df = pd.DataFrame(rows)

        def colour_verdict(val: str) -> str:
            if val == "AI Generated":
                return "background-color: rgba(255,68,68,0.15); color:#FF6666; font-weight:600;"
            if val == "Real":
                return "background-color: rgba(0,255,136,0.12); color:#00CC77; font-weight:600;"
            return ""

        styled = df.style.map(colour_verdict, subset=["Verdict"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        valid    = [r for r in batch_results if not r["error"]]
        n_ai     = sum(1 for r in valid if r["result"]["final_label"] == "AI Generated")
        n_real   = len(valid) - n_ai
        avg_conf = (
            sum(r["result"]["final_confidence"] for r in valid) / len(valid) * 100
            if valid else 0
        )

        st.markdown(f"""
<div class="stats-row">
    <span class="stat-chip">📊 Total: {len(batch_results)}</span>
    <span class="stat-chip ai">🤖 AI Generated: {n_ai}</span>
    <span class="stat-chip real">✅ Real: {n_real}</span>
    <span class="stat-chip">🎯 Avg Confidence: {avg_conf:.1f}%</span>
</div>
""", unsafe_allow_html=True)

        csv_str = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_str,
            file_name="artguard_batch_results.csv",
            mime="text/csv",
            key="csv_download",
        )

    elif not uploaded_files:
        st.markdown(
            '<div class="info-box">📦 Upload up to 20 images to run batch analysis.</div>',
            unsafe_allow_html=True,
        )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.15rem;font-weight:800;font-family:Inter,sans-serif;">'
            '🎨 ArtGuard AI</div>'
            f'<div class="sidebar-version">{APP_VERSION} · Model V4</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        
        st.toggle("🌙  Dark Mode", key="theme_toggle")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        with st.expander("🧠 Model Information", expanded=False):
            st.markdown(f"""
**Active Model:** `{MODEL_V4}`

Training data: **60,000 images**  
Task: Real vs AI-Generated Detection  
Output: Sigmoid score [0.0 – 1.0]
""")

        with st.expander("📖 How It Works", expanded=False):
            st.markdown("""
**Step 1:** Upload an image  
**Step 2:** Model V4 analyses the image  
**Step 3:** Score is compared against threshold  

The model outputs a probability score between 
0.0 and 1.0 indicating how likely the image is real.
""")

        with st.expander("⚖️ Confidence Threshold", expanded=False):
            st.markdown(f"""
| Score | Verdict |
|-------|---------|
| `>= {THRESHOLD}` | ✅ Real |
| `< {THRESHOLD}`  | 🤖 AI Generated |

Confidence = how far the score is from the 
threshold, scaled to [50%, 100%].
""")

        st.markdown(
            '<div class="sidebar-footer">Built with TensorFlow & Streamlit<br>'
            '© 2024 ArtGuard AI</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="ArtGuard AI",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Seed theme_toggle on first load only (True = dark mode on)
    if "theme_toggle" not in st.session_state:
        st.session_state["theme_toggle"] = True

    # Read theme directly from theme_toggle — Streamlit updates widget keys
    # before the script reruns, so this always reflects the NEW toggle state
    # immediately, fixing the one-click-behind lag.
    theme = "dark" if st.session_state["theme_toggle"] else "light"
    st.session_state["theme"] = theme

    inject_css(theme)

    defaults = {
        "single_result": None,
        "batch_results": None,
        "single_image":  None,
        "active_tab":    0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    render_sidebar()
    render_header()

    with st.spinner("⚡ Loading Model V4…"):
        model = load_model()

    tab1, tab2 = st.tabs(["🖼  Single Image", "📦  Batch Upload"])

    with tab1:
        render_single_tab(model, theme)

    with tab2:
        render_batch_tab(model, theme)


if __name__ == "__main__":
    main()
