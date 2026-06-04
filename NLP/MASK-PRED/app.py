"""
MaskGuard AI — Real-time Mask Detection Frontend
Single-file Streamlit app. Load model.pkl → preprocess → predict → render results.

Preprocessing matches original notebook exactly:
    cv2.resize(img, (128,128)) → /255 → np.reshape([1,128,128,3])
    argmax == 1  →  With Mask   ✅
    argmax == 0  →  Without Mask ❌

Fixes vs v1:
  1. CLASS_NAMES order corrected to match notebook:
       index 1 = With Mask, index 0 = Without Mask
  2. Sidebar expander text overlap fixed (wildcard CSS removed)
  3. Model loading: pickle → keras fallback
  4. predict() verbose=0
"""

from __future__ import annotations

import pickle
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image
import os

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      =  os.path.join(dir, "model.pkl")
IMG_SIZE        = (128, 128)
# Matches notebook exactly:  argmax==1 → With Mask,  argmax==0 → Without Mask
CLASS_NAMES     = ["Without Mask", "With Mask"]
CONFIDENCE_HIGH = 80.0
CONFIDENCE_MID  = 50.0


# ─────────────────────────────────────────────
# MODEL LOADING  (robust: pickle → keras fallback)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Try plain pickle first.
    If the unpickled object lacks .predict(), attempt keras load as fallback.
    Handles keras import inside try/except to support late binding.
    """
    try:
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)
        
        # If pickle gave us a real model, use it directly
        if hasattr(obj, "predict"):
            return obj
    except ImportError:
        # Keras not available during unpickling; try direct keras load
        pass
    except Exception as e:
        st.warning(f"Pickle load failed: {e}. Attempting Keras load...")

    # Fallback: maybe the .pkl is actually a keras SavedModel path or bytes
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as keras_err:
        pass

    raise RuntimeError(
        "Loaded object from model.pkl has no .predict() method. "
        "Ensure the file contains a trained Keras/sklearn model."
    )


# ─────────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Mirror the notebook pipeline exactly:
        cv2.resize(img, (128,128)) -> /255 -> np.reshape([1,128,128,3])

    The model was trained on cv2 BGR arrays, so we flip RGB->BGR to match.
    """
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)                       # (128, 128)
    arr = np.array(img, dtype=np.float32)            # (128, 128, 3) RGB
    arr = arr[:, :, ::-1]                            # RGB -> BGR (match cv2.imread)
    arr = arr / 255.0                                # scale to [0, 1]
    return np.reshape(arr, [1, 128, 128, 3])         # -> (1, 128, 128, 3)


def predict(model, image: Image.Image) -> tuple[str, float]:
    """Run inference; return (label, confidence_pct)."""
    arr   = preprocess_image(image)
    preds = model.predict(arr, verbose=0)            # verbose=0 = no progress bar
    class_idx  = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0])) * 100.0
    label = CLASS_NAMES[class_idx]
    return label, confidence


# ─────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────
def inject_css(theme: str) -> None:
    is_dark = theme == "dark"

    bg_main      = "#0A0A0F"  if is_dark else "#F0F4F8"
    bg_card      = "#1A1A2E"  if is_dark else "#FFFFFF"
    bg_card2     = "#12122A"  if is_dark else "#F7FAFF"
    text_primary = "#FFFFFF"  if is_dark else "#1A1A2E"
    text_muted   = "#8888AA"  if is_dark else "#5A6478"
    border_col   = "#2A2A4A"  if is_dark else "#D8E0EE"
    sidebar_bg   = "#0F0F20"  if is_dark else "#E8EDF5"
    tab_inactive = "#1E1E38"  if is_dark else "#DCE4F0"

    st.markdown(f"""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Hide Streamlit chrome ── */
    #MainMenu {{visibility: hidden;}}
    footer     {{visibility: hidden;}}
    header     {{visibility: hidden;}}
    .block-container {{padding-top: 1rem; padding-bottom: 2rem;}}

    /* ── Global base ── */
    html, body,
    [data-testid="stAppViewContainer"],
    .main {{
        background-color: {bg_main} !important;
        color: {text_primary} !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }}

    /* ── Sidebar base (no wildcard * override — that broke expanders) ── */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_col};
    }}
    /* Target only real text nodes, not SVG/internal Streamlit spans */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span:not([data-testid]),
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stToggle {{
        color: {text_primary} !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }}

    /* ── Expander header — clean layout, no overlap ── */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        color: {text_primary} !important;
        padding: 0.6rem 0.8rem !important;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {{
        flex-shrink: 0;
        color: {text_muted} !important;
    }}
    [data-testid="stExpander"] {{
        border: 1px solid {border_col} !important;
        border-radius: 12px !important;
        background: {bg_card2} !important;
        margin-bottom: 0.5rem !important;
        overflow: hidden !important;
    }}

    /* ── Hero banner ── */
    .hero-banner {{
        background: linear-gradient(135deg, #0A0A0F 0%, #1A1A2E 50%, #0D1B2A 100%);
        border: 1px solid #2A2A4A;
        border-radius: 20px;
        padding: 3rem 2rem 2.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 2rem;
    }}
    .hero-banner::before {{
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(ellipse at center, rgba(0,255,136,0.05) 0%, transparent 60%);
        pointer-events: none;
    }}
    .hero-emoji-wrapper {{
        position: relative;
        display: inline-block;
        margin-bottom: 1rem;
    }}
    .hero-emoji {{
        font-size: 4.5rem;
        display: block;
        position: relative;
        z-index: 2;
        animation: float 3s ease-in-out infinite;
    }}
    .pulse-ring {{
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        width: 90px; height: 90px;
        border: 2px solid rgba(0,255,136,0.6);
        border-radius: 50%;
        animation: pulse-ring 2s ease-out infinite;
        z-index: 1;
    }}
    .pulse-ring-2 {{ animation-delay: 0.6s;  border-color: rgba(0,255,136,0.3); }}
    .pulse-ring-3 {{ animation-delay: 1.2s;  border-color: rgba(0,255,136,0.15); }}

    .hero-title {{
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        margin: 0.25rem 0;
        line-height: 1;
    }}
    .hero-title span {{
        background: linear-gradient(90deg, #00FF88, #00CCFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .hero-subtitle {{
        font-size: 1.05rem;
        color: #8888AA;
        font-weight: 400;
        margin-top: 0.6rem;
    }}
    .neon-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, #00FF88, #00CCFF, transparent);
        margin: 1.5rem auto 0;
        max-width: 420px;
        opacity: 0.7;
    }}

    /* ── Keyframes ── */
    @keyframes pulse-ring {{
        0%   {{ transform: translate(-50%, -50%) scale(0.8); opacity: 1; }}
        100% {{ transform: translate(-50%, -50%) scale(2.4); opacity: 0; }}
    }}
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50%       {{ transform: translateY(-9px); }}
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(28px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes glow-pulse-green {{
        0%, 100% {{ box-shadow: 0 0 30px #00FF8866; }}
        50%       {{ box-shadow: 0 0 60px #00FF88AA, 0 0 100px #00FF8844; }}
    }}
    @keyframes glow-pulse-red {{
        0%, 100% {{ box-shadow: 0 0 30px #FF444466; }}
        50%       {{ box-shadow: 0 0 60px #FF4444AA, 0 0 100px #FF444444; }}
    }}

    /* ── Pill tab overrides ── */
    [data-testid="stTabs"] [role="tablist"] {{
        background: {tab_inactive};
        border-radius: 50px;
        padding: 4px;
        gap: 4px;
        border: 1px solid {border_col};
    }}
    [data-testid="stTabs"] [role="tab"] {{
        border-radius: 50px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1.4rem !important;
        color: {text_muted} !important;
        transition: all 0.2s ease;
    }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, #00FF88, #00CCFF) !important;
        color: #0A0A0F !important;
        box-shadow: 0 0 18px rgba(0,255,136,0.45);
    }}

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {{
        border: 2px dashed #00FF8855 !important;
        border-radius: 14px !important;
        background: {bg_card2} !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: #00FF88 !important;
        box-shadow: 0 0 20px rgba(0,255,136,0.15);
    }}

    /* ── Camera input ── */
    [data-testid="stCameraInput"] {{
        border: 2px solid {border_col} !important;
        border-radius: 14px !important;
        overflow: hidden;
    }}

    /* ── Detect button ── */
    .stButton > button {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.04em !important;
        border-radius: 50px !important;
        padding: 0.65rem 2.5rem !important;
        border: none !important;
        background: linear-gradient(135deg, #00FF88, #00CCFF) !important;
        color: #0A0A0F !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0,255,136,0.3) !important;
        width: 100% !important;
        margin-top: 0.75rem !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0,255,136,0.5) !important;
    }}
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}

    /* ── Result card ── */
    .result-card {{
        border-radius: 20px;
        padding: 2.5rem 2rem 1.5rem;
        text-align: center;
        animation: fadeInUp 0.5s ease forwards;
        position: relative;
        overflow: hidden;
    }}
    .result-card.mask {{
        background: linear-gradient(135deg, {bg_card}, #0D2A1E);
        border: 1.5px solid #00FF8866;
        animation: fadeInUp 0.5s ease forwards, glow-pulse-green 3s ease-in-out infinite;
    }}
    .result-card.nomask {{
        background: linear-gradient(135deg, {bg_card}, #2A0D0D);
        border: 1.5px solid #FF444466;
        animation: fadeInUp 0.5s ease forwards, glow-pulse-red 3s ease-in-out infinite;
    }}
    .result-emoji {{
        font-size: 4rem;
        display: block;
        margin-bottom: 0.5rem;
        animation: float 2.5s ease-in-out infinite;
    }}
    .result-label {{
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin: 0.2rem 0 0.6rem;
    }}
    .result-label.mask   {{ color: #00FF88; }}
    .result-label.nomask {{ color: #FF4444; }}
    .result-confidence-label {{
        font-size: 0.95rem;
        color: {text_muted};
        font-weight: 500;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }}
    .result-status-bar {{
        margin-top: 1.2rem;
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        display: inline-block;
    }}
    .result-status-bar.mask {{
        background: rgba(0,255,136,0.1);
        border: 1px solid #00FF8844;
        color: #00FF88;
    }}
    .result-status-bar.nomask {{
        background: rgba(255,68,68,0.1);
        border: 1px solid #FF444444;
        color: #FF4444;
    }}

    /* ── Progress bar ── */
    [data-testid="stProgress"] > div > div {{
        border-radius: 50px !important;
        height: 10px !important;
    }}

    /* ── Section label ── */
    .section-label {{
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {text_muted};
        margin-bottom: 0.75rem;
    }}

    /* ── Info pill ── */
    .info-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: {bg_card};
        border: 1px solid {border_col};
        border-radius: 50px;
        padding: 4px 14px;
        font-size: 0.82rem;
        color: {text_muted};
        margin: 3px 2px;
    }}
    .info-pill b {{ color: {text_primary}; font-weight: 600; }}

    /* ── Sidebar version badge ── */
    .sidebar-version {{
        display: inline-block;
        background: rgba(0,255,136,0.15);
        color: #00FF88;
        border: 1px solid #00FF8844;
        border-radius: 50px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        margin-left: 8px;
        vertical-align: middle;
    }}
    .sidebar-footer {{
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid {border_col};
        font-size: 0.8rem;
        color: {text_muted};
        text-align: center;
        line-height: 1.6;
    }}

    /* ── Image ── */
    [data-testid="stImage"] img {{
        border-radius: 12px;
        border: 1px solid {border_col};
    }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RENDER: HERO HEADER
# ─────────────────────────────────────────────
def render_header() -> None:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-emoji-wrapper">
            <span class="hero-emoji">😷</span>
            <div class="pulse-ring"></div>
            <div class="pulse-ring pulse-ring-2"></div>
            <div class="pulse-ring pulse-ring-3"></div>
        </div>
        <div class="hero-title">Mask<span>Guard</span> AI</div>
        <div class="hero-subtitle">Real-time mask detection powered by deep learning</div>
        <div class="neon-divider"></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RENDER: INPUT PANEL
# ─────────────────────────────────────────────
def render_input(model) -> None:
    tab_upload, tab_camera = st.tabs(["📁  Upload Image", "📷  Camera Capture"])

    # ── TAB 1: Upload ──────────────────────────────────────────────────────
    with tab_upload:
        st.markdown(
            '<div class="section-label">Drop or browse an image</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            label="",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded:
            pil_img = Image.open(BytesIO(uploaded.read())).convert("RGB")
            st.session_state["image"]      = pil_img
            st.session_state["input_mode"] = "upload"

            col_l, col_c, col_r = st.columns([1, 4, 1])
            with col_c:
                st.image(pil_img, caption="Preview", use_container_width=True)

            if st.button("🔍  Detect Mask", key="btn_upload"):
                with st.spinner("Running inference…"):
                    label, conf = predict(model, pil_img)
                st.session_state["result"]     = label
                st.session_state["confidence"] = conf
                st.rerun()

    # ── TAB 2: Camera ──────────────────────────────────────────────────────
    with tab_camera:
        st.markdown(
            '<div class="section-label">Capture a live snapshot</div>',
            unsafe_allow_html=True,
        )
        camera_img = st.camera_input(label="", label_visibility="collapsed")

        if camera_img:
            pil_img = Image.open(BytesIO(camera_img.getvalue())).convert("RGB")
            st.session_state["image"]      = pil_img
            st.session_state["input_mode"] = "camera"

            col_l, col_c, col_r = st.columns([1, 4, 1])
            with col_c:
                st.image(pil_img, caption="Captured Frame", use_container_width=True)

            if st.button("🔍  Detect Mask", key="btn_camera"):
                with st.spinner("Running inference…"):
                    label, conf = predict(model, pil_img)
                st.session_state["result"]     = label
                st.session_state["confidence"] = conf
                st.rerun()


# ─────────────────────────────────────────────
# RENDER: RESULT CARD
# ─────────────────────────────────────────────
def render_result() -> None:
    result     = st.session_state.get("result")
    confidence = st.session_state.get("confidence")
    if result is None:
        return

    is_mask  = result == "With Mask"
    card_cls = "mask" if is_mask else "nomask"
    emoji    = "✅" if is_mask else "❌"
    display  = "MASK DETECTED" if is_mask else "NO MASK DETECTED"
    status   = "✅ Safe — Mask is being worn properly" if is_mask else "⚠️ Alert — No mask detected"

    bar_color = (
        "#00FF88" if confidence >= CONFIDENCE_HIGH else
        "#FFD700" if confidence >= CONFIDENCE_MID  else
        "#FF4444"
    )

    st.markdown(f"""
    <div class="result-card {card_cls}">
        <span class="result-emoji">{emoji}</span>
        <div class="result-label {card_cls}">{display}</div>
        <div class="result-confidence-label">Confidence: {confidence:.1f}%</div>
        <div class="result-status-bar {card_cls}">{status}</div>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic progress-bar colour
    st.markdown(f"""
    <style>
    [data-testid="stProgress"] > div > div > div {{
        background: {bar_color} !important;
    }}
    </style>""", unsafe_allow_html=True)
    st.progress(min(int(confidence), 100))

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        if st.button("🔄  Analyse Another Image", key="btn_reset"):
            for key in ("result", "confidence", "image", "input_mode"):
                st.session_state.pop(key, None)
            st.rerun()


# ─────────────────────────────────────────────
# RENDER: SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        # Header
        st.markdown(f"""
        <div style="margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid #2A2A4A;">
            <span style="font-size:1.25rem;font-weight:700;">MaskGuard AI</span>
            <span class="sidebar-version">v1.0</span>
            <div style="font-size:0.8rem;margin-top:0.3rem;color:#8888AA;">
                CNN Mask Detection System
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Theme toggle
        st.markdown('<div class="section-label">Appearance</div>', unsafe_allow_html=True)
        dark_on = st.toggle(
            "Dark Mode",
            value=(st.session_state.get("theme", "dark") == "dark"),
            key="theme_toggle",
        )
        st.session_state["theme"] = "dark" if dark_on else "light"

        st.markdown("<br>", unsafe_allow_html=True)

        # Model info — plain text label (no emoji prefix that caused overlap)
        with st.expander("Model Information"):
            st.markdown("""
            <div style="margin-top:0.25rem;">
            <div class="info-pill">Architecture <b>CNN</b></div>
            <div class="info-pill">Format <b>Pickle</b></div>
            <div class="info-pill">Input <b>128 × 128</b></div>
            <div class="info-pill">Channels <b>RGB</b></div>
            <div class="info-pill">Classes <b>2</b></div>
            <div class="info-pill">Norm <b>÷ 255</b></div>
            </div>
            <div style="font-size:0.82rem;color:#8888AA;margin-top:0.75rem;line-height:1.7;">
                🔴 Class 0 → Without Mask<br>
                🟢 Class 1 → With Mask
            </div>
            """, unsafe_allow_html=True)

        # How to use — plain text label
        with st.expander("How to Use"):
            st.markdown("""
            **Step 1 — Choose input**  
            Use the *Upload* tab for saved images, or *Camera* for a live snapshot.

            **Step 2 — Preview & detect**  
            Review the preview, then click **Detect Mask**.

            **Step 3 — Read results**  
            The result card shows detection outcome and confidence score.
            """)

        # Footer
        st.markdown("""
        <div class="sidebar-footer">
            Built with ❤️ using<br>Streamlit &amp; TensorFlow
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="MaskGuard AI",
        page_icon="😷",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Session state defaults
    st.session_state.setdefault("theme",      "dark")
    st.session_state.setdefault("result",     None)
    st.session_state.setdefault("confidence", None)
    st.session_state.setdefault("image",      None)
    st.session_state.setdefault("input_mode", None)

    # 1. Inject CSS with current theme
    inject_css(st.session_state["theme"])

    # 2. Sidebar (theme toggle lives here; updates session state)
    render_sidebar()

    # 3. Re-inject CSS if theme was toggled this rerun
    inject_css(st.session_state["theme"])

    # 4. Hero header
    render_header()

    # 5. Load model (cached; only runs once)
    try:
        model = load_model()
    except FileNotFoundError:
        st.error(
            f"⚠️ **Model file not found.** "
            f"Place `{MODEL_PATH}` in the same directory as `app.py` and restart."
        )
        st.stop()
    except RuntimeError as e:
        st.error(f"⚠️ **Model load error:** {e}")
        st.stop()

    # 6. Input panel
    render_input(model)

    # 7. Result card (only when a prediction exists)
    if st.session_state.get("result"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label" style="text-align:center;letter-spacing:0.15em;">'
            'Detection Result</div>',
            unsafe_allow_html=True,
        )
        render_result()


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────
# requirements.txt
# ─────────────────────────────────────────────
# streamlit>=1.27.0
# tensorflow>=2.12.0
# pillow>=10.0.0
# numpy>=1.24.0
#
# pip install -r requirements.txt
# streamlit run app.py