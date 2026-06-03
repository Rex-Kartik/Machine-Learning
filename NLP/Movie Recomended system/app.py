import pickle
import difflib
import os
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────── CONSTANTS ────────────────────────────────── #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_PATH     = os.path.join(BASE_DIR, "movies.pkl")
SIMILARITY_PATH = os.path.join(BASE_DIR, "similarity (1).pkl")
TOP_N           = 10
FEATURED_N      = 8
ACCENT          = "#E50914"

# Free TMDB API key (public read-only v3 key — no auth needed for public data)
TMDB_API_KEY    = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_IMG_BASE   = "https://image.tmdb.org/t/p/w500"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"

CARD_GRADIENTS = [
    "linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%)",
    "linear-gradient(135deg,#200122 0%,#6f0000 100%)",
    "linear-gradient(135deg,#0f2027 0%,#203a43 50%,#2c5364 100%)",
    "linear-gradient(135deg,#232526 0%,#414345 100%)",
    "linear-gradient(135deg,#1f1c2c 0%,#928dab 100%)",
]

# ──────────────────────────────── DATA LAYER ──────────────────────────────── #
@st.cache_resource
def load_data() -> tuple:
    movies     = pickle.load(open(MOVIES_PATH,     "rb"))
    similarity = pickle.load(open(SIMILARITY_PATH, "rb"))
    return movies, similarity


def _titles(movies) -> list[str]:
    if hasattr(movies, "columns"):
        return movies["title"].tolist()
    return list(movies)


def get_recommendations(movie_title: str, movies, similarity: np.ndarray, top_n: int = TOP_N):
    titles  = _titles(movies)
    matches = difflib.get_close_matches(movie_title, titles, n=1, cutoff=0.4)
    if not matches:
        return [], movie_title
    matched = matches[0]
    idx     = titles.index(matched)
    scores  = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    recs    = [
        {"title": titles[i], "index": i, "rank": rank, "score": float(s)}
        for rank, (i, s) in enumerate(scores[1:top_n + 1], 1)
    ]
    return recs, matched


def get_featured(movies, n: int = FEATURED_N) -> list[dict]:
    titles = _titles(movies)
    return [{"title": t, "index": i} for i, t in enumerate(titles[:n])]


# ────────────────────────────────── CSS ───────────────────────────────────── #
def inject_css() -> None:
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800;900&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    components.html("""
    <style>
    #MainMenu, footer, header { visibility: hidden !important; }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem !important;
        max-width: 1280px !important;
    }
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
        background-color: #0E1117 !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] {
        background: #13131F !important;
        border-right: 1px solid rgba(255,255,255,0.07) !important;
    }
    div[data-testid="stTextInput"] input {
        background: #16162A !important;
        border: 2px solid rgba(229,9,20,0.45) !important;
        border-radius: 50px !important;
        color: #FFFFFF !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 1rem !important;
        padding: 0.85rem 1.5rem !important;
        box-shadow: 0 4px 20px rgba(229,9,20,0.1) !important;
        transition: all .3s ease !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #E50914 !important;
        box-shadow: 0 4px 32px rgba(229,9,20,0.38) !important;
        outline: none !important;
    }
    div[data-testid="stTextInput"] label { display: none !important; }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg,#E50914 0%,#c40812 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 50px !important;
        padding: .75rem 2.5rem !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: .03em !important;
        box-shadow: 0 6px 24px rgba(229,9,20,0.4) !important;
        transition: all .25s ease !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 36px rgba(229,9,20,0.55) !important;
    }
    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position:  200% 0; }
    }
    </style>
    """, height=0, scrolling=False)


# ──────────────────────────── CARD GRID (iframe) ─────────────────────────── #
# All card HTML is rendered inside a single components.html() call per grid.
# This completely bypasses Streamlit's markdown sanitiser.

import math as _math

def _grid_height(n_cards: int, cols: int) -> int:
    """
    Derive a pixel-accurate iframe height from card count + column count.
    Streamlit's components.html() height is fixed at creation time and cannot
    be changed by postMessage after mount, so we must compute it upfront.

    Card poster uses CSS aspect-ratio 2:3.
    Empirical column widths on a 1280-px wide layout (sidebar open):
      4-col → ~255px wide  → poster ~383px tall
      3-col → ~330px wide  → poster ~495px tall
    Add 90px for card body (title + tags + padding) and 14px gap per row.
    Add 20px bottom buffer so the last card never clips.
    """
    poster_h = 383 if cols == 4 else 495
    row_h    = poster_h + 90 + 14
    rows     = _math.ceil(n_cards / cols)
    return rows * row_h + 20


def _build_grid_html(movies_data: list[dict], cols: int, tmdb_key: str) -> str:
    """
    Renders a responsive card grid via a self-contained HTML page.
    Each card fetches its own poster + trailer key from TMDB on the client side.
    Clicking a card shows an embedded YouTube trailer in a modal.
    """
    cards_json_items = []
    for m in movies_data:
        title_esc = m["title"].replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
        rank      = m.get("rank", "")
        score     = m.get("score", None)
        idx       = m.get("index", 0)
        grad      = CARD_GRADIENTS[idx % len(CARD_GRADIENTS)]
        score_str = f"{score:.0%}" if score is not None else ""
        featured  = "true" if rank == "" else "false"
        cards_json_items.append(
            f'{{title:"{title_esc}",rank:"{rank}",score:"{score_str}",'
            f'featured:{featured},gradient:"{grad}"}}'
        )

    cards_json = "[" + ",".join(cards_json_items) + "]"
    col_width  = f"calc({100/cols}% - 14px)"

    return f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:transparent;font-family:'Poppins',sans-serif;overflow:visible;}}
.grid{{display:flex;flex-wrap:wrap;gap:14px;padding:4px 2px 12px;}}
.card{{
    width:{col_width};
    background:#1C1C2E;
    border-radius:16px;
    border:1px solid rgba(255,255,255,0.07);
    overflow:hidden;
    cursor:pointer;
    transition:transform .3s ease,box-shadow .3s ease,border-color .3s ease;
    box-shadow:0 2px 12px rgba(0,0,0,.35);
    flex-shrink:0;
}}
.card:hover{{
    transform:translateY(-6px) scale(1.025);
    box-shadow:0 16px 48px rgba(229,9,20,0.28);
    border-color:rgba(229,9,20,0.35);
}}
.poster{{
    width:100%;
    aspect-ratio:2/3;
    position:relative;
    overflow:hidden;
}}
/* Real poster image — stacked on top via z-index, hidden until loaded */
.poster img{{
    position:absolute;
    inset:0;
    width:100%;
    height:100%;
    object-fit:cover;
    display:block;
    opacity:0;
    transition:opacity .45s ease;
    z-index:2;
}}
.poster img.loaded{{
    opacity:1;
}}
/* Gradient fallback — always behind the image */
.poster-fallback{{
    position:absolute;
    inset:0;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:3rem;
    z-index:1;
}}
.poster-overlay{{
    position:absolute;
    inset:0;
    background:linear-gradient(to top,rgba(0,0,0,.85) 0%,transparent 55%);
    pointer-events:none;
    z-index:3;
}}
.rank-badge{{
    position:absolute;top:10px;left:10px;
    background:#E50914;color:#fff;
    font-size:.68rem;font-weight:700;
    padding:3px 10px;border-radius:20px;
    letter-spacing:.05em;z-index:5;
}}
.play-btn{{
    position:absolute;
    top:50%;left:50%;
    transform:translate(-50%,-50%) scale(0);
    background:rgba(229,9,20,0.9);
    border:none;border-radius:50%;
    width:56px;height:56px;
    display:flex;align-items:center;justify-content:center;
    cursor:pointer;
    transition:transform .25s ease,background .2s;
    z-index:6;
    pointer-events:none;
    box-shadow:0 4px 20px rgba(0,0,0,.5);
}}
.card:hover .play-btn{{
    transform:translate(-50%,-50%) scale(1);
    pointer-events:auto;
}}
.play-btn:hover{{ background:rgba(229,9,20,1); }}
.play-btn svg{{ width:22px;height:22px;fill:#fff;margin-left:3px; }}
.card-body{{padding:.8rem 1rem 1rem;}}
.card-title{{
    font-size:.92rem;font-weight:600;color:#fff;
    margin:0 0 .4rem;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
    line-height:1.3;
}}
.tags{{display:flex;gap:.35rem;flex-wrap:wrap;}}
.tag{{font-size:.65rem;font-weight:600;padding:2px 9px;border-radius:20px;letter-spacing:.04em;}}
.tag-genre{{background:rgba(229,9,20,.15);color:#E50914;border:1px solid rgba(229,9,20,.3);}}
.tag-feat{{background:rgba(255,200,0,.13);color:#FFD700;border:1px solid rgba(255,200,0,.28);}}
.tag-score{{background:rgba(80,200,120,.13);color:#50C878;border:1px solid rgba(80,200,120,.28);}}
/* Trailer overlay */
.trailer-overlay{{
    display:none;
    position:fixed;inset:0;
    background:rgba(0,0,0,.88);
    z-index:9999;
    align-items:center;justify-content:center;
    flex-direction:column;
    animation:fadeIn .2s ease;
}}
.trailer-overlay.active{{display:flex;}}
@keyframes fadeIn{{from{{opacity:0}}to{{opacity:1}}}}
.trailer-box{{
    background:#0E1117;
    border-radius:20px;
    border:1px solid rgba(229,9,20,0.3);
    overflow:hidden;
    width:min(800px,94vw);
    box-shadow:0 24px 80px rgba(0,0,0,.8);
    position:relative;
}}
.trailer-header{{
    display:flex;align-items:center;justify-content:space-between;
    padding:1rem 1.2rem .75rem;
    border-bottom:1px solid rgba(255,255,255,.07);
}}
.trailer-title{{font-size:1rem;font-weight:700;color:#fff;}}
.trailer-close{{
    background:rgba(229,9,20,.15);border:1px solid rgba(229,9,20,.3);
    color:#E50914;border-radius:50%;width:34px;height:34px;
    font-size:1.1rem;cursor:pointer;display:flex;align-items:center;justify-content:center;
    transition:background .2s;
}}
.trailer-close:hover{{background:rgba(229,9,20,.35);}}
.trailer-frame{{width:100%;aspect-ratio:16/9;border:none;}}
.no-trailer{{
    padding:2.5rem;text-align:center;
    color:rgba(255,255,255,.5);font-size:.9rem;
}}
.loading-spinner{{
    padding:2.5rem;text-align:center;color:#E50914;font-size:.9rem;
}}
</style>
</head>
<body>

<div class="grid" id="grid"></div>

<!-- Trailer Modal -->
<div class="trailer-overlay" id="trailerOverlay">
  <div class="trailer-box">
    <div class="trailer-header">
      <span class="trailer-title" id="trailerTitle"></span>
      <button class="trailer-close" onclick="closeTrailer()">✕</button>
    </div>
    <div id="trailerContent"></div>
  </div>
</div>

<script>
const TMDB_KEY  = "{tmdb_key}";
const IMG_BASE  = "https://image.tmdb.org/t/p/w500";
const cards     = {cards_json};
const posterCache = {{}};
const trailerCache = {{}};

async function fetchPoster(title) {{
    if (posterCache[title] !== undefined) return posterCache[title];
    try {{
        const r = await fetch(
            `https://api.themoviedb.org/3/search/movie?api_key=${{TMDB_KEY}}&query=${{encodeURIComponent(title)}}&page=1`
        );
        const d = await r.json();
        const path = d.results?.[0]?.poster_path || null;
        posterCache[title] = path;
        return path;
    }} catch(e) {{ posterCache[title] = null; return null; }}
}}

async function fetchTrailerKey(title) {{
    if (trailerCache[title] !== undefined) return trailerCache[title];
    try {{
        const r = await fetch(
            `https://api.themoviedb.org/3/search/movie?api_key=${{TMDB_KEY}}&query=${{encodeURIComponent(title)}}&page=1`
        );
        const d  = await r.json();
        const id = d.results?.[0]?.id;
        if (!id) {{ trailerCache[title] = null; return null; }}
        const v  = await fetch(
            `https://api.themoviedb.org/3/movie/${{id}}/videos?api_key=${{TMDB_KEY}}`
        );
        const vd = await v.json();
        const yt = vd.results?.find(x => x.site==="YouTube" && (x.type==="Trailer"||x.type==="Teaser"));
        trailerCache[title] = yt?.key || null;
        return trailerCache[title];
    }} catch(e) {{ trailerCache[title] = null; return null; }}
}}

function closeTrailer() {{
    document.getElementById("trailerOverlay").classList.remove("active");
    document.getElementById("trailerContent").innerHTML = "";
}}

async function openTrailer(title) {{
    const overlay = document.getElementById("trailerOverlay");
    const content = document.getElementById("trailerContent");
    document.getElementById("trailerTitle").textContent = title;
    content.innerHTML = '<div class="loading-spinner">🎬 Loading trailer…</div>';
    overlay.classList.add("active");
    const key = await fetchTrailerKey(title);
    if (key) {{
        content.innerHTML = `<iframe class="trailer-frame"
            src="https://www.youtube.com/embed/${{key}}?autoplay=1&rel=0"
            allow="autoplay;encrypted-media" allowfullscreen></iframe>`;
    }} else {{
        content.innerHTML = '<div class="no-trailer">🎬 No trailer available for this movie.</div>';
    }}
}}

document.getElementById("trailerOverlay").addEventListener("click", function(e) {{
    if (e.target === this) closeTrailer();
}});
document.addEventListener("keydown", e => {{ if(e.key==="Escape") closeTrailer(); }});

async function buildCard(m, idx) {{
    const div = document.createElement("div");
    div.className = "card";

    const rankBadge = m.rank
        ? `<div class="rank-badge">#${{m.rank}}</div>`
        : "";

    const scorePill = m.score
        ? `<span class="tag tag-score">⚡ ${{m.score}}</span>`
        : "";

    const featPill = m.featured === "true"
        ? `<span class="tag tag-feat">⭐ Featured</span>`
        : "";

    div.innerHTML = `
        <div class="poster" id="poster-${{idx}}" style="background:${{m.gradient}};">
            <div class="poster-fallback">🎬</div>
            <div class="poster-overlay"></div>
            ${{rankBadge}}
            <button class="play-btn" onclick="event.stopPropagation();openTrailer('${{m.title.replace(/'/g,"\\'")}}')">
                <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
            </button>
        </div>
        <div class="card-body">
            <div class="card-title" title="${{m.title}}">${{m.title}}</div>
            <div class="tags">
                <span class="tag tag-genre">🎬 Movie</span>
                ${{featPill}}
                ${{scorePill}}
            </div>
        </div>
    `;

    div.addEventListener("click", () => openTrailer(m.title));
    return div;
}}

function hydratePosters() {{
    // Fire all poster fetches in parallel — no awaiting in sequence
    cards.forEach(async (m, idx) => {{
        const path = await fetchPoster(m.title);
        if (!path) return;
        const poster = document.getElementById(`poster-${{idx}}`);
        if (!poster) return;
        const img = document.createElement("img");
        img.alt = m.title;
        // Insert BEFORE overlay so z-index layering works:
        // fallback(z1) → img(z2, fades in) → overlay(z3) → badge(z5) → playBtn(z6)
        const overlay = poster.querySelector(".poster-overlay");
        poster.insertBefore(img, overlay);
        img.onload  = () => img.classList.add("loaded");   // fade in on load
        img.onerror = () => img.remove();                  // silently drop on error
        img.src = IMG_BASE + path;                         // set src AFTER handlers
    }});
}}

// Auto-resize iframe height to match content — eliminates all fixed-height guessing
function setupResizeObserver() {{
    if (!window.ResizeObserver) return;
    const ro = new ResizeObserver(() => {{
        const h = document.documentElement.scrollHeight;
        window.parent.postMessage({{type:"streamlit:setFrameHeight", height:h}}, "*");
    }});
    ro.observe(document.body);
}}

async function init() {{
    const grid = document.getElementById("grid");
    const frag = document.createDocumentFragment();
    for (let i = 0; i < cards.length; i++) {{
        frag.appendChild(await buildCard(cards[i], i));
    }}
    grid.appendChild(frag);
    hydratePosters();      // non-blocking poster fetch
    setupResizeObserver(); // auto-fit iframe height
}}

init();
</script>
</body>
</html>"""


# ──────────────────────────────── SIDEBAR ─────────────────────────────────── #
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f'<div style="font-size:1.5rem;font-weight:900;color:{ACCENT};'
            f'letter-spacing:-.02em;margin-bottom:.5rem;">🎬 CineMatch</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.selectbox("🎨 Theme", ["Dark"], index=0, key="theme_select", disabled=True,
                     help="More themes coming soon")
        st.divider()
        st.markdown(
            '<p style="font-size:.82rem;line-height:1.75;opacity:.65;">'
            "<b>How it works</b><br>"
            "Type any movie title in the search bar. CineMatch uses a precomputed "
            "cosine similarity matrix to find your 10 closest matches instantly.<br><br>"
            "<b>Click any card</b> to watch its trailer."
            "</p>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────── HERO ────────────────────────────────────── #
def render_hero() -> tuple[str, bool]:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0E1117 0%,#1a1a2e 50%,#16213e 100%);
          padding:4rem 2rem 3rem;border-radius:0 0 32px 32px;text-align:center;
          position:relative;overflow:hidden;margin-bottom:2rem;
          box-shadow:0 16px 64px rgba(0,0,0,.5);">
          <div style="position:absolute;inset:0;
            background:linear-gradient(90deg,transparent,rgba(229,9,20,.1),transparent);
            background-size:200% 100%;animation:shimmer 4s ease-in-out infinite;"></div>
          <span style="font-size:3.5rem;display:block;margin-bottom:.4rem;">🎬</span>
          <h1 style="font-size:clamp(2.4rem,5vw,4rem);font-weight:900;color:#fff;
            letter-spacing:-.02em;margin:0;text-shadow:0 4px 24px rgba(229,9,20,.5);
            font-family:'Poppins',sans-serif;">
            Cine<span style="color:#E50914;">Match</span>
          </h1>
          <p style="font-size:1.05rem;font-weight:300;color:#8888AA;
            margin:.5rem 0 0;letter-spacing:.04em;">
            Discover your next favourite film &nbsp;·&nbsp; Click any card to watch the trailer
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, col_m, _ = st.columns([1, 3, 1])
    with col_m:
        query   = st.text_input("search",
                                placeholder="e.g. The Dark Knight, Inception, Parasite…",
                                label_visibility="collapsed",
                                key="search_input")
        clicked = st.button("🔍 Find Movies", use_container_width=True)
    return query.strip(), clicked


# ────────────────────────────── FEATURED ──────────────────────────────────── #
def render_featured(movies) -> None:
    st.markdown(
        f'<div style="font-size:1.5rem;font-weight:700;margin:1rem 0 .75rem;'
        f'display:flex;align-items:center;gap:.5rem;">'
        f'<span style="display:inline-block;width:4px;height:1.3rem;background:{ACCENT};border-radius:2px;"></span>'
        f'⭐ Featured Today</div>',
        unsafe_allow_html=True,
    )
    featured = get_featured(movies, FEATURED_N)
    html     = _build_grid_html(featured, cols=4, tmdb_key=TMDB_API_KEY)
    components.html(html, height=_grid_height(FEATURED_N, 4), scrolling=False)

    st.markdown(
        f'<hr style="height:1px;background:linear-gradient(90deg,transparent,{ACCENT}55,transparent);'
        f'border:none;margin:1rem 0 1.5rem;">',
        unsafe_allow_html=True,
    )


# ────────────────────────────── RESULTS ───────────────────────────────────── #
def render_results(results: list[dict], matched_title: str) -> None:
    st.markdown(
        f'<div style="font-size:1.5rem;font-weight:700;margin:.5rem 0 .75rem;'
        f'display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;">'
        f'<span style="display:inline-block;width:4px;height:1.3rem;background:{ACCENT};border-radius:2px;"></span>'
        f'🎯 Because you searched for &ldquo;<em>{matched_title}</em>&rdquo;</div>',
        unsafe_allow_html=True,
    )
    if not results:
        st.markdown(
            '<div style="text-align:center;padding:4rem 2rem;opacity:.6;">'
            '<div style="font-size:4rem;margin-bottom:1rem;">🎬</div>'
            '<h3 style="font-size:1.4rem;font-weight:600;margin-bottom:.5rem;">No movies found</h3>'
            "<p>Try a different title or check your spelling.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    html = _build_grid_html(results, cols=3, tmdb_key=TMDB_API_KEY)
    components.html(html, height=_grid_height(len(results), 3), scrolling=False)


# ────────────────────────────────── MAIN ──────────────────────────────────── #
def main() -> None:
    st.set_page_config(
        page_title="CineMatch",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    for key, val in [("results", []), ("matched_title", ""), ("searched", False)]:
        if key not in st.session_state:
            st.session_state[key] = val

    inject_css()
    render_sidebar()

    movies, similarity = load_data()

    query, clicked = render_hero()

    if clicked and query:
        with st.spinner("🎬 Finding your movies…"):
            recs, matched = get_recommendations(query, movies, similarity, TOP_N)
            st.session_state.results       = recs
            st.session_state.matched_title = matched
            st.session_state.searched      = True

    render_featured(movies)

    if st.session_state.searched:
        render_results(st.session_state.results, st.session_state.matched_title)


if __name__ == "__main__":
    main()


# ─────────────────────────── requirements.txt ─────────────────────────────── #
# streamlit>=1.27.0
# numpy>=1.24.0
#
# streamlit run app.py