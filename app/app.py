"""Streamlit UI for the Doctor's Prescription Reader — sci-fi HUD edition."""

import base64
import html
import io
import sys
import time
from pathlib import Path

# Ensure the project root is importable when run via `streamlit run app/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from PIL import Image, ImageDraw

from app import utils
from app.image_processing import to_pil
from src.detection import WordDetector, doctr_available

st.set_page_config(page_title="Prescription Reader", page_icon="💊", layout="centered")

# Keep the scan animation on screen at least this long, even if inference is fast.
MIN_SCAN_SECONDS = 1.2

NEON = "#00F0FF"
NEON_GREEN = "#00FF9C"

# --- global sci-fi styling -------------------------------------------
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
:root {
    --neon: #00F0FF;
    --neon-green: #00FF9C;
    --neon-dim: rgba(0, 240, 255, 0.35);
}
.stApp {
    background:
        radial-gradient(ellipse at 50% -10%, rgba(0, 240, 255, 0.10), transparent 55%),
        linear-gradient(rgba(0, 240, 255, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 240, 255, 0.035) 1px, transparent 1px),
        #050A12;
    background-size: 100% 100%, 34px 34px, 34px 34px, 100% 100%;
}
h1, h2, h3, .hud-font {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 3px;
    color: var(--neon) !important;
    text-shadow: 0 0 8px rgba(0, 240, 255, 0.7), 0 0 24px rgba(0, 240, 255, 0.35);
    text-transform: uppercase;
}
p, li, .stCaption, .stMarkdown, label, td, th {
    font-family: 'Share Tech Mono', monospace;
}
.hud-chips { display: flex; gap: 10px; flex-wrap: wrap; margin: 4px 0 14px 0; }
.hud-chip {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 2px;
    padding: 4px 12px;
    border: 1px solid var(--neon-dim);
    border-radius: 3px;
    color: var(--neon);
    background: rgba(0, 240, 255, 0.06);
    box-shadow: inset 0 0 12px rgba(0, 240, 255, 0.08);
}
.hud-chip.offline {
    color: #FF5470;
    border-color: rgba(255, 84, 112, 0.5);
    background: rgba(255, 84, 112, 0.06);
}
.stButton > button {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--neon);
    background: rgba(0, 240, 255, 0.05);
    border: 1px solid var(--neon-dim);
    border-radius: 3px;
    transition: all 0.2s ease;
}
.stButton > button:hover:enabled {
    color: #050A12;
    background: var(--neon);
    box-shadow: 0 0 18px rgba(0, 240, 255, 0.8);
    border-color: var(--neon);
}
/* --- the scanner frame ------------------------------------------- */
.scan-frame {
    position: relative;
    border: 1px solid var(--neon-dim);
    box-shadow: 0 0 24px rgba(0, 240, 255, 0.15), inset 0 0 40px rgba(0, 240, 255, 0.05);
    background: #02060B;
    padding: 14px;
    margin: 10px 0 18px 0;
}
.scan-frame img { display: block; width: 100%; }
.scan-frame .crt {
    position: absolute; inset: 14px;
    background: repeating-linear-gradient(
        0deg, rgba(0, 240, 255, 0.05) 0px, rgba(0, 240, 255, 0.05) 1px,
        transparent 1px, transparent 4px);
    pointer-events: none;
}
.scan-frame .corner {
    position: absolute; width: 26px; height: 26px;
    border: 2px solid var(--neon);
    filter: drop-shadow(0 0 6px var(--neon));
}
.scan-frame .tl { top: -2px; left: -2px; border-right: none; border-bottom: none; }
.scan-frame .tr { top: -2px; right: -2px; border-left: none; border-bottom: none; }
.scan-frame .bl { bottom: -2px; left: -2px; border-right: none; border-top: none; }
.scan-frame .br { bottom: -2px; right: -2px; border-left: none; border-top: none; }
.scan-frame .scan-line {
    position: absolute; left: 14px; right: 14px; height: 3px; top: 14px;
    background: linear-gradient(90deg, transparent, var(--neon-green), transparent);
    box-shadow: 0 0 14px 3px rgba(0, 255, 156, 0.8);
    animation: scan-sweep 2.2s ease-in-out infinite alternate;
}
.scan-frame .scan-trail {
    position: absolute; left: 14px; right: 14px; height: 70px; top: 14px;
    background: linear-gradient(to top, rgba(0, 255, 156, 0.22), transparent);
    transform: translateY(-100%);
    animation: scan-sweep 2.2s ease-in-out infinite alternate;
    pointer-events: none;
}
@keyframes scan-sweep {
    0%   { top: 14px; }
    100% { top: calc(100% - 17px); }
}
.scan-status {
    font-family: 'Share Tech Mono', monospace;
    color: var(--neon-green);
    letter-spacing: 3px;
    font-size: 0.85rem;
    margin-top: 12px;
    animation: hud-blink 1s steps(2, start) infinite;
}
@keyframes hud-blink { 50% { opacity: 0.25; } }
.frame-caption {
    font-family: 'Share Tech Mono', monospace;
    color: var(--neon);
    letter-spacing: 3px;
    font-size: 0.8rem;
    margin-top: 12px;
    opacity: 0.85;
}
/* --- results table ------------------------------------------------ */
table.hud-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Share Tech Mono', monospace;
    margin: 8px 0 18px 0;
}
table.hud-table th {
    color: var(--neon);
    letter-spacing: 2px;
    font-size: 0.75rem;
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--neon-dim);
}
table.hud-table td {
    color: #C8E6F5;
    padding: 8px 10px;
    border-bottom: 1px solid rgba(0, 240, 255, 0.12);
    font-size: 0.9rem;
}
table.hud-table tr:hover td { background: rgba(0, 240, 255, 0.06); }
td .medname { color: var(--neon-green); text-shadow: 0 0 8px rgba(0, 255, 156, 0.6); }
.hud-bar {
    width: 90px; height: 8px;
    border: 1px solid var(--neon-dim);
    background: rgba(0, 240, 255, 0.05);
    display: inline-block; vertical-align: middle; margin-right: 8px;
}
.hud-bar > div {
    height: 100%;
    background: linear-gradient(90deg, var(--neon), var(--neon-green));
    box-shadow: 0 0 8px rgba(0, 255, 156, 0.7);
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("💊 Prescription Reader")
st.caption("UPLOAD SPECIMEN → SCAN WORD REGIONS → DECODE MEDICINES")

model_ready = utils.model_is_available()
detector_backend = "doctr" if doctr_available() else "numpy"

# --- HUD status chips -------------------------------------------------
st.markdown(
    f"""
<div class="hud-chips">
  <span class="hud-chip {'' if model_ready else 'offline'}">
    NEURAL OCR: {"ONLINE" if model_ready else "OFFLINE"}</span>
  <span class="hud-chip">DETECTOR: {detector_backend.upper()}</span>
  <span class="hud-chip">SYS: OPERATIONAL</span>
</div>
""",
    unsafe_allow_html=True,
)

if not model_ready:
    st.warning(
        "Fine-tuned model not found in `models/trocr_prescription_model/`. "
        "You can still preview **word detection** below; recognition unlocks "
        "once the model + `torch` are added."
    )


# --- cached heavy resources -------------------------------------------
@st.cache_resource(show_spinner=False)
def get_detector(backend):
    """Load the word detector once per server, not on every click."""
    return WordDetector(backend=backend)


@st.cache_resource(show_spinner=False)
def get_pipeline(backend):
    """Load the full pipeline (incl. the TrOCR model) once per server."""
    from src.pipeline import PrescriptionPipeline

    return PrescriptionPipeline(detector_backend=backend)


# --- helpers ----------------------------------------------------------
def _img_to_b64(image, max_width=900):
    """Base64-encode a (possibly downscaled) copy of the image for HTML embeds."""
    img = image.copy()
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def hud_frame(image, caption="", scanning=False, status="SCANNING SPECIMEN"):
    """Render an image inside the neon HUD frame; scanning=True adds the sweep."""
    b64 = _img_to_b64(image)
    scan_bits = (
        f'<div class="scan-trail"></div><div class="scan-line"></div>'
        f'<div class="scan-status">▌ {html.escape(status)} ...</div>'
        if scanning
        else (f'<div class="frame-caption">◈ {html.escape(caption)}</div>' if caption else "")
    )
    return f"""
<div class="scan-frame">
  <img src="data:image/png;base64,{b64}"/>
  <div class="crt"></div>
  <div class="corner tl"></div><div class="corner tr"></div>
  <div class="corner bl"></div><div class="corner br"></div>
  {scan_bits}
</div>
"""


def run_with_scanner(placeholder, image, status, work):
    """Show the scan animation in `placeholder` while `work()` runs."""
    placeholder.markdown(
        hud_frame(image, scanning=True, status=status), unsafe_allow_html=True
    )
    start = time.time()
    result = work()
    # Let the sweep play for a moment even when inference is instant.
    leftover = MIN_SCAN_SECONDS - (time.time() - start)
    if leftover > 0:
        time.sleep(leftover)
    placeholder.empty()
    return result


def draw_boxes(image, regions):
    """Return a copy of the page with detected boxes drawn on it."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for i, region in enumerate(regions):
        x0, y0, x1, y1 = region["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=NEON_GREEN, width=2)
        draw.text((x0, max(0, y0 - 10)), str(i + 1), fill=NEON_GREEN)
    return canvas


def medicines_table(medicines):
    """Neon HUD table for the recognition results."""
    rows = ""
    for i, m in enumerate(medicines):
        ocr = m["ocr_confidence"] * 100
        match = m["match_score"] * 100
        rows += f"""
<tr>
  <td>{i + 1:02d}</td>
  <td>{html.escape(str(m["raw_text"]))}</td>
  <td><span class="medname">{html.escape(str(m["matched_brand"]))}</span></td>
  <td>{html.escape(str(m["generic"]))}</td>
  <td><span class="hud-bar"><div style="width:{ocr:.0f}%"></div></span>{ocr:.0f}%</td>
  <td><span class="hud-bar"><div style="width:{match:.0f}%"></div></span>{match:.0f}%</td>
</tr>"""
    return f"""
<table class="hud-table">
  <thead><tr>
    <th>#</th><th>READ AS</th><th>MEDICINE</th><th>GENERIC</th>
    <th>OCR CONF</th><th>MATCH</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


# --- input ------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload a prescription image", type=["png", "jpg", "jpeg"]
)

samples = utils.list_sample_images()
sample_choice = None
if samples and uploaded is None:
    sample_choice = st.selectbox(
        "...or load a bundled sample",
        [None] + samples,
        format_func=lambda p: "— select a sample —" if p is None else p.name,
    )

source = uploaded if uploaded is not None else sample_choice

if source is not None:
    image = to_pil(Image.open(source))
    st.markdown(
        hud_frame(image, caption="SPECIMEN LOADED — AWAITING SCAN"),
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    detect_clicked = col1.button("① Scan word regions")
    read_clicked = col2.button("② Decode medicines", disabled=not model_ready)

    scanner_slot = st.empty()

    # Stage 1: detection preview (works without the model) ------------
    if detect_clicked:
        def _detect():
            return get_detector(detector_backend).detect(image)

        regions = run_with_scanner(
            scanner_slot, image, "SCANNING SPECIMEN · LOCATING WORD REGIONS", _detect
        )
        st.subheader(f"{len(regions)} targets acquired")
        st.markdown(
            hud_frame(draw_boxes(image, regions), caption="WORD REGIONS LOCKED"),
            unsafe_allow_html=True,
        )
        if regions:
            st.caption("PREVIEW OF INDIVIDUAL CROPS (FIRST 12):")
            cols = st.columns(6)
            for i, region in enumerate(regions[:12]):
                cols[i % 6].image(region["crop"], caption=f"{i + 1:02d}")

    # Stage 2: full recognition (needs the model) ---------------------
    if read_clicked:
        def _read():
            return get_pipeline(detector_backend).read_page(image)

        medicines = run_with_scanner(
            scanner_slot, image, "NEURAL OCR ENGAGED · DECODING HANDWRITING", _read
        )
        st.subheader(f"Decode complete — {len(medicines)} medicine(s)")
        if medicines:
            st.markdown(
                hud_frame(draw_boxes(image, medicines), caption="MEDICINE REGIONS MATCHED"),
                unsafe_allow_html=True,
            )
            st.markdown(medicines_table(medicines), unsafe_allow_html=True)
        else:
            st.info(
                "No known medicines matched. Try the detection preview to check "
                "segmentation, or extend the medicine list in `medicine_lookup.py`."
            )
