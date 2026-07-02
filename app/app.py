"""Streamlit UI for the Doctor's Prescription Reader."""

import sys
from pathlib import Path

# Ensure the project root is importable when run via `streamlit run app/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from PIL import Image, ImageDraw

from app import utils
from app.image_processing import to_pil
from src.detection import WordDetector, doctr_available

st.set_page_config(page_title="Prescription Reader", page_icon="💊")

st.title("💊 Doctor's Prescription Reader")
st.caption("Upload a prescription → detect word regions → read medicine names")

model_ready = utils.model_is_available()
detector_backend = "doctr" if doctr_available() else "numpy"

# --- status banners --------------------------------------------------
if not model_ready:
    st.warning(
        "Fine-tuned model not found in `models/trocr_prescription_model/`. "
        "You can still preview **word detection** below; recognition unlocks "
        "once the model + `torch` are added."
    )
st.caption(f"Detection backend: **{detector_backend}**"
           + ("" if detector_backend == "doctr"
              else "  ·  install `python-doctr` for higher-quality detection"))


def draw_boxes(image, regions):
    """Return a copy of the page with detected boxes drawn on it."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for i, region in enumerate(regions):
        x0, y0, x1, y1 = region["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, max(0, y0 - 10)), str(i + 1), fill="red")
    return canvas


uploaded = st.file_uploader(
    "Upload a prescription image", type=["png", "jpg", "jpeg"]
)

if uploaded is not None:
    image = to_pil(Image.open(uploaded))
    st.image(image, caption="Uploaded prescription", use_container_width=True)

    col1, col2 = st.columns(2)
    detect_clicked = col1.button("① Detect word regions")
    read_clicked = col2.button("② Read medicines", disabled=not model_ready)

    # Stage 1: detection preview (works without the model) ------------
    if detect_clicked:
        with st.spinner("Detecting word regions..."):
            detector = WordDetector(backend=detector_backend)
            regions = detector.detect(image)
        st.subheader(f"Detected {len(regions)} word regions")
        st.image(draw_boxes(image, regions), use_container_width=True)
        if regions:
            st.caption("Preview of individual crops (first 12):")
            cols = st.columns(6)
            for i, region in enumerate(regions[:12]):
                cols[i % 6].image(region["crop"], caption=str(i + 1))

    # Stage 2: full recognition (needs the model) ---------------------
    if read_clicked:
        with st.spinner("Detecting + reading medicines..."):
            from src.pipeline import PrescriptionPipeline

            pipeline = PrescriptionPipeline(detector_backend=detector_backend)
            medicines = pipeline.read_page(image)

        st.subheader(f"Found {len(medicines)} medicine(s)")
        if medicines:
            st.image(
                draw_boxes(image, medicines),
                caption="Matched medicine regions",
                use_container_width=True,
            )
            st.dataframe(
                [
                    {
                        "#": i + 1,
                        "Read as": m["raw_text"],
                        "Medicine": m["matched_brand"],
                        "Generic": m["generic"],
                        "OCR conf.": f"{m['ocr_confidence'] * 100:.0f}%",
                        "Match": f"{m['match_score'] * 100:.0f}%",
                    }
                    for i, m in enumerate(medicines)
                ],
                use_container_width=True,
            )
        else:
            st.info(
                "No known medicines matched. Try the detection preview to check "
                "segmentation, or extend the medicine list in `medicine_lookup.py`."
            )
