"""Simple Streamlit interface for the baseline CNN brain tumor classifier."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "baseline_cnn_final.keras"
SUPPORTED_IMAGE_TYPES = ("png", "jpg", "jpeg")
DEFAULT_CLASS_ORDER: List[str] = ["glioma", "notumor", "meningioma", "pituitary"]

st.set_page_config(page_title="Brain Tumor MRI Detector", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner=False)
def load_baseline_model() -> tf.keras.Model:
    """Load the single baseline CNN checkpoint."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Couldn't find 'baseline_cnn_final.keras' inside the models folder."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def parse_class_labels(raw: str) -> List[str]:
    labels = [label.strip() for label in raw.split(",") if label.strip()]
    return labels or DEFAULT_CLASS_ORDER.copy()


def infer_spatial_size(model: tf.keras.Model) -> Tuple[int, int]:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if not input_shape:
        return (256, 256)

    height, width = None, None
    if len(input_shape) >= 3:
        dims = [dim for dim in input_shape if isinstance(dim, int) or dim is None]
        if len(dims) >= 2:
            height = dims[-3] if len(dims) >= 3 else dims[0]
            width = dims[-2] if len(dims) >= 2 else dims[1]

    if not height or not width:
        return (256, 256)

    return (int(height), int(width))


def prepare_image(image: Image.Image, target_size: Tuple[int, int]) -> Tuple[np.ndarray, Image.Image]:
    image = image.convert("RGB")
    resized = image.resize(target_size[::-1], Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    batch = np.expand_dims(array, axis=0)
    return batch, resized


def ensure_class_cardinality(labels: Sequence[str], target: int) -> List[str]:
    labels = list(labels)
    if len(labels) >= target:
        return labels[:target]

    missing = [f"class_{idx}" for idx in range(len(labels), target)]
    return labels + missing


def render_predictions(model: tf.keras.Model, image: Image.Image, class_labels: Sequence[str]) -> None:
    target_size = infer_spatial_size(model)
    batch, _ = prepare_image(image, target_size)

    with st.spinner("Analyzing image..."):
        predictions = model.predict(batch, verbose=0)[0]

    probs = tf.nn.softmax(predictions).numpy().tolist()
    labels = ensure_class_cardinality(class_labels, len(probs))
    ranked = sorted(zip(labels, probs), key=lambda item: item[1], reverse=True)

    top_label, top_prob = ranked[0]
    st.success(f"Prediction: **{top_label}** ({top_prob:.1%} confidence)")

    st.markdown(
        "- Common risk factors include prior radiation exposure and certain genetic syndromes.\n"
        "- Typical treatments mix surgery, radiation, and chemotherapy under neuro-oncology care.\n"
        "- Lifestyle focus: seizure precautions, corticosteroid management, regular neuro follow-ups."
    )

    st.bar_chart({label: prob for label, prob in ranked})

    st.subheader("Learn more")
    st.markdown(
        "- [Brain Tumor Types & Grades (American Brain Tumor Association)](https://www.abta.org/about-brain-tumors/brain-tumor-101/types/).\n"
        "- [Treatment planning and options (Johns Hopkins Medicine)](https://www.hopkinsmedicine.org/health/conditions-and-diseases/brain-tumor/brain-tumor-treatment).\n"
        "- [Living with brain tumors: precautions & support (Cancer Research UK)](https://www.cancerresearchuk.org/about-cancer/brain-tumours/living-with).\n"
        "- [Comprehensive glioma care guide (National Brain Tumor Society)](https://braintumor.org/brain-tumors/types-of-brain-tumors/glioblastoma/)."
    )


def main() -> None:
    st.title("🧠 Baseline Brain Tumor Detector")
    st.caption("Demonstration-only tool: upload an MRI slice and see how the baseline CNN responds.")

    st.markdown(
        "**How it works:**\n"
        "1. Locate a trained model in `models/baseline_cnn_final.keras`.\n"
        "2. Upload a single axial MRI slice (JPG or PNG).\n"
        "3. Wait a moment for the baseline CNN to estimate the tumor class."
    )

    class_labels = DEFAULT_CLASS_ORDER

    uploaded = st.file_uploader(
        "Upload a JPG or PNG MRI image",
        type=list(SUPPORTED_IMAGE_TYPES),
        help="High-contrast brain slices produce the best results.",
    )

    try:
        model = load_baseline_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if uploaded is None:
        st.info("Upload an MRI slice to get a prediction.")
        return

    image = Image.open(uploaded)
    render_predictions(model=model, image=image, class_labels=class_labels)

    st.markdown("---")
    st.caption(
        "Reminder: This tool is for experimentation only. Always confirm findings with a medical professional."
    )


if __name__ == "__main__":
    main()

