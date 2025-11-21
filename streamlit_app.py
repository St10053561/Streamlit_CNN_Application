"""Simple Streamlit interface for the baseline CNN brain tumor classifier."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
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
    batch, resized = prepare_image(image, target_size)

    preview, charts = st.columns([1, 1])
    with preview:
        st.image(resized, caption="Uploaded MRI slice", use_column_width=True)
        st.caption(f"Resized to {target_size[0]}×{target_size[1]} (HxW)")

    with st.spinner("Analyzing image..."):
        predictions = model.predict(batch, verbose=0)[0]

    probs = tf.nn.softmax(predictions).numpy().tolist()
    labels = ensure_class_cardinality(class_labels, len(probs))
    ranked = sorted(zip(labels, probs), key=lambda item: item[1], reverse=True)

    top_label, top_prob = ranked[0]
    st.success(f"Prediction: **{top_label}** ({top_prob:.1%} confidence)")

    with charts:
        df = pd.DataFrame(ranked, columns=["Class", "Probability"])
        st.bar_chart(df, x="Class", y="Probability")

    with st.expander("Show raw probabilities"):
        st.json({label: float(prob) for label, prob in ranked})


def main() -> None:
    st.title("🧠 Baseline Brain Tumor Detector")
    st.caption("Upload an MRI slice and let the baseline CNN estimate the tumor type.")

    st.sidebar.header("Model settings")
    class_text = st.sidebar.text_input(
        "Class order (CSV)",
        value=",".join(DEFAULT_CLASS_ORDER),
        help="Adjust if your baseline model predicts classes in a different order.",
    )
    class_labels = parse_class_labels(class_text)

    st.sidebar.header("Need an image?")
    st.sidebar.markdown(
        "Use any T1-weighted slice from the Brain Tumor MRI dataset or your own sample, "
        "then drag-and-drop it below."
    )

    uploaded = st.file_uploader(
        "Drop a JPG or PNG MRI image",
        type=list(SUPPORTED_IMAGE_TYPES),
        help="Single axial slice works best.",
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

