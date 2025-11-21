"""Interactive Streamlit front-end for Keras brain tumor classifiers."""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
PARQUET_SOURCES: Dict[str, Path] = {
    "Training": BASE_DIR / "training_records.parquet",
    "Validation": BASE_DIR / "validation_records.parquet",
    "Testing": BASE_DIR / "testing_records.parquet",
}
DEFAULT_CLASS_ORDER: List[str] = ["glioma", "notumor", "meningioma", "pituitary"]
SUPPORTED_MODEL_SUFFIXES = (".keras", ".h5", ".hdf5")
SUPPORTED_IMAGE_TYPES = ("png", "jpg", "jpeg")


@dataclass
class SampleSelection:
    """Represents an image pulled from one of the parquet sources."""

    image: Image.Image
    label: Optional[str]
    partition: str
    row_index: int


st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
)


def discover_models() -> List[Path]:
    if not MODEL_DIR.exists():
        return []

    candidates = []
    for suffix in SUPPORTED_MODEL_SUFFIXES:
        candidates.extend(MODEL_DIR.rglob(f"*{suffix}"))

    return sorted({p.resolve() for p in candidates if p.is_file() and not p.name.startswith('.')})


@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def load_partition(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def parse_class_labels(raw: str) -> List[str]:
    labels = [label.strip() for label in raw.split(',') if label.strip()]
    return labels or DEFAULT_CLASS_ORDER.copy()


def infer_spatial_size(model: tf.keras.Model) -> Tuple[int, int]:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if not input_shape:
        return (256, 256)

    height, width = None, None
    if len(input_shape) >= 3:
        # Expecting (batch, height, width, channels) or (height, width, channels)
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


def decode_base64_image(encoded: str) -> Image.Image:
    data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return image


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    for column in df.columns:
        lower = column.lower()
        if any(key in lower for key in ("label", "class", "category")):
            return column
    return None


def sample_from_partition(partition: str, label_filter: Optional[str], row_index: int) -> SampleSelection:
    path = PARQUET_SOURCES[partition]
    df = load_partition(str(path))
    label_column = detect_label_column(df)

    filtered = df if not label_column or not label_filter else df[df[label_column] == label_filter]
    if filtered.empty:
        raise ValueError("No samples available for the selected filter.")

    row_index = max(0, min(row_index, len(filtered) - 1))
    record = filtered.iloc[row_index]

    if "processed_image" not in record:
        raise ValueError("The parquet record does not contain a 'processed_image' column.")

    try:
        image = decode_base64_image(record["processed_image"])
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError("Failed to decode the stored base64 image.") from exc

    label_value = record[label_column] if label_column else None
    return SampleSelection(image=image, label=label_value, partition=partition, row_index=row_index)


def render_predictions(
    model: tf.keras.Model,
    image: Image.Image,
    class_labels: Sequence[str],
    sample: Optional[SampleSelection],
) -> None:
    target_size = infer_spatial_size(model)
    batch, resized = prepare_image(image, target_size)

    col_preview, col_table = st.columns([1, 1.5])
    with col_preview:
        caption = "Uploaded image"
        if sample:
            caption = f"{sample.partition} sample"
            if sample.label:
                caption += f" ({sample.label})"
        st.image(resized, caption=caption, use_column_width=True)

        st.caption(f"Model input size inferred as {target_size[0]}x{target_size[1]} (HxW)")

    with st.spinner("Running inference..."):
        predictions = model.predict(batch, verbose=0)[0]

    probs = tf.nn.softmax(predictions).numpy().tolist()
    labels = ensure_class_cardinality(class_labels, len(probs))
    entries = sorted(zip(labels, probs), key=lambda item: item[1], reverse=True)

    df = pd.DataFrame(entries, columns=["Class", "Probability"])
    with col_table:
        st.subheader("Predicted probabilities")
        st.bar_chart(df.set_index("Class"))
        st.dataframe(df.style.format({"Probability": "{:.4f}"}), use_container_width=True)

    top_label, top_prob = entries[0]
    st.success(f"Predicted: {top_label} (confidence {top_prob:.2%})")

    st.subheader("Raw prediction vector")
    mapped = {label: float(prob) for label, prob in zip(labels, probs)}
    st.json(mapped)

    with st.expander("Model metadata", expanded=False):
        st.write(
            {
                "Input shape": model.input_shape,
                "Output shape": model.output_shape,
                "Trainable params": f"{model.count_params():,}",
                "Classes": labels,
            }
        )


def sidebar_controls(models: List[Path]) -> Tuple[Optional[Path], List[str], Optional[Image.Image], Optional[SampleSelection]]:
    st.sidebar.header("1️⃣ Model & labels")
    selected_model = st.sidebar.selectbox(
        "Choose a saved Keras model",
        options=[None] + models,
        format_func=lambda path: "— Select model —" if path is None else path.name,
    )
    if not models:
        st.sidebar.warning("Drop trained .keras/.h5 files inside the 'models' folder to begin.")

    class_text = st.sidebar.text_input(
        "Class order (CSV)",
        value=",".join(DEFAULT_CLASS_ORDER),
        help="Override the default class order if your model outputs a different ordering.",
    )
    class_labels = parse_class_labels(class_text)

    st.sidebar.header("2️⃣ Image source")
    source = st.sidebar.radio(
        "Provide an image via",
        options=("Upload", "Dataset sample"),
    )

    chosen_image: Optional[Image.Image] = None
    sample_meta: Optional[SampleSelection] = None

    if source == "Upload":
        uploaded = st.sidebar.file_uploader(
            "Upload a JPG or PNG MRI slice",
            type=list(SUPPORTED_IMAGE_TYPES),
            accept_multiple_files=False,
        )
        if uploaded is not None:
            chosen_image = Image.open(uploaded).convert("RGB")
            st.sidebar.success("Image uploaded.")
    else:
        available_partitions = [name for name, path in PARQUET_SOURCES.items() if path.exists()]
        if not available_partitions:
            st.sidebar.info("No parquet datasets detected. Keep the .parquet files in the repo root to enable samples.")
        else:
            partition = st.sidebar.selectbox("Dataset partition", options=available_partitions)
            df = load_partition(str(PARQUET_SOURCES[partition]))
            label_column = detect_label_column(df)

            label_options: Iterable[str]
            if label_column:
                unique_labels = sorted(df[label_column].dropna().unique().tolist())
                label_options = ["All labels"] + unique_labels
            else:
                label_options = ["All labels"]
            label_choice = st.sidebar.selectbox("Filter by label", options=label_options)
            label_filter = None if label_choice == "All labels" else label_choice

            filtered_df = df if not label_column or not label_filter else df[df[label_column] == label_filter]
            total_rows = len(filtered_df)
            if total_rows == 0:
                st.sidebar.warning("No rows match the selected label filter.")
            else:
                key_prefix = f"sample_idx_{partition}_{label_choice}"
                if key_prefix not in st.session_state:
                    st.session_state[key_prefix] = 0

                max_index = total_rows - 1
                sample_index = st.sidebar.number_input(
                    "Row index",
                    min_value=0,
                    max_value=max_index,
                    value=int(st.session_state[key_prefix]),
                    step=1,
                    key=key_prefix,
                )
                st.session_state[key_prefix] = int(sample_index)

                if st.sidebar.button("Random sample", key=f"random_{key_prefix}"):
                    st.session_state[key_prefix] = int(np.random.randint(0, max_index + 1))
                    st.experimental_rerun()

                try:
                    sample_meta = sample_from_partition(
                        partition=partition,
                        label_filter=label_filter,
                        row_index=int(sample_index),
                    )
                    chosen_image = sample_meta.image
                    st.sidebar.success(
                        f"Loaded {partition} row {int(sample_index)}" +
                        (f" ({sample_meta.label})" if sample_meta.label else "")
                    )
                except ValueError as exc:
                    st.sidebar.error(str(exc))

    return selected_model, class_labels, chosen_image, sample_meta


def main() -> None:
    st.title("🧠 Brain Tumor MRI Streamlit Workbench")
    st.caption(
        "Interact with your trained Keras models, upload MRI slices, or preview embedded samples "
        "from the training/validation/testing parquet files."
    )

    models = discover_models()
    selected_model, class_labels, image, sample_meta = sidebar_controls(models)

    if not selected_model:
        st.info("Select a model from the sidebar to begin.")
        return

    with st.spinner(f"Loading {selected_model.name}..."):
        model = load_keras_model(str(selected_model))

    if image is None:
        st.warning("Upload an image or pick a dataset sample to run predictions.")
        return

    render_predictions(model=model, image=image, class_labels=class_labels, sample=sample_meta)


if __name__ == "__main__":
    main()
