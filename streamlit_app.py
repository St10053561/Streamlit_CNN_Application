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

TUMOR_GUIDANCE = {
    "glioma": {
        "title": "Glioma / Astrocytoma",
        "definition": [
            "Gliomas are primary brain tumors that begin in the glial support cells and range from slow-growing to "
            "high-grade aggressive lesions."
        ],
        "risk_factors": [
            "Inherited tumor-suppressor syndromes such as Li-Fraumeni or NF1 raise glial mutation risk.",
            "Previous cranial ionizing radiation is strongly associated with high-grade gliomas.",
            "Incidence rises with age and is slightly higher in males and people of European descent.",
        ],
        "care_focus": [
            "Discuss maximal-safe surgical resection followed by chemoradiation (temozolomide + focal RT).",
            "Ask about MGMT, IDH, and 1p/19q markers to understand expected response and prognosis.",
            "Plan for seizure prophylaxis, corticosteroid tapering, and neuro-rehab as needed.",
        ],
        "sources": [
            (
                "Mayo Clinic – Glioma overview",
                "https://www.mayoclinic.org/diseases-conditions/glioma/diagnosis-treatment/drc-20350255",
            ),
            (
                "Clever Land Clinic – Adult glioma treatment (PDQ)",
                "https://my.clevelandclinic.org/health/diseases/21969-glioma",
            ),
        ],
    },
    "meningioma": {
        "title": "Meningioma",
        "risk_factors": [
            "History of head/neck radiation or dural scarring increases meningeal overgrowth risk.",
            "Neurofibromatosis type 2 (NF2) and TRAF7/KLF4 mutations are common genetic drivers.",
            "Incidence is higher in women, suggesting estrogen/progesterone influence.",
        ],
        "care_focus": [
            "Monitor small asymptomatic lesions with annual MRI before committing to intervention.",
            "Consider microsurgical resection or stereotactic radiosurgery when mass effect appears.",
            "Watch for visual field loss, cranial nerve palsies, or venous sinus involvement.",
        ],
        "sources": [
            (
                "American Brain Tumor Association – Meningioma facts",
                "https://www.abta.org/tumor_types/meningioma/",
            ),
            (
                "Mayo Clinic – Meningioma diagnosis & treatment",
                "https://www.mayoclinic.org/diseases-conditions/meningioma/diagnosis-treatment/drc-20355647",
            ),
        ],
    },
    "pituitary": {
        "title": "Pituitary adenoma",
        "risk_factors": [
            "Family history of MEN1 or familial isolated pituitary adenoma syndromes.",
            "Chronic estrogen/testosterone therapy and endocrine disruptors can stimulate adenoma growth.",
            "Prior cranial irradiation or traumatic brain injury affecting the sella.",
        ],
        "care_focus": [
            "Order full endocrine panels to detect hormone excess or deficiency from the lesion.",
            "Discuss transsphenoidal surgery vs. medical therapy (dopamine agonists, somatostatin analogs).",
            "Track visual fields and pituitary hormone levels during follow-up.",
        ],
        "sources": [
            (
                "Mayo Clinic – About Pituitary",
                "https://www.mayoclinic.org/diseases-conditions/pituitary-tumors/diagnosis-treatment/drc-20350553",
            ),
            (
                "Endocrine Society – Pituitary adenoma clinical practice guidance",
                "https://www.endocrine.org/patient-engagement/endocrine-library/pituitary-tumors",
            ),
        ],
    },
    "notumor": {
        "title": "No tumor detected",
        "risk_factors": [
            "AI prediction shows no tumor, yet unresolved headaches, seizures, or deficits still need workup.",
            "Migraines, infections, vascular events, or autoimmune disease can mimic tumor symptoms.",
            "Family history of tumors or prior radiation may justify periodic MRI surveillance despite a clear scan.",
        ],
        "care_focus": [
            "Share this scan with your neurologist/radiologist to confirm the absence of lesions.",
            "Continue lifestyle steps: blood-pressure control, sleep hygiene, seizure precautions if prescribed.",
            "Seek urgent care if symptoms worsen or new neurologic signs emerge.",
        ],
        "sources": [
            (
                "Mayo Clinic – Brain tumour symptoms and red flags",
                "https://www.mayoclinic.org/diseases-conditions/brain-tumor/diagnosis-treatment/drc-20350088",
            ),
        ],
    },
    "default": {
        "title": "Brain tumor overview",
        "risk_factors": [
            "Previous cranial radiation, immune suppression, and certain hereditary syndromes.",
            "Environmental exposures such as petrochemicals or heavy metals remain under study.",
            "Age, sex, and ethnicity patterns vary by tumor subtype.",
        ],
        "care_focus": [
            "Discuss multidisciplinary care involving neurosurgery, neuro-oncology, and rehab.",
            "Clarify tumor grade, molecular markers, and available clinical trials.",
            "Address symptom relief: seizure control, steroids for edema, cognitive support.",
        ],
        "sources": [
            (
                "American Brain Tumor Association – Tumor types",
                "https://www.abta.org/about-brain-tumors/brain-tumor-101/types/",
            ),
            (
                "Johns Hopkins Medicine – Brain tumor treatment planning",
                "https://www.hopkinsmedicine.org/health/conditions-and-diseases/brain-tumor/brain-tumor-treatment",
            ),
        ],
    },
}

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


def render_guidance(tumor_label: str) -> None:
    info = TUMOR_GUIDANCE.get((tumor_label or "").lower(), TUMOR_GUIDANCE["default"])

    st.markdown(f"#### Guidance for {info['title']}")

    risk_factors = info.get("risk_factors", [])
    if risk_factors:
        st.markdown("**Risk factors & clinical cues**")
        st.markdown("\n".join(f"- {item}" for item in risk_factors))

    care_focus = info.get("care_focus", [])
    if care_focus:
        st.markdown("**Care considerations**")
        st.markdown("\n".join(f"- {item}" for item in care_focus))

    sources = info.get("sources", [])
    if sources:
        st.markdown(f"**Sources specific to {info['title']}**")
        st.markdown("\n".join(f"- [{label}]({url})" for label, url in sources))


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

    st.bar_chart({label: prob for label, prob in ranked})

    render_guidance(top_label)


def main() -> None:
    st.title("🧠 Baseline Brain Tumor Detector")
    st.caption("Demonstration-only tool: upload an MRI slice and see how the baseline CNN responds.")

    st.markdown(
        "**How it works:**\n"
        "1. Locate a trained model in `models/baseline_cnn_final.keras`.\n"
        "2. Upload a single axial MRI slice (JPG or PNG).\n"
        "3. Wait a moment for the baseline CNN to estimate the tumor class."
    )

    st.info(
        "Every prediction surfaces tumor-specific risk factors, care considerations, and credible sources to review with your clinician."
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

