# Streamlit CNN Application

A lightweight Streamlit front-end for experimenting with saved Keras/TensorFlow brain-tumour classifiers. Point the app at the `.keras` models stored in `models/`, upload your own MRI slice or browse embedded samples from the parquet datasets, then inspect the predicted class distribution, raw logits, and model metadata without writing extra code.

## Key capabilities

- 🔍 **Model autodiscovery** – every `.keras`, `.h5`, or `.hdf5` file placed in `models/` appears in the UI.
- 🖼️ **Flexible image inputs** – upload PNG/JPG files or pull examples directly from `training_records.parquet`, `validation_records.parquet`, or `testing_records.parquet` (base64-encoded images + labels).
- 🧮 **Automatic preprocessing** – images are converted to RGB, resized to the model's expected input shape, and scaled to `[0, 1]`.
- 📊 **Clear predictions** – probability bars, sortable tables, top-class callouts, and raw JSON outputs ready for downstream logging.
- ⚙️ **Custom class ordering** – override the default `glioma, notumor, meningioma, pituitary` order straight from the sidebar, per model.
- ☁️ **Deploy anywhere** – run locally, inside Docker, or on Streamlit Community Cloud with the provided manifest.

## Repository layout

```
.
├── models/                     # Drop your trained .keras/.h5 checkpoints here
├── streamlit_app.py            # Streamlit entry point
├── requirements.txt            # Shared dependency manifest
├── training_records.parquet    # Optional dataset sample store (base64 image, label)
├── validation_records.parquet  # Optional dataset sample store
├── testing_records.parquet     # Optional dataset sample store
└── README.md                   # You're here
```

## A → Z setup & usage

1. **Clone or download** this project and `cd` into `Streamlit_CNN_Application`.
2. **Create a Python environment** (choose your own tool; commands below use Conda and `venv`).

   ```powershell
   # Conda
   conda create -n streamlit-cnn python=3.11 -y
   conda activate streamlit-cnn

   # Or built-in venv
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies** from the manifest (TensorFlow, pandas, streamlit, etc.):

   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > 💡 If you hit install errors for `pyspark`/`pyarrow` on Windows, install them one at a time (`pip install pyarrow pyspark`) or remove them if you do not use Spark features.

4. **Prepare models** – copy your trained `.keras`/`.h5` checkpoints into the `models/` folder. The filename becomes the dropdown label.
5. **(Optional) Keep dataset metadata** – leave the provided parquet files (or add your own) in the repo root so the app can surface sample images via the sidebar.
6. **Run the Streamlit UI**:

   ```powershell
   streamlit run streamlit_app.py
   ```

7. **Use the sidebar** to:
   - pick a model,
   - override the class order if needed (`glioma,notumor,meningioma,pituitary` by default),
   - choose an uploaded file or browse dataset samples (filter by partition + label, jump to any row, or pull a random sample).
8. **Inspect the output** – the main area shows the resized preview, probability bars/table, top prediction with confidence, JSON payload, and model metadata (input/output shapes, parameter count).

## Quick test checklist

- Upload any MRI slice (`.jpg`, `.jpeg`, `.png`) to ensure the pipeline runs end-to-end.
- Use `testing_records.parquet` → `Random sample` to quickly sanity-check the bundled models.
- Verify that the predicted class matches expectations; if it does not, double-check the class order text field.

## Validation & health checks

- **Static import check**: `python -m compileall streamlit_app.py`
- **Optional linting**: `python -m pylint streamlit_app.py` (requires `pylint`)
- **Streamlit smoke test**: `streamlit run streamlit_app.py --server.headless true --server.port 8888`

## Docker (optional but convenient)

A minimal Dockerfile is provided for reproducible deployments. Build and run it like so:

```powershell
docker build -t streamlit-cnn .
docker run --rm -p 8501:8501 -v ${PWD}/models:/app/models streamlit-cnn
```

Override the bind mount with your own models directory as needed.

## Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), create a new app that points to `streamlit_app.py` on the `main` branch.
3. Ensure `requirements.txt` is present so Streamlit installs everything automatically.
4. Configure environment secrets (if any) via the Streamlit Cloud dashboard.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| "No models found" warning | `models/` folder empty or different extension | Drop `.keras`/`.h5` models into `models/` (subfolders supported). |
| Model load failure referencing custom layers | Model was trained with custom objects | Edit `load_keras_model` in `streamlit_app.py` to pass `custom_objects={...}` or export without custom layers. |
| Predictions look misaligned | Class order differs from default | Use the "Class order (CSV)" input to match your model's output order. |
| Dataset samples unavailable | Parquet files missing | Keep the `*_records.parquet` files in the repo root or update `PARQUET_SOURCES`. |

## Next ideas

- Auto-detect preprocessing pipelines per model (e.g., `tf.keras.applications.xception.preprocess_input`).
- Pull label metadata from training checkpoints (e.g., `labels.json`) when available.
- Batch scoring for entire folders plus CSV export.
- Grad-CAM / saliency visualisations for interpretability.

Happy experimenting! 🧪
