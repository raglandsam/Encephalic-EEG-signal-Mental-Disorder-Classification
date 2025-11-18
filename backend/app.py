# backend/app.py

import os
import shutil
import uuid
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ============================================================
#  MODEL DOWNLOAD BOOTSTRAP (HUGGINGFACE DIRECT LINKS)
# ============================================================

BASE_DIR = os.path.dirname(__file__)
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# HuggingFace dataset model URLs
models_to_download = {
    "global_tangent_space.pkl": "https://huggingface.co/datasets/hysam50epc/raglandsam-EEG_models/resolve/main/global_tangent_space.pkl",
    "csp_pipeline.pkl": "https://huggingface.co/datasets/hysam50epc/raglandsam-EEG_models/resolve/main/csp_pipeline.pkl",
    "scaler.pkl": "https://huggingface.co/datasets/hysam50epc/raglandsam-EEG_models/resolve/main/scaler.pkl",
    "svm_model.pkl": "https://huggingface.co/datasets/hysam50epc/raglandsam-EEG_models/resolve/main/svm_model.pkl",
}

print(f"\n=== DOWNLOADING MODELS INTO {MODEL_DIR} ===")
for file_name, url in models_to_download.items():
    file_path = os.path.join(MODEL_DIR, file_name)

    if not os.path.exists(file_path):
        print(f"🔽 Downloading {file_name} ...")

        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(r.content)

        except Exception as e:
            raise RuntimeError(f"Failed to download {file_name}: {e}")
    else:
        print(f"✔ {file_name} already exists.")

print("\n=== FINAL MODEL_DIR CONTENTS ===")
for f in sorted(os.listdir(MODEL_DIR)):
    print(" -", repr(f))
print("=================================\n")

# ============================================================
#  IMPORT INFERENCE LOGIC
# ============================================================

from preprocessing import preprocess_eeg_file
from inference_svm import predict_npz


# ============================================================
#  FASTAPI APP
# ============================================================

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="MODMA MDD Classifier API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  MAIN PIPELINE ROUTE
# ============================================================

@app.post("/full-pipeline")
async def full_pipeline(file: UploadFile = File(...), run_infer: bool = True):

    if not (file.filename.lower().endswith(".raw") or file.filename.lower().endswith(".npz")):
        raise HTTPException(400, "Only .raw or .npz files are supported.")

    # Save uploaded file
    uid = uuid.uuid4().hex[:8]
    filename = f"{uid}__{file.filename}"
    saved_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(500, f"Failed to save uploaded file: {e}")

    # Preprocessing
    try:
        out = preprocess_eeg_file(saved_path, output_dir=UPLOAD_DIR)
        if isinstance(out, tuple):
            npz_path, info = out
        else:
            npz_path = out
            info = {}
    except Exception as e:
        raise HTTPException(500, f"Preprocessing failed: {e}")

    # Inference
    if run_infer:
        try:
            result = predict_npz(npz_path)
            result["preprocess_info"] = info
        except Exception as e:
            raise HTTPException(500, f"Inference failed: {e}")
    else:
        result = {"message": "Preprocessing complete", "npz_path": npz_path}

    return result

# ============================================================
#  LOCAL DEV ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
