# backend/app.py

import os
import shutil
import uuid
import gdown

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# ============================================================
#  MODEL DOWNLOAD BOOTSTRAP (RUNS BEFORE ANY INFERENCE IMPORTS)
# ============================================================

BASE_DIR = os.path.dirname(__file__)

# Clean MODEL_DIR environment variable if present
env_model_dir = os.environ.get("MODEL_DIR", "").strip()
MODEL_DIR = env_model_dir if env_model_dir else os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

models_to_download = {
    "global_tangent_space.pkl": "https://drive.google.com/uc?id=1_xt2cdcn-mB4hHtRJjytnNwJAIEajrro",
    "scaler.pkl": "https://drive.google.com/uc?id=15Sd4YZTFicoXH8sM-UJtpOQHqW6ZTUlG",
    "svm_model.pkl": "https://drive.google.com/uc?id=1WnKvt4DkfhAtLMhvmRShYWzFFDDO1NsV",
    "csp_pipeline.pkl": "https://drive.google.com/uc?id=1UmZfd7BWpA-WcLKiZ_OM9JZ1-0hatejF",
}

print(f"\n=== DOWNLOADING MODELS INTO {MODEL_DIR} ===")
for key, url in models_to_download.items():
    clean_name = key.strip()        # strip newline if any
    file_path = os.path.join(MODEL_DIR, clean_name)
    if not os.path.exists(file_path):
        print(f"🔽 Downloading {clean_name} → {file_path}")
        gdown.download(url, file_path, quiet=False)
    else:
        print(f"✔ {clean_name} already exists.")

print("\n=== FINAL MODEL_DIR CONTENTS ===")
for f in sorted(os.listdir(MODEL_DIR)):
    print(" -", repr(f))
print("=================================\n")

# ============================================================
#  AFTER MODELS EXIST, IMPORT INFERENCE MODULE
# ============================================================

from preprocessing import preprocess_eeg_file
from inference_svm import predict_npz


# ============================================================
#  FASTAPI APP
# ============================================================

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "public")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="MODMA MDD Classifier API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  ROUTES
# ============================================================

@app.post("/full-pipeline")
async def full_pipeline(file: UploadFile = File(...), run_infer: bool = True):

    # FILE TYPE CHECK
    if not (file.filename.lower().endswith(".raw")
            or file.filename.lower().endswith(".npz")):
        raise HTTPException(status_code=400,
            detail=f"Unsupported file type '{file.filename}'. Only .raw or .npz allowed."
        )

    # SAVE FILE
    uid = uuid.uuid4().hex[:8]
    filename = f"{uid}__{file.filename}"
    saved_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(500, f"Failed to save uploaded file: {e}")

    # PREPROCESS
    try:
        preproc_out = preprocess_eeg_file(saved_path, output_dir=UPLOAD_DIR)
        if isinstance(preproc_out, tuple):
            npz_path, info = preproc_out
        else:
            npz_path = preproc_out
            info = {}
    except Exception as e:
        raise HTTPException(500, f"Preprocessing error: {e}")

    # INFERENCE
    if run_infer:
        try:
            result = predict_npz(npz_path)
            result["preprocess_info"] = info
        except Exception as e:
            raise HTTPException(500, f"Inference failed: {e}")
    else:
        result = {"message": "Preprocessing complete", "npz_path": npz_path, "info": info}

    return result


# ============================================================
#  STATIC FRONTEND
# ============================================================

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


# ============================================================
#  LOCAL DEV ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
