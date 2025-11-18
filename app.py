import os
import shutil
import uuid
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join("backend", "models")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="MODMA MDD Classifier API")

# Serve static frontend
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# HF model URLs
models_to_download = {
    "global_tangent_space.pkl": "https://huggingface.co/datasets/hysam50epc/raglandsam-EEG_models/resolve/main/global_tangent_space.pkl",
    "csp_pipeline.pkl": "https://huggingface.co/datasets/hysam50epc/raglandsam-EEG_models/resolve/main/csp_pipeline.pkl",
    "scaler.pkl": "https://huggingface.co/datasets/hysam50epc/raglandsam-EEG_models/resolve/main/scaler.pkl",
    "svm_model.pkl": "https://huggingface.co/datasets/hysam50epc/raglandsam-EEG_models/resolve/main/svm_model.pkl",
}

print(f"=== DOWNLOADING MODELS INTO {MODEL_DIR} ===")
for fname, url in models_to_download.items():
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        r = requests.get(url)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded {fname}")
    else:
        print(f"{fname} already exists")

from backend.preprocessing import preprocess_eeg_file
from backend.inference_svm import predict_npz

@app.post("/full-pipeline")
async def pipeline(file: UploadFile = File(...), run_infer: bool = True):

    if not (file.filename.endswith(".raw") or file.filename.endswith(".npz")):
        raise HTTPException(400, "Upload .raw or .npz only")

    uid = uuid.uuid4().hex[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{uid}__{file.filename}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    npz_path, info = preprocess_eeg_file(save_path)

    if run_infer:
        result = predict_npz(npz_path)
        result["preprocess_info"] = info
        return result

    return {"npz_path": npz_path, "info": info}

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000)
