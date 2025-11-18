# backend/app.py
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import gdown
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

models_to_download = {
    "global_tangent_space.pkl": "https://drive.google.com/uc?id=1_xt2cdcn-mB4hHtRJjytnNwJAIEajrro",
    "scaler.pkl": "https://drive.google.com/uc?id=15Sd4YZTFicoXH8sM-UJtpOQHqW6ZTUlG",
    "svm_model.pkl": "https://drive.google.com/uc?id=1WnKvt4DkfhAtLMhvmRShYWzFFDDO1NsV",
    "csp_pipeline.pkl": "https://drive.google.com/uc?id=1UmZfd7BWpA-WcLKiZ_OM9JZ1-0hatejF",
}

# Download missing model files at startup
for filename, url in models_to_download.items():
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"🔽 Downloading {filename} ...")
        gdown.download(url, path, quiet=False)
    else:
        print(f"✔ {filename} already exists.")

from .preprocessing import preprocess_eeg_file
from .inference_svm import predict_npz   # your SVM inference script

#MODIFIED
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(__file__)
# Allow overriding upload directory via env var for cloud deployment
UPLOAD_DIR = os.environ.get('UPLOAD_DIR', os.path.join(BASE_DIR, "uploads"))
FRONTEND_DIR = os.path.join(BASE_DIR, "public")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="MODMA MDD Classifier API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Upload + Preprocess + Infer endpoint
@app.post("/full-pipeline")
async def full_pipeline(file: UploadFile = File(...), run_infer: bool = True):

    # Ensure correct file types (support .raw and preprocessed .npz only)
    if not (file.filename.lower().endswith(".raw")
            or file.filename.lower().endswith(".npz")):

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.filename}'. "
                   "Upload only .raw or preprocessed .npz files."
        )

    # Save upload
    uid = uuid.uuid4().hex[:8]
    filename = f"{uid}__{file.filename}"
    saved_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {e}"
        )

    # Preprocess
    try:
        preproc_out = preprocess_eeg_file(saved_path, output_dir=UPLOAD_DIR)
        # support both returns: (npz_path, info) or single npz_path
        if isinstance(preproc_out, tuple) and len(preproc_out) == 2:
            npz_path, info = preproc_out
        else:
            npz_path = preproc_out
            info = {}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing error: {str(e)}"
        )

    # Inference if requested
    if run_infer:
        try:
            result = predict_npz(npz_path)
            if isinstance(result, dict):
                result["preprocess_info"] = info
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(e)}"
            )
    else:
        result = {"message": "Preprocessing completed", "npz_path": npz_path, "preprocess_info": info}

    return result


# Serve frontend static files (must be mounted last so catch-all doesn't interfere with API routes)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


# -------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
