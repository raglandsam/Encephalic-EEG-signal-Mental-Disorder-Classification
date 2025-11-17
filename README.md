# Encephalic EEG Analyzer

AI-powered mental health classification system using EEG signals. Classifies Major Depressive Disorder (MDD) vs. Healthy Control (HC). The model is developed using the MODMA dataset.

Dataset: https://modma.lzu.edu.cn/data/application/

Quick start (single-process)

1. Create and activate a Python virtual environment, then install dependencies:

```powershell
cd C:\Modma_dataset\EEG_128channels_ERP_lanzhou_2015\EEG_app
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the server from the repository root so package imports resolve:

```powershell
python -m uvicorn Backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Open the UI at `http://localhost:8000/` and use the web form to upload `.raw` or `.npz` files for analysis.

Project structure (relevant folders)

- `Backend/` — FastAPI server, EEG preprocessing, SVM classification, and static frontend (`Backend/public`)
- `App/`, `Model_development/` — experimental / development artifacts (not required for deployment)

File formats

- `.raw` — EGI NetStation EEG (recommended for raw uploads)
- `.npz` — Pre-processed NumPy archive (skip preprocessing if using this)

Environment variables (optional)

Set these if you want to override defaults (paths are relative to `Backend/` by default):

```powershell
setx MODEL_DIR "C:\path\to\models"
setx UPLOAD_DIR "C:\path\to\uploads"
setx PREPROCESS_OUTPUT_DIR "C:\path\to\processed"
```

Important

This system is for research only. Results are not medical diagnoses. Consult a healthcare professional for clinical decisions.

Documentation

- Backend API: `Backend/README.md`
