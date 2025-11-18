# backend/svminference.py
import os
import numpy as np
import joblib
from collections import Counter
from mne.filter import filter_data
from pyriemann.estimation import Covariances
from scipy.stats import skew, kurtosis
 # optional if you want xgb path too

# ----- CONFIG: point these to your model files in backend/models/
BASE_DIR = os.path.dirname(__file__)
# Allow overriding model dir via env var for cloud deployment
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(BASE_DIR, "models"))

CSP_PATH = os.path.join(MODEL_DIR, "csp_pipeline.pkl")
TS_PATH  = os.path.join(MODEL_DIR, "global_tangent_space.pkl")  # name as saved
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
SVM_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")  # or svm_pipeline.pkl

SFREQ = 250
BAND = (8, 30)
EPS = 1e-6

# Load artifacts once - with error handling
def _load_models():
    try:
        _csp = joblib.load(CSP_PATH)
        _ts = joblib.load(TS_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _svm = joblib.load(SVM_PATH)
        return _csp, _ts, _scaler, _svm
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {e}. Ensure all .pkl files exist in {MODEL_DIR}")

try:
    _csp, _ts, _scaler, _svm = _load_models()
except Exception as e:
    print(f"⚠️ WARNING: Model loading failed - {e}")
    _csp = _ts = _scaler = _svm = None

def _bandpass_trials(X):
    Xf = np.zeros_like(X)
    for i in range(X.shape[0]):
        Xf[i] = filter_data(X[i], sfreq=SFREQ, l_freq=BAND[0], h_freq=BAND[1], method='iir', verbose=False)
    return Xf

def _apply_csp(X_raw):
    Xf = _bandpass_trials(X_raw)
    Xcsp = _csp.transform(Xf)
    if Xcsp.ndim == 3:
        Xcsp = Xcsp.mean(axis=2)
    return Xcsp

def _apply_riemann(X_raw):
    cov_est = Covariances(estimator='oas')
    covs = cov_est.fit_transform(X_raw)  # shape (n_trials, ch, ch)
    covs += np.eye(covs.shape[1]) * EPS
    X_ts = _ts.transform(covs)
    return X_ts

def _compute_stats(X_csp):
    # X_csp: (trials, comps)
    mean = X_csp.mean(axis=0)      # (n_comps,)
    std  = X_csp.std(axis=0)
    sk   = skew(X_csp, axis=0)
    kt   = kurtosis(X_csp, axis=0, fisher=False)
    stats_vec = np.hstack([mean, std, sk, kt])  # (24,)
    stats = np.tile(stats_vec, (X_csp.shape[0], 1))
    return stats

def predict_npz(npz_path):
    if _svm is None:
        raise RuntimeError("SVM model not loaded. Check that all model .pkl files exist in models/ directory")
    
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    if X.ndim == 4:
        X = X.squeeze(-1)
    subject = data.get("subject", "unknown")
    # Steps
    X_csp = _apply_csp(X)
    X_riem = _apply_riemann(X)
    X_stats = _compute_stats(X_csp)
    X_fused = np.hstack([X_csp, X_riem, X_stats])  # -> (trials, 8286)
    X_scaled = _scaler.transform(X_fused)

    preds = _svm.predict(X_scaled)
    if hasattr(_svm, "predict_proba"):
        probs = _svm.predict_proba(X_scaled)[:,1]
    else:
        # fallback: use decision_function -> map with sigmoid
        df = _svm.decision_function(X_scaled)
        probs = 1/(1+np.exp(-df))

    maj = int(Counter(preds).most_common(1)[0][0])
    avg_prob = float(np.mean(probs))

    result = {
        "subject": str(subject),
        "label": "HC" if maj == 0 else "MDD",
        "prob": avg_prob,
        "votes": dict(Counter(map(int, preds))),
        "feature_stats": {
            "csp_mean": X_csp.mean(axis=0).tolist(),
            "riemann_norm_mean": float(np.linalg.norm(X_riem, axis=1).mean())
        }
    }
    return result
