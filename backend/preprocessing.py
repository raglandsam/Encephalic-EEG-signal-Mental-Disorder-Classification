# backend/preprocessing.py
import os
import numpy as np
import mne

STANDARD_CHANNELS = [f"E{i}" for i in range(1, 129)]
MIN_CHANNELS_REQUIRED = 120        # reject broken/incomplete files
VALID_SFREQ_RANGE = (200, 300)     # MODMA dataset is ~250Hz
TMIN, TMAX = -0.2, 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Allow overriding processed output dir via env var for cloud deployment
OUTPUT_DIR = os.environ.get('PREPROCESS_OUTPUT_DIR', os.path.join(BASE_DIR, "processed"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------
def preprocess_eeg_file(file_path: str, output_dir: str = None):
    if output_dir is None:
        output_dir = OUTPUT_DIR

    fname = os.path.basename(file_path)
    print(f"\n[PREPROCESS] Received file: {fname}")

    info = {
        "file": fname,
        "channels_before": None,
        "channels_after": None,
        "sampling_rate": None,
        "epoch_count": None,
        "message": ""
    }

    # 1. LOAD
    if file_path.lower().endswith(".raw"):
        raw = mne.io.read_raw_egi(file_path, preload=True)
    elif file_path.lower().endswith(".fif"):
        raw = mne.io.read_raw_fif(file_path, preload=True)
    elif file_path.lower().endswith(".npz"):
        info["message"] = "File already preprocessed (.npz)"
        return file_path, info
    else:
        raise ValueError(f"Unsupported EEG file format: {file_path}")

    info["channels_before"] = len(raw.ch_names)
    info["sampling_rate"] = raw.info["sfreq"]

    # 2. CHANNEL FIXING
    if "E129" in raw.ch_names:
        raw.drop_channels(["E129"])

    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    raw.set_montage(montage)

    extra = [c for c in raw.ch_names if c not in STANDARD_CHANNELS]
    if len(extra) > 0:
        raw.drop_channels(extra)

    for ch in STANDARD_CHANNELS:
        if ch not in raw.ch_names:
            info_msg = f"Missing channel added: {ch}"
            print(info_msg)
            info["message"] += info_msg + " | "

            zero = np.zeros((1, raw.n_times))
            raw.add_channels([mne.io.RawArray(zero, mne.create_info([ch], raw.info["sfreq"], 'eeg'))])

    raw.reorder_channels(STANDARD_CHANNELS)

    info["channels_after"] = len(raw.ch_names)

    # 3. FILTER
    raw.filter(1., 45., fir_design="firwin", verbose=False)

    # 4. EPOCHING
    try:
        events, event_id = mne.events_from_annotations(raw)
        cues = {k: v for k, v in event_id.items() if "cue" in k.lower()}
        if len(cues) > 0:
            epochs = mne.Epochs(raw, events, cues, tmin=TMIN, tmax=TMAX, baseline=(None, 0), preload=True)
            data = epochs.get_data()
        else:
            data = raw.get_data().T[np.newaxis, :, :]
    except Exception as e:
        info["message"] += f"Epoching failed: {e}"
        data = raw.get_data().T[np.newaxis, :, :]

    info["epoch_count"] = data.shape[0]

    subject_id = os.path.splitext(fname)[0]
    out_path = os.path.join(output_dir, f"{subject_id}_preprocessed.npz")
    np.savez(out_path, X=data, y=np.zeros(data.shape[0]), subject=subject_id)

    return out_path, info
