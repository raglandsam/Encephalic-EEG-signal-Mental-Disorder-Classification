const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const spinner = document.getElementById('spinner');
const resultBox = document.getElementById('result');
const labelEl = document.getElementById('label');
const probFill = document.getElementById('probFill');
const probPercent = document.getElementById('probPercent');
const jsonOut = document.getElementById('jsonOut');
const statusBadge = document.getElementById('status-badge');
const uploadArea = document.querySelector('.upload-area');

// HuggingFace backend endpoint (FIXED)
const API_URL = "/api/full-pipeline";

// Click upload area → open file picker
uploadArea.onclick = () => fileInput.click();

// Show selected filename
fileInput.onchange = (e) => {
  const file = e.target.files[0];
  if (file) {
    uploadArea.style.borderColor = 'var(--accent-color)';
    const fileLabel = document.getElementById("uploadedFileName");
    if (fileLabel) {
      fileLabel.textContent = `📄 Selected File: ${file.name}`;
      fileLabel.style.opacity = 1;
    }
  }
};

// Upload handler
uploadBtn.onclick = async () => {
  const f = fileInput.files[0];
  if (!f) {
    alert('📁 Please select an EEG file first!');
    return;
  }

  spinner.classList.remove('hidden');
  resultBox.classList.add('hidden');
  uploadBtn.disabled = true;
  uploadBtn.style.opacity = '0.6';

  const form = new FormData();
  form.append('file', f);

  try {
    const resp = await fetch(API_URL, { method: 'POST', body: form });

    if (!resp.ok) {
      throw new Error(`Server returned ${resp.status}`);
    }

    const data = await resp.json();

    spinner.classList.add('hidden');
    resultBox.classList.remove('hidden');

    // ---- Prediction ----
    const label = data.label || "Unknown";
    const prob = data.prob ?? 0;

    labelEl.textContent = label;
    probPercent.textContent = Math.round(prob * 100);

    setTimeout(() => {
      probFill.style.width = `${prob * 100}%`;
    }, 100);

    // ---- Status Badge ----
    statusBadge.classList.remove("mdd", "hc");

    if (label === "MDD") {
      statusBadge.textContent = "⚠️ Major Depressive Disorder";
      statusBadge.classList.add("mdd");
    } else {
      statusBadge.textContent = "✓ Healthy Control";
      statusBadge.classList.add("hc");
    }

    // ---- Pretty JSON ----
    jsonOut.textContent = JSON.stringify(data, null, 2);

    // Smooth scroll to result
    setTimeout(() => {
      resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 200);

  } catch (err) {
    spinner.classList.add('hidden');
    alert('❌ Upload failed: ' + err.message);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.style.opacity = '1';
  }
};

// Reset UI
function resetForm() {
  fileInput.value = '';
  resultBox.classList.add('hidden');
  uploadArea.style.borderColor = '';
  probFill.style.width = '0%';
  probPercent.textContent = '0';
  statusBadge.classList.remove('mdd', 'hc');
}
