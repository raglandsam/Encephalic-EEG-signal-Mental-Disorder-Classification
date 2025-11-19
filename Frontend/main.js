const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadBtn = document.getElementById('uploadBtn');
const spinner = document.getElementById('spinner');
const resultBox = document.getElementById('result');
const labelEl = document.getElementById('label');
const probFill = document.getElementById('probFill');
const probPercent = document.getElementById('probPercent');
const jsonOut = document.getElementById('jsonOut');
const statusBadge = document.getElementById('status-badge');

const API_URL = "/api/full-pipeline";

uploadArea.addEventListener("click", () => {
  fileInput.click();    // bulletproof single trigger
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    document.getElementById("uploadedFileName").textContent =
      `📄 Selected File: ${file.name}`;
  }
});

uploadBtn.onclick = async () => {
  const f = fileInput.files[0];
  if (!f) { alert("📁 Please select an EEG file first!"); return; }

  spinner.classList.remove('hidden');
  resultBox.classList.add('hidden');
  uploadBtn.disabled = true;
  uploadBtn.style.opacity = '0.6';

  const form = new FormData();
  form.append('file', f);

  try {
    const resp = await fetch(API_URL, { method: 'POST', body: form });
    if (!resp.ok) throw new Error(`Server returned ${resp.status}`);

    const data = await resp.json();

    spinner.classList.add('hidden');
    resultBox.classList.remove('hidden');

    const label = data.label ?? "Unknown";
    const prob = data.prob ?? 0;

    labelEl.textContent = label;
    probPercent.textContent = Math.round(prob * 100);

    setTimeout(() => { probFill.style.width = `${prob * 100}%`; }, 50);

    statusBadge.className = "status";
    if (label === "MDD") {
      statusBadge.textContent = "⚠️ Major Depressive Disorder";
      statusBadge.classList.add("mdd");
    } else {
      statusBadge.textContent = "✓ Healthy Control";
      statusBadge.classList.add("hc");
    }

    jsonOut.textContent = JSON.stringify(data, null, 2);

  } catch (err) {
    spinner.classList.add('hidden');
    alert("❌ Upload failed: " + err.message);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.style.opacity = "1";
  }
};

function resetForm() {
  fileInput.value = "";
  resultBox.classList.add("hidden");
  document.getElementById("uploadedFileName").textContent = "No file selected";
  probFill.style.width = "0%";
  statusBadge.className = "status";
}
