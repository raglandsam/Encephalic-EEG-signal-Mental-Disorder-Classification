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

// Click upload area to select file
uploadArea.onclick = () => fileInput.click();

// File input change handler
fileInput.onchange = (e) => {
  const file = e.target.files[0];
  if (file) {
    uploadArea.style.borderColor = 'var(--accent-color)';
    
    // SHOW FILE NAME
    const fileLabel = document.getElementById("uploadedFileName");
    if (fileLabel) {
      fileLabel.textContent = `📄 Selected File: ${file.name}`;
      fileLabel.style.opacity = 1;
    }
  }
};


uploadBtn.onclick = async () => {
  const f = fileInput.files[0];
  if (!f) { 
    alert('📁 Please select an EEG file to analyze');
    return; 
  }

  spinner.classList.remove('hidden');
  resultBox.classList.add('hidden');
  uploadBtn.disabled = true;
  uploadBtn.style.opacity = '0.6';

  const form = new FormData();
  form.append('file', f);

  try {
    const resp = await fetch('/full-pipeline', { method: 'POST', body: form, timeout: 180000 });
    const data = await resp.json();
    spinner.classList.add('hidden');

    if (data.error) {
      alert('⚠️ Error: ' + (data.error || 'Processing failed'));
      return;
    }

    resultBox.classList.remove('hidden');
    
    // Parse prediction
    const prediction = data.label || data.prediction || 'Unknown';
    const probability = data.prob !== undefined ? data.prob : data.probability || 0;
    
    labelEl.textContent = prediction;
    probPercent.textContent = Math.round(probability * 100);
    
    // Animate probability bar
    setTimeout(() => {
      probFill.style.width = `${(probability * 100)}%`;
    }, 100);

    // Set status badge
    statusBadge.textContent = prediction === 'MDD' ? '⚠️ Major Depressive Disorder' : '✓ Healthy Control';
    statusBadge.classList.add(prediction.toLowerCase());

    // Format JSON nicely
    jsonOut.textContent = JSON.stringify(data, null, 2);
    
    // Scroll to results
    setTimeout(() => {
      resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
  } catch (err) {
    spinner.classList.add('hidden');
    alert('❌ Upload failed: ' + err.message);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.style.opacity = '1';
  }
};

function resetForm() {
  fileInput.value = '';
  resultBox.classList.add('hidden');
  uploadArea.style.borderColor = '';
  probFill.style.width = '0%';
  probPercent.textContent = '0';
  statusBadge.classList.remove('mdd', 'hc');
}
