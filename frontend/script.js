// ===========================
// API BASE URL
// ===========================

// Backend is always on port 8000, same host as frontend
const API_BASE = '';
const sessionId = crypto.randomUUID(); // Page memory only; refresh starts fresh.
const nativeFetch = window.fetch.bind(window);
const fetch = (url, options = {}) => {
  return nativeFetch(url, {...options, headers: {...options.headers, 'X-Session-Id': sessionId}});
};

// ===========================
// DOM elements
// ===========================

// Dataset
const classNameInput = document.getElementById("class-name-input");
const uploadInput = document.getElementById("upload-input");
const uploadBtn = document.getElementById("upload-btn");
const uploadStatusEl = document.getElementById("upload-status");

// Training
const trainBtn = document.getElementById("train-btn");
const trainingStatusEl = document.getElementById("training-status");
const progressFillEl = document.getElementById("progress-bar-fill");
const progressPercentEl = document.getElementById("progress-percent");
const deviceInfoEl = document.getElementById("device-info");
const metricsTableBody = document.querySelector("#metrics-table tbody");

// Charts
let accuracyChart = null;
let lossChart = null;

// Prediction
const predictUploadInput = document.getElementById("predict-upload-input");
const predictUploadBtn = document.getElementById("predict-upload-btn");
const webcamVideo = document.getElementById("webcam-video");
const webcamCanvas = document.getElementById("webcam-canvas");
const webcamCaptureBtn = document.getElementById("webcam-capture-btn");
const predictionsContainer = document.getElementById("predictions-container");

// Theme toggle
const themeToggleInput = document.getElementById("theme-toggle-input");

// ONNX export
const onnxExportBtn = document.getElementById("onnx-export-btn");

// Training status polling
let trainingPollInterval = null;

// ===========================
// Theme
// ===========================

function applyThemeFromStorage() {
  const stored = null;
  const prefersDark =
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;

  const shouldBeDark = stored
    ? stored === "dark"
    : prefersDark
    ? true
    : false;

  document.body.classList.toggle("dark", shouldBeDark);
  themeToggleInput.checked = shouldBeDark;
}

themeToggleInput.addEventListener("change", () => {
  const isDark = themeToggleInput.checked;
  document.body.classList.toggle("dark", isDark);
});

applyThemeFromStorage();

// ===========================
// Dataset upload
// ===========================

uploadBtn.addEventListener("click", async () => {
  const className = classNameInput.value.trim();
  const files = uploadInput.files;

  if (!className) {
    uploadStatusEl.textContent = "Please enter a class name.";
    return;
  }
  if (!files || files.length === 0) {
    uploadStatusEl.textContent = "Please select at least one image.";
    return;
  }

  uploadStatusEl.textContent = "Uploading images...";
  let uploadedCount = 0;

  for (const file of files) {
    const formData = new FormData();
    formData.append("class_name", className);
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        uploadedCount++;
      } else {
        const errData = await res.json().catch(() => ({}));
        uploadStatusEl.textContent = errData.detail || errData.error || 'Upload failed.';
        return;
      }
    } catch (err) {
      console.error("Upload error:", err);
    }
  }

  const countsResponse = await fetch('/dataset');
  const counts = await countsResponse.json();
  uploadStatusEl.textContent = `Uploaded ${uploadedCount} / ${files.length}. Classes: ${Object.entries(counts).map(([name,count]) => `${name}: ${count}`).join(', ')}. Train again to use this dataset.`;
  predictionsContainer.textContent = '';
  metricsTableBody.textContent = '';
});

// ===========================
// Training
// ===========================

trainBtn.addEventListener("click", async () => {
  trainingStatusEl.textContent = "Starting training...";
  progressFillEl.style.width = "0%";
  progressPercentEl.textContent = "0%";
  deviceInfoEl.textContent = "";
  metricsTableBody.innerHTML = "";

  if (accuracyChart) {
    accuracyChart.destroy();
    accuracyChart = null;
  }
  if (lossChart) {
    lossChart.destroy();
    lossChart = null;
  }

  try {
    const res = await fetch(`${API_BASE}/train`, {
      method: "POST",
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      console.error("Error starting training:", data);
      trainingStatusEl.textContent =
        data.detail || data.error || "Failed to start training.";
      return;
    }

    trainingStatusEl.textContent = "Training started...";
    trainBtn.disabled = true;
  } catch (err) {
    console.error("Error starting training (network):", err);
    trainingStatusEl.textContent =
      "Error starting training (network). Check backend.";
    return;
  }

  if (trainingPollInterval) clearInterval(trainingPollInterval);
  trainingPollInterval = setInterval(fetchTrainingStatus, 1000);
});

async function fetchTrainingStatus() {
  try {
    const res = await fetch(`${API_BASE}/training/status`);
    const data = await res.json();

    const status = data.status || "unknown";
    trainingStatusEl.textContent = `Status: ${status}${data.stage ? ' · '+data.stage : ''}`;
    if (status === 'idle') {
      clearInterval(trainingPollInterval); trainingPollInterval = null; trainBtn.disabled = false;
      trainingStatusEl.textContent = 'Session reset or expired. Re-upload your images and train again.';
      return;
    }

    if (data.device) {
      const d = data.device;
      deviceInfoEl.textContent = d.gpu_available
        ? `GPU detected (${d.num_gpus}) – ${d.gpus.join(", ")}`
        : "GPU not detected – using CPU";
    }

    if (data.cnn) {
      const { current_epoch, total_epochs, val_accuracy, loss, val_loss } =
        data.cnn;
      const pct = total_epochs
        ? Math.round((current_epoch / total_epochs) * 100)
        : 0;

      progressFillEl.style.width = `${pct}%`;
      progressPercentEl.textContent = `${pct}%`;

      trainingStatusEl.textContent = `Status: ${status} · Epoch ${current_epoch} / ${total_epochs} · val_acc: ${
        typeof val_accuracy === 'number' ? val_accuracy.toFixed(3) : "–"
      }`;
    }

    if (status === "finished" || status === "error") {
      clearInterval(trainingPollInterval);
      trainingPollInterval = null;
      trainBtn.disabled = false;

      if (status === "finished" && data.results) {
        populateMetricsAndCharts(data.results);
      } else if (status === "error") {
        trainingStatusEl.textContent = `Status: error – ${
          data.error || "Unknown error"
        }`;
        console.error("Training error:", data.error);
      }
    }
  } catch (err) {
    console.error("Error fetching training status:", err);
  }
}

function populateMetricsAndCharts(results) {
  metricsTableBody.innerHTML = "";

  const modelsOrder = ["logreg", "rf", "cnn"];

  modelsOrder.forEach((key) => {
    if (!results[key]) return;
    const model = results[key];
    const tr = document.createElement("tr");

    const nameTd = document.createElement("td");
    nameTd.textContent = model.name || key;

    const accTd = document.createElement("td");
    const acc =
      typeof model.accuracy === "number"
        ? (model.accuracy * 100).toFixed(2) + "%"
        : "–";
    accTd.textContent = acc;

    tr.appendChild(nameTd);
    tr.appendChild(accTd);
    metricsTableBody.appendChild(tr);
  });

  const cnn = results.cnn;
  if (!cnn || !cnn.history) return;

  const accCanvas = document.getElementById("accuracyChart");
  const lossCanvas = document.getElementById("lossChart");

  const epochs = cnn.history.accuracy
    ? cnn.history.accuracy.map((_, i) => i + 1)
    : [];

  if (!epochs.length) return;

  accuracyChart = new Chart(accCanvas.getContext("2d"), {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Train Accuracy",
          data: cnn.history.accuracy,
          fill: false,
        },
        {
          label: "Val Accuracy",
          data: cnn.history.val_accuracy,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
      },
      scales: {
        y: {
          min: 0,
          max: 1,
        },
      },
    },
  });

  lossChart = new Chart(lossCanvas.getContext("2d"), {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Train Loss",
          data: cnn.history.loss,
          fill: false,
        },
        {
          label: "Val Loss",
          data: cnn.history.val_loss,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
      },
    },
  });
}

// ===========================
// Prediction: upload
// ===========================

predictUploadBtn.addEventListener("click", async () => {
  const file = predictUploadInput.files[0];
  if (!file) {
    alert("Please select an image.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/predict/upload`, {
      method: "POST",
      body: formData,
    });

    let data = null;
    try {
      data = await res.json();
    } catch (e) {
      console.error("Failed to parse JSON:", e);
    }

    if (!res.ok) {
      console.error("Server error on predict:", data, "status:", res.status);
      alert((data && data.error) || `Server error: ${res.status}`);
      return;
    }

    if (!data) {
      alert("Empty response from server.");
      return;
    }

    renderPredictions(data);
  } catch (err) {
    console.error("Predict error (network/CORS):", err);
    alert(
      "Prediction error (network). Check backend is running at " + API_BASE
    );
  }
});

// ===========================
// Webcam setup & capture
// ===========================

async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 320, height: 240 },
    });
    webcamVideo.srcObject = stream;
  } catch (err) {
    console.warn("Webcam not available:", err);
  }
}

webcamCaptureBtn.addEventListener("click", async () => {
  if (!webcamVideo.srcObject) {
    alert("Webcam not available or permission denied.");
    return;
  }

  const width = webcamVideo.videoWidth || 320;
  const height = webcamVideo.videoHeight || 240;

  webcamCanvas.width = width;
  webcamCanvas.height = height;
  const ctx = webcamCanvas.getContext("2d");
  ctx.drawImage(webcamVideo, 0, 0, width, height);

  const blob = await new Promise((resolve) =>
    webcamCanvas.toBlob(resolve, "image/jpeg")
  );
  const formData = new FormData();
  formData.append("file", blob, "webcam.jpg");

  try {
    const res = await fetch(`${API_BASE}/predict/upload`, {
      method: "POST",
      body: formData,
    });

    let data = null;
    try {
      data = await res.json();
    } catch (e) {
      console.error("Failed to parse JSON (webcam):", e);
    }

    if (!res.ok) {
      console.error("Server error on webcam predict:", data);
      alert((data && data.error) || `Server error: ${res.status}`);
      return;
    }

    if (!data) {
      alert("Empty response from server.");
      return;
    }

    renderPredictions(data);
  } catch (err) {
    console.error("Predict error (webcam, network/CORS):", err);
    alert(
      "Prediction error (webcam, network). Check backend is running at " +
        API_BASE
    );
  }
});

document.getElementById('webcam-start-btn').addEventListener('click', initWebcam);
document.getElementById('webcam-stop-btn').addEventListener('click', () => {
  webcamVideo.srcObject?.getTracks().forEach(track => track.stop());
  webcamVideo.srcObject = null;
});
window.addEventListener('pagehide', () => webcamVideo.srcObject?.getTracks().forEach(track => track.stop()));

// ===========================
// Render predictions
// ===========================

function renderPredictions(predictionData) {
  predictionsContainer.innerHTML = "";

  const order = ["logreg", "rf", "cnn"];

  order.forEach((key) => {
    const model = predictionData[key];
    if (!model) return;

    const card = document.createElement("div");
    card.className = "prediction-card";

    const header = document.createElement("div");
    header.className = "prediction-header";

    const modelNameSpan = document.createElement("span");
    modelNameSpan.className = "model-name";
    modelNameSpan.textContent =
      key === "logreg"
        ? "Logistic Regression"
        : key === "rf"
        ? "Random Forest"
        : "CNN";

    const classSpan = document.createElement("span");
    classSpan.className = "class-name";
    classSpan.textContent = model.class_name ?? `Class ${model.class_index}`;

    header.appendChild(modelNameSpan);
    header.appendChild(classSpan);
    card.appendChild(header);

    if (model.probs && model.probs.length) {
      const maxProb = Math.max(...model.probs);
      const probContainer = document.createElement("div");
      probContainer.className = "prob-bar-container";

      const bar = document.createElement("div");
      bar.className = "prob-bar";

      const barFill = document.createElement("div");
      barFill.className = "prob-bar-fill";
      barFill.style.width = `${Math.round(maxProb * 100)}%`;

      bar.appendChild(barFill);

      const label = document.createElement("div");
      label.className = "prob-label";
      label.textContent = `Confidence: ${(maxProb * 100).toFixed(2)}%`;

      probContainer.appendChild(bar);
      probContainer.appendChild(label);

      card.appendChild(probContainer);
    }

    predictionsContainer.appendChild(card);
  });
}

// ===========================
// ONNX export
// ===========================

onnxExportBtn.addEventListener("click", async () => {
  try {
    const res = await fetch(`${API_BASE}/export/onnx`, {
      method: "POST",
    });
    if (!res.ok) { const error = await res.json(); throw new Error(error.detail || 'Train first.'); }
    const url = URL.createObjectURL(await res.blob());
    const link = document.createElement('a');
    link.href = url; link.download = 'visionforge-onnx.zip'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (err) {
    console.error("ONNX export error:", err);
    alert(err.message || 'ONNX export failed.');
  }
});

document.getElementById('reset-session-btn').addEventListener('click', async () => {
  const response = await fetch('/session', {method: 'DELETE'});
  if (!response.ok) { const error = await response.json(); alert(error.detail); return; }
  location.reload();
});
