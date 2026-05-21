const state = {
  models: [],
  file: null,
  previewUrl: "",
  jobId: null,
  pollHandle: null,
  inputResolution: null,
  busy: false,
};

const elements = {
  deviceBadge: document.getElementById("deviceBadge"),
  modelCount: document.getElementById("modelCount"),
  refreshModelsButton: document.getElementById("refreshModelsButton"),
  imageInput: document.getElementById("imageInput"),
  dropZone: document.getElementById("dropZone"),
  inputFileLabel: document.getElementById("inputFileLabel"),
  modelSelect: document.getElementById("modelSelect"),
  nativeScaleHint: document.getElementById("nativeScaleHint"),
  nativeScaleValue: document.getElementById("nativeScaleValue"),
  factorInput: document.getElementById("factorInput"),
  factorValue: document.getElementById("factorValue"),
  tileSelect: document.getElementById("tileSelect"),
  originalResolution: document.getElementById("originalResolution"),
  targetResolution: document.getElementById("targetResolution"),
  progressMessage: document.getElementById("progressMessage"),
  progressPercent: document.getElementById("progressPercent"),
  progressFill: document.getElementById("progressFill"),
  statusBanner: document.getElementById("statusBanner"),
  originalPreview: document.getElementById("originalPreview"),
  originalPlaceholder: document.getElementById("originalPlaceholder"),
  originalMeta: document.getElementById("originalMeta"),
  resultPreview: document.getElementById("resultPreview"),
  resultPlaceholder: document.getElementById("resultPlaceholder"),
  resultMeta: document.getElementById("resultMeta"),
  resultResolution: document.getElementById("resultResolution"),
  downloadButton: document.getElementById("downloadButton"),
  startButton: document.getElementById("startButton"),
  upscaleForm: document.getElementById("upscaleForm"),
};

function formatFactor(value) {
  return `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 })}x`;
}

function formatPixels(width, height) {
  if (!width || !height) {
    return "Waiting for image";
  }
  return `${width.toLocaleString()} × ${height.toLocaleString()}`;
}

function formatBytes(bytes) {
  if (!bytes) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function setStatus(kind, message) {
  elements.statusBanner.dataset.kind = kind;
  elements.statusBanner.textContent = message;
}

function setProgress(progress, message) {
  const safeProgress = Math.max(0, Math.min(100, Math.round(progress)));
  elements.progressFill.style.width = `${safeProgress}%`;
  elements.progressPercent.textContent = `${safeProgress}%`;
  elements.progressMessage.textContent = message;
}

function getSelectedModel() {
  return state.models.find((model) => model.key === elements.modelSelect.value) || null;
}

function updateEstimate() {
  const factor = Number(elements.factorInput.value);
  elements.factorValue.textContent = formatFactor(factor);

  const selectedModel = getSelectedModel();
  if (selectedModel) {
    const nativeScale = formatFactor(selectedModel.native_scale);
    elements.nativeScaleHint.textContent = nativeScale;
    elements.nativeScaleValue.textContent = nativeScale;
  } else {
    elements.nativeScaleHint.textContent = "Native scale pending";
    elements.nativeScaleValue.textContent = "Waiting for model";
  }

  if (!state.inputResolution) {
    elements.targetResolution.textContent = "Waiting for image";
    return;
  }

  const width = Math.max(1, Math.round(state.inputResolution.width * factor));
  const height = Math.max(1, Math.round(state.inputResolution.height * factor));
  elements.targetResolution.textContent = formatPixels(width, height);
  if (!state.busy) {
    elements.resultMeta.textContent = `Estimated output: ${formatPixels(width, height)}`;
  }
}

function updateBusyState() {
  elements.modelSelect.disabled = state.busy || !state.models.length;
  elements.factorInput.disabled = state.busy;
  elements.tileSelect.disabled = state.busy;
  elements.imageInput.disabled = state.busy;
  elements.startButton.disabled = state.busy || !state.models.length;
  elements.startButton.textContent = state.busy ? "Upscaling..." : "Start Upscaling";
  elements.refreshModelsButton.disabled = state.busy;
}

function clearPolling() {
  if (state.pollHandle) {
    window.clearInterval(state.pollHandle);
    state.pollHandle = null;
  }
}

function resetOutputPreview() {
  elements.resultPreview.hidden = true;
  elements.resultPreview.removeAttribute("src");
  elements.resultPlaceholder.hidden = false;
  elements.resultMeta.textContent = "Run a job to generate a result preview";
  elements.resultResolution.textContent = "Waiting to render";
  elements.downloadButton.classList.add("is-disabled");
  elements.downloadButton.removeAttribute("href");
  elements.downloadButton.removeAttribute("download");
  elements.downloadButton.setAttribute("aria-disabled", "true");
}

function applyFile(file) {
  if (!file) {
    return;
  }
  if (!file.type.startsWith("image/")) {
    setStatus("error", "Please choose a supported image file.");
    return;
  }

  state.file = file;
  resetOutputPreview();
  setProgress(0, "Waiting for a job");

  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }
  state.previewUrl = URL.createObjectURL(file);

  elements.originalPreview.src = state.previewUrl;
  elements.originalPreview.hidden = false;
  elements.originalPlaceholder.hidden = true;
  elements.inputFileLabel.textContent = `${file.name} · ${formatBytes(file.size)}`;

  const imageProbe = new Image();
  imageProbe.onload = () => {
    state.inputResolution = {
      width: imageProbe.naturalWidth,
      height: imageProbe.naturalHeight,
    };
    elements.originalResolution.textContent = formatPixels(imageProbe.naturalWidth, imageProbe.naturalHeight);
    elements.originalMeta.textContent = `${formatPixels(imageProbe.naturalWidth, imageProbe.naturalHeight)} · ${formatBytes(file.size)}`;
    updateEstimate();
    setStatus("idle", "Image ready. Choose a model and start upscaling.");
  };
  imageProbe.src = state.previewUrl;
}

async function readJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed with status ${response.status}.`);
  }
  return data;
}

function populateModelSelect() {
  elements.modelSelect.replaceChildren();
  if (!state.models.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No models found";
    elements.modelSelect.append(option);
    return;
  }

  for (const model of state.models) {
    const option = document.createElement("option");
    option.value = model.key;
    option.textContent = `${model.label} · ${formatFactor(model.native_scale)}`;
    elements.modelSelect.append(option);
  }
}

async function fetchModels() {
  setStatus("working", "Scanning the model folder...");
  try {
    const payload = await readJson(await fetch("/api/models"));
    state.models = payload.models || [];
    elements.deviceBadge.textContent = payload.device || "Device unavailable";
    elements.modelCount.textContent = `${state.models.length} model${state.models.length === 1 ? "" : "s"} detected`;
    populateModelSelect();
    updateBusyState();
    updateEstimate();

    if (state.models.length) {
      setStatus("idle", "Ready when you are.");
    } else {
      setStatus("error", "No .pth model files were found in the model folder.");
    }
  } catch (error) {
    state.models = [];
    populateModelSelect();
    updateBusyState();
    setStatus("error", error.message || "Failed to scan the model folder.");
  }
}

async function updateJobStatus() {
  if (!state.jobId) {
    return;
  }

  try {
    const payload = await readJson(await fetch(`/api/jobs/${state.jobId}`));
    setProgress(payload.progress || 0, payload.message || "Working");

    if (payload.original_width && payload.original_height) {
      elements.originalResolution.textContent = formatPixels(payload.original_width, payload.original_height);
    }

    if (payload.status === "completed") {
      clearPolling();
      state.busy = false;
      updateBusyState();
      elements.resultPreview.src = `${payload.output_url}?t=${Date.now()}`;
      elements.resultPreview.hidden = false;
      elements.resultPlaceholder.hidden = true;
      elements.resultResolution.textContent = formatPixels(payload.result_width, payload.result_height);
      elements.resultMeta.textContent = `${formatPixels(payload.result_width, payload.result_height)} · ${payload.model_key}`;
      elements.downloadButton.href = payload.output_url;
      elements.downloadButton.download = payload.output_url.split("/").pop();
      elements.downloadButton.classList.remove("is-disabled");
      elements.downloadButton.removeAttribute("aria-disabled");
      setStatus("success", `Upscale finished with ${payload.model_key}.`);
      return;
    }

    if (payload.status === "failed") {
      clearPolling();
      state.busy = false;
      updateBusyState();
      setStatus("error", payload.error || "The upscale job failed.");
    }
  } catch (error) {
    clearPolling();
    state.busy = false;
    updateBusyState();
    setStatus("error", error.message || "Unable to read job progress.");
  }
}

async function startUpscale(event) {
  event.preventDefault();
  if (state.busy) {
    return;
  }
  if (!state.file) {
    setStatus("error", "Choose an input image before starting.");
    return;
  }
  if (!elements.modelSelect.value) {
    setStatus("error", "Select a model from the model folder first.");
    return;
  }

  clearPolling();
  resetOutputPreview();
  state.busy = true;
  updateBusyState();
  setStatus("working", "Uploading the image and starting the ESRGAN pass...");
  setProgress(6, "Creating job");

  const formData = new FormData();
  formData.append("image", state.file);
  formData.append("model", elements.modelSelect.value);
  formData.append("upscale_factor", elements.factorInput.value);
  formData.append("tile_size", elements.tileSelect.value);

  try {
    const payload = await readJson(
      await fetch("/api/upscale", {
        method: "POST",
        body: formData,
      })
    );
    state.jobId = payload.job_id;
    await updateJobStatus();
    state.pollHandle = window.setInterval(updateJobStatus, 700);
  } catch (error) {
    state.busy = false;
    updateBusyState();
    setStatus("error", error.message || "Unable to start the upscale job.");
    setProgress(0, "Waiting for a job");
  }
}

function handleDrop(event) {
  event.preventDefault();
  elements.dropZone.classList.remove("is-dragover");
  const file = event.dataTransfer?.files?.[0];
  if (!file) {
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(file);
  elements.imageInput.files = transfer.files;
  applyFile(file);
}

elements.imageInput.addEventListener("change", () => {
  applyFile(elements.imageInput.files?.[0] || null);
});
elements.factorInput.addEventListener("input", updateEstimate);
elements.modelSelect.addEventListener("change", updateEstimate);
elements.refreshModelsButton.addEventListener("click", fetchModels);
elements.upscaleForm.addEventListener("submit", startUpscale);
elements.dropZone.addEventListener("dragenter", (event) => {
  event.preventDefault();
  elements.dropZone.classList.add("is-dragover");
});
elements.dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.dropZone.classList.add("is-dragover");
});
elements.dropZone.addEventListener("dragleave", (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("is-dragover");
});
elements.dropZone.addEventListener("drop", handleDrop);

window.addEventListener("beforeunload", () => {
  clearPolling();
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }
});

updateBusyState();
setProgress(0, "Waiting for a job");
fetchModels();