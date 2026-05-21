const READY_STATUS = "ready";
const TERMINAL_BATCH_STATUSES = new Set(["completed", "completed_with_errors"]);

const state = {
  models: [],
  items: [],
  batchId: null,
  pollHandle: null,
  busy: false,
  focusedItemId: null,
  compare: {
    zoom: 1,
    offsetX: 0,
    offsetY: 0,
    split: 50,
    canPan: false,
    dragging: false,
    splitDragging: false,
    lastX: 0,
    lastY: 0,
    panPointerId: null,
    splitPointerId: null,
    boundItemId: null,
  },
};

const elements = {
  deviceBadge: document.getElementById("deviceBadge"),
  modelCount: document.getElementById("modelCount"),
  refreshModelsButton: document.getElementById("refreshModelsButton"),
  clearQueueButton: document.getElementById("clearQueueButton"),
  imageInput: document.getElementById("imageInput"),
  dropZone: document.getElementById("dropZone"),
  inputFileLabel: document.getElementById("inputFileLabel"),
  modelSelect: document.getElementById("modelSelect"),
  nativeScaleHint: document.getElementById("nativeScaleHint"),
  nativeScaleValue: document.getElementById("nativeScaleValue"),
  factorInput: document.getElementById("factorInput"),
  factorValue: document.getElementById("factorValue"),
  tileSelect: document.getElementById("tileSelect"),
  allowSub100Zoom: document.getElementById("allowSub100Zoom"),
  selectedCountValue: document.getElementById("selectedCountValue"),
  originalResolution: document.getElementById("originalResolution"),
  targetResolution: document.getElementById("targetResolution"),
  progressMessage: document.getElementById("progressMessage"),
  progressPercent: document.getElementById("progressPercent"),
  progressFill: document.getElementById("progressFill"),
  batchStats: document.getElementById("batchStats"),
  queueSummary: document.getElementById("queueSummary"),
  queueEmptyState: document.getElementById("queueEmptyState"),
  queueList: document.getElementById("queueList"),
  statusBanner: document.getElementById("statusBanner"),
  originalPreview: document.getElementById("originalPreview"),
  originalPlaceholder: document.getElementById("originalPlaceholder"),
  originalMeta: document.getElementById("originalMeta"),
  resultPreview: document.getElementById("resultPreview"),
  resultPlaceholder: document.getElementById("resultPlaceholder"),
  resultMeta: document.getElementById("resultMeta"),
  resultResolution: document.getElementById("resultResolution"),
  downloadButton: document.getElementById("downloadButton"),
  comparePlaceholder: document.getElementById("comparePlaceholder"),
  compareViewport: document.getElementById("compareViewport"),
  compareCanvasPan: document.getElementById("compareCanvasPan"),
  compareCanvas: document.getElementById("compareCanvas"),
  compareBeforeImage: document.getElementById("compareBeforeImage"),
  compareAfterImage: document.getElementById("compareAfterImage"),
  compareAfterMask: document.getElementById("compareAfterMask"),
  compareDivider: document.getElementById("compareDivider"),
  compareHint: document.getElementById("compareHint"),
  zoomOutButton: document.getElementById("zoomOutButton"),
  zoomInButton: document.getElementById("zoomInButton"),
  zoomLabel: document.getElementById("zoomLabel"),
  resetViewButton: document.getElementById("resetViewButton"),
  startButton: document.getElementById("startButton"),
  upscaleForm: document.getElementById("upscaleForm"),
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function pluralize(count, singular, plural = `${singular}s`) {
  return count === 1 ? singular : plural;
}

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

function createLocalId(index) {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${index}-${Math.random().toString(16).slice(2)}`;
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

function getFocusedItem() {
  if (!state.items.length) {
    return null;
  }

  const focusedItem = state.items.find((item) => item.id === state.focusedItemId);
  if (focusedItem) {
    return focusedItem;
  }

  state.focusedItemId = state.items[0].id;
  return state.items[0];
}

function getCounts() {
  return state.items.reduce(
    (counts, item) => {
      counts.total += 1;
      counts[item.status] = (counts[item.status] || 0) + 1;
      return counts;
    },
    { total: 0, [READY_STATUS]: 0, queued: 0, running: 0, completed: 0, failed: 0 }
  );
}

function updateModelHints() {
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
}

function updateSummaryMetrics() {
  const counts = getCounts();
  if (!counts.total) {
    elements.selectedCountValue.textContent = "No images selected";
    elements.queueSummary.textContent = "No files selected";
    elements.batchStats.textContent = "No active batch";
    return;
  }

  elements.selectedCountValue.textContent = `${counts.total} ${pluralize(counts.total, "image")} selected`;

  const summaryParts = [`${counts.total} selected`];
  if (counts.completed) {
    summaryParts.push(`${counts.completed} ready`);
  }
  if (counts.running) {
    summaryParts.push(`${counts.running} processing`);
  }
  if (counts.queued) {
    summaryParts.push(`${counts.queued} queued`);
  }
  if (counts.failed) {
    summaryParts.push(`${counts.failed} failed`);
  }
  if (counts[READY_STATUS] && !state.busy) {
    summaryParts.push(`${counts[READY_STATUS]} pending`);
  }

  elements.queueSummary.textContent = summaryParts.join(" • ");
  elements.batchStats.textContent = `${counts.completed} ready • ${counts.failed} failed`;
}

function disableDownloadButton() {
  elements.downloadButton.classList.add("is-disabled");
  elements.downloadButton.removeAttribute("href");
  elements.downloadButton.removeAttribute("download");
  elements.downloadButton.setAttribute("aria-disabled", "true");
}

function updateEstimate() {
  updateModelHints();

  const item = getFocusedItem();
  if (!item || !item.width || !item.height) {
    elements.targetResolution.textContent = state.items.length ? "Reading image size" : "Waiting for image";
    return;
  }

  const factor = Number(elements.factorInput.value);
  const width = Math.max(1, Math.round(item.width * factor));
  const height = Math.max(1, Math.round(item.height * factor));
  elements.targetResolution.textContent = formatPixels(width, height);

  if (!state.busy && !item.outputUrl) {
    elements.resultMeta.textContent = `Estimated output: ${formatPixels(width, height)}`;
  }
}

function updateBusyState() {
  const hasItems = state.items.length > 0;
  elements.modelSelect.disabled = state.busy || !state.models.length;
  elements.factorInput.disabled = state.busy;
  elements.tileSelect.disabled = state.busy;
  elements.imageInput.disabled = state.busy;
  elements.refreshModelsButton.disabled = state.busy;
  elements.clearQueueButton.disabled = state.busy || !hasItems;
  elements.startButton.disabled = state.busy || !state.models.length || !hasItems;

  if (state.busy) {
    elements.startButton.textContent = `Processing ${state.items.length} ${pluralize(state.items.length, "image")}...`;
  } else if (state.items.length > 1) {
    elements.startButton.textContent = "Start Batch Upscaling";
  } else {
    elements.startButton.textContent = "Start Upscaling";
  }
}

function clearPolling() {
  if (state.pollHandle) {
    window.clearInterval(state.pollHandle);
    state.pollHandle = null;
  }
}

function updateCompareControlState(enabled) {
  elements.compareDivider.disabled = !enabled;
  elements.zoomOutButton.disabled = !enabled;
  elements.zoomInButton.disabled = !enabled;
  elements.resetViewButton.disabled = !enabled;
}

function resetCompareState(resetSplit = true) {
  state.compare.zoom = 1;
  state.compare.offsetX = 0;
  state.compare.offsetY = 0;
  state.compare.canPan = false;
  state.compare.dragging = false;
  state.compare.splitDragging = false;
  state.compare.panPointerId = null;
  state.compare.splitPointerId = null;
  if (resetSplit) {
    state.compare.split = 50;
  }
}

function getMinimumZoom() {
  return elements.allowSub100Zoom.checked ? 0.6 : 1;
}

function getCompareMetrics(item) {
  const rect = elements.compareViewport.getBoundingClientRect();
  const { width, height } = getCompareDimensions(item);
  const baseScale = rect.width > 0 && rect.height > 0 ? Math.min(rect.width / width, rect.height / height) : 1;
  const scaledWidth = width * baseScale * state.compare.zoom;
  const scaledHeight = height * baseScale * state.compare.zoom;
  const maxOffsetX = Math.max(0, (scaledWidth - rect.width) / 2);
  const maxOffsetY = Math.max(0, (scaledHeight - rect.height) / 2);

  return {
    rect,
    width,
    height,
    baseScale,
    effectiveScale: baseScale * state.compare.zoom,
    scaledWidth,
    scaledHeight,
    maxOffsetX,
    maxOffsetY,
  };
}

function revokeItemUrls() {
  for (const item of state.items) {
    if (item.previewUrl) {
      URL.revokeObjectURL(item.previewUrl);
    }
  }
}

function clearQueue() {
  if (state.busy) {
    return;
  }

  clearPolling();
  revokeItemUrls();
  state.items = [];
  state.batchId = null;
  state.focusedItemId = null;
  state.compare.boundItemId = null;
  resetCompareState();
  elements.imageInput.value = "";
  elements.inputFileLabel.textContent = "PNG, JPG, WEBP, BMP";
  renderQueue();
  syncFocusedItem();
  updateBusyState();
  setProgress(0, "Waiting for a job");
  setStatus(state.models.length ? "idle" : "error", state.models.length ? "Choose images to begin." : "No .pth model files were found in the model folder.");
}

function createItemFromFile(file, index) {
  return {
    id: createLocalId(index),
    file,
    previewUrl: URL.createObjectURL(file),
    width: null,
    height: null,
    status: READY_STATUS,
    progress: 0,
    message: "Ready to queue",
    jobId: null,
    outputUrl: null,
    resultWidth: null,
    resultHeight: null,
    error: null,
    outputToken: 0,
  };
}

function measureItem(item) {
  return new Promise((resolve) => {
    const probe = new Image();
    probe.onload = () => {
      item.width = probe.naturalWidth;
      item.height = probe.naturalHeight;
      resolve();
    };
    probe.onerror = () => resolve();
    probe.src = item.previewUrl;
  });
}

async function applyFiles(fileList) {
  const incomingFiles = Array.from(fileList || []);
  const supportedFiles = incomingFiles.filter((file) => file.type.startsWith("image/"));
  const skippedCount = incomingFiles.length - supportedFiles.length;

  if (!supportedFiles.length) {
    setStatus("error", incomingFiles.length ? "Please choose supported image files." : "No files selected.");
    return;
  }

  clearPolling();
  revokeItemUrls();
  state.batchId = null;
  state.busy = false;
  state.items = supportedFiles.map((file, index) => createItemFromFile(file, index));
  state.focusedItemId = state.items[0].id;
  state.compare.boundItemId = null;
  resetCompareState();
  updateBusyState();

  const totalBytes = state.items.reduce((sum, item) => sum + item.file.size, 0);
  elements.inputFileLabel.textContent = `${state.items.length} ${pluralize(state.items.length, "image")} • ${formatBytes(totalBytes)}`;

  renderQueue();
  syncFocusedItem();
  setProgress(0, "Waiting for a job");

  await Promise.all(state.items.map((item) => measureItem(item)));

  renderQueue();
  syncFocusedItem();

  if (skippedCount) {
    setStatus(
      "idle",
      `${state.items.length} ${pluralize(state.items.length, "image")} ready. ${skippedCount} unsupported ${pluralize(skippedCount, "file")} ignored.`
    );
  } else {
    setStatus("idle", `${state.items.length} ${pluralize(state.items.length, "image")} ready. Choose a model and start the batch.`);
  }
}

function getItemStatusLabel(item) {
  if (item.status === READY_STATUS) {
    return "Ready";
  }
  if (item.status === "running") {
    return `${item.progress}%`;
  }
  if (item.status === "completed") {
    return "Ready";
  }
  if (item.status === "failed") {
    return "Failed";
  }
  return item.status.charAt(0).toUpperCase() + item.status.slice(1);
}

function getItemDetailText(item) {
  if (item.status === "completed" && item.resultWidth && item.resultHeight) {
    return `Output ${formatPixels(item.resultWidth, item.resultHeight)}`;
  }

  if (item.status === "failed") {
    return item.error || "Processing failed";
  }

  if (item.status === "running") {
    return item.message || "Processing";
  }

  if (item.status === "queued") {
    return "Queued for processing";
  }

  return item.width && item.height ? `Original ${formatPixels(item.width, item.height)}` : "Reading image size";
}

function setFocusedItem(itemId) {
  if (state.focusedItemId === itemId) {
    return;
  }

  state.focusedItemId = itemId;
  state.compare.boundItemId = null;
  resetCompareState(false);
  renderQueue();
  syncFocusedItem();
}

function renderQueue() {
  updateSummaryMetrics();
  const hasItems = state.items.length > 0;
  elements.queueEmptyState.hidden = hasItems;
  elements.queueList.hidden = !hasItems;
  elements.queueList.replaceChildren();

  if (!hasItems) {
    return;
  }

  for (const item of state.items) {
    const card = document.createElement("article");
    card.className = `queue-item${item.id === state.focusedItemId ? " is-active" : ""}`;
    card.dataset.status = item.status;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-pressed", item.id === state.focusedItemId ? "true" : "false");
    card.addEventListener("click", () => setFocusedItem(item.id));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        setFocusedItem(item.id);
      }
    });

    const header = document.createElement("div");
    header.className = "queue-item-header";

    const copy = document.createElement("div");
    copy.className = "queue-item-copy";

    const title = document.createElement("strong");
    title.className = "queue-item-title";
    title.textContent = item.file.name;

    const meta = document.createElement("p");
    meta.className = "queue-item-meta";
    const sizeLabel = item.width && item.height ? formatPixels(item.width, item.height) : "Reading size";
    meta.textContent = `${sizeLabel} • ${formatBytes(item.file.size)}`;

    copy.append(title, meta);

    const badge = document.createElement("span");
    badge.className = "queue-status-badge";
    badge.textContent = getItemStatusLabel(item);

    header.append(copy, badge);

    const track = document.createElement("div");
    track.className = "queue-progress";

    const fill = document.createElement("div");
    fill.className = "queue-progress-fill";
    fill.style.width = `${Math.max(4, item.progress || 0)}%`;
    if (item.status === READY_STATUS) {
      fill.style.width = "0%";
    }
    track.append(fill);

    const footer = document.createElement("div");
    footer.className = "queue-item-footer";

    const detail = document.createElement("span");
    detail.className = `queue-item-detail${item.status === "failed" ? " is-error" : ""}`;
    detail.textContent = getItemDetailText(item);

    footer.append(detail);

    if (item.outputUrl) {
      const download = document.createElement("a");
      download.className = "queue-download";
      download.href = item.outputUrl;
      download.download = item.outputUrl.split("/").pop() || "upscaled.png";
      download.textContent = "Download";
      download.addEventListener("click", (event) => event.stopPropagation());
      footer.append(download);
    }

    card.append(header, track, footer);
    elements.queueList.append(card);
  }
}

function enableDownloadButton(item) {
  if (!item.outputUrl) {
    disableDownloadButton();
    return;
  }

  elements.downloadButton.href = item.outputUrl;
  elements.downloadButton.download = item.outputUrl.split("/").pop() || "upscaled.png";
  elements.downloadButton.classList.remove("is-disabled");
  elements.downloadButton.removeAttribute("aria-disabled");
}

function getCompareDimensions(item) {
  return {
    width: Math.max(1, item.resultWidth || Math.round((item.width || 1) * Number(elements.factorInput.value))),
    height: Math.max(1, item.resultHeight || Math.round((item.height || 1) * Number(elements.factorInput.value))),
  };
}

function updateCompareTransform() {
  const item = getFocusedItem();
  if (!item || !item.outputUrl || elements.compareViewport.hidden) {
    return;
  }

  const metrics = getCompareMetrics(item);
  state.compare.offsetX = clamp(state.compare.offsetX, -metrics.maxOffsetX, metrics.maxOffsetX);
  state.compare.offsetY = clamp(state.compare.offsetY, -metrics.maxOffsetY, metrics.maxOffsetY);
  state.compare.canPan = metrics.maxOffsetX > 0 || metrics.maxOffsetY > 0;

  elements.compareCanvasPan.style.transform = `translate(calc(-50% + ${state.compare.offsetX}px), calc(-50% + ${state.compare.offsetY}px))`;
  elements.compareCanvas.style.transform = `scale(${metrics.effectiveScale})`;
  elements.compareCanvas.style.width = `${metrics.width}px`;
  elements.compareCanvas.style.height = `${metrics.height}px`;
  elements.compareCanvas.style.setProperty("--split-position", `${state.compare.split}%`);
  elements.compareViewport.classList.toggle("is-pannable", state.compare.canPan);
  elements.zoomLabel.textContent = `${Math.round(state.compare.zoom * 100)}%`;
  elements.compareDivider.value = `${Math.round(state.compare.split)}`;
}

function renderCompare() {
  const item = getFocusedItem();
  const canCompare = Boolean(item?.previewUrl && item?.outputUrl);
  updateCompareControlState(canCompare);

  if (!canCompare || !item) {
    elements.compareViewport.hidden = true;
    elements.comparePlaceholder.hidden = false;
    elements.compareHint.textContent = item
      ? item.status === "failed"
        ? item.error || "This item failed, so there is no compare view to inspect."
        : item.status === "running"
          ? "This image is still processing. The compare workspace unlocks as soon as the output is ready."
          : "Run the batch to unlock zoom, pan, and split view."
      : "Select images to build a batch, then pick a processed item to inspect.";
    return;
  }

  if (state.compare.boundItemId !== item.id) {
    state.compare.boundItemId = item.id;
    resetCompareState();
  }

  elements.compareBeforeImage.src = item.previewUrl;
  const afterSource = `${item.outputUrl}?t=${item.outputToken || 0}`;
  if (elements.compareAfterImage.getAttribute("src") !== afterSource) {
    elements.compareAfterImage.src = afterSource;
  }

  elements.compareViewport.hidden = false;
  elements.comparePlaceholder.hidden = true;
  elements.compareHint.textContent = `Viewing ${item.file.name}. Drag the divider to compare. Drag to pan only after zooming in enough to crop the image.`;
  requestAnimationFrame(updateCompareTransform);
}

function syncFocusedItem() {
  updateSummaryMetrics();
  updateEstimate();

  const item = getFocusedItem();
  if (!item) {
    elements.originalPreview.hidden = true;
    elements.originalPreview.removeAttribute("src");
    elements.originalPlaceholder.hidden = false;
    elements.originalMeta.textContent = "No image selected yet";
    elements.originalResolution.textContent = "Waiting for image";
    elements.targetResolution.textContent = "Waiting for image";

    elements.resultPreview.hidden = true;
    elements.resultPreview.removeAttribute("src");
    elements.resultPlaceholder.hidden = false;
    elements.resultMeta.textContent = "Run a batch to generate a result preview";
    elements.resultResolution.textContent = "Waiting to render";
    elements.resultPlaceholder.textContent = "When the selected item finishes its ESRGAN pass, the rendered output and download link appear here.";
    disableDownloadButton();
    renderCompare();
    return;
  }

  const sizeLabel = item.width && item.height ? formatPixels(item.width, item.height) : "Reading image size";
  elements.originalPreview.src = item.previewUrl;
  elements.originalPreview.hidden = false;
  elements.originalPlaceholder.hidden = true;
  elements.originalMeta.textContent = `${item.file.name} • ${sizeLabel} • ${formatBytes(item.file.size)}`;
  elements.originalResolution.textContent = sizeLabel;

  if (item.outputUrl) {
    elements.resultPreview.src = `${item.outputUrl}?t=${item.outputToken || 0}`;
    elements.resultPreview.hidden = false;
    elements.resultPlaceholder.hidden = true;
    elements.resultResolution.textContent = formatPixels(item.resultWidth, item.resultHeight);
    elements.resultMeta.textContent = `${formatPixels(item.resultWidth, item.resultHeight)} • ${getSelectedModel()?.label || item.modelKey || "ESRGAN output"}`;
    enableDownloadButton(item);
  } else {
    elements.resultPreview.hidden = true;
    elements.resultPreview.removeAttribute("src");
    elements.resultPlaceholder.hidden = false;
    elements.resultPlaceholder.textContent =
      item.status === "failed"
        ? item.error || "This item failed to render."
        : item.status === "running"
          ? `${item.message}. Keep this item selected to inspect the result as soon as it finishes.`
          : item.status === "queued"
            ? "This image is queued and waiting for its turn in the batch."
            : "Run the batch to generate a result preview for the selected image.";
    elements.resultMeta.textContent = item.status === READY_STATUS ? "Run a batch to generate a result preview" : item.message || "Waiting for render";
    elements.resultResolution.textContent =
      item.status === "running" ? `${item.progress}% rendered` : item.status === "failed" ? "Render failed" : "Waiting to render";
    disableDownloadButton();
  }

  renderCompare();
}

function extractErrorMessage(data, status) {
  if (typeof data.detail === "string") {
    return data.detail;
  }

  if (Array.isArray(data.detail)) {
    return data.detail
      .map((item) => (typeof item?.msg === "string" ? item.msg : JSON.stringify(item)))
      .join("; ");
  }

  return `Request failed with status ${status}.`;
}

async function readJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(extractErrorMessage(data, response.status));
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
    option.textContent = `${model.label} • ${formatFactor(model.native_scale)}`;
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
      setStatus("idle", state.items.length ? "Models refreshed. Ready when you are." : "Ready when you are.");
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

function mergeBatchSnapshot(snapshot) {
  const itemsByJobId = new Map((snapshot.items || []).map((item) => [item.id, item]));
  let focusCandidate = null;

  for (const item of state.items) {
    const serverItem = item.jobId ? itemsByJobId.get(item.jobId) : null;
    if (!serverItem) {
      continue;
    }

    item.status = serverItem.status || item.status;
    item.progress = serverItem.progress || 0;
    item.message = serverItem.message || item.message;
    item.width = serverItem.original_width || item.width;
    item.height = serverItem.original_height || item.height;
    item.resultWidth = serverItem.result_width || item.resultWidth;
    item.resultHeight = serverItem.result_height || item.resultHeight;
    item.error = serverItem.error || null;

    if (serverItem.output_url && item.outputUrl !== serverItem.output_url) {
      item.outputToken = Date.now();
    }
    item.outputUrl = serverItem.output_url || item.outputUrl;

    if (!focusCandidate && (item.status === "running" || item.status === "completed")) {
      focusCandidate = item;
    }
  }

  const focusedItem = getFocusedItem();
  if ((!focusedItem || (!focusedItem.outputUrl && focusedItem.status !== "running")) && focusCandidate) {
    state.focusedItemId = focusCandidate.id;
    state.compare.boundItemId = null;
  }
}

async function updateBatchStatus() {
  if (!state.batchId) {
    return;
  }

  try {
    const snapshot = await readJson(await fetch(`/api/batches/${state.batchId}`));
    mergeBatchSnapshot(snapshot);
    renderQueue();
    syncFocusedItem();
    setProgress(snapshot.progress || 0, snapshot.message || "Working");

    if (!TERMINAL_BATCH_STATUSES.has(snapshot.status)) {
      setStatus("working", snapshot.message || "Processing batch...");
      return;
    }

    clearPolling();
    state.busy = false;
    updateBusyState();

    if (snapshot.status === "completed") {
      const completedCount = snapshot.completed_count || 0;
      setStatus(
        "success",
        `Finished ${completedCount} ${pluralize(completedCount, "image")}. Select any queue card to inspect or download it.`
      );
    } else {
      setStatus("error", snapshot.message || "The batch finished with some errors.");
    }
  } catch (error) {
    clearPolling();
    state.busy = false;
    updateBusyState();
    setStatus("error", error.message || "Unable to read batch progress.");
  }
}

async function startUpscale(event) {
  event.preventDefault();
  if (state.busy) {
    return;
  }
  if (!state.items.length) {
    setStatus("error", "Choose one or more input images before starting.");
    return;
  }
  if (!elements.modelSelect.value) {
    setStatus("error", "Select a model from the model folder first.");
    return;
  }

  clearPolling();
  state.busy = true;
  state.batchId = null;
  state.compare.boundItemId = null;
  resetCompareState();

  for (const item of state.items) {
    item.status = "queued";
    item.progress = 0;
    item.message = "Queued for processing";
    item.jobId = null;
    item.outputUrl = null;
    item.resultWidth = null;
    item.resultHeight = null;
    item.error = null;
    item.outputToken = 0;
  }

  updateBusyState();
  renderQueue();
  syncFocusedItem();
  setStatus("working", `Uploading ${state.items.length} ${pluralize(state.items.length, "image")} and creating the batch...`);
  setProgress(5, "Creating batch");

  const formData = new FormData();
  for (const item of state.items) {
    formData.append("images", item.file, item.file.name);
  }
  formData.append("model", elements.modelSelect.value);
  formData.append("upscale_factor", elements.factorInput.value);
  formData.append("tile_size", elements.tileSelect.value);

  try {
    const payload = await readJson(
      await fetch("/api/upscale/batch", {
        method: "POST",
        body: formData,
      })
    );

    state.batchId = payload.batch_id;
    (payload.items || []).forEach((serverItem, index) => {
      const item = state.items[index];
      if (item) {
        item.jobId = serverItem.job_id;
      }
    });

    renderQueue();
    await updateBatchStatus();

    if (state.busy) {
      state.pollHandle = window.setInterval(updateBatchStatus, 700);
    }
  } catch (error) {
    state.busy = false;
    updateBusyState();
    for (const item of state.items) {
      item.status = READY_STATUS;
      item.message = "Ready to queue";
      item.progress = 0;
      item.jobId = null;
    }
    renderQueue();
    syncFocusedItem();
    setStatus("error", error.message || "Unable to start the batch.");
    setProgress(0, "Waiting for a job");
  }
}

function updateCompareZoom(nextZoom, focalPoint = null) {
  const clampedZoom = clamp(nextZoom, getMinimumZoom(), 6);
  if (clampedZoom === state.compare.zoom) {
    updateCompareTransform();
    return;
  }

  const item = getFocusedItem();
  if (focalPoint && item && !elements.compareViewport.hidden) {
    const metrics = getCompareMetrics(item);
    const centerX = metrics.rect.left + metrics.rect.width / 2 + state.compare.offsetX;
    const centerY = metrics.rect.top + metrics.rect.height / 2 + state.compare.offsetY;
    const ratio = clampedZoom / state.compare.zoom;
    state.compare.offsetX -= (focalPoint.x - centerX) * (ratio - 1);
    state.compare.offsetY -= (focalPoint.y - centerY) * (ratio - 1);
  }

  state.compare.zoom = clampedZoom;
  updateCompareTransform();
}

function handleCompareWheel(event) {
  if (elements.compareViewport.hidden) {
    return;
  }

  event.preventDefault();
  const factor = event.deltaY < 0 ? 1.12 : 0.88;
  updateCompareZoom(state.compare.zoom * factor, { x: event.clientX, y: event.clientY });
}

function startComparePan(event) {
  if (elements.compareViewport.hidden || !state.compare.canPan || state.compare.splitDragging) {
    return;
  }

  state.compare.dragging = true;
  state.compare.panPointerId = event.pointerId;
  state.compare.lastX = event.clientX;
  state.compare.lastY = event.clientY;
  elements.compareViewport.classList.add("is-dragging");
  elements.compareViewport.setPointerCapture(event.pointerId);
}

function moveComparePan(event) {
  if (!state.compare.dragging) {
    return;
  }

  state.compare.offsetX += event.clientX - state.compare.lastX;
  state.compare.offsetY += event.clientY - state.compare.lastY;
  state.compare.lastX = event.clientX;
  state.compare.lastY = event.clientY;
  updateCompareTransform();
}

function stopComparePan() {
  if (!state.compare.dragging) {
    return;
  }

  state.compare.dragging = false;
  elements.compareViewport.classList.remove("is-dragging");
  if (state.compare.panPointerId !== null) {
    try {
      elements.compareViewport.releasePointerCapture(state.compare.panPointerId);
    } catch {
      // Ignore release errors when the pointer capture is already cleared.
    }
  }
  state.compare.panPointerId = null;
}

function handleDrop(event) {
  event.preventDefault();
  elements.dropZone.classList.remove("is-dragover");
  const files = Array.from(event.dataTransfer?.files || []);
  if (!files.length) {
    return;
  }

  const transfer = new DataTransfer();
  for (const file of files) {
    transfer.items.add(file);
  }
  elements.imageInput.files = transfer.files;
  applyFiles(files);
}

elements.imageInput.addEventListener("change", () => {
  applyFiles(elements.imageInput.files);
});
elements.factorInput.addEventListener("input", () => {
  updateEstimate();
  if (!elements.compareViewport.hidden) {
    updateCompareTransform();
  }
});
elements.modelSelect.addEventListener("change", updateEstimate);
elements.allowSub100Zoom.addEventListener("change", () => {
  if (!elements.allowSub100Zoom.checked && state.compare.zoom < 1) {
    updateCompareZoom(1);
    return;
  }

  updateCompareTransform();
});
elements.refreshModelsButton.addEventListener("click", fetchModels);
elements.clearQueueButton.addEventListener("click", clearQueue);
elements.upscaleForm.addEventListener("submit", startUpscale);
elements.compareDivider.addEventListener("pointerdown", (event) => {
  event.stopPropagation();
});
elements.compareDivider.addEventListener("mousedown", (event) => {
  event.stopPropagation();
});
elements.compareDivider.addEventListener("input", () => {
  state.compare.split = Number(elements.compareDivider.value);
  updateCompareTransform();
});
elements.zoomOutButton.addEventListener("click", () => updateCompareZoom(state.compare.zoom / 1.15));
elements.zoomInButton.addEventListener("click", () => updateCompareZoom(state.compare.zoom * 1.15));
elements.resetViewButton.addEventListener("click", () => {
  resetCompareState(false);
  updateCompareTransform();
});
elements.compareViewport.addEventListener("wheel", handleCompareWheel, { passive: false });
elements.compareViewport.addEventListener("pointerdown", startComparePan);
elements.compareViewport.addEventListener("pointermove", moveComparePan);
elements.compareViewport.addEventListener("pointerup", stopComparePan);
elements.compareViewport.addEventListener("pointercancel", stopComparePan);
elements.compareViewport.addEventListener("lostpointercapture", stopComparePan);
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

window.addEventListener("resize", updateCompareTransform);
window.addEventListener("beforeunload", () => {
  clearPolling();
  revokeItemUrls();
});

updateCompareControlState(false);
updateBusyState();
renderQueue();
syncFocusedItem();
setProgress(0, "Waiting for a job");
fetchModels();