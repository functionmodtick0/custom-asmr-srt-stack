const state = {
  master: null,
  translated: null,
  selectedId: null,
  audioUrl: null,
  stopTimer: null,
  audioBuffer: null,
};

const els = {
  fileInput: document.getElementById("fileInput"),
  translatedInput: document.getElementById("translatedInput"),
  dropZone: document.getElementById("dropZone"),
  audioPlayer: document.getElementById("audioPlayer"),
  waveformCanvas: document.getElementById("waveformCanvas"),
  segmentList: document.getElementById("segmentList"),
  segmentCount: document.getElementById("segmentCount"),
  selectedLabel: document.getElementById("selectedLabel"),
  statusLabel: document.getElementById("statusLabel"),
  statusText: document.getElementById("statusText"),
  modelButton: document.getElementById("modelButton"),
  startButton: document.getElementById("startButton"),
  retranscribeButton: document.getElementById("retranscribeButton"),
  exportMasterButton: document.getElementById("exportMasterButton"),
  exportTranslationButton: document.getElementById("exportTranslationButton"),
  importTranslatedButton: document.getElementById("importTranslatedButton"),
  exportSrtButton: document.getElementById("exportSrtButton"),
  modelDialog: document.getElementById("modelDialog"),
  endpointInput: document.getElementById("endpointInput"),
  modelInput: document.getElementById("modelInput"),
  apiKeyInput: document.getElementById("apiKeyInput"),
  saveModelButton: document.getElementById("saveModelButton"),
};

function setStatus(label, text, isError = false) {
  els.statusLabel.textContent = label;
  els.statusText.textContent = text;
  els.statusLabel.style.color = isError ? "var(--danger)" : "var(--accent-strong)";
}

async function apiPost(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.error || `request failed: ${response.status}`);
  }
  return body;
}

function downloadJson(filename, value) {
  downloadText(filename, `${JSON.stringify(value, null, 2)}\n`, "application/json");
}

function downloadText(filename, content, type = "text/plain") {
  const blob = new Blob([content], { type: `${type};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function msToClock(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const milliseconds = String(ms % 1000).padStart(3, "0");
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${milliseconds}`;
}

function loadModelSettings() {
  const raw = localStorage.getItem("customAsmrModelSettings");
  if (!raw) return;
  try {
    const settings = JSON.parse(raw);
    els.endpointInput.value = settings.endpoint || "";
    els.modelInput.value = settings.model || "";
    els.apiKeyInput.value = settings.apiKey || "";
  } catch {
    localStorage.removeItem("customAsmrModelSettings");
  }
}

function saveModelSettings() {
  localStorage.setItem(
    "customAsmrModelSettings",
    JSON.stringify({
      endpoint: els.endpointInput.value.trim(),
      model: els.modelInput.value.trim(),
      apiKey: els.apiKeyInput.value,
    }),
  );
  els.modelDialog.close();
  setStatus("저장됨", "모델 설정을 저장했습니다.");
}

function setMaster(master, label) {
  state.master = master;
  state.translated = null;
  state.selectedId = master.segments[0]?.id || null;
  render();
  drawWaveform();
  setStatus("로드됨", label);
}

async function handleFile(file) {
  const lowerName = file.name.toLowerCase();
  if (lowerName.endsWith(".srt")) {
    const content = await file.text();
    const master = await apiPost("/api/srt-to-json", {
      content,
      source_language: "ja",
      source_file: file.name,
    });
    setMaster(master, `${file.name}에서 ${master.segments.length}개 segment를 만들었습니다.`);
    return;
  }

  if (lowerName.endsWith(".json")) {
    const parsed = JSON.parse(await file.text());
    if (parsed.format === "custom-asmr-master-v1") {
      setMaster(parsed, `${file.name}을 열었습니다.`);
      return;
    }
    if (parsed.format === "custom-asmr-translated-v1") {
      state.translated = parsed;
      render();
      setStatus("번역 로드됨", `${file.name}을 가져왔습니다.`);
      return;
    }
    throw new Error("지원하지 않는 JSON format입니다.");
  }

  if (file.type.startsWith("audio/")) {
    await loadAudio(file);
    setStatus("오디오 로드됨", file.name);
    return;
  }

  throw new Error("지원하지 않는 파일입니다.");
}

async function loadAudio(file) {
  if (state.audioUrl) {
    URL.revokeObjectURL(state.audioUrl);
  }
  state.audioUrl = URL.createObjectURL(file);
  els.audioPlayer.src = state.audioUrl;

  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContext();
    state.audioBuffer = await context.decodeAudioData(await file.arrayBuffer());
    await context.close();
  } catch {
    state.audioBuffer = null;
  }
  drawWaveform();
}

function render() {
  const segments = state.master?.segments || [];
  els.segmentCount.textContent = `${segments.length} segments`;
  els.selectedLabel.textContent = state.selectedId ? state.selectedId : "선택 없음";
  els.retranscribeButton.disabled = !state.selectedId;
  els.exportMasterButton.disabled = !state.master;
  els.exportTranslationButton.disabled = !state.master;
  els.exportSrtButton.disabled = !state.master;

  if (!segments.length) {
    els.segmentList.innerHTML = '<div class="empty-state">segment 없음</div>';
    return;
  }

  els.segmentList.replaceChildren(...segments.map(renderSegment));
}

function renderSegment(segment) {
  const row = document.createElement("div");
  row.className = `segment-row${segment.id === state.selectedId ? " is-selected" : ""}`;
  row.dataset.id = segment.id;

  const time = document.createElement("div");
  time.className = "segment-time";
  time.innerHTML = `<div>${msToClock(segment.start_ms)}</div><div>${msToClock(segment.end_ms)}</div>`;

  const meta = document.createElement("div");
  meta.className = "segment-meta";
  const channel = document.createElement("span");
  channel.className = "channel";
  channel.textContent = segment.channel;
  const kind = document.createElement("span");
  kind.className = "kind";
  kind.textContent = segment.kind;
  meta.append(channel, kind);

  const text = document.createElement("textarea");
  text.className = "segment-text";
  text.value = segment.text;
  text.addEventListener("input", () => {
    segment.text = text.value;
  });
  text.addEventListener("focus", () => selectSegment(segment.id, false));

  row.addEventListener("click", (event) => {
    if (event.target !== text) {
      selectSegment(segment.id, true);
    }
  });
  row.append(time, meta, text);
  return row;
}

function selectSegment(id, play) {
  state.selectedId = id;
  render();
  drawWaveform();
  const segment = state.master?.segments.find((item) => item.id === id);
  if (play && segment && els.audioPlayer.src) {
    playSegment(segment);
  }
}

function playSegment(segment) {
  window.clearTimeout(state.stopTimer);
  els.audioPlayer.currentTime = segment.start_ms / 1000;
  els.audioPlayer.play();
  state.stopTimer = window.setTimeout(() => {
    els.audioPlayer.pause();
  }, Math.max(0, segment.end_ms - segment.start_ms));
}

function drawWaveform() {
  const canvas = els.waveformCanvas;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#0b0d0c";
  ctx.fillRect(0, 0, rect.width, rect.height);

  if (state.audioBuffer) {
    drawAudioBuffer(ctx, rect.width, rect.height);
  } else {
    drawSegmentTimeline(ctx, rect.width, rect.height);
  }
}

function drawAudioBuffer(ctx, width, height) {
  const data = state.audioBuffer.getChannelData(0);
  const step = Math.max(1, Math.floor(data.length / width));
  ctx.strokeStyle = "#7dd9b9";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    let peak = 0;
    for (let i = 0; i < step; i += 1) {
      peak = Math.max(peak, Math.abs(data[x * step + i] || 0));
    }
    const y = (height / 2) * peak;
    ctx.moveTo(x, height / 2 - y);
    ctx.lineTo(x, height / 2 + y);
  }
  ctx.stroke();
  drawSegmentTimeline(ctx, width, height);
}

function drawSegmentTimeline(ctx, width, height) {
  const segments = state.master?.segments || [];
  if (!segments.length) {
    ctx.fillStyle = "#343a34";
    ctx.fillRect(24, height / 2 - 1, width - 48, 2);
    return;
  }

  const duration = Math.max(...segments.map((segment) => segment.end_ms), state.master?.audio?.duration_ms || 0);
  for (const segment of segments) {
    const x = (segment.start_ms / duration) * width;
    const w = Math.max(2, ((segment.end_ms - segment.start_ms) / duration) * width);
    const isSelected = segment.id === state.selectedId;
    ctx.fillStyle = isSelected ? "rgba(168, 240, 215, 0.95)" : "rgba(125, 217, 185, 0.32)";
    ctx.fillRect(x, height - 34, w, 14);
  }
}

async function exportTranslationJson() {
  const exported = await apiPost("/api/export-translation-json", { master: state.master });
  downloadJson("translation.json", exported);
  setStatus("내보냄", "translation.json을 만들었습니다.");
}

async function exportSrt() {
  const response = await apiPost("/api/json-to-srt", {
    master: state.master,
    translated: state.translated,
  });
  downloadText("export.srt", response.content, "application/x-subrip");
  setStatus("내보냄", "export.srt를 만들었습니다.");
}

async function safeRun(task) {
  try {
    await task();
  } catch (error) {
    setStatus("오류", error.message || String(error), true);
  }
}

els.fileInput.addEventListener("change", () => {
  const file = els.fileInput.files?.[0];
  if (file) safeRun(() => handleFile(file));
  els.fileInput.value = "";
});

els.translatedInput.addEventListener("change", () => {
  const file = els.translatedInput.files?.[0];
  if (file) safeRun(() => handleFile(file));
  els.translatedInput.value = "";
});

for (const eventName of ["dragenter", "dragover"]) {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.classList.add("is-over");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("is-over");
  });
}

els.dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) safeRun(() => handleFile(file));
});

els.modelButton.addEventListener("click", () => els.modelDialog.showModal());
els.saveModelButton.addEventListener("click", saveModelSettings);
els.importTranslatedButton.addEventListener("click", () => els.translatedInput.click());
els.exportMasterButton.addEventListener("click", () => {
  if (state.master) downloadJson("master.json", state.master);
});
els.exportTranslationButton.addEventListener("click", () => safeRun(exportTranslationJson));
els.exportSrtButton.addEventListener("click", () => safeRun(exportSrt));
els.startButton.addEventListener("click", () => {
  const model = els.modelInput.value.trim();
  const endpoint = els.endpointInput.value.trim();
  if (!model || !endpoint) {
    setStatus("모델 필요", "모델 설정에서 Endpoint URL과 Model ID를 입력하세요.", true);
    return;
  }
  setStatus("대기", "전사 어댑터는 다음 개발 단위에서 연결됩니다.");
});
els.retranscribeButton.addEventListener("click", () => {
  if (!state.selectedId) return;
  setStatus("대기", `${state.selectedId} 재전사는 다음 개발 단위에서 연결됩니다.`);
});

window.addEventListener("resize", drawWaveform);
loadModelSettings();
render();
drawWaveform();
