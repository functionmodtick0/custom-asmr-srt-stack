const state = {
  projectId: null,
  master: null,
  translated: null,
  selectedId: null,
  audioUrl: null,
  hasAudio: false,
  stopTimer: null,
  audioBuffer: null,
  saveTimer: null,
  reviewPack: null,
  reviewPackSelectedIndex: null,
  reviewCaseSet: null,
  reviewCaseReference: null,
};

const els = {
  fileInput: document.getElementById("fileInput"),
  translatedInput: document.getElementById("translatedInput"),
  dropZone: document.getElementById("dropZone"),
  audioPlayer: document.getElementById("audioPlayer"),
  reviewPackPathInput: document.getElementById("reviewPackPathInput"),
  loadReviewPackButton: document.getElementById("loadReviewPackButton"),
  waveformCanvas: document.getElementById("waveformCanvas"),
  segmentList: document.getElementById("segmentList"),
  segmentCount: document.getElementById("segmentCount"),
  selectedLabel: document.getElementById("selectedLabel"),
  statusLabel: document.getElementById("statusLabel"),
  statusText: document.getElementById("statusText"),
  modelButton: document.getElementById("modelButton"),
  startButton: document.getElementById("startButton"),
  retranscribeButton: document.getElementById("retranscribeButton"),
  sourceCaseButton: document.getElementById("sourceCaseButton"),
  reviewDoneButton: document.getElementById("reviewDoneButton"),
  caseListButton: document.getElementById("caseListButton"),
  nextCaseButton: document.getElementById("nextCaseButton"),
  exportMasterButton: document.getElementById("exportMasterButton"),
  exportTranslationButton: document.getElementById("exportTranslationButton"),
  importTranslatedButton: document.getElementById("importTranslatedButton"),
  exportSrtButton: document.getElementById("exportSrtButton"),
  modelDialog: document.getElementById("modelDialog"),
  adapterInput: document.getElementById("adapterInput"),
  endpointInput: document.getElementById("endpointInput"),
  modelInput: document.getElementById("modelInput"),
  apiKeyInput: document.getElementById("apiKeyInput"),
  validateModelButton: document.getElementById("validateModelButton"),
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

function loadModelSettings() {
  const raw = localStorage.getItem("customAsmrModelSettings");
  if (!raw) {
    syncModelFormForAdapter();
    return;
  }
  try {
    const settings = JSON.parse(raw);
    els.adapterInput.value = settings.adapter || "openai-compatible";
    els.endpointInput.value = settings.endpoint || "";
    els.modelInput.value = settings.model || "";
    els.apiKeyInput.value = settings.apiKey || "";
    syncModelFormForAdapter();
  } catch {
    localStorage.removeItem("customAsmrModelSettings");
    syncModelFormForAdapter();
  }
}

function saveModelSettings() {
  localStorage.setItem(
    "customAsmrModelSettings",
    JSON.stringify({
      endpoint: els.endpointInput.value.trim(),
      adapter: els.adapterInput.value,
      model: els.modelInput.value.trim(),
      apiKey: els.apiKeyInput.value,
    }),
  );
  els.modelDialog.close();
  setStatus("저장됨", "모델 설정을 저장했습니다.");
}

async function validateModelSettings() {
  const model = getModelSettings();
  const result = await apiPost("/api/model/validate", { model });
  setStatus("확인됨", `${result.adapter} / ${result.model_id}`);
}

function getModelSettings() {
  return {
    adapter: els.adapterInput.value,
    endpoint_url: els.endpointInput.value.trim(),
    model_id: els.modelInput.value.trim(),
    api_key: els.apiKeyInput.value,
  };
}

function isLocalAdapter(adapter) {
  return ["local-transformers", "local-qwen-asr", "local-qwen-hf-asr", "local-cohere-asr", "local-granite-asr"].includes(
    adapter,
  );
}

function modelSettingsReady(model) {
  if (!model.model_id) return false;
  return isLocalAdapter(model.adapter) || Boolean(model.endpoint_url);
}

function modelSettingsRequirementText(model) {
  if (isLocalAdapter(model.adapter)) {
    return "모델 설정에서 Model ID를 입력하세요.";
  }
  return "모델 설정에서 Endpoint URL과 Model ID를 입력하세요.";
}

function syncModelFormForAdapter() {
  const isLocal = isLocalAdapter(els.adapterInput.value);
  els.endpointInput.disabled = isLocal;
  els.apiKeyInput.disabled = isLocal;
  els.endpointInput.placeholder = isLocal ? "사용하지 않음" : "http://127.0.0.1:8000/v1";
  els.modelInput.placeholder = modelPlaceholderForAdapter(els.adapterInput.value);
  if (isLocal) {
    els.endpointInput.value = "";
    els.apiKeyInput.value = "";
  }
}

function modelPlaceholderForAdapter(adapter) {
  if (adapter === "local-qwen-asr") return "Qwen/Qwen3-ASR-1.7B";
  if (adapter === "local-qwen-hf-asr") return "/models/qwen3-asr-1.7b-hf";
  if (adapter === "local-cohere-asr") return "/models/cohere-transcribe-03-2026";
  if (adapter === "local-granite-asr") return ".casrt/models/granite-speech-4.1-2b-de575db64086f84fdc79da4932d1076e965bc546";
  if (adapter === "local-transformers") return "google/gemma-4-E4B-it";
  return "gemma-4-e4b";
}

function setMaster(master, label, projectId = state.projectId, hasAudio = state.hasAudio) {
  state.projectId = projectId;
  state.hasAudio = hasAudio;
  state.master = master;
  state.translated = null;
  state.reviewPack = null;
  state.reviewPackSelectedIndex = null;
  state.reviewCaseSet = null;
  state.reviewCaseReference = null;
  state.selectedId = firstReviewOrFirstSegmentId(master);
  render();
  drawWaveform();
  setStatus("로드됨", label);
}

function firstReviewOrFirstSegmentId(master) {
  return master.segments.find((segment) => segment.needs_review)?.id || master.segments[0]?.id || null;
}

async function handleFile(file) {
  const lowerName = file.name.toLowerCase();
  if (lowerName.endsWith(".srt")) {
    const content = await file.text();
    if (state.projectId && state.hasAudio && !state.master) {
      const master = await apiPost("/api/srt-to-json", {
        content,
        source_language: "ja",
        source_file: file.name,
      });
      const saved = await apiPost("/api/projects/save-master", {
        project_id: state.projectId,
        master,
      });
      setMaster(
        saved.master,
        `${file.name}에서 ${saved.master.segments.length}개 segment를 현재 오디오 project에 연결했습니다.`,
        state.projectId,
        true,
      );
      return;
    }
    const project = await apiPost("/api/projects/import-srt", {
      content,
      source_language: "ja",
      source_file: file.name,
    });
    setMaster(
      project.master,
      `${file.name}에서 ${project.master.segments.length}개 segment를 만들었습니다.`,
      project.project_id,
      Boolean(project.metadata?.has_audio),
    );
    return;
  }

  if (lowerName.endsWith(".json")) {
    const parsed = JSON.parse(await file.text());
    if (parsed.format === "custom-asmr-master-v1") {
      if (state.projectId && state.hasAudio && !state.master) {
        const saved = await apiPost("/api/projects/save-master", {
          project_id: state.projectId,
          master: parsed,
        });
        setMaster(saved.master, `${file.name}을 현재 오디오 project에 연결했습니다.`, state.projectId, true);
        return;
      }
      const project = await apiPost("/api/projects/import-master-json", { master: parsed });
      setMaster(project.master, `${file.name}을 열었습니다.`, project.project_id, Boolean(project.metadata?.has_audio));
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
    const project = await apiPost("/api/projects/upload-audio", {
      file_name: file.name,
      mime_type: file.type || "application/octet-stream",
      content_base64: await fileToBase64(file),
    });
    state.projectId = project.project_id;
    state.hasAudio = true;
    state.master = null;
    state.translated = null;
    state.reviewPack = null;
    state.reviewPackSelectedIndex = null;
    state.reviewCaseSet = null;
    state.reviewCaseReference = null;
    state.selectedId = null;
    render();
    setStatus("오디오 로드됨", `${file.name} project를 만들었습니다.`);
    return;
  }

  throw new Error("지원하지 않는 파일입니다.");
}

async function fileToBase64(file) {
  const buffer = await file.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
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
  if (state.reviewPack) {
    renderReviewPack();
    return;
  }
  if (state.reviewCaseSet && !state.reviewCaseReference) {
    renderReviewCaseSet();
    return;
  }

  const segments = state.master?.segments || [];
  els.segmentCount.textContent = state.reviewCaseReference
    ? `${state.reviewCaseReference.caseId} · ${segments.length} segments`
    : `${segments.length} segments`;
  els.selectedLabel.textContent = state.selectedId ? state.selectedId : "선택 없음";
  els.retranscribeButton.hidden = Boolean(state.reviewCaseReference);
  els.sourceCaseButton.hidden = true;
  els.sourceCaseButton.disabled = true;
  els.reviewDoneButton.hidden = !state.reviewCaseReference;
  els.caseListButton.hidden = !state.reviewCaseReference;
  els.nextCaseButton.hidden = !state.reviewCaseReference;
  els.nextCaseButton.disabled = nextReviewCaseIndex() === null;
  updateSelectedActionState();
  els.exportMasterButton.disabled = !state.master;
  els.exportTranslationButton.disabled = !state.master;
  els.exportSrtButton.disabled = !state.master;

  if (!segments.length) {
    els.segmentList.innerHTML = '<div class="empty-state">segment 없음</div>';
    return;
  }

  els.segmentList.replaceChildren(...segments.map(renderSegment));
}

function renderReviewPack() {
  const items = state.reviewPack.items || [];
  const selectedItem = items[state.reviewPackSelectedIndex];
  const sourceItem = reviewPackSelectedOrDefaultSourceItem();
  const caseCount = state.reviewPack.case_count;
  const nextCaseId = state.reviewPack.next_case_id;
  const summaryParts = [`${items.length} review clips`];
  if (Number.isInteger(caseCount)) {
    summaryParts.push(`${caseCount} cases`);
  }
  summaryParts.push(...reviewPackDurationSummaryParts(state.reviewPack.duration_summary));
  if (nextCaseId) {
    summaryParts.push(`next ${nextCaseId}`);
  }
  els.segmentCount.textContent = summaryParts.join(" · ");
  els.selectedLabel.textContent =
    state.reviewPackSelectedIndex === null ? "선택 없음" : reviewPackSelectedLabel(selectedItem);
  els.retranscribeButton.disabled = true;
  els.retranscribeButton.hidden = false;
  els.sourceCaseButton.hidden = !items.some((item) => reviewPackSourceTarget(item));
  els.sourceCaseButton.disabled = !reviewPackSourceTarget(sourceItem);
  els.reviewDoneButton.hidden = true;
  els.reviewDoneButton.disabled = true;
  els.caseListButton.hidden = true;
  els.nextCaseButton.hidden = true;
  els.exportMasterButton.disabled = true;
  els.exportTranslationButton.disabled = true;
  els.exportSrtButton.disabled = true;

  if (!items.length) {
    els.segmentList.innerHTML = '<div class="empty-state">review clip 없음</div>';
    return;
  }

  els.segmentList.replaceChildren(...items.map(renderReviewPackItem));
}

function renderReviewCaseSet() {
  const items = state.reviewCaseSet.items || [];
  const reviewFlagCount = items.reduce((total, item) => total + (item.review_count || 0), 0);
  const reviewDurationMs = items.reduce((total, item) => total + (item.review_duration_ms || 0), 0);
  els.segmentCount.textContent = `${items.length} review cases · ${reviewFlagCount} flags · ${formatDuration(reviewDurationMs)}`;
  els.selectedLabel.textContent = "case 선택";
  els.retranscribeButton.disabled = true;
  els.retranscribeButton.hidden = false;
  els.sourceCaseButton.hidden = true;
  els.sourceCaseButton.disabled = true;
  els.reviewDoneButton.hidden = true;
  els.reviewDoneButton.disabled = true;
  els.caseListButton.hidden = true;
  els.nextCaseButton.hidden = true;
  els.exportMasterButton.disabled = true;
  els.exportTranslationButton.disabled = true;
  els.exportSrtButton.disabled = true;

  if (!items.length) {
    els.segmentList.innerHTML = '<div class="empty-state">review case 없음</div>';
    return;
  }

  els.segmentList.replaceChildren(...items.map(renderReviewCaseItem));
}

function renderReviewPackItem(item, index) {
  const row = document.createElement("div");
  row.className = `review-pack-row${index === state.reviewPackSelectedIndex ? " is-selected" : ""}`;
  row.dataset.index = String(index);

  const meta = document.createElement("div");
  meta.className = "review-pack-meta";
  meta.append(
    textBlock("review-rank", `#${item.priority_rank || index + 1}`),
    textBlock("review-detail", formatMsRange(item.start_ms, item.end_ms)),
    textBlock("review-detail", item.case_id || "single"),
    textBlock("review-detail", item.reference_id || "ref -"),
  );

  const reasons = document.createElement("div");
  reasons.className = "review-reasons";
  reasons.append(
    textBlock("review-detail", formatScore(item.priority_score)),
    textBlock("reason-list", Array.isArray(item.reasons) ? item.reasons.join(" / ") : "reason 없음"),
  );

  const texts = document.createElement("div");
  texts.className = "review-pack-texts";
  texts.append(reviewTextLine("REF", item.reference_channel, item.reference_text));
  if (reviewPackHasCandidate(item)) {
    texts.append(
      reviewTextLine(
        reviewPackSecondaryLabel(item),
        item.candidate_channel,
        reviewPackSecondaryText(item),
      ),
    );
  }

  row.addEventListener("click", () => selectReviewPackItem(index, true));
  row.append(meta, reasons, texts);
  return row;
}

function renderReviewCaseItem(item, index) {
  const row = document.createElement("div");
  row.className = `review-case-row${(item.review_count || 0) > 0 ? " needs-review" : ""}`;
  row.dataset.index = String(index);

  const meta = document.createElement("div");
  meta.className = "review-pack-meta";
  meta.append(
    textBlock("review-rank", item.id || `case ${index + 1}`),
    textBlock("review-detail", item.reference_type || state.reviewCaseSet.reference_type || "unspecified"),
    textBlock("review-detail", formatDuration(item.duration_ms)),
  );

  const counts = document.createElement("div");
  counts.className = "review-reasons";
  counts.append(
    textBlock("review-detail", `${item.segments ?? 0} segments`),
    textBlock("reason-list", `${item.review_count ?? 0} review flags`),
    textBlock("review-detail", formatDuration(item.review_duration_ms)),
  );

  const source = document.createElement("div");
  source.className = "review-pack-texts";
  const nextReview = firstReviewSegment(item);
  source.append(
    reviewTextLine("AUDIO", null, item.audio),
    reviewTextLine("REF", null, item.reference),
    reviewTextLine("NEXT", nextReview?.channel, reviewSegmentPreview(nextReview)),
  );

  row.addEventListener("click", () => loadReviewCaseItem(index));
  row.append(meta, counts, source);
  return row;
}

function firstReviewSegment(item) {
  const segments = item?.reference_master?.segments;
  if (!Array.isArray(segments)) return null;
  return segments.find((segment) => segment?.needs_review) || null;
}

function reviewSegmentPreview(segment) {
  if (!segment) return "검수 flag 없음";
  return `${formatMsRange(segment.start_ms, segment.end_ms)} · ${segment.text || "-"}`;
}

function reviewPackHasCandidate(item) {
  return Boolean(item?.candidate_id || item?.candidate_channel || item?.candidate_text);
}

function reviewPackIsReferenceAuditItem(item) {
  return Array.isArray(item?.reasons)
    && item.reasons.some((reason) => typeof reason === "string" && reason.startsWith("reference-"));
}

function reviewPackIsReferenceChannelAuditItem(item) {
  return Array.isArray(item?.reasons)
    && item.reasons.some((reason) => typeof reason === "string" && reason.startsWith("reference-channel-energy-"));
}

function reviewPackSecondaryLabel(item) {
  if (reviewPackIsReferenceChannelAuditItem(item)) return "ENERGY";
  return reviewPackIsReferenceAuditItem(item) ? "REF2" : "CAND";
}

function reviewPackSecondaryText(item) {
  if (reviewPackIsReferenceChannelAuditItem(item)) {
    return reviewPackChannelEvidenceText(item);
  }
  if (reviewPackIsReferenceAuditItem(item) && !item?.candidate_text) {
    return item?.candidate_id;
  }
  return item?.candidate_text;
}

function reviewPackChannelEvidenceText(item) {
  const parts = [];
  const left = formatDbfs(item?.left_dbfs);
  const right = formatDbfs(item?.right_dbfs);
  const delta = formatDb(item?.delta_db);
  if (left) parts.push(`L ${left}`);
  if (right) parts.push(`R ${right}`);
  if (delta) parts.push(`delta ${delta}`);
  return parts.join(" · ") || item?.candidate_text || item?.candidate_id;
}

function reviewPackSourceHintText(item) {
  if (!reviewPackIsReferenceAuditItem(item)) return null;
  const reasonText = Array.isArray(item?.reasons) ? item.reasons.join(" / ") : "reference audit";
  if (reviewPackIsReferenceChannelAuditItem(item)) {
    const verdict = item?.candidate_channel ? `ENERGY ${item.candidate_channel}` : "ENERGY -";
    return `${item.case_id || "case -"}/${item.reference_id || "ref -"} · ${verdict} · ${reviewPackChannelEvidenceText(item)}`;
  }
  return `${item.case_id || "case -"}/${item.reference_id || "ref -"} · ${reasonText} · ${reviewPackReferenceAuditEvidenceText(item)}`;
}

function reviewPackReferenceAuditEvidenceText(item) {
  const parts = [];
  if (item?.reference_channel) parts.push(`REF ${item.reference_channel}`);
  if (item?.candidate_id || item?.candidate_channel) {
    const channel = item?.candidate_channel ? ` ${item.candidate_channel}` : "";
    parts.push(`REF2${channel} ${item?.candidate_id || "ref -"}`);
  }
  if (Number.isFinite(item?.overlap_ms)) {
    parts.push(`overlap ${formatDuration(item.overlap_ms)}`);
  } else if (Number.isFinite(item?.start_ms) && Number.isFinite(item?.end_ms)) {
    parts.push(`duration ${formatDuration(item.end_ms - item.start_ms)}`);
  }
  return parts.join(" · ") || formatMsRange(item?.start_ms, item?.end_ms);
}

function reviewPackSecondaryReferenceId(item) {
  if (reviewPackIsReferenceChannelAuditItem(item)) return null;
  if (!reviewPackIsReferenceAuditItem(item)) return null;
  return typeof item?.candidate_id === "string" && item.candidate_id ? item.candidate_id : null;
}

function formatDbfs(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} dBFS` : null;
}

function formatDb(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} dB` : null;
}

function reviewPackSourceTarget(item) {
  const caseIndexPath = item?.source_case_index || state.reviewPack?.source_case_index;
  if (!caseIndexPath || !item?.case_id || !item?.reference_id) return null;
  return {
    caseIndexPath,
    caseId: item.case_id,
    segmentId: item.reference_id,
  };
}

function reviewPackSelectedOrDefaultSourceItem() {
  const selected = state.reviewPack?.items?.[state.reviewPackSelectedIndex];
  if (reviewPackSourceTarget(selected)) return selected;
  if (state.reviewPackSelectedIndex !== null) return null;
  const items = state.reviewPack?.items || [];
  const nextCaseId = state.reviewPack?.next_case_id;
  if (nextCaseId) {
    const nextCaseItem = items.find((item) => item?.case_id === nextCaseId && reviewPackSourceTarget(item));
    if (nextCaseItem) return nextCaseItem;
  }
  return null;
}

function textBlock(className, value) {
  const block = document.createElement("span");
  block.className = className;
  block.textContent = value === undefined || value === null || value === "" ? "-" : String(value);
  return block;
}

function reviewTextLine(label, channel, text) {
  const line = document.createElement("p");
  line.className = "review-text-line";
  const marker = document.createElement("span");
  marker.textContent = channel ? `${label} ${channel}` : label;
  const body = document.createElement("strong");
  body.textContent = text || "-";
  line.append(marker, body);
  return line;
}

function reviewPackSelectedLabel(item) {
  if (!item) return "선택 없음";
  return `#${item.priority_rank || state.reviewPackSelectedIndex + 1} ${formatMsRange(item.start_ms, item.end_ms)}`;
}

function formatScore(score) {
  return typeof score === "number" ? `score ${score.toFixed(1)}` : "score -";
}

function formatDuration(ms) {
  return Number.isFinite(ms) ? formatMs(ms) : "duration -";
}

function reviewPackDurationSummaryParts(summary) {
  if (!summary || typeof summary !== "object") return [];
  const clipDurationMs = finiteNumber(summary.clip_duration_ms_sum);
  if (!Number.isFinite(clipDurationMs)) return [];
  const parts = [`listen ${formatDuration(clipDurationMs)}`];
  const sourceDurationMs = finiteNumber(summary.source_item_duration_ms_sum);
  const effectiveDurationMs = finiteNumber(summary.effective_item_duration_ms_sum);
  const focusItemCount = finiteNumber(summary.focus_item_count);
  if (
    Number.isFinite(sourceDurationMs) &&
    Number.isFinite(effectiveDurationMs) &&
    Number.isFinite(focusItemCount) &&
    focusItemCount > 0 &&
    effectiveDurationMs !== sourceDurationMs
  ) {
    parts.push(`focus ${formatDuration(effectiveDurationMs)}/${formatDuration(sourceDurationMs)}`);
  }
  return parts;
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatMsRange(startMs, endMs) {
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return "time -";
  return `${formatMs(startMs)} - ${formatMs(endMs)}`;
}

function formatMs(value) {
  const total = Math.max(0, Math.round(value));
  const minutes = Math.floor(total / 60000);
  const seconds = Math.floor((total % 60000) / 1000);
  const millis = total % 1000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function renderSegment(segment) {
  const row = document.createElement("div");
  row.className = `segment-row${segment.id === state.selectedId ? " is-selected" : ""}${
    segment.needs_review ? " needs-review" : ""
  }${
    segment.id === state.reviewCaseReference?.secondarySegmentId ? " is-secondary-reference" : ""
  }`;
  row.dataset.id = segment.id;

  const time = document.createElement("div");
  time.className = "segment-time";
  time.append(
    renderTimeInput(segment, "start_ms", "S"),
    renderTimeInput(segment, "end_ms", "E"),
  );

  const meta = document.createElement("div");
  meta.className = "segment-meta";
  const channel = document.createElement("select");
  channel.className = "channel-select";
  for (const value of ["L", "R", "MIX"]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    channel.append(option);
  }
  channel.value = ["L", "R", "MIX"].includes(segment.channel) ? segment.channel : "MIX";
  channel.addEventListener("change", () => {
    selectSegment(segment.id, false);
    segment.channel = channel.value;
    scheduleSaveMaster();
    drawWaveform();
  });
  const kind = document.createElement("span");
  kind.className = "kind";
  kind.textContent = segment.kind;
  const review = document.createElement("label");
  review.className = "review-toggle";
  const reviewInput = document.createElement("input");
  reviewInput.type = "checkbox";
  reviewInput.checked = Boolean(segment.needs_review);
  reviewInput.addEventListener("change", () => {
    selectSegment(segment.id, false);
    segment.needs_review = reviewInput.checked;
    row.classList.toggle("needs-review", segment.needs_review);
    updateSelectedActionState();
    scheduleSaveMaster();
  });
  const reviewText = document.createElement("span");
  reviewText.textContent = "검수";
  review.append(reviewInput, reviewText);
  meta.append(channel, kind, review);

  const text = document.createElement("textarea");
  text.className = "segment-text";
  text.value = segment.text;
  text.addEventListener("input", () => {
    segment.text = text.value;
    scheduleSaveMaster();
  });
  text.addEventListener("focus", () => selectSegment(segment.id, false));

  row.addEventListener("click", (event) => {
    if (!isInteractiveTarget(event.target)) {
      selectSegment(segment.id, true);
    }
  });
  row.append(time, meta, text);
  return row;
}

function renderTimeInput(segment, key, labelText) {
  const label = document.createElement("label");
  label.className = "time-input";
  const marker = document.createElement("span");
  marker.textContent = labelText;
  const input = document.createElement("input");
  input.type = "number";
  input.min = "0";
  input.step = "1";
  input.value = String(segment[key]);
  input.addEventListener("change", () => commitSegmentTime(segment, key, input));
  input.addEventListener("focus", () => selectSegment(segment.id, false));
  label.append(marker, input);
  return label;
}

function commitSegmentTime(segment, key, input) {
  const nextValue = Number(input.value);
  const nextStart = key === "start_ms" ? nextValue : segment.start_ms;
  const nextEnd = key === "end_ms" ? nextValue : segment.end_ms;
  if (!Number.isInteger(nextValue) || nextValue < 0 || nextEnd <= nextStart) {
    input.value = String(segment[key]);
    setStatus("시간 오류", "start/end ms를 확인하세요.", true);
    return;
  }
  segment[key] = nextValue;
  scheduleSaveMaster();
  drawWaveform();
}

function isInteractiveTarget(target) {
  return target instanceof Element && Boolean(target.closest("textarea, input, select, button, label"));
}

function selectSegment(id, play) {
  state.selectedId = id;
  syncSelectedSegment();
  drawWaveform();
  const segment = state.master?.segments.find((item) => item.id === id);
  if (play && segment && els.audioPlayer.src) {
    playSegment(segment);
  }
}

function selectReviewPackItem(index, play) {
  state.reviewPackSelectedIndex = index;
  syncSelectedReviewPackItem();
  const item = state.reviewPack?.items?.[index];
  if (play && item?.clip_url) {
    window.clearTimeout(state.stopTimer);
    els.audioPlayer.src = item.clip_url;
    els.audioPlayer.play();
  }
}

function syncSelectedReviewPackItem() {
  const item = state.reviewPack?.items?.[state.reviewPackSelectedIndex];
  els.selectedLabel.textContent = reviewPackSelectedLabel(item);
  els.sourceCaseButton.disabled = !reviewPackSourceTarget(reviewPackSelectedOrDefaultSourceItem());
  for (const row of els.segmentList.querySelectorAll(".review-pack-row")) {
    row.classList.toggle("is-selected", Number(row.dataset.index) === state.reviewPackSelectedIndex);
  }
}

function syncSelectedSegment() {
  els.selectedLabel.textContent = state.selectedId ? state.selectedId : "선택 없음";
  updateSelectedActionState();
  const secondarySegmentId = state.reviewCaseReference?.secondarySegmentId;
  for (const row of els.segmentList.querySelectorAll(".segment-row")) {
    row.classList.toggle("is-selected", row.dataset.id === state.selectedId);
    row.classList.toggle("is-secondary-reference", row.dataset.id === secondarySegmentId);
  }
}

function updateSelectedActionState() {
  const segment = selectedSegment();
  els.retranscribeButton.disabled = !segment;
  els.reviewDoneButton.disabled = !state.reviewCaseReference || !segment?.needs_review;
}

function selectedSegment() {
  return state.master?.segments.find((item) => item.id === state.selectedId) || null;
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

function scheduleSaveMaster() {
  window.clearTimeout(state.saveTimer);
  if (!state.master) return;
  state.saveTimer = window.setTimeout(() => {
    safeRun(saveCurrentMasterNow);
  }, 500);
}

async function saveCurrentMasterNow() {
  window.clearTimeout(state.saveTimer);
  if (!state.master) return null;
  if (state.reviewCaseReference) {
    const result = await apiPost("/api/review-case/save-reference", {
      case_index_path: state.reviewCaseReference.caseIndexPath,
      case_id: state.reviewCaseReference.caseId,
      master: state.master,
    });
    syncReviewCaseItemAfterSave(result);
    return result;
  }
  if (state.projectId) {
    return apiPost("/api/projects/save-master", {
      project_id: state.projectId,
      master: state.master,
    });
  }
  return null;
}

function syncReviewCaseItemAfterSave(result) {
  const item = state.reviewCaseSet?.items?.[state.reviewCaseReference?.itemIndex];
  if (!item) return;
  item.segments = result.segments;
  item.review_count = result.review_count;
  item.review_duration_ms = result.review_duration_ms;
  item.reference_master = state.master;
}

async function markSelectedReviewDone() {
  const segment = selectedSegment();
  if (!segment || !state.reviewCaseReference) return;
  segment.needs_review = false;
  const nextReviewId = nextReviewSegmentId(segment.id);
  if (nextReviewId !== null) {
    state.selectedId = nextReviewId;
  }
  render();
  drawWaveform();
  const result = await saveCurrentMasterNow();
  const remaining = result?.review_count ?? countReviewSegments();
  setStatus(
    remaining > 0 ? "검수 저장됨" : "case 검수 완료",
    remaining > 0 ? `${remaining}개 검수 flag가 남았습니다.` : "이 case의 검수 flag를 모두 처리했습니다.",
  );
}

function nextReviewSegmentId(currentId) {
  const segments = state.master?.segments || [];
  const currentIndex = segments.findIndex((segment) => segment.id === currentId);
  if (currentIndex < 0) return null;
  for (let index = currentIndex + 1; index < segments.length; index += 1) {
    if (segments[index].needs_review) return segments[index].id;
  }
  for (let index = 0; index < currentIndex; index += 1) {
    if (segments[index].needs_review) return segments[index].id;
  }
  return null;
}

function countReviewSegments() {
  return (state.master?.segments || []).filter((segment) => segment.needs_review).length;
}

async function startTranscription() {
  if (!state.projectId) {
    setStatus("파일 필요", "먼저 오디오 파일을 여세요.", true);
    return;
  }
  const model = getModelSettings();
  if (!modelSettingsReady(model)) {
    setStatus("모델 필요", modelSettingsRequirementText(model), true);
    return;
  }

  setStatus("분석 중", "오디오 채널과 chunk를 준비합니다.");
  await apiPost("/api/projects/analyze-audio", { project_id: state.projectId });
  setStatus("전사 중", "모델에 오디오를 보내고 있습니다.");
  const result = await apiPost("/api/projects/transcribe", {
    project_id: state.projectId,
    source_language: "ja",
    model,
  });
  setMaster(result.master, `${result.master.segments.length}개 segment를 저장했습니다.`, state.projectId);
}

async function loadReviewPath() {
  const path = els.reviewPackPathInput.value.trim();
  if (!path) {
    setStatus("경로 필요", "review pack 또는 case set 경로를 입력하세요.", true);
    return;
  }
  const review = await apiPost("/api/review/load", { path });
  if (state.audioUrl) {
    URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = null;
  }
  window.clearTimeout(state.stopTimer);
  els.audioPlayer.removeAttribute("src");
  els.audioPlayer.load();
  state.projectId = null;
  state.hasAudio = false;
  state.master = null;
  state.translated = null;
  state.selectedId = null;
  state.audioBuffer = null;
  state.reviewCaseReference = null;
  if (review.kind === "review-pack") {
    state.reviewPack = review;
    state.reviewPackSelectedIndex = null;
    state.reviewCaseSet = null;
    render();
    drawWaveform();
    setStatus("Review pack", `${review.items.length}개 clip을 priority 순서로 불러왔습니다.`);
    return;
  }
  if (review.kind === "review-case-set") {
    state.reviewPack = null;
    state.reviewPackSelectedIndex = null;
    state.reviewCaseSet = review;
    state.reviewCaseReference = null;
    render();
    drawWaveform();
    setStatus("Review cases", `${review.items.length}개 case를 불러왔습니다.`);
    return;
  }
  throw new Error("지원하지 않는 review path입니다.");
}

function loadReviewCaseItem(index, selectedSegmentId = null, secondarySegmentId = null) {
  const caseSet = state.reviewCaseSet;
  const item = caseSet?.items?.[index];
  if (!caseSet || !item?.reference_master) return;
  if (state.audioUrl) {
    URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = null;
  }
  window.clearTimeout(state.stopTimer);
  els.audioPlayer.src = item.audio_url;
  state.projectId = null;
  state.hasAudio = true;
  state.master = item.reference_master;
  state.translated = null;
  state.selectedId = segmentIdOrFirstReview(state.master, selectedSegmentId);
  state.audioBuffer = null;
  state.reviewPack = null;
  state.reviewPackSelectedIndex = null;
  state.reviewCaseReference = {
    caseIndexPath: caseSet.case_index_path,
    caseId: item.id,
    itemIndex: index,
    secondarySegmentId,
  };
  render();
  drawWaveform();
  setStatus("Review case", `${item.id} reference를 열었습니다. 수정은 case reference에 자동 저장됩니다.`);
}

async function openSelectedReviewPackSourceCase() {
  const item = reviewPackSelectedOrDefaultSourceItem();
  const target = reviewPackSourceTarget(item);
  if (!target) return;
  const caseSet = await apiPost("/api/review-case/load", { path: target.caseIndexPath });
  const caseIndex = (caseSet.items || []).findIndex((caseItem) => caseItem.id === target.caseId);
  if (caseIndex < 0) {
    throw new Error(`source case를 찾을 수 없습니다: ${target.caseId}`);
  }
  state.reviewCaseSet = caseSet;
  loadReviewCaseItem(caseIndex, target.segmentId, reviewPackSecondaryReferenceId(item));
  const sourceHint = reviewPackSourceHintText(item);
  if (sourceHint) {
    setStatus("Review case", sourceHint);
  }
}

async function returnToReviewCases() {
  if (!state.reviewCaseSet) return;
  await saveCurrentMasterNow();
  window.clearTimeout(state.stopTimer);
  els.audioPlayer.removeAttribute("src");
  els.audioPlayer.load();
  state.master = null;
  state.translated = null;
  state.selectedId = null;
  state.hasAudio = false;
  state.audioBuffer = null;
  state.reviewCaseReference = null;
  render();
  drawWaveform();
  setStatus("Review cases", `${state.reviewCaseSet.items.length}개 case 목록으로 돌아왔습니다.`);
}

async function openNextReviewCase() {
  const nextIndex = nextReviewCaseIndex();
  if (nextIndex === null) return;
  await saveCurrentMasterNow();
  loadReviewCaseItem(nextIndex);
}

function nextReviewCaseIndex() {
  const items = state.reviewCaseSet?.items || [];
  const currentIndex = state.reviewCaseReference?.itemIndex;
  if (currentIndex === undefined || currentIndex === null) return null;
  for (let index = currentIndex + 1; index < items.length; index += 1) {
    if ((items[index].review_count || 0) > 0) return index;
  }
  return currentIndex + 1 < items.length ? currentIndex + 1 : null;
}

function segmentIdOrFirstReview(master, segmentId) {
  const segments = master?.segments || [];
  if (segmentId && segments.some((segment) => segment.id === segmentId)) {
    return segmentId;
  }
  return firstReviewOrFirstSegmentId(master);
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
els.adapterInput.addEventListener("change", syncModelFormForAdapter);
els.validateModelButton.addEventListener("click", () => safeRun(validateModelSettings));
els.saveModelButton.addEventListener("click", saveModelSettings);
els.importTranslatedButton.addEventListener("click", () => els.translatedInput.click());
els.loadReviewPackButton.addEventListener("click", () => safeRun(loadReviewPath));
els.sourceCaseButton.addEventListener("click", () => safeRun(openSelectedReviewPackSourceCase));
els.reviewDoneButton.addEventListener("click", () => safeRun(markSelectedReviewDone));
els.caseListButton.addEventListener("click", () => safeRun(returnToReviewCases));
els.nextCaseButton.addEventListener("click", () => safeRun(openNextReviewCase));
els.reviewPackPathInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    safeRun(loadReviewPath);
  }
});
els.exportMasterButton.addEventListener("click", () => {
  if (state.master) downloadJson("master.json", state.master);
});
els.exportTranslationButton.addEventListener("click", () => safeRun(exportTranslationJson));
els.exportSrtButton.addEventListener("click", () => safeRun(exportSrt));
els.startButton.addEventListener("click", () => {
  safeRun(startTranscription);
});
els.retranscribeButton.addEventListener("click", () => {
  if (!state.selectedId) return;
  safeRun(async () => {
    const model = getModelSettings();
    if (!modelSettingsReady(model)) {
      setStatus("모델 필요", modelSettingsRequirementText(model), true);
      return;
    }
    setStatus("재전사 중", `${state.selectedId} segment를 모델에 보내고 있습니다.`);
    const result = await apiPost("/api/projects/retranscribe-segment", {
      project_id: state.projectId,
      segment_id: state.selectedId,
      source_language: "ja",
      model,
    });
    setMaster(result.master, "선택 segment를 재전사했습니다.", state.projectId);
  });
});

window.addEventListener("resize", drawWaveform);
loadModelSettings();
render();
drawWaveform();
