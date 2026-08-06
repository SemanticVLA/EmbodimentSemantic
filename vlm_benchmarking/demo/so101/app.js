"use strict";

const state = {
  health: null,
  frames: [],
  frameIndex: 0,
  payload: null,
  image: null,
  playing: false,
  timer: null,
  videoLoop: null,
  videoReadyKey: "",
  continuousPayloads: new Map(),
  payloadCache: new Map(),
  payloadRequests: new Map(),
  requestToken: 0,
  workToken: 0,
  recording: false,
  editorDirty: false,
  rightPanel: "eval",
  worklist: [],
  worklistSummary: null,
  pipelineStatus: null,
};

const els = Object.fromEntries([
  "buildStatus", "modeBanner", "camera", "evidenceMode", "overlayMode", "task", "episode",
  "frame", "frameValue", "subject", "showArrows", "showLabels", "showBboxes",
  "continuousVideo", "loop", "playButton", "stillButton", "videoButton", "sceneStatus", "sceneCanvas",
  "sourceVideo", "loadingOverlay", "loadingText", "tripletSubtitle", "tripletCount",
  "metrics", "metadataSummary", "coverageNotice", "worklistEditFilter",
  "worklistValidationFilter", "worklistStats", "annotationTab", "evalTab",
  "annotationPanel", "evalPanel", "graphEditorPanel", "annotationReadOnlyNotice",
  "editorState", "frameBadges", "validationDetails", "graphEditorRows",
  "saveGraphButton", "resetGraphButton", "nextGraphButton", "exportCsvButton",
  "exportStatus", "tripletList"
].map((id) => [id, document.getElementById(id)]));
const ctx = els.sceneCanvas.getContext("2d");
const common = window.DemoCommon;
const API_ROOT = so101ApiRoot();
const RELATION_VALUES = [
  "is_left_of",
  "is_right_of",
  "is_in_front_of",
  "is_behind",
  "is_on_top_of",
  "is_below_of",
  "is_inside",
  "contains",
];
const INVERSE_RELATIONS = {
  is_left_of: "is_right_of",
  is_right_of: "is_left_of",
  is_in_front_of: "is_behind",
  is_behind: "is_in_front_of",
  is_on_top_of: "is_below_of",
  is_below_of: "is_on_top_of",
  is_inside: "contains",
  contains: "is_inside",
};
const startupSelection = {
  task: new URLSearchParams(window.location.search).get("task"),
  episode: new URLSearchParams(window.location.search).get("episode"),
  taskApplied: false,
  episodeApplied: false,
};

async function fetchJson(url, options = {}) {
  return common.fetchJson(url, options);
}

async function postJson(path, payload = {}) {
  const body = JSON.stringify(payload);
  let lastError = null;
  for (const root of so101ApiRoots()) {
    try {
      return await fetchJson(common.apiUrl(root, path), {
        method: "POST",
        cache: "no-store",
        redirect: "error",
        headers: {"Content-Type": "application/json"},
        body,
      });
    } catch (error) {
      lastError = error;
      const message = String(error?.message || "");
      if (!message.startsWith("404 ") && !message.startsWith("405 ")) {
        throw error;
      }
    }
  }
  throw lastError || new Error("POST request failed");
}

function apiUrl(path) {
  return common.apiUrl(API_ROOT, path);
}

function so101ApiRoot() {
  const path = window.location.pathname.replace(/\/$/, "");
  if (path === "/so101" || path.startsWith("/so101/")) return "/so101/api";
  return common.scopedApiRoot("api/");
}

function so101ApiRoots() {
  const primary = so101ApiRoot();
  const fallback = primary === "/so101/api" ? "/api" : "/so101/api";
  return [primary, fallback].filter((root, index, roots) => root && roots.indexOf(root) === index);
}

function query(params) {
  return common.query(params);
}

function setLoading(active, text) {
  els.loadingOverlay.hidden = !active;
  els.loadingText.textContent = text || "Loading";
}

function cancellationError() {
  const error = new Error("Playback canceled");
  error.name = "AbortError";
  return error;
}

function isCancellation(error) {
  return error?.name === "AbortError";
}

function selectionContext() {
  return [
    els.task.value,
    els.episode.value,
    els.camera.value,
    els.evidenceMode.value,
  ].join("|");
}

function isWorkCurrent(token) {
  return token === state.workToken;
}

function cancelActiveWork() {
  state.workToken += 1;
  state.requestToken += 1;
  stopPlayback();
  state.recording = false;
  setLoading(false);
}

function setOptions(select, items, valueKey = "id", labelKey = "name") {
  common.setOptions(select, items, valueKey, labelKey);
}

function selectedSpeed() {
  return Number(document.querySelector('input[name="speed"]:checked').value);
}

function wait(ms) {
  return common.wait(ms);
}

function downloadBlob(blob, filename) {
  common.downloadBlob(blob, filename);
}

function safeFilename(value) {
  return common.safeFilename(value);
}

function stillFilename() {
  return `${safeFilename(els.task.value)}_${safeFilename(els.episode.value)}_${safeFilename(els.camera.value)}_frame_${currentFrame()}.png`;
}

function videoFilename() {
  return `${safeFilename(els.task.value)}_${safeFilename(els.episode.value)}_${safeFilename(els.camera.value)}_${selectedSpeed()}fps.webm`;
}

function setActionBusy(button, active, label) {
  button.disabled = active;
  button.textContent = label;
}

function currentFrame() {
  return state.frames[state.frameIndex] ?? 0;
}

function framePayloadKey(frame) {
  return [
    els.task.value,
    els.episode.value,
    els.camera.value,
    els.evidenceMode.value,
    frame,
  ].join("|");
}

function clonePayload(payload) {
  return JSON.parse(JSON.stringify(payload));
}

function fetchFramePayload(frame) {
  const key = framePayloadKey(frame);
  if (state.payloadCache.has(key)) return Promise.resolve(clonePayload(state.payloadCache.get(key)));
  if (state.payloadRequests.has(key)) return state.payloadRequests.get(key);
  const request = fetchJson(apiUrl(`frame?${query({
    task: els.task.value,
    episode: els.episode.value,
    camera: els.camera.value,
    frame,
    mode: els.evidenceMode.value,
  })}`)).then((payload) => {
    state.payloadCache.set(key, payload);
    return clonePayload(payload);
  }).finally(() => state.payloadRequests.delete(key));
  state.payloadRequests.set(key, request);
  return request;
}

function syncFrameControls(index) {
  state.frameIndex = index;
  const frame = currentFrame();
  els.frame.value = String(index);
  els.frameValue.textContent = `${frame} | ${index + 1} / ${state.frames.length}`;
}

async function initialize() {
  setLoading(true, "Loading dataset");
  state.health = await fetchJson(apiUrl("health"));
  state.pipelineStatus = await fetchJson(apiUrl("pipeline-status"));
  const modeLabel = state.health.demo_mode_label ? `${state.health.demo_mode_label} | ` : "";
  els.buildStatus.textContent = `${modeLabel}${state.health.episodes} episodes | ${state.health.sampled_frames} sampled views | ${state.health.proxy_frames} proxy frames`;
  renderModeBanner();
  renderCoverageNotice();
  state.rightPanel = canEditGraphs() ? "annotation" : "eval";
  renderRightPanel();
  const taskData = await fetchJson(apiUrl("tasks"));
  setOptions(els.task, taskData.tasks);
  if (startupSelection.task && [...els.task.options].some((option) => option.value === startupSelection.task)) {
    els.task.value = startupSelection.task;
  }
  startupSelection.taskApplied = true;
  await loadEpisodes();
  setLoading(false);
}

async function loadEpisodes() {
  state.workToken += 1;
  stopPlayback();
  const data = await fetchJson(apiUrl(`episodes?${query({task: els.task.value})}`));
  setOptions(els.episode, data.episodes);
  if (!startupSelection.episodeApplied && startupSelection.episode
      && [...els.episode.options].some((option) => option.value === startupSelection.episode)) {
    els.episode.value = startupSelection.episode;
  }
  startupSelection.episodeApplied = true;
  await loadFrames();
}

async function loadFrames() {
  state.workToken += 1;
  stopPlayback();
  const data = await fetchJson(apiUrl(`frames?${query({task: els.task.value, episode: els.episode.value, camera: els.camera.value})}`));
  state.frames = data.frames;
  syncFrameControls(0);
  els.frame.min = "0";
  els.frame.max = String(Math.max(0, state.frames.length - 1));
  resetVideo();
  await loadFrame(true);
  await refreshWorklist();
}

async function loadFrame(drawImage) {
  if (!state.frames.length) {
    state.payload = null;
    drawEmpty("No sampled frames");
    return;
  }
  const token = ++state.requestToken;
  const frame = currentFrame();
  syncFrameControls(state.frameIndex);
  setLoading(true, "Loading frame");
  try {
    const payload = await fetchFramePayload(frame);
    if (token !== state.requestToken) return;
    state.payload = payload;
    state.editorDirty = false;
    ensureGraphPairs(state.payload);
    rememberSelectionControls();
    updateSubjects(payload.visible_objects);
    updatePanels();
    if (drawImage) {
      state.image = await loadImage(payload.image_url);
      if (token !== state.requestToken) return;
      drawScene(state.image);
    }
  } catch (error) {
    if (isCancellation(error) || token !== state.requestToken) return;
    drawEmpty(error.message);
  } finally {
    if (token === state.requestToken) setLoading(false);
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Unable to load sampled image"));
    image.src = url;
  });
}

function updateSubjects(objects) {
  const previous = els.subject.value;
  const names = ["All", ...objects];
  els.subject.replaceChildren();
  for (const name of names) {
    const option = document.createElement("option");
    option.value = name === "All" ? "all" : name;
    option.textContent = pretty(name);
    els.subject.appendChild(option);
  }
  els.subject.value = names.includes(previous) ? previous : (objects.includes("black_bowl") ? "black_bowl" : "all");
}

function pretty(value) {
  return common.prettyObjectName(value);
}

function relationLabel(value) {
  return common.relationLabel(value);
}

function tripletKey(item) {
  return `${item.subject}\u0000${item.relation}\u0000${item.object}`;
}

function canEditGraphs() {
  return Boolean(state.health?.graph_editing_enabled);
}

function renderModeBanner() {
  if (!els.modeBanner || !state.health) return;
  if (state.health.demo_mode === "online") {
    els.modeBanner.textContent = "Online cached showcase: reduced episode_0, read-only";
  } else {
    els.modeBanner.textContent = "Offline localhost tool: annotations and CSV export enabled";
  }
  els.modeBanner.hidden = false;
}

function renderRightPanel() {
  const annotationEnabled = canEditGraphs();
  if (!annotationEnabled && state.rightPanel === "annotation") {
    state.rightPanel = "eval";
  }
  const showAnnotation = state.rightPanel === "annotation";
  if (els.annotationPanel) els.annotationPanel.hidden = !showAnnotation;
  if (els.evalPanel) els.evalPanel.hidden = showAnnotation;
  if (els.annotationTab) {
    els.annotationTab.disabled = !annotationEnabled;
    els.annotationTab.title = annotationEnabled ? "Open annotation tools" : "Annotation is only available in offline mode";
    els.annotationTab.setAttribute("aria-selected", String(showAnnotation));
    els.annotationTab.tabIndex = showAnnotation ? 0 : -1;
  }
  if (els.evalTab) {
    els.evalTab.setAttribute("aria-selected", String(!showAnnotation));
    els.evalTab.tabIndex = showAnnotation ? -1 : 0;
  }
}

function selectRightPanel(name) {
  if (name === "annotation" && !canEditGraphs()) return;
  state.rightPanel = name === "annotation" ? "annotation" : "eval";
  renderRightPanel();
}

function renderCoverageNotice() {
  if (!els.coverageNotice) return;
  const wrist = state.pipelineStatus?.coverage_by_camera?.wrist || state.health?.coverage_by_camera?.wrist;
  if (!wrist || wrist.complete) {
    els.coverageNotice.hidden = true;
    els.coverageNotice.textContent = "";
    return;
  }
  els.coverageNotice.hidden = false;
  els.coverageNotice.textContent = `Wrist proxy coverage is partial: ${wrist.proxy_frames} / ${wrist.sampled_frames} sampled frames.`;
}

function worklistQueryParams() {
  return {
    task: els.task.value,
    episode: els.episode.value,
    camera: els.camera.value,
    mode: els.evidenceMode.value,
    edit_status: els.worklistEditFilter?.value || "all",
    validation_status: els.worklistValidationFilter?.value || "all",
  };
}

async function refreshWorklist() {
  if (!els.worklistStats) return;
  try {
    const payload = await fetchJson(apiUrl(`worklist?${query(worklistQueryParams())}`));
    state.worklist = payload.items || [];
    state.worklistSummary = payload.summary || null;
    renderWorklistStats(payload);
  } catch (error) {
    els.worklistStats.textContent = error.message;
  }
}

function renderWorklistStats(payload = {}) {
  if (!els.worklistStats) return;
  const summary = payload.summary || state.worklistSummary || {};
  const count = Number(payload.count ?? state.worklist.length ?? 0);
  els.worklistStats.textContent = [
    `${count} shown`,
    `${summary.edited_frames || 0} edited`,
    `${summary.invalid_edit_frames || 0} invalid`,
    `${summary.frames || 0} total`,
  ].join(" | ");
}

function compactOutputPath(path) {
  const value = String(path || "");
  if (!value) return "";
  const normalized = value.replace(/\\/g, "/");
  const marker = "/output/";
  const lower = normalized.toLowerCase();
  const index = lower.lastIndexOf(marker);
  if (index >= 0) return normalized.slice(index + 1);
  return normalized.split("/").filter(Boolean).slice(-3).join("/");
}

function formatCount(value) {
  return Number(value || 0).toLocaleString("en-US");
}

function setWorkflowStatus(message, {title = "", tone = ""} = {}) {
  if (!els.exportStatus) return;
  els.exportStatus.textContent = message || "";
  els.exportStatus.title = title || message || "";
  els.exportStatus.className = `export-status ${tone}`.trim();
}

function blockUnsavedEdits(action) {
  if (!state.editorDirty) return false;
  stopPlayback();
  setWorkflowStatus(`Save or reset the current frame before ${action}.`);
  return true;
}

function rememberSelectionControls() {
  for (const select of [els.task, els.episode, els.camera, els.evidenceMode]) {
    if (select) select.dataset.previousValue = select.value;
  }
}

function setEditorDirty(value) {
  state.editorDirty = value;
  if (!els.editorState) return;
  const invalid = (state.payload?.validation_errors || []).length > 0;
  const saved = invalid ? "Invalid edit" : (state.payload?.manual_edit ? `Saved edit r${state.payload.edit_revision || 1}` : "Generated");
  els.editorState.textContent = value ? "Unsaved" : saved;
  if (els.resetGraphButton && state.payload) {
    els.resetGraphButton.disabled = !state.payload.manual_edit && !value;
  }
  if (els.nextGraphButton) {
    els.nextGraphButton.disabled = value || !state.frames.length || state.frameIndex >= state.frames.length - 1;
  }
  renderFrameBadges();
}

function badge(label, tone = "") {
  const item = document.createElement("span");
  item.className = `frame-badge ${tone}`.trim();
  item.textContent = label;
  return item;
}

function renderFrameBadges() {
  if (!els.frameBadges || !state.payload) return;
  const payload = state.payload;
  els.frameBadges.replaceChildren();
  if (state.editorDirty) els.frameBadges.appendChild(badge("Unsaved", "warn"));
  els.frameBadges.appendChild(payload.manual_edit ? badge("Edited", "good") : badge("Generated"));
  if ((payload.validation_errors || []).length) els.frameBadges.appendChild(badge("Invalid", "bad"));
  if (payload.stale_edit) els.frameBadges.appendChild(badge("Stale", "bad"));
}

function renderValidationDetails() {
  if (!els.validationDetails || !state.payload) return;
  const errors = state.payload.validation_errors || [];
  els.validationDetails.className = errors.length ? "validation-details bad" : "validation-details";
  els.validationDetails.textContent = errors.length ? errors.join(" ") : "Validation OK";
}

function recomputePredictionMetrics(payload) {
  if (!payload) return;
  const proxy = new Set((payload.proxy_relations || []).map(tripletKey));
  let tp = 0;
  let fp = 0;
  for (const item of payload.prediction_relations || []) {
    item.correct = proxy.has(tripletKey(item));
    if (item.correct) tp += 1;
    else fp += 1;
  }
  const fn = Math.max(0, proxy.size - tp);
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  payload.metrics = {tp, fp, fn, precision, recall, f1};
}

function updateStatusLine() {
  const payload = state.payload;
  if (!payload) return;
  const proxyCount = payload.proxy_relations.length;
  const predCount = payload.prediction_relations.length;
  const editText = payload.manual_edit || state.editorDirty ? " | edited" : "";
  els.sceneStatus.textContent = `${pretty(payload.camera)} | frame ${payload.frame} | ${payload.proxy_available ? proxyCount + " Proxy GT triplets" : "Proxy GT not generated"} | ${payload.prediction_available ? predCount + " Gemini triplets" : "prediction unavailable"}${editText}`;
}

function refreshGraphView() {
  recomputePredictionMetrics(state.payload);
  updateStatusLine();
  renderFrameBadges();
  renderValidationDetails();
  const relations = activeRelations();
  els.tripletCount.textContent = `${relations.length} shown`;
  common.renderMetrics(els.metrics, state.payload?.metrics);
  renderTriplets(relations);
  const source = state.playing && els.continuousVideo.checked ? els.sourceVideo : state.image;
  if (source) drawScene(source);
}

function pairKey(subject, object) {
  return [subject, object].sort().join("\u0000");
}

function relationDirectionKey(subject, object) {
  return `${subject}\u0000${object}`;
}

function graphPairsFromRelations(relations) {
  const output = [];
  const seen = new Set();
  const byDirection = new Map();
  for (const item of relations || []) {
    byDirection.set(relationDirectionKey(item.subject, item.object), item);
  }
  for (const item of relations || []) {
    const key = pairKey(item.subject, item.object);
    if (seen.has(key)) continue;
    seen.add(key);
    const inverse = byDirection.get(relationDirectionKey(item.object, item.subject));
    output.push({
      subject: item.subject,
      relation: item.relation,
      object: item.object,
      inverse_relation: inverse?.relation || "",
      edited: item.source === "manual_edit" || inverse?.source === "manual_edit",
      original_relation: item.original_relation || "",
      original_inverse_relation: inverse?.original_relation || "",
    });
  }
  return output;
}

function ensureGraphPairs(payload) {
  if (!payload) return;
  if (!Array.isArray(payload.graph_pairs)) {
    payload.graph_pairs = graphPairsFromRelations(payload.proxy_relations || []);
  }
}

function applyGraphPairsToPayload() {
  if (!state.payload) return;
  ensureGraphPairs(state.payload);
  const previous = new Map();
  for (const item of state.payload.proxy_relations || []) {
    previous.set(relationDirectionKey(item.subject, item.object), item);
  }
  const output = [];
  for (const pair of state.payload.graph_pairs || []) {
    const inverseRelation = INVERSE_RELATIONS[pair.relation];
    if (!inverseRelation) continue;
    const forwardBase = previous.get(relationDirectionKey(pair.subject, pair.object)) || {};
    const inverseBase = previous.get(relationDirectionKey(pair.object, pair.subject)) || {};
    output.push({
      ...forwardBase,
      subject: pair.subject,
      relation: pair.relation,
      object: pair.object,
      source: "manual_edit",
      confidence: Number.isFinite(Number(forwardBase.confidence)) ? Number(forwardBase.confidence) : 1,
      metadata_gates: Array.isArray(forwardBase.metadata_gates) ? [...forwardBase.metadata_gates] : [],
      evidence: {...(forwardBase.evidence || {})},
    });
    output.push({
      ...inverseBase,
      subject: pair.object,
      relation: inverseRelation,
      object: pair.subject,
      source: "manual_edit",
      confidence: Number.isFinite(Number(inverseBase.confidence)) ? Number(inverseBase.confidence) : 1,
      metadata_gates: Array.isArray(inverseBase.metadata_gates) ? [...inverseBase.metadata_gates] : [],
      evidence: {...(inverseBase.evidence || {})},
    });
    pair.inverse_relation = inverseRelation;
    pair.edited = true;
  }
  state.payload.proxy_relations = output;
}

function graphEditPayload() {
  if (!state.payload) throw new Error("No frame is loaded.");
  ensureGraphPairs(state.payload);
  return {
    task: state.payload.task,
    episode: state.payload.episode,
    frame: state.payload.frame,
    camera: state.payload.camera,
    mode: state.payload.mode || els.evidenceMode.value,
    base_graph_hash: state.payload.base_graph_hash,
    pairs: (state.payload.graph_pairs || []).map((item) => ({
      subject: item.subject,
      relation: item.relation,
      object: item.object,
    })),
  };
}

function renderGraphEditor() {
  if (!els.graphEditorPanel) return;
  const editable = canEditGraphs() && state.payload?.editable;
  if (els.annotationReadOnlyNotice) {
    els.annotationReadOnlyNotice.hidden = editable;
  }
  if (!editable) {
    els.graphEditorRows.replaceChildren();
    els.graphEditorRows.hidden = true;
    els.saveGraphButton.disabled = true;
    els.nextGraphButton.disabled = true;
    els.resetGraphButton.disabled = true;
    els.exportCsvButton.disabled = true;
    setEditorDirty(false);
    return;
  }
  els.graphEditorRows.hidden = false;
  ensureGraphPairs(state.payload);
  els.graphEditorRows.replaceChildren();
  const pairs = state.payload.graph_pairs || [];
  if (!pairs.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No proxy relations for this frame.";
    els.graphEditorRows.appendChild(empty);
  }
  pairs.forEach((pair, index) => {
    const row = document.createElement("div");
    row.className = "editor-row";

    const subject = document.createElement("span");
    subject.className = "editor-entity";
    subject.title = pair.subject;
    subject.textContent = pretty(pair.subject);

    const select = document.createElement("select");
    select.dataset.index = String(index);
    for (const value of RELATION_VALUES) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = relationLabel(value);
      select.appendChild(option);
    }
    select.value = RELATION_VALUES.includes(pair.relation) ? pair.relation : "is_left_of";
    select.addEventListener("change", () => {
      const item = state.payload?.graph_pairs?.[index];
      if (!item) return;
      item.relation = select.value;
      item.inverse_relation = INVERSE_RELATIONS[select.value] || "";
      state.payload.manual_edit = true;
      applyGraphPairsToPayload();
      setEditorDirty(true);
      refreshGraphView();
    });

    const object = document.createElement("span");
    object.className = "editor-entity";
    object.title = pair.object;
    object.textContent = pretty(pair.object);

    row.append(subject, select, object);
    els.graphEditorRows.appendChild(row);
  });
  setEditorDirty(state.editorDirty);
  els.saveGraphButton.disabled = !pairs.length;
  els.resetGraphButton.disabled = !state.payload.manual_edit && !state.editorDirty;
  if (els.nextGraphButton) {
    els.nextGraphButton.disabled = state.editorDirty || state.frameIndex >= state.frames.length - 1;
  }
  els.exportCsvButton.disabled = false;
  if ((state.payload.validation_errors || []).length) {
    setWorkflowStatus(state.payload.validation_errors.join(" "));
  }
}

function cacheCurrentPayload(payload) {
  const normalized = clonePayload(payload);
  ensureGraphPairs(normalized);
  state.payload = normalized;
  state.payloadCache.set(framePayloadKey(payload.frame), clonePayload(normalized));
}

async function saveGraphEdit() {
  if (!canEditGraphs()) return;
  const button = els.saveGraphButton;
  const originalText = button.textContent;
  setActionBusy(button, true, "Saving...");
  try {
    const payload = await postJson("graph-edits", graphEditPayload());
    cacheCurrentPayload(payload);
    state.editorDirty = false;
    updateSubjects(payload.visible_objects);
    updatePanels();
    if (state.image) drawScene(state.image);
    setWorkflowStatus(`Saved frame ${payload.frame}.`);
    await refreshWorklist();
    return payload;
  } catch (error) {
    setWorkflowStatus(error.message, {tone: "bad"});
    return null;
  } finally {
    setActionBusy(button, false, originalText);
  }
}

async function resetGraphEdit() {
  if (!canEditGraphs() || !state.payload) return;
  const originalText = els.resetGraphButton.textContent;
  setActionBusy(els.resetGraphButton, true, "Resetting...");
  try {
    const payload = await postJson("graph-edits/reset", graphEditPayload());
    cacheCurrentPayload(payload);
    state.editorDirty = false;
    updateSubjects(payload.visible_objects);
    updatePanels();
    if (state.image) drawScene(state.image);
    setWorkflowStatus(`Reset frame ${payload.frame} to generated graph.`);
    await refreshWorklist();
  } catch (error) {
    setWorkflowStatus(error.message, {tone: "bad"});
  } finally {
    setActionBusy(els.resetGraphButton, false, originalText);
  }
}

async function exportGraphCsvs() {
  if (!canEditGraphs()) return;
  if (state.editorDirty) {
    setWorkflowStatus("Save or reset the current frame before exporting CSVs.");
    return;
  }
  const originalText = els.exportCsvButton.textContent;
  setActionBusy(els.exportCsvButton, true, "Exporting...");
  try {
    const payload = await postJson("export-csv", {});
    const fullPath = payload.output_dir || "";
    const shortPath = compactOutputPath(fullPath);
    setWorkflowStatus(
      `Exported ${formatCount(payload.rows)} rows${shortPath ? ` to ${shortPath}` : ""}.`,
      {title: fullPath}
    );
  } catch (error) {
    setWorkflowStatus(error.message, {tone: "bad"});
  } finally {
    setActionBusy(els.exportCsvButton, false, originalText);
  }
}

async function moveToFrameIndex(index, {requireClean = true} = {}) {
  if (requireClean && blockUnsavedEdits("changing frames")) return;
  if (!state.frames.length) return;
  const clamped = Math.max(0, Math.min(state.frames.length - 1, index));
  if (clamped === state.frameIndex) return;
  stopPlayback();
  state.frameIndex = clamped;
  syncFrameControls(state.frameIndex);
  await loadFrame(true);
}

async function moveToNextFrame(options = {}) {
  if (options.requireClean !== false && blockUnsavedEdits("moving to the next frame")) return;
  if (state.frameIndex >= state.frames.length - 1) {
    setWorkflowStatus("Already on the last frame.");
    return;
  }
  await moveToFrameIndex(state.frameIndex + 1, {requireClean: false});
}

function evidenceLabel(item) {
  const evidence = item.evidence || {};
  if (evidence.basis !== "bbox_midpoint_arrow") return "";
  if (evidence.semantic_camera === "agent_view" && evidence.visibility_camera === "wrist") {
    const displayDx = Number(evidence.display_arrow_dx_norm);
    const displayDy = Number(evidence.display_arrow_dy_norm);
    const parts = ["semantic: agent GT", "visibility: wrist bbox"];
    if (Number.isFinite(displayDx) && Number.isFinite(displayDy)) {
      parts.push(`display arrow dx ${displayDx.toFixed(2)}, dy ${displayDy.toFixed(2)}`);
    }
    return parts.join(" | ");
  }
  const dx = Number(evidence.arrow_dx_norm);
  const dy = Number(evidence.arrow_dy_norm);
  const margin = Number(evidence.axis_margin);
  const iomin = Number(evidence.iomin);
  const parts = [];
  if (Number.isFinite(dx) && Number.isFinite(dy)) {
    parts.push(`arrow dx ${dx.toFixed(2)}, dy ${dy.toFixed(2)}`);
  }
  if (evidence.dominant_axis) parts.push(String(evidence.dominant_axis).replace("_", "/"));
  if (Number.isFinite(margin)) parts.push(`margin ${margin.toFixed(2)}`);
  if (Number.isFinite(iomin) && iomin > 0) parts.push(`IoMin ${iomin.toFixed(2)}`);
  return parts.join(" | ");
}

function activeRelations() {
  if (!state.payload) return [];
  const presentation = els.overlayMode.value;
  const output = [];
  if (presentation === "proxy" || presentation === "compare") {
    for (const item of state.payload.proxy_relations) output.push({...item, layer: "proxy", correct: true});
  }
  if (presentation === "prediction" || presentation === "compare") {
    for (const item of state.payload.prediction_relations) output.push({...item, layer: "prediction"});
  }
  const subject = els.subject.value;
  return subject === "all" ? output : output.filter((item) => item.subject === subject);
}

function drawScene(source, bboxOverride = null) {
  if (!state.payload || !source) return;
  els.sceneCanvas.width = state.payload.width;
  els.sceneCanvas.height = state.payload.height;
  ctx.clearRect(0, 0, els.sceneCanvas.width, els.sceneCanvas.height);
  ctx.drawImage(source, 0, 0, els.sceneCanvas.width, els.sceneCanvas.height);
  const bboxes = bboxOverride || state.payload.bboxes;
  if (els.showBboxes.checked) drawBboxes(bboxes);
  if (els.showArrows.checked) {
    const occupiedLabels = [];
    for (const relation of activeRelations()) {
      drawRelation(relation, bboxes, occupiedLabels);
    }
  }
}

async function exportStill() {
  if (!state.payload || !state.image) return;
  if (blockUnsavedEdits("exporting a still")) return;
  setActionBusy(els.stillButton, true, "Exporting...");
  try {
    await new Promise((resolve, reject) => {
      els.sceneCanvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error("Still export failed"));
          return;
        }
        downloadBlob(blob, stillFilename());
        resolve();
      }, "image/png");
    });
  } catch (error) {
    els.sceneStatus.textContent = error.message;
  } finally {
    setActionBusy(els.stillButton, false, "Export Still");
  }
}

function mediaRecorderType() {
  return common.mediaRecorderType();
}

async function downloadVideo() {
  if (!window.MediaRecorder || !els.sceneCanvas.captureStream) {
    els.sceneStatus.textContent = "Video download is not supported by this browser.";
    return;
  }
  if (!state.frames.length || state.recording) return;
  if (blockUnsavedEdits("downloading a video")) return;
  stopPlayback();
  state.recording = true;
  const originalIndex = state.frameIndex;
  const fps = selectedSpeed();
  const frameMs = 1000 / fps;
  const chunks = [];
  setActionBusy(els.videoButton, true, "Preparing Video...");
  els.playButton.disabled = true;
  try {
    const stream = els.sceneCanvas.captureStream(fps);
    const mimeType = mediaRecorderType();
    const recorder = new MediaRecorder(stream, mimeType ? {mimeType} : undefined);
    const stopped = new Promise((resolve) => {
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) chunks.push(event.data);
      };
      recorder.onstop = resolve;
    });
    recorder.start();
    for (let index = 0; index < state.frames.length; index += 1) {
      if (!state.recording) break;
      state.frameIndex = index;
      syncFrameControls(index);
      await loadFrame(true);
      if (!state.recording) break;
      els.sceneStatus.textContent = `Recording video ${Math.round(((index + 1) / state.frames.length) * 100)}%`;
      await wait(frameMs);
    }
    await wait(Math.max(350, frameMs * 3));
    recorder.stop();
    stream.getTracks().forEach((track) => track.stop());
    await stopped;
    downloadBlob(new Blob(chunks, {type: mimeType || "video/webm"}), videoFilename());
  } catch (error) {
    els.sceneStatus.textContent = error.message;
  } finally {
    state.recording = false;
    setActionBusy(els.videoButton, false, "Download Video");
    els.playButton.disabled = false;
    state.frameIndex = originalIndex;
    syncFrameControls(originalIndex);
    await loadFrame(true);
  }
}

function drawBboxes(bboxes) {
  ctx.save();
  ctx.lineWidth = 1.5;
  ctx.font = "11px Inter, sans-serif";
  for (const [name, item] of Object.entries(bboxes)) {
    const [x1, y1, x2, y2] = item.bbox;
    ctx.strokeStyle = "#f3c969";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = "#111";
    ctx.fillText(pretty(name), x1 + 3, Math.max(12, y1 - 4));
  }
  ctx.restore();
}

function center(item) {
  return common.centerFromBbox(item);
}

function drawRelation(relation, bboxes, occupiedLabels) {
  const from = bboxes[relation.subject];
  const to = bboxes[relation.object];
  if (!from || !to) return;
  const [x1, y1] = center(from);
  const [x2, y2] = center(to);
  const color = relation.layer === "proxy" ? "#00a66b" : (relation.correct ? "#008f5b" : "#d43c2f");
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = relation.layer === "prediction" ? 2.5 : 2;
  common.drawArrow(ctx, x1, y1, x2, y2);
  if (els.showLabels.checked) {
    const label = relationLabel(relation.relation);
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    ctx.font = "600 11px Inter, sans-serif";
    const width = ctx.measureText(label).width + 6;
    const position = placeCanvasLabel(mx, my, width, 15, occupiedLabels);
    ctx.fillStyle = "rgba(255,255,255,.88)";
    ctx.fillRect(position.left, position.top, width, 15);
    ctx.fillStyle = color;
    ctx.textAlign = "center";
    ctx.fillText(label, position.left + width / 2, position.top + 11);
  }
  ctx.restore();
}

function placeCanvasLabel(centerX, centerY, width, height, occupied) {
  return common.placeCanvasLabel(els.sceneCanvas, centerX, centerY, width, height, occupied);
}

function drawEmpty(message) {
  els.sceneCanvas.width = 640;
  els.sceneCanvas.height = 480;
  ctx.fillStyle = "#111418";
  ctx.fillRect(0, 0, 640, 480);
  ctx.fillStyle = "#d4dae1";
  ctx.font = "14px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(message, 320, 240);
}

function updatePanels() {
  const payload = state.payload;
  if (!payload) return;
  const presentation = els.overlayMode.value;
  recomputePredictionMetrics(payload);
  updateStatusLine();
  renderFrameBadges();
  renderValidationDetails();
  els.tripletSubtitle.textContent = presentation === "proxy" ? "2D Proxy GT" : (presentation === "prediction" ? "Gemini prediction" : "Proxy GT vs Gemini");
  const relations = activeRelations();
  els.tripletCount.textContent = `${relations.length} shown`;
  common.renderMetrics(els.metrics, payload.metrics);
  common.renderMetadataSummary(els.metadataSummary, payload.metadata, "Metadata signals have not been extracted.");
  renderGraphEditor();
  renderTriplets(relations);
}

function renderTriplets(relations) {
  common.renderGroupedTriplets(els.tripletList, relations, {
    selectedSubject: els.subject.value,
    pretty,
    emptyText: "No relations for this selection.",
  });
}

function renderCurrent() {
  const source = state.playing && els.continuousVideo.checked ? els.sourceVideo : state.image;
  if (source) drawScene(source);
  updatePanels();
}

function stopPlayback() {
  state.playing = false;
  clearTimeout(state.timer);
  state.timer = null;
  if (!els.sourceVideo.paused) els.sourceVideo.pause();
  if (state.videoLoop) cancelAnimationFrame(state.videoLoop);
  state.videoLoop = null;
  els.playButton.textContent = "Play";
}

async function togglePlayback() {
  if (state.playing) {
    stopPlayback();
    return;
  }
  if (blockUnsavedEdits("starting playback")) return;
  const token = state.workToken;
  if (state.frameIndex >= state.frames.length - 1) {
    state.frameIndex = 0;
    await loadFrame(true);
    if (!state.frames.length || !isWorkCurrent(token)) return;
  }
  state.playing = true;
  els.playButton.textContent = "Pause";
  if (els.continuousVideo.checked && state.payload && state.payload.video_url) {
    await playContinuous();
  } else {
    playSamples();
  }
}

function playSamples() {
  if (!state.playing) return;
  const delay = 1000 / selectedSpeed();
  state.timer = setTimeout(async () => {
    if (!state.playing) return;
    if (state.frameIndex >= state.frames.length - 1) {
      if (!els.loop.checked) { stopPlayback(); return; }
      state.frameIndex = 0;
    } else {
      state.frameIndex += 1;
    }
    await loadFrame(true);
    if (!state.playing) return;
    playSamples();
  }, delay);
}

async function prepareVideo(token = state.workToken) {
  if (!state.payload || !state.payload.video_url) throw new Error("Source video is unavailable");
  const key = `${state.payload.task}|${state.payload.episode}|${state.payload.camera}`;
  if (state.videoReadyKey !== key) {
    els.sourceVideo.src = state.payload.video_url;
    els.sourceVideo.load();
    await new Promise((resolve, reject) => {
      els.sourceVideo.onloadedmetadata = resolve;
      els.sourceVideo.onerror = () => reject(new Error("Browser could not decode the source video"));
    });
    if (!isWorkCurrent(token)) throw cancellationError();
    state.videoReadyKey = key;
  }
  if (!isWorkCurrent(token)) throw cancellationError();
  els.sourceVideo.currentTime = state.payload.video_start + currentFrame() / state.payload.fps;
}

async function prepareContinuousPayloads() {
  const token = state.workToken;
  const context = selectionContext();
  const entries = [];
  for (const frame of state.frames) {
    if (!state.playing || !isWorkCurrent(token) || context !== selectionContext()) {
      throw cancellationError();
    }
    const payload = await fetchFramePayload(frame);
    if (!state.playing || !isWorkCurrent(token) || context !== selectionContext()) {
      throw cancellationError();
    }
    entries.push([frame, payload]);
    await wait(0);
  }
  state.continuousPayloads = new Map(entries);
}

async function playContinuous() {
  const token = state.workToken;
  try {
    setLoading(true, "Preparing synchronized overlays");
    await Promise.all([prepareVideo(token), prepareContinuousPayloads()]);
    if (!state.playing || !isWorkCurrent(token)) throw cancellationError();
    els.sourceVideo.playbackRate = Math.min(2, Math.max(.25, selectedSpeed() / 30));
    await els.sourceVideo.play();
    if (!state.playing || !isWorkCurrent(token)) throw cancellationError();
    setLoading(false);
    continuousLoop();
  } catch (error) {
    setLoading(false);
    stopPlayback();
    if (!isCancellation(error)) els.sceneStatus.textContent = error.message;
  }
}

function sampleBracket(frame) {
  const last = state.frames.length - 1;
  if (last <= 0 || frame <= state.frames[0]) return {left: 0, right: 0, alpha: 0};
  if (frame >= state.frames[last]) return {left: last, right: last, alpha: 0};
  let low = 0;
  let high = last;
  while (low + 1 < high) {
    const middle = Math.floor((low + high) / 2);
    if (state.frames[middle] <= frame) low = middle;
    else high = middle;
  }
  const span = Math.max(1, state.frames[high] - state.frames[low]);
  return {left: low, right: high, alpha: Math.max(0, Math.min(1, (frame - state.frames[low]) / span))};
}

function interpolateBboxes(leftPayload, rightPayload, alpha) {
  if (!leftPayload || !rightPayload || alpha <= 0) return leftPayload?.bboxes || rightPayload?.bboxes || {};
  const output = {};
  const names = new Set([
    ...Object.keys(leftPayload.bboxes || {}),
    ...Object.keys(rightPayload.bboxes || {}),
  ]);
  for (const name of names) {
    const left = leftPayload.bboxes[name];
    const right = rightPayload.bboxes[name];
    if (!left || !right) {
      output[name] = left || right;
      continue;
    }
    output[name] = {
      ...left,
      bbox: left.bbox.map((value, index) => value + alpha * (right.bbox[index] - value)),
      source: "demo_temporal_interpolation",
      tracking_confidence: Math.min(left.tracking_confidence, right.tracking_confidence),
    };
  }
  return output;
}

function continuousLoop() {
  if (!state.playing || !els.continuousVideo.checked) return;
  const payload = state.payload;
  const relativeFrame = Math.max(0, (els.sourceVideo.currentTime - payload.video_start) * payload.fps);
  const bracket = sampleBracket(relativeFrame);
  const nextIndex = bracket.alpha < .5 ? bracket.left : bracket.right;
  if (nextIndex !== state.frameIndex) {
    syncFrameControls(nextIndex);
    const nextPayload = state.continuousPayloads.get(currentFrame());
    if (nextPayload) {
      state.payload = nextPayload;
      updateSubjects(nextPayload.visible_objects);
      updatePanels();
    }
  }
  if (els.sourceVideo.currentTime >= payload.video_end - .02) {
    if (els.loop.checked) {
      syncFrameControls(0);
      const firstPayload = state.continuousPayloads.get(currentFrame());
      if (firstPayload) {
        state.payload = firstPayload;
        updateSubjects(firstPayload.visible_objects);
        updatePanels();
      }
      els.sourceVideo.currentTime = payload.video_start;
    } else {
      stopPlayback();
      return;
    }
  }
  if (els.sourceVideo.readyState >= 2) {
    const leftPayload = state.continuousPayloads.get(state.frames[bracket.left]);
    const rightPayload = state.continuousPayloads.get(state.frames[bracket.right]);
    drawScene(els.sourceVideo, interpolateBboxes(leftPayload, rightPayload, bracket.alpha));
  }
  state.videoLoop = requestAnimationFrame(continuousLoop);
}

function resetVideo() {
  stopPlayback();
  state.videoReadyKey = "";
  state.continuousPayloads = new Map();
  els.sourceVideo.removeAttribute("src");
  els.sourceVideo.load();
}

function guardedSelectChange(select, action, handler) {
  const previous = select.dataset.previousValue ?? select.value;
  if (blockUnsavedEdits(action)) {
    select.value = previous;
    return;
  }
  select.dataset.previousValue = select.value;
  handler();
}

function isEditingControl(target) {
  return target instanceof HTMLInputElement
    || target instanceof HTMLSelectElement
    || target instanceof HTMLTextAreaElement
    || target instanceof HTMLButtonElement;
}

function handleKeydown(event) {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveGraphEdit();
    return;
  }
  if (isEditingControl(event.target)) return;
  if (event.key.toLowerCase() === "n") {
    event.preventDefault();
    moveToNextFrame();
    return;
  }
  if (event.key === "ArrowRight") {
    event.preventDefault();
    moveToFrameIndex(state.frameIndex + 1);
    return;
  }
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    moveToFrameIndex(state.frameIndex - 1);
  }
}

els.task.addEventListener("change", () => guardedSelectChange(els.task, "changing tasks", loadEpisodes));
els.episode.addEventListener("change", () => guardedSelectChange(els.episode, "changing episodes", loadFrames));
els.camera.addEventListener("change", () => guardedSelectChange(els.camera, "changing cameras", loadFrames));
els.evidenceMode.addEventListener("change", () => guardedSelectChange(els.evidenceMode, "changing graph modes", () => {
  stopPlayback();
  state.continuousPayloads = new Map();
  loadFrame(true);
}));
els.overlayMode.addEventListener("change", renderCurrent);
els.subject.addEventListener("change", renderCurrent);
els.showArrows.addEventListener("change", renderCurrent);
els.showLabels.addEventListener("change", renderCurrent);
els.showBboxes.addEventListener("change", renderCurrent);
els.frame.addEventListener("input", async () => {
  if (blockUnsavedEdits("changing frames")) {
    syncFrameControls(state.frameIndex);
    return;
  }
  stopPlayback();
  state.frameIndex = Number(els.frame.value);
  await loadFrame(true);
});
els.playButton.addEventListener("click", togglePlayback);
els.stillButton.addEventListener("click", exportStill);
els.videoButton.addEventListener("click", downloadVideo);
els.saveGraphButton.addEventListener("click", (event) => {
  event.preventDefault();
  saveGraphEdit();
});
els.resetGraphButton.addEventListener("click", (event) => {
  event.preventDefault();
  resetGraphEdit();
});
els.nextGraphButton.addEventListener("click", (event) => {
  event.preventDefault();
  moveToNextFrame();
});
els.exportCsvButton.addEventListener("click", (event) => {
  event.preventDefault();
  exportGraphCsvs();
});
els.annotationTab.addEventListener("click", () => selectRightPanel("annotation"));
els.evalTab.addEventListener("click", () => selectRightPanel("eval"));
for (const select of [els.worklistEditFilter, els.worklistValidationFilter]) {
  select.addEventListener("change", refreshWorklist);
}
els.continuousVideo.addEventListener("change", stopPlayback);
document.addEventListener("keydown", handleKeydown);
window.addEventListener("message", (event) => {
  if (event.origin === window.location.origin && event.data?.type === "demo:stop") cancelActiveWork();
});
window.addEventListener("pagehide", cancelActiveWork);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) cancelActiveWork();
});

initialize().catch((error) => {
  setLoading(false);
  drawEmpty(error.message);
  els.buildStatus.textContent = error.message;
});
