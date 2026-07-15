"use strict";

const DEMOS = {
  libero: {
    label: "LIBERO",
    title: "LIBERO Demo",
    appUrl: "libero/",
    healthUrl: "api/health",
    bundleUrl: "data/libero.json",
  },
  so101: {
    label: "SO101",
    title: "SO101 Demo",
    appUrl: "so101/",
    healthUrl: "so101/api/health",
    bundleUrl: "data/so101.json",
  },
};
const QUERY = new URLSearchParams(window.location.search);
const HASH_PARTS = window.location.hash.replace("#", "").split("/");

const elements = Object.fromEntries([
  "portalTitle", "liberoTab", "so101Tab", "liveView",
  "demoFrame", "staticView", "sampleMode", "sampleCamera", "sampleTask",
  "sampleSequence", "sampleFrame", "sampleFrameValue", "sampleArrows", "sampleLabels",
  "sampleBboxes", "sampleSceneStatus", "sampleCanvas", "sampleLoading",
  "sampleGraphType", "sampleTripletCount", "sampleMetricsScope", "sampleMetrics",
  "sampleMetadataSummary", "sampleTripletList",
].map((id) => [id, document.getElementById(id)]));

const context = elements.sampleCanvas.getContext("2d");
const state = {
  demo: HASH_PARTS[0] in DEMOS
    ? HASH_PARTS[0]
    : "libero",
  runtime: new Map(),
  bundles: new Map(),
  bundle: null,
  sequenceId: null,
  frameIndex: 0,
  image: null,
  requestId: 0,
};

function timeoutSignal(milliseconds) {
  const controller = new AbortController();
  window.setTimeout(() => controller.abort(), milliseconds);
  return controller.signal;
}

async function backendAvailable(config) {
  try {
    const response = await fetch(config.healthUrl, {
      cache: "no-store",
      signal: timeoutSignal(1400),
    });
    if (!response.ok) return false;
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) return false;
    await response.json();
    return true;
  } catch {
    return false;
  }
}

async function selectDemo(name, updateHash = true) {
  if (!(name in DEMOS)) return;
  state.demo = name;
  state.sequenceId = null;
  state.frameIndex = 0;
  const config = DEMOS[name];
  document.title = config.title;
  elements.portalTitle.textContent = config.title;
  for (const [key, value] of Object.entries(DEMOS)) {
    const tab = elements[`${key}Tab`];
    const active = key === name;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  if (updateHash) history.replaceState(null, "", `#${name}`);
  let available = state.runtime.get(name);
  if (available === undefined) {
    available = await backendAvailable(config);
    state.runtime.set(name, available);
  }

  if (available) {
    showLive(config);
    return;
  }
  await showStatic(config, available);
}

function showLive(config) {
  state.requestId += 1;
  elements.liveView.hidden = false;
  elements.staticView.hidden = true;
  const nextUrl = new URL(config.appUrl, window.location.href).href;
  if (elements.demoFrame.src !== nextUrl) elements.demoFrame.src = nextUrl;
}

async function showStatic(config, backendIsAvailable = false) {
  elements.liveView.hidden = true;
  elements.staticView.hidden = false;
  elements.sampleLoading.hidden = false;
  try {
    let bundle = state.bundles.get(state.demo);
    if (!bundle) {
      const response = await fetch(config.bundleUrl);
      if (!response.ok) throw new Error(`Sample data unavailable (${response.status})`);
      bundle = normalizeBundle(await response.json());
      state.bundles.set(state.demo, bundle);
    }
    state.bundle = bundle;
    populateStaticControls(bundle);
    await renderStaticFrame();
  } catch (error) {
    elements.sampleLoading.textContent = error.message;
    elements.sampleSceneStatus.textContent = error.message;
  }
}

function populateStaticControls(bundle) {
  const cameras = uniqueOptions(bundle.sequences, "camera", "camera_label");
  setOptions(elements.sampleCamera, cameras);
  const requestedCamera = QUERY.get("camera") || HASH_PARTS[1];
  const defaultCamera = cameras.some((item) => item.id === requestedCamera)
    ? requestedCamera
    : cameras.some((item) => item.id === bundle.default_camera)
      ? bundle.default_camera
    : cameras[0]?.id;
  elements.sampleCamera.value = defaultCamera || "";
  elements.sampleGraphType.textContent = bundle.graph_label;
  populateTaskControls(bundle);
}

function normalizeBundle(bundle) {
  if (Array.isArray(bundle.sequences)) return bundle;
  const sequence = {
    id: `${bundle.task}:${bundle.sequence_label}:${bundle.camera}`,
    task: bundle.task,
    task_label: bundle.task_label,
    sequence: bundle.sequence_label,
    sequence_label: bundle.sequence_label,
    camera: bundle.camera,
    camera_label: bundle.camera_label,
    frames: bundle.frames || [],
  };
  return {
    ...bundle,
    schema_version: 2,
    default_camera: bundle.camera,
    default_task: bundle.task,
    default_sequence: bundle.sequence_label,
    sequence_count: 1,
    frame_count: sequence.frames.length,
    sequences: [sequence],
  };
}

function uniqueOptions(items, idKey, nameKey) {
  const options = new Map();
  for (const item of items) {
    if (!options.has(item[idKey])) {
      options.set(item[idKey], {id: item[idKey], name: item[nameKey]});
    }
  }
  return [...options.values()];
}

function populateTaskControls(bundle) {
  const camera = elements.sampleCamera.value;
  const sequences = bundle.sequences.filter((item) => item.camera === camera);
  const tasks = uniqueOptions(sequences, "task", "task_label");
  const previousTask = currentSequence()?.task;
  const requestedTask = QUERY.get("task");
  setOptions(elements.sampleTask, tasks);
  const preferredTask = tasks.some((item) => item.id === previousTask)
    ? previousTask
    : tasks.some((item) => item.id === requestedTask)
      ? requestedTask
    : tasks.some((item) => item.id === bundle.default_task)
      ? bundle.default_task
      : tasks[0]?.id;
  elements.sampleTask.value = preferredTask || "";
  populateSequenceControls(bundle);
}

function populateSequenceControls(bundle) {
  const camera = elements.sampleCamera.value;
  const task = elements.sampleTask.value;
  const sequences = bundle.sequences.filter(
    (item) => item.camera === camera && item.task === task,
  );
  const options = uniqueOptions(sequences, "sequence", "sequence_label");
  const previousSequence = currentSequence()?.sequence;
  const requestedSequence = QUERY.get("sequence");
  setOptions(elements.sampleSequence, options);
  const preferredSequence = options.some((item) => item.id === previousSequence)
    ? previousSequence
    : options.some((item) => item.id === requestedSequence)
      ? requestedSequence
    : options.some((item) => item.id === bundle.default_sequence)
      ? bundle.default_sequence
      : options[0]?.id;
  elements.sampleSequence.value = preferredSequence || "";
  activateSequence(bundle, camera, task, elements.sampleSequence.value);
}

function activateSequence(bundle, camera, task, sequenceName) {
  const sequence = bundle.sequences.find(
    (item) => item.camera === camera
      && item.task === task
      && item.sequence === sequenceName,
  ) || null;
  state.sequenceId = sequence?.id || null;
  const requestedFrame = Number(QUERY.get("frame") || HASH_PARTS[2]);
  const requestedIndex = Number.isFinite(requestedFrame)
    ? sequence?.frames.findIndex((item) => item.frame === requestedFrame) ?? -1
    : -1;
  state.frameIndex = requestedIndex >= 0 ? requestedIndex : 0;
  const frameCount = sequence?.frames.length || 0;
  elements.sampleMode.textContent = `${bundle.sequence_count} sequences / ${frameCount} sampled frames`;
  elements.sampleFrame.min = "0";
  elements.sampleFrame.max = String(Math.max(0, frameCount - 1));
  elements.sampleFrame.value = String(state.frameIndex);
  elements.sampleFrame.disabled = frameCount < 2;
}

function setOptions(select, options) {
  select.replaceChildren();
  for (const item of options) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.name;
    option.title = item.name;
    select.appendChild(option);
  }
}

function currentFrame() {
  return currentSequence()?.frames[state.frameIndex] || null;
}

function currentSequence() {
  return state.bundle?.sequences.find((item) => item.id === state.sequenceId) || null;
}

async function renderStaticFrame() {
  const frame = currentFrame();
  const sequence = currentSequence();
  if (!frame || !sequence) return;
  const requestId = ++state.requestId;
  elements.sampleLoading.hidden = false;
  elements.sampleLoading.textContent = "Loading";
  elements.sampleFrame.value = String(state.frameIndex);
  elements.sampleFrameValue.textContent = `${frame.frame} | ${state.frameIndex + 1} / ${sequence.frames.length}`;
  elements.sampleSceneStatus.textContent = `${sequence.camera_label} | ${sequence.sequence_label} | frame ${frame.frame}`;
  renderTriplets(frame.relations || []);
  try {
    const image = await loadImage(new URL(frame.image, window.location.href).href);
    if (requestId !== state.requestId) return;
    state.image = image;
    drawFrame(frame, image);
    elements.sampleLoading.hidden = true;
  } catch (error) {
    if (requestId !== state.requestId) return;
    elements.sampleLoading.textContent = error.message;
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Unable to load sample image"));
    image.src = url;
  });
}

function drawFrame(frame, image = state.image) {
  if (!image) return;
  elements.sampleCanvas.width = frame.width || image.naturalWidth;
  elements.sampleCanvas.height = frame.height || image.naturalHeight;
  context.clearRect(0, 0, elements.sampleCanvas.width, elements.sampleCanvas.height);
  context.drawImage(image, 0, 0, elements.sampleCanvas.width, elements.sampleCanvas.height);
  if (elements.sampleBboxes.checked) drawBboxes(frame.bboxes || []);
  if (elements.sampleArrows.checked) drawRelations(frame.relations || []);
}

function drawBboxes(bboxes) {
  const scale = Math.max(1, elements.sampleCanvas.width / 640);
  context.save();
  context.lineWidth = 1.5 * scale;
  context.strokeStyle = "#f0c84b";
  context.fillStyle = "rgba(10, 14, 18, 0.76)";
  context.font = `${11 * scale}px Arial`;
  for (const item of bboxes) {
    const [x1, y1, x2, y2] = item.bbox;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = prettyName(item.object);
    const width = context.measureText(label).width + 7 * scale;
    const top = Math.max(0, y1 - 15 * scale);
    context.fillRect(x1, top, width, 15 * scale);
    context.fillStyle = "#f7fafb";
    context.fillText(label, x1 + 3 * scale, top + 11 * scale);
    context.fillStyle = "rgba(10, 14, 18, 0.76)";
  }
  context.restore();
}

function drawRelations(relations) {
  const scale = Math.max(1, elements.sampleCanvas.width / 640);
  const occupiedLabels = [];
  context.save();
  context.lineWidth = 2 * scale;
  context.strokeStyle = "#00b77e";
  context.fillStyle = "#00b77e";
  context.font = `bold ${11 * scale}px Arial`;
  for (const relation of relations) {
    const [x1, y1] = relation.start;
    const [x2, y2] = relation.end;
    drawArrow(x1, y1, x2, y2, scale);
    if (elements.sampleLabels.checked) {
      const x = x1 * 0.56 + x2 * 0.44;
      const y = y1 * 0.56 + y2 * 0.44;
      drawRelationLabel(relation.label, x, y, scale, occupiedLabels);
    }
  }
  context.restore();
}

function drawArrow(x1, y1, x2, y2, scale) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 9 * scale;
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.stroke();
  context.beginPath();
  context.moveTo(x2, y2);
  context.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
  context.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
  context.closePath();
  context.fill();
}

function drawRelationLabel(text, x, y, scale, occupied) {
  const width = context.measureText(text).width + 8 * scale;
  const height = 16 * scale;
  const position = placeStaticLabel(x, y, width, height, occupied, scale);
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.9)";
  context.fillRect(position.left, position.top, width, height);
  context.fillStyle = "#087a55";
  context.fillText(text, position.left + 4 * scale, position.top + 12 * scale);
  context.restore();
}

function placeStaticLabel(x, y, width, height, occupied, scale) {
  const offsets = [0, -18, 18, -36, 36, -54, 54].map((value) => value * scale);
  for (const offset of offsets) {
    const left = Math.max(0, Math.min(elements.sampleCanvas.width - width, x - 3 * scale));
    const top = Math.max(0, Math.min(elements.sampleCanvas.height - height, y - 12 * scale + offset));
    const candidate = {left, top, right: left + width, bottom: top + height};
    const overlaps = occupied.some((item) => !(
      candidate.right + 3 * scale < item.left || candidate.left > item.right + 3 * scale
      || candidate.bottom + 3 * scale < item.top || candidate.top > item.bottom + 3 * scale
    ));
    if (!overlaps) {
      occupied.push(candidate);
      return candidate;
    }
  }
  const fallback = {
    left: Math.max(0, Math.min(elements.sampleCanvas.width - width, x - 3 * scale)),
    top: Math.max(0, Math.min(elements.sampleCanvas.height - height, y - 12 * scale)),
  };
  occupied.push({...fallback, right: fallback.left + width, bottom: fallback.top + height});
  return fallback;
}

function renderTriplets(relations) {
  elements.sampleTripletCount.textContent = String(relations.length);
  DemoCommon.renderMetrics(elements.sampleMetrics, null);
  DemoCommon.renderMetadataSummary(
    elements.sampleMetadataSummary,
    null,
    "Static GitHub Pages sample | prediction metrics unavailable",
  );
  DemoCommon.renderGroupedTriplets(elements.sampleTripletList, relations, {
    pretty: prettyName,
    emptyText: "No visible relations",
  });
}

function prettyName(value) {
  const raw = String(value || "");
  const indexedBowl = /^akita_black_bowl_\d+$/.test(raw);
  return raw
    .replace(/_(\d+)$/, indexedBowl ? " $1" : "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

for (const name of Object.keys(DEMOS)) {
  elements[`${name}Tab`].addEventListener("click", () => selectDemo(name));
}

elements.sampleFrame.addEventListener("input", () => {
  state.frameIndex = Number(elements.sampleFrame.value);
  renderStaticFrame();
});

elements.sampleCamera.addEventListener("change", () => {
  populateTaskControls(state.bundle);
  renderStaticFrame();
});

elements.sampleTask.addEventListener("change", () => {
  populateSequenceControls(state.bundle);
  renderStaticFrame();
});

elements.sampleSequence.addEventListener("change", () => {
  activateSequence(
    state.bundle,
    elements.sampleCamera.value,
    elements.sampleTask.value,
    elements.sampleSequence.value,
  );
  renderStaticFrame();
});

for (const control of [elements.sampleArrows, elements.sampleLabels, elements.sampleBboxes]) {
  control.addEventListener("change", () => drawFrame(currentFrame()));
}

window.addEventListener("hashchange", () => {
  const name = window.location.hash.replace("#", "").split("/")[0];
  if (name in DEMOS && name !== state.demo) selectDemo(name, false);
});

window.addEventListener("resize", () => {
  const shouldShowLive = state.runtime.get(state.demo);
  if (shouldShowLive === !elements.liveView.hidden) return;
  selectDemo(state.demo, false);
});

selectDemo(state.demo, false);
