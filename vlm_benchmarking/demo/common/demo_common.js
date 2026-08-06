"use strict";

window.DemoCommon = (() => {
  async function fetchJson(url, options = {}) {
    const response = await fetch(url, {cache: "no-store", ...options});
    const text = await response.text();
    let payload = {};
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = {error: text.slice(0, 240)};
      }
    }
    if (!response.ok || payload.error) {
      const detail = payload.error || response.statusText || `HTTP ${response.status}`;
      throw new Error(`${response.status} ${detail}`.trim());
    }
    return payload;
  }

  function scopedApiRoot(scope = "api/") {
    return new URL(scope, window.location.href).pathname.replace(/\/$/, "");
  }

  function apiUrl(root, path) {
    return `${root.replace(/\/$/, "")}/${String(path).replace(/^\//, "")}`;
  }

  function query(params) {
    return new URLSearchParams(params).toString();
  }

  function fillSelect(select, items, getValue, getLabel) {
    select.replaceChildren();
    for (const item of items) {
      const option = document.createElement("option");
      option.value = getValue(item);
      option.textContent = getLabel(item);
      select.appendChild(option);
    }
  }

  function setOptions(select, items, valueKey = "id", labelKey = "name") {
    fillSelect(select, items, item => item[valueKey], item => item[labelKey]);
  }

  function hasOption(select, value) {
    return Array.from(select.options).some(option => option.value === value);
  }

  function prettyObjectName(value) {
    return String(value || "")
      .replace(/^akita_/, "")
      .replace(/^is_/, "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, char => char.toUpperCase());
  }

  function relationLabel(value) {
    const labels = {
      is_left_of: "left",
      is_right_of: "right",
      is_in_front_of: "front",
      is_behind: "behind",
      is_on_top_of: "top",
      is_below_of: "below",
      is_inside: "inside",
      contains: "contains",
    };
    return labels[value] || value;
  }

  function formatPercent(value) {
    return `${Math.round(Number(value || 0) * 100)}%`;
  }

  function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  function safeFilename(value) {
    return String(value || "").replace(/[^a-z0-9_-]+/gi, "_");
  }

  function setButtonBusy(button, value) {
    if (!button) return;
    button.classList.toggle("is-loading", value);
    button.setAttribute("aria-busy", value ? "true" : "false");
  }

  async function withButtonBusy(button, work) {
    setButtonBusy(button, true);
    try {
      return await work();
    } finally {
      setButtonBusy(button, false);
    }
  }

  function mediaRecorderType() {
    const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    return types.find(type => MediaRecorder.isTypeSupported(type)) || "";
  }

  function centerFromBbox(item) {
    const bbox = Array.isArray(item) ? item : item.bbox;
    const [x1, y1, x2, y2] = bbox;
    return [(x1 + x2) / 2, (y1 + y2) / 2];
  }

  function drawArrow(ctx, x1, y1, x2, y2, scale = 1) {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const head = 8 * scale;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }

  function drawEdgeLabel(ctx, text, x, y, color, scale = 1) {
    const metrics = ctx.measureText(text);
    const width = metrics.width + 6 * scale;
    const height = 14 * scale;
    ctx.save();
    ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
    ctx.fillRect(x - 2 * scale, y - 11 * scale, width, height);
    ctx.fillStyle = color;
    ctx.fillText(text, x + 1 * scale, y);
    ctx.restore();
  }

  function placeCanvasLabel(canvas, centerX, centerY, width, height, occupied) {
    const offsets = [0, -18, 18, -36, 36, -54, 54];
    for (const offset of offsets) {
      const left = Math.max(0, Math.min(canvas.width - width, centerX - width / 2));
      const top = Math.max(0, Math.min(canvas.height - height, centerY - 8 + offset));
      const candidate = {left, top, right: left + width, bottom: top + height};
      const overlaps = occupied.some(item => !(
        candidate.right + 3 < item.left || candidate.left > item.right + 3
        || candidate.bottom + 3 < item.top || candidate.top > item.bottom + 3
      ));
      if (!overlaps) {
        occupied.push(candidate);
        return candidate;
      }
    }
    const fallback = {
      left: Math.max(0, Math.min(canvas.width - width, centerX - width / 2)),
      top: Math.max(0, Math.min(canvas.height - height, centerY - 8)),
    };
    occupied.push({...fallback, right: fallback.left + width, bottom: fallback.top + height});
    return fallback;
  }

  function formatMetricValue(value) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(2) : "-";
  }

  function renderMetrics(container, metrics) {
    if (!container) return;
    const m = metrics || {};
    const items = [
      ["TP", m.tp],
      ["FP", m.fp],
      ["FN", m.fn],
      ["Precision", m.precision],
      ["Recall", m.recall],
      ["F1", m.f1],
    ];
    container.replaceChildren();
    for (const [name, value] of items) {
      const metric = document.createElement("div");
      metric.className = "metric";
      const number = document.createElement("strong");
      number.textContent = ["TP", "FP", "FN"].includes(name)
        ? (Number.isFinite(Number(value)) ? String(value) : "-")
        : formatMetricValue(value);
      const label = document.createElement("span");
      label.textContent = name;
      metric.append(number, label);
      container.appendChild(metric);
    }
  }

  function renderMetadataSummary(container, metadata, fallbackText = "Metadata signals are not available.") {
    if (!container) return;
    if (metadata) {
      const reliable = metadata.metadata_reliable !== false;
      const phase = prettyObjectName(metadata.phase || "unknown");
      const gates = (metadata.gates || []).join(", ") || "none";
      container.className = `metadata-summary ${reliable ? "good" : "warn"}`;
      container.textContent = `Phase: ${phase} | metadata ${reliable ? "reliable" : "fallback"} | gates: ${gates}`;
      return;
    }
    container.className = "metadata-summary warn";
    container.textContent = fallbackText;
  }

  function renderGroupedTriplets(container, items, options = {}) {
    if (!container) return;
    const triplets = items || [];
    const pretty = options.pretty || prettyObjectName;
    const selectedSubject = String(options.selectedSubject || "").toLowerCase();
    const selectedOnly = selectedSubject && selectedSubject !== "all";
    container.replaceChildren();
    if (!triplets.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state triplet-scope";
      empty.textContent = options.emptyText || "No triplets for this frame";
      container.appendChild(empty);
      return;
    }

    const groups = new Map();
    for (const triplet of triplets) {
      if (!groups.has(triplet.subject)) groups.set(triplet.subject, []);
      groups.get(triplet.subject).push(triplet);
    }
    const orderedGroups = Array.from(groups.entries()).sort(([a], [b]) => {
      if (selectedOnly && a.toLowerCase() === selectedSubject) return -1;
      if (selectedOnly && b.toLowerCase() === selectedSubject) return 1;
      return a.localeCompare(b);
    });
    for (const [subject, groupItems] of orderedGroups) {
      const details = document.createElement("details");
      details.className = "triplet-group";
      details.open = !selectedOnly || subject.toLowerCase() === selectedSubject;
      const summary = document.createElement("summary");
      const title = document.createElement("span");
      title.textContent = pretty(subject);
      title.title = subject;
      const count = document.createElement("span");
      count.className = "group-count";
      count.textContent = `${groupItems.length}`;
      summary.append(title, count);
      details.appendChild(summary);

      const list = document.createElement("div");
      list.className = "triplet-list";
      for (const triplet of groupItems) {
        const row = document.createElement("div");
        const classes = ["triplet-row"];
        if (triplet.visible === false) classes.push("is-muted");
        if (triplet.correct === false) classes.push("is-wrong");
        row.className = classes.join(" ");

        const subj = document.createElement("span");
        subj.className = "triplet-entity";
        subj.title = triplet.subject;
        subj.textContent = pretty(triplet.subject);
        const rel = document.createElement("span");
        rel.className = "relation-chip";
        rel.title = triplet.relation;
        rel.textContent = triplet.label || relationLabel(triplet.relation);
        const obj = document.createElement("span");
        obj.className = "triplet-entity";
        obj.title = triplet.object;
        obj.textContent = pretty(triplet.object);
        row.append(subj, rel, obj);

        if (options.showProvenance && triplet.source) {
          const provenance = document.createElement("span");
          provenance.className = "provenance";
          const confidence = Number.isFinite(Number(triplet.confidence))
            ? ` | confidence ${Number(triplet.confidence).toFixed(2)}`
            : "";
          provenance.textContent = `${triplet.source}${confidence}`;
          row.appendChild(provenance);
        }
        list.appendChild(row);
      }
      details.appendChild(list);
      container.appendChild(details);
    }
  }

  return {
    fetchJson,
    scopedApiRoot,
    apiUrl,
    query,
    fillSelect,
    setOptions,
    hasOption,
    prettyObjectName,
    relationLabel,
    formatPercent,
    wait,
    downloadBlob,
    safeFilename,
    setButtonBusy,
    withButtonBusy,
    mediaRecorderType,
    centerFromBbox,
    drawArrow,
    drawEdgeLabel,
    placeCanvasLabel,
    renderMetrics,
    renderMetadataSummary,
    renderGroupedTriplets,
  };
})();
