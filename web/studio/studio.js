const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function formData(form) {
  const data = new FormData(form);
  const out = {};
  for (const [key, value] of data.entries()) {
    out[key] = value;
  }
  for (const input of form.querySelectorAll('input[type="checkbox"]')) {
    out[input.name] = input.checked;
  }
  return out;
}

function optionalPath(value) {
  return value && value.trim() ? value.trim() : null;
}

function fileUrl(path) {
  return `/api/file?path=${encodeURIComponent(path)}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

const regionEditor = {
  canvas: null,
  context: null,
  source: new Image(),
  comparison: new Image(),
  sourceReady: false,
  comparisonReady: false,
  tool: "polygon",
  locks: [],
  draft: [],
  dragStart: null,
  pointerDown: false,
  history: [],
  future: [],
};

function editorValue(id) {
  return document.getElementById(id).value;
}

function editorNumber(id) {
  return Number(editorValue(id));
}

function pointerPoint(event) {
  const rect = regionEditor.canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  };
}

function hexRgb(value) {
  const hex = value.replace("#", "");
  return {
    r: Number.parseInt(hex.slice(0, 2), 16),
    g: Number.parseInt(hex.slice(2, 4), 16),
    b: Number.parseInt(hex.slice(4, 6), 16),
  };
}

function regionStyle() {
  const semantic = editorValue("region-semantic");
  const mode = editorValue("region-mode");
  return {
    id: `${semantic}_${Date.now()}_${regionEditor.locks.length + 1}`,
    label: semantic,
    semantic,
    mode,
    color: mode === "colorize" ? hexRgb(editorValue("region-color")) : null,
    strength: editorNumber("region-strength"),
    feather_radius: editorNumber("region-feather"),
    enabled: true,
  };
}

function snapshotRegions() {
  regionEditor.history.push(JSON.stringify(regionEditor.locks));
  if (regionEditor.history.length > 100) regionEditor.history.shift();
  regionEditor.future = [];
}

function restoreRegions(serialized) {
  regionEditor.locks = JSON.parse(serialized);
  regionEditor.draft = [];
  renderRegions();
}

function undoRegion() {
  const previous = regionEditor.history.pop();
  if (!previous) return;
  regionEditor.future.push(JSON.stringify(regionEditor.locks));
  restoreRegions(previous);
}

function redoRegion() {
  const next = regionEditor.future.pop();
  if (!next) return;
  regionEditor.history.push(JSON.stringify(regionEditor.locks));
  restoreRegions(next);
}

function normalizedShape(type, start, end, points = []) {
  if (type === "rectangle") {
    return {
      shape: "rectangle",
      x: Math.min(start.x, end.x),
      y: Math.min(start.y, end.y),
      width: Math.abs(end.x - start.x),
      height: Math.abs(end.y - start.y),
    };
  }
  if (type === "ellipse") {
    return {
      shape: "ellipse",
      center_x: (start.x + end.x) / 2,
      center_y: (start.y + end.y) / 2,
      radius_x: Math.abs(end.x - start.x) / 2,
      radius_y: Math.abs(end.y - start.y) / 2,
    };
  }
  return { shape: "polygon", points };
}

function commitRegion(region) {
  snapshotRegions();
  regionEditor.locks.push({ ...regionStyle(), region });
  regionEditor.draft = [];
  renderRegions();
}

function finishPolygon() {
  if (regionEditor.draft.length < 3) return;
  commitRegion(normalizedShape("polygon", null, null, regionEditor.draft.slice()));
}

function shapePath(context, region, width, height) {
  context.beginPath();
  if (region.shape === "rectangle") {
    context.rect(region.x * width, region.y * height, region.width * width, region.height * height);
  } else if (region.shape === "ellipse") {
    context.ellipse(
      region.center_x * width,
      region.center_y * height,
      Math.max(1, region.radius_x * width),
      Math.max(1, region.radius_y * height),
      0,
      0,
      Math.PI * 2,
    );
  } else if (region.points.length) {
    context.moveTo(region.points[0].x * width, region.points[0].y * height);
    region.points.slice(1).forEach((point) => context.lineTo(point.x * width, point.y * height));
    context.closePath();
  }
}

function lockCss(lock) {
  if (lock.mode === "preserve") return "#ffffff";
  if (lock.mode === "exclude") return "#d43f52";
  const color = lock.color || { r: 40, g: 104, b: 188 };
  return `rgb(${color.r},${color.g},${color.b})`;
}

function drawRegionLock(lock) {
  if (!lock.enabled) return;
  const context = regionEditor.context;
  shapePath(context, lock.region, regionEditor.canvas.width, regionEditor.canvas.height);
  const color = lockCss(lock);
  context.fillStyle = color.replace("rgb", "rgba").replace(")", ",0.24)");
  if (color.startsWith("#")) context.fillStyle = `${color}3d`;
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.fill();
  context.stroke();
}

function drawRegionCanvas() {
  if (!regionEditor.context) return;
  const { context, canvas } = regionEditor;
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (regionEditor.sourceReady) context.drawImage(regionEditor.source, 0, 0, canvas.width, canvas.height);
  const opacity = editorNumber("region-compare-opacity");
  if (regionEditor.comparisonReady && opacity > 0) {
    context.save();
    context.globalAlpha = opacity;
    context.drawImage(regionEditor.comparison, 0, 0, canvas.width, canvas.height);
    context.restore();
  }
  regionEditor.locks.forEach(drawRegionLock);
  if (regionEditor.draft.length) {
    context.beginPath();
    context.moveTo(regionEditor.draft[0].x * canvas.width, regionEditor.draft[0].y * canvas.height);
    regionEditor.draft.slice(1).forEach((point) => context.lineTo(point.x * canvas.width, point.y * canvas.height));
    context.strokeStyle = editorValue("region-color");
    context.lineWidth = 2;
    context.stroke();
  }
}

function renderRegionLayers() {
  const list = document.getElementById("region-layers");
  list.innerHTML = "";
  [...regionEditor.locks].reverse().forEach((lock, reverseIndex) => {
    const index = regionEditor.locks.length - reverseIndex - 1;
    const row = document.createElement("div");
    row.className = "layer-row";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = lock.enabled;
    toggle.addEventListener("change", () => {
      snapshotRegions();
      lock.enabled = toggle.checked;
      drawRegionCanvas();
    });
    const name = document.createElement("span");
    name.textContent = `${lock.semantic} · ${lock.mode}`;
    const remove = document.createElement("button");
    remove.className = "layer-delete";
    remove.textContent = "×";
    remove.title = "Delete region";
    remove.addEventListener("click", () => {
      snapshotRegions();
      regionEditor.locks.splice(index, 1);
      renderRegions();
    });
    row.append(toggle, name, remove);
    list.append(row);
  });
}

function renderRegions() {
  drawRegionCanvas();
  renderRegionLayers();
  document.getElementById("region-status").textContent =
    `${regionEditor.locks.length} saved region(s), ${regionEditor.draft.length} draft point(s)`;
}

async function loadRegionPage() {
  const sourcePath = editorValue("region-source");
  regionEditor.sourceReady = false;
  regionEditor.comparisonReady = false;
  await new Promise((resolve, reject) => {
    regionEditor.source.onload = resolve;
    regionEditor.source.onerror = reject;
    regionEditor.source.src = `${fileUrl(sourcePath)}&t=${Date.now()}`;
  });
  regionEditor.sourceReady = true;
  const maxWidth = 1200;
  const scale = Math.min(1, maxWidth / regionEditor.source.naturalWidth);
  regionEditor.canvas.width = Math.max(1, Math.round(regionEditor.source.naturalWidth * scale));
  regionEditor.canvas.height = Math.max(1, Math.round(regionEditor.source.naturalHeight * scale));
  const comparisonPath = editorValue("region-compare").trim();
  if (comparisonPath) {
    try {
      await new Promise((resolve, reject) => {
        regionEditor.comparison.onload = resolve;
        regionEditor.comparison.onerror = reject;
        regionEditor.comparison.src = `${fileUrl(comparisonPath)}&t=${Date.now()}`;
      });
      regionEditor.comparisonReady = true;
    } catch (_) {
      regionEditor.comparisonReady = false;
    }
  }
  renderRegions();
}

function pointInsideLock(point, lock) {
  const region = lock.region;
  if (region.shape === "rectangle") {
    return point.x >= region.x && point.x <= region.x + region.width && point.y >= region.y && point.y <= region.y + region.height;
  }
  if (region.shape === "ellipse") {
    const dx = (point.x - region.center_x) / Math.max(region.radius_x, 0.0001);
    const dy = (point.y - region.center_y) / Math.max(region.radius_y, 0.0001);
    return dx * dx + dy * dy <= 1;
  }
  let inside = false;
  for (let i = 0, j = region.points.length - 1; i < region.points.length; j = i, i += 1) {
    const a = region.points[i];
    const b = region.points[j];
    if ((a.y > point.y) !== (b.y > point.y) && point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

function regionPointerDown(event) {
  event.preventDefault();
  const point = pointerPoint(event);
  if (regionEditor.tool === "fill") {
    const index = regionEditor.locks.findLastIndex((lock) => pointInsideLock(point, lock));
    if (index >= 0) {
      snapshotRegions();
      const region = regionEditor.locks[index].region;
      regionEditor.locks[index] = { ...regionStyle(), region };
      renderRegions();
    }
    return;
  }
  if (regionEditor.tool === "erase") {
    const index = regionEditor.locks.findLastIndex((lock) => pointInsideLock(point, lock));
    if (index >= 0) {
      snapshotRegions();
      regionEditor.locks.splice(index, 1);
      renderRegions();
    }
    return;
  }
  if (regionEditor.tool === "polygon") {
    regionEditor.draft.push(point);
    renderRegions();
    return;
  }
  regionEditor.pointerDown = true;
  regionEditor.dragStart = point;
  regionEditor.draft = [point];
  regionEditor.canvas.setPointerCapture(event.pointerId);
}

function regionPointerMove(event) {
  if (!regionEditor.pointerDown) return;
  const point = pointerPoint(event);
  if (regionEditor.tool === "brush") {
    const previous = regionEditor.draft.at(-1);
    if (!previous || Math.hypot(point.x - previous.x, point.y - previous.y) > 0.004) regionEditor.draft.push(point);
  } else {
    regionEditor.draft = [regionEditor.dragStart, point];
  }
  drawRegionCanvas();
}

function regionPointerUp(event) {
  if (!regionEditor.pointerDown) return;
  regionEditor.pointerDown = false;
  const point = pointerPoint(event);
  if (regionEditor.tool === "brush") {
    if (regionEditor.draft.length >= 3) finishPolygon();
  } else if (Math.hypot(point.x - regionEditor.dragStart.x, point.y - regionEditor.dragStart.y) > 0.005) {
    commitRegion(normalizedShape(regionEditor.tool, regionEditor.dragStart, point));
  } else {
    regionEditor.draft = [];
    renderRegions();
  }
}

async function saveRegionDocument() {
  const documentValue = {
    version: 1,
    page_id: editorValue("region-source").split(/[\\/]/).pop().replace(/\.[^.]+$/, ""),
    coordinate_space: "normalized",
    source_width: regionEditor.source.naturalWidth || null,
    source_height: regionEditor.source.naturalHeight || null,
    locks: regionEditor.locks,
  };
  const result = await api("/api/brain/region-locks/save", {
    method: "POST",
    body: JSON.stringify({ output_path: editorValue("region-output"), document: documentValue }),
  });
  document.getElementById("region-status").textContent = `Saved ${result.locks} region(s) to ${result.output_path}`;
}

async function loadConsistencyDashboard() {
  const response = await fetch(`${fileUrl(editorValue("consistency-report"))}&t=${Date.now()}`);
  if (!response.ok) throw new Error("Could not load brain report");
  const report = await response.json();
  document.getElementById("consistency-summary").innerHTML = [
    ["Consistency", `${Math.round((report.consistency_score || 0) * 100)}%`],
    ["Pages", report.analyzed_pages || 0],
    ["Palette locks", (report.palette_locks || []).length],
    ["Warnings", (report.warnings || []).length],
  ].map(([label, value]) => `<div class="metric"><strong>${value}</strong>${label}</div>`).join("");
  const pages = [...(report.pages || [])]
    .map((page) => ({ ...page, risk: (page.palette_drift_score || 0) * 0.72 + (page.tone_drift_score || 0) * 0.28 }))
    .sort((a, b) => b.risk - a.risk);
  document.getElementById("consistency-pages").innerHTML = pages.map((page) =>
    `<div class="risk-row"><strong>${String(page.index).padStart(3, "0")}</strong><div class="risk-bar"><span style="width:${Math.min(100, page.risk * 100)}%"></span></div><span>${Math.round(page.risk * 100)}%</span></div>`,
  ).join("");
}

function initializeRegionEditor() {
  regionEditor.canvas = document.getElementById("region-canvas");
  regionEditor.context = regionEditor.canvas.getContext("2d");
  regionEditor.canvas.addEventListener("pointerdown", regionPointerDown);
  regionEditor.canvas.addEventListener("pointermove", regionPointerMove);
  regionEditor.canvas.addEventListener("pointerup", regionPointerUp);
  document.querySelectorAll(".region-tool").forEach((button) => button.addEventListener("click", () => {
    regionEditor.tool = button.dataset.tool;
    document.querySelectorAll(".region-tool").forEach((item) => item.classList.toggle("active", item === button));
  }));
  document.getElementById("region-load").addEventListener("click", () => loadRegionPage().catch((error) => document.getElementById("region-status").textContent = error.message));
  document.getElementById("region-save").addEventListener("click", () => saveRegionDocument().catch((error) => document.getElementById("region-status").textContent = error.message));
  document.getElementById("region-finish").addEventListener("click", finishPolygon);
  document.getElementById("region-undo").addEventListener("click", undoRegion);
  document.getElementById("region-redo").addEventListener("click", redoRegion);
  document.getElementById("region-clear").addEventListener("click", () => { snapshotRegions(); regionEditor.locks = []; renderRegions(); });
  document.getElementById("region-compare-opacity").addEventListener("input", drawRegionCanvas);
  document.getElementById("consistency-load").addEventListener("click", () => loadConsistencyDashboard().catch((error) => document.getElementById("consistency-summary").textContent = error.message));
  window.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      event.shiftKey ? redoRegion() : undoRegion();
    }
  });
}

function activate(view) {
  $$(".tab").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  $$(".view").forEach((section) => section.classList.toggle("active", section.id === `view-${view}`));
}

async function refreshHealth() {
  try {
    const health = await api("/health");
    $("#health").textContent = `${health.status} | ${health.role}`;
  } catch (error) {
    $("#health").textContent = error.message;
  }
}

async function loadStyles() {
  const styles = await api("/api/styles");
  $("#styles-list").innerHTML = styles
    .map((style) => `<div class="style-pill"><strong>${style.id}</strong><br>${style.name}</div>`)
    .join("");
}

async function runInspect() {
  const data = formData($("#inspect-form"));
  const result = await api("/api/inspect", {
    method: "POST",
    body: JSON.stringify({
      request: {
        input_path: data.input_path,
        debug_dir: optionalPath(data.debug_dir),
      },
    }),
  });
  $("#inspect-output").textContent = pretty({
    classification: result.page_classification,
    size: result.size,
    preprocess: result.preprocess,
    panels: result.panels.length,
    tiles: result.tiles.length,
    debug_files: result.debug_files,
  });
}

async function queueColorize() {
  const data = formData($("#colorize-form"));
  const request = {
    input_path: data.input_path,
    output_path: data.output_path,
    debug_dir: optionalPath(data.debug_dir),
    style_path: optionalPath(data.style_path),
    user_hints_path: optionalPath(data.user_hints_path),
    region_locks_path: optionalPath(data.region_locks_path),
    use_mock_backend: data.use_mock_backend,
  };
  const job = await api("/api/colorize", {
    method: "POST",
    body: JSON.stringify({ request, ads_token: null, premium_account_id: null }),
  });
  $("#colorize-output").textContent = pretty(job);
  await pollJob(job.id, data.output_path);
}

async function pollJob(id, outputPath) {
  for (let i = 0; i < 80; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    const job = await api(`/api/jobs/${id}`);
    $("#colorize-output").textContent = pretty(job);
    if (job.status === "complete") {
      const image = $("#preview-image");
      image.src = `${fileUrl(outputPath)}&t=${Date.now()}`;
      image.style.display = "block";
      await refreshJobs();
      return;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "job failed");
    }
  }
  throw new Error("job timed out");
}

async function queueLinkColorize() {
  const data = formData($("#link-form"));
  const request = {
    input_path: "",
    output_path: "",
    style_path: optionalPath(data.style_path),
    region_locks_path: optionalPath(data.region_locks_path),
    use_mock_backend: data.use_mock_backend,
  };
  const job = await api("/api/link/colorize", {
    method: "POST",
    body: JSON.stringify({
      source_url: data.source_url,
      output_dir: data.output_dir,
      downloads_dir: optionalPath(data.downloads_dir),
      max_pages: Number(data.max_pages || 8),
      continue_on_error: true,
      template: request,
      user_agent: null,
    }),
  });
  $("#link-output").textContent = pretty(job);
  await pollLinkJob(job.id);
}

async function pollLinkJob(id) {
  for (let i = 0; i < 120; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    const job = await api(`/api/jobs/${id}`);
    $("#link-output").textContent = pretty(job);
    if (job.status === "complete") {
      const result = await api(`/api/jobs/${id}/result`);
      $("#link-output").textContent = pretty(result);
      const firstPage = result.result?.pages?.find((page) => page.status === "complete");
      if (firstPage?.output_path) {
        const image = $("#link-preview-image");
        image.src = `${fileUrl(firstPage.output_path)}&t=${Date.now()}`;
        image.style.display = "block";
      }
      await refreshJobs();
      return;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "link job failed");
    }
  }
  throw new Error("link job timed out");
}

async function importChapter() {
  const data = formData($("#chapter-form"));
  const result = await api("/api/chapter/import", {
    method: "POST",
    body: JSON.stringify({
      input_path: data.input_path,
      output_dir: data.output_dir,
      manifest_path: optionalPath(data.manifest_path),
      max_pages: Number(data.max_pages || 500),
      overwrite: true,
    }),
  });
  $("#chapter-output").textContent = pretty(result);
}

async function exportChapter() {
  const data = formData($("#chapter-form"));
  const result = await api("/api/chapter/export", {
    method: "POST",
    body: JSON.stringify({
      input_dir: data.export_input_dir,
      output_file: data.export_file,
      manifest_path: optionalPath(data.manifest_path),
    }),
  });
  $("#chapter-output").textContent = pretty(result);
}

async function colorizeChapter() {
  const data = formData($("#chapter-form"));
  const job = await api("/api/chapter/colorize", {
    method: "POST",
    body: JSON.stringify({
      input_path: data.input_path,
      work_dir: data.work_dir,
      output_dir: data.color_output_dir,
      export_path: optionalPath(data.export_file),
      gallery_path: optionalPath(data.gallery_path),
      manifest_path: optionalPath(data.manifest_path),
      brain_report_path: optionalPath(data.brain_report_path),
      locked_style_path: optionalPath(data.locked_style_path),
      palette_memory_path: optionalPath(data.palette_memory_path),
      palette_memory_style_path: optionalPath(data.palette_memory_style_path),
      safety_report_path: optionalPath(data.safety_report_path),
      max_pages: Number(data.max_pages || 500),
      resume: data.resume,
      safety_gate: true,
      enforce_safety_gate: data.enforce_safety_gate,
      consistency_retry: data.consistency_retry,
      retry_drift_threshold: Number(data.retry_drift_threshold || 0.18),
      retry_tone_threshold: 0.34,
      retry_max_pages: Number(data.retry_max_pages || 24),
      template: {
        input_path: "",
        output_path: "",
        style_path: optionalPath(data.style_path),
        region_locks_path: optionalPath(data.region_locks_path),
        use_mock_backend: data.use_mock_backend,
      },
    }),
  });
  $("#chapter-output").textContent = pretty(job);
  await pollChapterJob(job.id);
}

async function pollChapterJob(id) {
  for (let i = 0; i < 180; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    const job = await api(`/api/jobs/${id}`);
    $("#chapter-output").textContent = pretty(job);
    if (job.status === "complete") {
      const result = await api(`/api/jobs/${id}/result`);
      $("#chapter-output").textContent = pretty(result);
      await refreshJobs();
      return;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "chapter job failed");
    }
  }
  throw new Error("chapter job timed out");
}

async function refreshJobs() {
  const jobs = await api("/api/jobs");
  $("#jobs-table").innerHTML = jobs
    .map(
      (job) => `<tr>
        <td>${job.id}</td>
        <td>${job.status}</td>
        <td>${job.output_path || ""}</td>
        <td><button data-job="${job.id}" class="result-btn">Open</button></td>
      </tr>`,
    )
    .join("");
  $$(".result-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const result = await api(`/api/jobs/${button.dataset.job}/result`);
      $("#inspect-output").textContent = pretty(result);
      activate("inspect");
    });
  });
}

async function saveCorrection() {
  const data = formData($("#correction-form"));
  const payload = {
    project: data.project,
    page: data.page,
    x: Number(data.x),
    y: Number(data.y),
    width: Number(data.width),
    height: Number(data.height),
    label: optionalPath(data.label),
    old: data.old,
    new: data.new,
    character: optionalPath(data.character),
  };
  const result = await api("/api/corrections", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  $("#memory-output").textContent = pretty(result);
}

async function exportCorrectionHints() {
  const data = formData($("#correction-form"));
  const result = await api("/api/projects/export-hints", {
    method: "POST",
    body: JSON.stringify({
      project: data.project,
      page: data.page,
      output: data.hints_output,
    }),
  });
  $("#memory-output").textContent = pretty(result);
}

function renderPaletteMemory(memory) {
  const entries = memory.entries || [];
  $("#palette-memory-list").innerHTML = entries
    .map(
      (entry) => `<button class="swatch" data-entry="${entry.id}" title="${entry.id}">
        <span style="background:${entry.color}"></span>
        <strong>${entry.role}</strong>
        <small>${entry.color} ${Math.round((entry.confidence || 0) * 100)}%</small>
      </button>`,
    )
    .join("");
  $$(".swatch").forEach((button) => {
    button.addEventListener("click", () => {
      const entry = entries.find((item) => item.id === button.dataset.entry);
      if (!entry) return;
      const form = $("#correction-form");
      form.elements.palette_edit_id.value = entry.id;
      form.elements.palette_edit_role.value = entry.role;
      form.elements.palette_edit_color.value = entry.color;
      form.elements.palette_edit_tolerance.value = entry.tolerance || 12;
      form.elements.palette_edit_locked.checked = Boolean(entry.locked);
      form.elements.palette_edit_notes.value = entry.notes || "";
    });
  });
}

async function buildPaletteMemory() {
  const data = formData($("#correction-form"));
  const memory = await api("/api/brain/palette-memory", {
    method: "POST",
    body: JSON.stringify({
      brain_report_path: data.palette_brain_report,
      output_path: data.palette_memory_output,
      style_output_path: optionalPath(data.palette_style_output),
      base_style_path: optionalPath(data.palette_base_style),
      id: optionalPath(data.palette_memory_id),
      name: optionalPath(data.palette_memory_name),
      max_entries: 12,
      min_confidence: 0.18,
    }),
  });
  renderPaletteMemory(memory);
  $("#memory-output").textContent = pretty(memory);
}

async function editPaletteMemory() {
  const data = formData($("#correction-form"));
  const memory = await api("/api/brain/palette-memory/edit", {
    method: "POST",
    body: JSON.stringify({
      memory_path: data.palette_memory_output,
      style_output_path: optionalPath(data.palette_style_output),
      base_style_path: optionalPath(data.palette_base_style),
      edits: [
        {
          id: data.palette_edit_id,
          role: optionalPath(data.palette_edit_role),
          color: optionalPath(data.palette_edit_color),
          locked: data.palette_edit_locked,
          confidence: 0.99,
          tolerance: Number(data.palette_edit_tolerance || 12),
          notes: optionalPath(data.palette_edit_notes),
        },
      ],
    }),
  });
  renderPaletteMemory(memory);
  $("#memory-output").textContent = pretty(memory);
}

async function runDoctor() {
  const data = formData($("#doctor-form"));
  const result = await api("/api/doctor", {
    method: "POST",
    body: JSON.stringify({
      config_path: data.config_path,
      styles_dir: data.styles_dir,
      metadata_path: data.metadata_path,
      jobs_dir: data.jobs_dir,
      studio_dir: data.studio_dir,
      sample_input: data.sample_input,
      python: data.python,
    }),
  });
  $("#doctor-output").textContent = pretty(result);
}

function wire() {
  initializeRegionEditor();
  $$(".tab").forEach((button) => button.addEventListener("click", () => activate(button.dataset.view)));
  $("#inspect-run").addEventListener("click", () => runInspect().catch((error) => $("#inspect-output").textContent = error.message));
  $("#colorize-run").addEventListener("click", () => queueColorize().catch((error) => $("#colorize-output").textContent = error.message));
  $("#link-run").addEventListener("click", () => queueLinkColorize().catch((error) => $("#link-output").textContent = error.message));
  $("#chapter-import").addEventListener("click", () => importChapter().catch((error) => $("#chapter-output").textContent = error.message));
  $("#chapter-export").addEventListener("click", () => exportChapter().catch((error) => $("#chapter-output").textContent = error.message));
  $("#chapter-colorize").addEventListener("click", () => colorizeChapter().catch((error) => $("#chapter-output").textContent = error.message));
  $("#jobs-refresh").addEventListener("click", () => refreshJobs());
  $("#correction-save").addEventListener("click", () => saveCorrection().catch((error) => $("#memory-output").textContent = error.message));
  $("#correction-export-hints").addEventListener("click", () => exportCorrectionHints().catch((error) => $("#memory-output").textContent = error.message));
  $("#palette-memory-build").addEventListener("click", () => buildPaletteMemory().catch((error) => $("#memory-output").textContent = error.message));
  $("#palette-memory-edit").addEventListener("click", () => editPaletteMemory().catch((error) => $("#memory-output").textContent = error.message));
  $("#doctor-run").addEventListener("click", () => runDoctor().catch((error) => $("#doctor-output").textContent = error.message));
}

wire();
refreshHealth();
loadStyles().catch((error) => $("#styles-list").textContent = error.message);
refreshJobs().catch(() => {});
