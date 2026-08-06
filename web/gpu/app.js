// MangaHue GPU Studio — client-side DDColor inference with WebGPU, then
// server finishing via /api/gpu/colorize. Port of the Rust/Python worker
// pipeline (ddcolor_onnx_worker.py) to JS with the same conventions.

const MODELS = {
  full: {
    url: "/models/ddcolor-full-fp16.onnx",
    size: "436 MB",
    label: "DDColor-full fp16 (best quality)",
  },
  tiny: {
    url: "/models/ddcolor-tiny-fp16.onnx",
    size: "130 MB",
    label: "DDColor-tiny fp16 (faster load)",
  },
};
const MODEL_SIZE = 512;
const TILE = 512;
const OVERLAP = 48;
const AB_SCALE_DEFAULT = 1.0;

const $ = (id) => document.getElementById(id);
const logEl = $("log");
const barEl = $("progressBar");
const barWrapEl = $("progressWrap");

function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function setProgress(pct, text) {
  barWrapEl.hidden = false;
  barEl.style.width = pct + "%";
  $("progressText").textContent = text;
}

function hideProgress() {
  barWrapEl.hidden = true;
}

let session = null;
let sourceCanvas = null;
let abScale = AB_SCALE_DEFAULT;
let modelBytes = null; // cached downloaded model (per tab session)

// ── Color math (matches ddcolor_onnx_worker.py) ──────────────────────────
function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
function linearToSrgb(c) {
  c = Math.max(0, Math.min(1, c));
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}
function rgbToLabL(r, g, b) {
  const lr = srgbToLinear(r), lg = srgbToLinear(g), lb = srgbToLinear(b);
  const y = 0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb;
  const delta = 6 / 29;
  const f = y > delta ** 3 ? Math.cbrt(y) : y / (3 * delta * delta) + 4 / 29;
  return 116 * f - 16;
}
function labToRgb(l, a, b) {
  const fy = (l + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const delta = 6 / 29;
  const inv = (t) => (t > delta ? t ** 3 : 3 * delta * delta * (t - 4 / 29));
  const x = inv(fx) * 0.95047, y = inv(fy), z = inv(fz) * 1.08883;
  const r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
  const g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
  const bl = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;
  return [linearToSrgb(r), linearToSrgb(g), linearToSrgb(bl)];
}
function neutralRgbFromL(l) { return labToRgb(l, 0, 0); }

// ── Model download + session ─────────────────────────────────────────────
async function fetchModelWithProgress(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`model download failed: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) setProgress((received / total) * 100, `${(received / 1048576).toFixed(0)} / ${(total / 1048576).toFixed(0)} MB`);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

async function initModel() {
  const key = $("modelSelect").value;
  const model = MODELS[key];
  const t0 = performance.now();
  log("model: " + model.label);
  if (!modelBytes) {
    log("downloading " + model.url + " …");
    modelBytes = await fetchModelWithProgress(model.url);
    hideProgress();
    log(`downloaded ${(modelBytes.length / 1048576).toFixed(0)} MB in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  }
  const ep = [];
  if (navigator.gpu) {
    ep.push("webgpu");
    log("WebGPU available — using your graphics card");
  } else {
    log("WebGPU NOT available — falling back to WASM (slow)");
  }
  ep.push("wasm");
  const s0 = performance.now();
  session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ep,
    graphOptimizationLevel: "all",
  });
  log(`session created in ${((performance.now() - s0) / 1000).toFixed(1)}s`);
  await calibrateAbScale();
  $("colorizeBtn").disabled = false;
  $("modelSelect").disabled = false;
}

async function calibrateAbScale() {
  // Saturated warmup input: raw tanh graphs report |ab| <= ~1; graphs with
  // the scale baked in report far above 2.
  const h = MODEL_SIZE, w = MODEL_SIZE;
  const input = new Float32Array(1 * 3 * h * w);
  const rgb = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) rgb[i * 3] = 1;       // red half
  for (let i = h / 2 * w; i < h * w; i++) {              // green half
    rgb[i * 3] = 0; rgb[i * 3 + 1] = 1; rgb[i * 3 + 2] = 0;
  }
  for (let i = 0; i < h * w; i++) {
    input[i * 3] = rgb[i * 3];
    input[i * 3 + 1] = rgb[i * 3 + 1];
    input[i * 3 + 2] = rgb[i * 3 + 2];
  }
  const tensor = new ort.Tensor("float32", input, [1, 3, h, w]);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const ab = out[session.outputNames[0]].data;
  let peak = 0;
  for (let i = 0; i < ab.length; i++) peak = Math.max(peak, Math.abs(ab[i]));
  abScale = peak <= 2 ? 110.0 : 1.0;
  log("ab scale calibration: " + abScale);
}

// ── Tiled inference ──────────────────────────────────────────────────────
function tilePositions(size, tile, overlap) {
  const step = tile - overlap;
  const pos = [];
  for (let p = 0; p + tile <= size; p += step) pos.push(p);
  const last = size - tile;
  if (last > 0 && pos[pos.length - 1] !== last) pos.push(last);
  return pos.length ? pos : [0];
}

async function runModelOnTile(x, y) {
  // Draw the tile into a 512 canvas, read pixels.
  const canvas = new OffscreenCanvas(MODEL_SIZE, MODEL_SIZE);
  const c2 = canvas.getContext("2d", { willReadFrequently: true });
  c2.drawImage(sourceCanvas, x, y, TILE, TILE, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const data = c2.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;

  // Build neutral RGB (gray via LAB L) input, exactly like the worker.
  const n = MODEL_SIZE * MODEL_SIZE;
  const input = new Float32Array(1 * 3 * MODEL_SIZE * MODEL_SIZE);
  for (let i = 0; i < n; i++) {
    const r = data[i * 4] / 255, g = data[i * 4 + 1] / 255, b = data[i * 4 + 2] / 255;
    const l = rgbToLabL(r, g, b);
    const neu = neutralRgbFromL(l);
    input[i * 3] = neu[0];
    input[i * 3 + 1] = neu[1];
    input[i * 3 + 2] = neu[2];
  }
  const tensor = new ort.Tensor("float32", input, [1, 3, MODEL_SIZE, MODEL_SIZE]);
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const abData = out[session.outputNames[0]].data; // [2, 512, 512]
  const scale = abScale;

  // Merge ab with the ORIGINAL tile's LAB L and convert back to RGB.
  const outCanvas = new OffscreenCanvas(MODEL_SIZE, MODEL_SIZE);
  const oc = outCanvas.getContext("2d", { willReadFrequently: true });
  const outImg = oc.createImageData(MODEL_SIZE, MODEL_SIZE);
  for (let i = 0; i < n; i++) {
    const r = data[i * 4] / 255, g = data[i * 4 + 1] / 255, b = data[i * 4 + 2] / 255;
    const L = rgbToLabL(r, g, b);
    const a = abData[i] * scale;
    const bb = abData[n + i] * scale;
    const [or, og, ob] = labToRgb(L, a, bb);
    outImg.data[i * 4] = Math.round(or * 255);
    outImg.data[i * 4 + 1] = Math.round(og * 255);
    outImg.data[i * 4 + 2] = Math.round(ob * 255);
    outImg.data[i * 4 + 3] = 255;
  }
  oc.putImageData(outImg, 0, 0);
  return outCanvas;
}

async function colorizeAllTiles(progress) {
  const W = sourceCanvas.width, H = sourceCanvas.height;
  const xs = tilePositions(W, TILE, OVERLAP);
  const ys = tilePositions(H, TILE, OVERLAP);
  // Weighted merge (mirrors the Rust merge_rgb_tiles): edge pixels of a
  // tile weigh 0.5, interior 1.0; overlapping pixels normalize by weight.
  const acc = new Float32Array(W * H * 3);
  const weights = new Float32Array(W * H);
  let done = 0;
  const total = xs.length * ys.length;
  for (const ty of ys) {
    for (const tx of xs) {
      const t0 = performance.now();
      const tile = await runModelOnTile(tx, ty);
      const tileCtx = tile.getContext("2d", { willReadFrequently: true });
      const tileImg = tileCtx.getImageData(0, 0, TILE, TILE);
      const tw = Math.min(TILE, W - tx), th = Math.min(TILE, H - ty);
      for (let y = 0; y < th; y++) {
        const wy = (y === 0 || y === TILE - 1) ? 0.5 : 1.0;
        for (let x = 0; x < tw; x++) {
          const wx = (x === 0 || x === TILE - 1) ? 0.5 : 1.0;
          const w = wy * wx;
          const src = (y * TILE + x) * 4;
          const dst = ((ty + y) * W + (tx + x)) * 3;
          acc[dst] += tileImg.data[src] * w;
          acc[dst + 1] += tileImg.data[src + 1] * w;
          acc[dst + 2] += tileImg.data[src + 2] * w;
          weights[(ty + y) * W + (tx + x)] += w;
        }
      }
      done++;
      progress(done, total, (performance.now() - t0) / 1000);
    }
  }
  const out = new OffscreenCanvas(W, H);
  const ctx = out.getContext("2d");
  const outImg = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const w = weights[i] || 1;
    outImg.data[i * 4] = Math.round(acc[i * 3] / w);
    outImg.data[i * 4 + 1] = Math.round(acc[i * 3 + 1] / w);
    outImg.data[i * 4 + 2] = Math.round(acc[i * 3 + 2] / w);
    outImg.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(outImg, 0, 0);
  return out;
}

// ── Upload + finish on the server ────────────────────────────────────────
async function finishOnServer(draftCanvas) {
  log("uploading to the engine for finishing…");
  const draftBlob = await draftCanvas.convertToBlob({ type: "image/png" });
  const pageBlob = await sourceCanvas.convertToBlob({ type: "image/jpeg", quality: 0.95 });
  const fd = new FormData();
  fd.append("page", pageBlob, "page.jpg");
  fd.append("draft", draftBlob, "draft.png");
  fd.append("preset", $("preset").value);
  fd.append("style", $("style").value);
  const palette = $("palette").value;
  if (palette) fd.append("palette", "configs/palettes/" + palette + ".yaml");
  const resp = await fetch("/api/gpu/colorize", {
    method: "POST",
    headers: { "x-mangahue-local": "1" },
    body: fd,
  });
  const json = await resp.json();
  if (!resp.ok) throw new Error(json.error || json.detail || resp.statusText);
  return json;
}

// ── UI wiring ────────────────────────────────────────────────────────────
async function handleFile(file) {
  const bitmap = await createImageBitmap(file);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  sourceCanvas = canvas;
  const img = document.createElement("img");
  img.src = URL.createObjectURL(file);
  const drop = $("drop");
  drop.innerHTML = "";
  drop.appendChild(img);
  $("fileInfo").textContent = `${bitmap.width} × ${bitmap.height} px · ${(file.size / 1048576).toFixed(2)} MB`;
  $("colorizeBtn").disabled = session == null;
}

$("drop").addEventListener("click", () => {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.onchange = () => input.files[0] && handleFile(input.files[0]);
  input.click();
});
["dragover", "drop"].forEach((ev) =>
  $("drop").addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === "dragover") $("drop").classList.add("drag");
    else {
      $("drop").classList.remove("drag");
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    }
  })
);

$("modelSelect").addEventListener("change", async () => {
  localStorage.setItem("mangahue-model", $("modelSelect").value);
  $("colorizeBtn").disabled = true;
  modelBytes = null;
  session = null;
  try {
    await initModel();
  } catch (e) {
    log("model load failed: " + e.message);
  }
});

$("colorizeBtn").addEventListener("click", async () => {
  if (!sourceCanvas || !session) return;
  const btn = $("colorizeBtn");
  btn.disabled = true;
  log("colorizing on your GPU…");
  try {
    const t0 = performance.now();
    const draft = await colorizeAllTiles((d, t, s) => {
      if (d % 2 === 0 || d === t) log(`tile ${d}/${t} · ${s.toFixed(2)}s`);
    });
    log(`draft done in ${((performance.now() - t0) / 1000).toFixed(1)}s — sending to the engine`);
    const draftBlob = await draft.convertToBlob({ type: "image/png" });
    $("draftImg").src = URL.createObjectURL(draftBlob);
    const res = await finishOnServer(draft);
    log("finished!");
    $("resultImg").src = res.result_url;
    $("verdict").textContent = res.verdict || "—";
  } catch (e) {
    log("ERROR: " + e.message);
  } finally {
    btn.disabled = false;
  }
});

(async () => {
  const saved = localStorage.getItem("mangahue-model");
  if (saved && MODELS[saved]) $("modelSelect").value = saved;
  $("modelSelect").disabled = true;
  if (!navigator.gpu) {
    $("gpuBadge").textContent = "WebGPU not available (fallback: WASM, slower)";
    $("gpuBadge").className = "badge no";
  } else {
    $("gpuBadge").textContent = "WebGPU ready — using your GPU";
    $("gpuBadge").className = "badge ok";
  }
  try {
    await initModel();
  } catch (e) {
    log("model load failed: " + e.message);
    $("gpuBadge").textContent = "model load failed";
  }
})();
