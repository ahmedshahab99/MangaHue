# MangaHue Professional Roadmap — "One Tiny Model, Full Studio Grade"

Status: approved direction (2026-08-07). This is the plan to make MangaHue the
most professional manga colorization project *defined by its mini model*:
a small, hint-conditioned model that runs everywhere (2-core CPU VPS, browser
WebGPU, laptop) while the deterministic engine makes the output look
reference-grade. Everything else in the project already exists to support this
story: region graph, protected layers, palette memory, correction loop, webapp,
Telegram, API.

---

## 1. Vision and Positioning

**Product statement.** "MangaHue: a professional manga colorizer whose entire
model is smaller than a single manga page, yet whose output passes reference
comparison."

**The one-sentence spec every developer should remember:**
> One ONNX file, ≤ 35 MB int8 / ≤ 60 MB fp16, hint-conditioned, 512² tiles,
> CPU-friendly ops only, quality ≥ 90% of DDColor-tiny on our own scorer.

**Why a mini model is the right strategy (not a compromise):**
- Distribution: browsers (WebGPU/WebNN), phones, cheap VPS, offline tools — no
  400 MB downloads, no GPU required anywhere.
- Speed: 2-core VPS today spends ~20 s/page on DDColor-tiny. A 35 MB int8
  model targets ~3–8 s/page (2–5×), and ~0.5–1.5 s/page on a user's GPU.
- Trust: a small auditable ONNX graph is inspectable, quantizable, and safe to
  ship to end users — a professional story, not an engineering limitation.
- The moat is not the model size; it is the **loop**: hints → model → engine
  finishing → user corrections → memory → fine-tune. The mini model makes that
  loop fast enough to be interactive.

---

## 2. The Mini Model — Target Specification (v1)

| Property | Target | Measured by |
|---|---|---|
| File size | ≤ 60 MB fp16, ≤ 35 MB int8 | file size of validated ONNX |
| Latency (2-core CPU VPS) | ≤ 8 s per 512² tile | bench suite on VPS |
| Latency (mid-range WebGPU) | ≤ 1.5 s per 512² tile | ort-web benchmark harness |
| Quality vs DDColor-tiny | ≥ 90 % of tiny's score | `benchmark_quality.py` on fixed page set |
| Reference-match score | ≥ 85/100 on the private benchmark set | reference scorer |
| Contract | 5ch in (lineart, hint R/G/B, mask), 3ch out, 0–1 | `model.mangahue.json` validation |
| Quantization | int8 dynamic or static with **graph validation** | export gate (see §6) |
| Ops | WebGPU-safe (Conv/BN→Conv+Scale, no exotic ops) | ONNX opset audit |

### 2.1 Architecture decision

Start from the existing `HintUNet` (5→3 channels, base=48) in
`python/mangahue_train/models.py` and upgrade it deliberately:

1. **Teacher**: DDColor-tiny (fp16 ONNX, in production) — free teacher, no
   download, same domain. Optionally also distil from the reference-CGAN
   (already on the VPS) on a small curated set to inherit its "match a
   reference" behaviour later.
2. **Student v1** (`HintUNetV2`): keep the 5-channel contract; changes:
   - Depthwise-separable convs in the encoder (3–4× params cut),
   - Light channel attention at the bottleneck (24-ch),
   - Nearest-upsample decoder with skip concat (already the shape),
   - Target params: ~3–6 M (≈ 12–24 MB fp32, ≈ 3–6 MB int8 core + IO),
     sized so the int8 file stays ≤ 35 MB with margin.
3. **Training-only extras**: small PatchGAN discriminator on 96² crops
   (discarded at export); no extra inference cost.
4. **Palette conditioning (v1.5)**: optional 64-dim palette embedding added at
   the bottleneck (from k-means of hint map or user palette). Keeps the same
   file contract; the engine already computes palettes.

### 2.2 Why not just ship DDColor-tiny?

DDColor-tiny is unconditional (no hints) and 135 MB. The professional features
— character memory, per-series palettes, user corrections — all flow through
the 5-channel hint contract. The mini model is the only path that keeps hints
*and* shrinks the artifact. DDColor-tiny stays as the ensemble/comparison
teacher and fallback.

---

## 3. Data — the real moat (Phase 0)

Professional output is a data problem before it is a model problem. Build a
curated, licensed, versioned dataset:

| Source | Use | Licensing posture |
|---|---|---|
| Colored pages collected so far (nhentai/MangaDex/etc.) | pretraining pairs | training-only, never redistributed; document provenance |
| Artist-licensed / public-domain colored manga (Aozora? official free chapters, CC art) | eval + small train | keep a clean eval set that CAN be shown publicly |
| Synthetic pairs from grayscale→colorized→re-colorized (DDColor-tiny + reference-CGAN) | distillation data | ours |
| User corrections from the webapp (once live) | fine-tune | opt-in checkbox; the product asks |

**Dataset requirements (engineering):**
- Versioned manifest (parquet/JSONL + hashes) with train/eval split,
- Fixed 10-page **private benchmark set** (never trains, never changes) — the
  one number that gates every release,
- Hint augmentation: for every pair, synthesize 3–5 random hint variants
  (wrong-region hints, missing regions, over-confident masks) so the model
  learns to *resist bad hints* — this is what makes it professional in the
  hands of real users,
- Screentone/text masks from the engine's protected-layer output as
  auxiliary supervision.

**KPI:** ≥ 5k curated pairs at v1 train, ≥ 200 eval pages, 100 % provenance
documented.

---

## 4. Training pipeline (Phases 1–2)

Already present in `python/mangahue_train/` (train.py, dataset.py,
prepare_dataset.py, export_onnx.py) — upgrade, don't rebuild:

1. **Loss stack (distillation):**
   - `L1` in LAB (chroma-focused, matches DDColor conventions),
   - Perceptual `LPIPS` on 256² crops (small net, CPU-evaluable at train time
     on a rented GPU),
   - Palette/histogram loss: match teacher's chroma histogram per region,
   - Optional adversarial (PatchGAN) with 0.1 weight — turn on only if it
     improves the reference score (measured, not assumed),
   - Hint-robustness loss: run 20 % of batches with degraded hints and weight
     their loss lower when the hint is wrong (learn to ignore).
2. **Schedule:** 100–200 epochs on 5k pairs at 512²; cosine LR; EMA of weights
   for export; checkpoint every eval-best.
3. **Compute:** rent one GPU (T4/A10, ~$0.5–1/h, 1–2 weeks wall time for v1;
   fine-tunes are hours). Local CPU training is for smoke tests only.
4. **Export gate** (`export_onnx.py` upgraded):
   - fp32 → fp16 → int8 (dynamic, then static with calibration from eval set),
   - **Every artifact must pass**: shape contract, 5ch in/3ch out, 0–1 range,
     spatial multiple of 32, and a numeric diff vs the fp32 model < tolerance —
     this prevents another corrupt-int8 incident,
   - Opset audit script listing non-WebGPU ops (reject on export, not in the
     browser).
5. **Regression gate:** every candidate runs `benchmark_quality.py` on the
   10-page private set + the VPS latency suite. A candidate that wins quality
   but loses speed > 20 % is rejected — both numbers are release criteria.

**KPI (v1 shipped):** ≤ 35 MB int8, ≤ 8 s/tile CPU, ≥ 90 % of DDColor-tiny
quality, all three export variants validated, WebGPU demo runs in-browser.

---

## 5. Product integration (parallel track)

The professional finish is in how the model meets users:

1. **Webapp (`web/app`)**: model is served from `/models` (browser-cacheable);
   quality tiers become *model tiers* — "Fast (mini int8)", "Studio
   (mini fp16 + engine finishing)", "Reference (reference-CGAN, premium)".
   The existing fast/studio/masterpiece toggle already maps 1:1.
2. **WebGPU mode (reactivate `web/gpu`)**: mini model first-class citizen —
   it is the one model small enough to download in seconds. Draft on GPU,
   finishing on the VPS (already built, parked; flip it back on with the mini
   model as default and ddcolor-full as optional).
3. **Memory loop (the moat)**: webapp corrections → `/api/corrections` →
   export-training-manifest → monthly fine-tune → new mini model. Show users
   "this model remembers your series" once consistency is measurable
   (same character hair across pages: within ΔE < 8).
4. **Telegram**: switch default backend to the mini model; keep DDColor-tiny
   for ensemble in studio mode. 2× throughput on the same VPS.
5. **API**: the sandboxed, authed API (already hardened) is the single
   contract; model artifacts ship via `/models` with a `model.mangahue.json`
   manifest the engine validates on load (exists).

**Monetization (Phase 4 of the product, not day 1):** credits for pages;
"Studio" tier = reference-CGAN minutes; "API" tier = batch/chapter.
ProductionStore already has tiers, tokens, quotas — wire pricing onto it.

---

## 6. Professional packaging & ops

- **Benchmarks page** on the webapp root: reference score, speed table
  (VPS/WebGPU/mobile), sample gallery with before/after — public, regenerated
  per release. This is the "most professional" visible artifact.
- **Release process:** tag `model-vX.Y.Z` → gates (quality, speed, export
  validity, webapp smoke) → deploy binary + model + manifest together;
  rollback = previous tag. CI exists in `.github`; add the model gate job.
- **Observability:** job telemetry (per-stage timing, tiles, model name) into
  `/metrics`; page-time SLO alerting on the VPS (2-core: ≤ 10 s/page p95).
- **Security:** current state is already hardened (auth-gated API, sandboxed
  jobs, SSRF/traversal blocked, CORS allowlist). Keep the model files
  out of git/LFS entirely (the 94 GB LFS incident is the cautionary tale) —
  models ship as release artifacts, never through the repo.
- **Docs:** one-page "model card" per release (training data summary,
  measured trade-offs, failure modes, license) — the sign of a professional
  project.

---

## 7. Phases, timeline, KPIs

| Phase | Duration | Deliverable | Gate (must pass to continue) |
|---|---|---|---|
| **0 — Baseline & data** | wk 1 | dataset v0 (≥5k pairs + 10-page private set + hint augmentation), teacher pinned (DDColor-tiny), scorer calibrated | scorer reproduces tiny's score; dataset manifests versioned |
| **1 — Mini v1** | wk 2–3 | HintUNetV2 distilled, 3 export variants **all validated**, WebGPU demo | ≤35 MB int8, ≤8 s/tile CPU, ≥90 % of tiny quality, export diff gate |
| **2 — Hints & memory** | wk 3–4 | hint-robustness trained in; palette embedding v1.5; character consistency eval | same-character ΔE < 8 across pages; bad-hint resistance test passes |
| **3 — Deployment everywhere** | wk 4–5 | webapp model tiers + benchmarks page; Telegram switch; WebGPU mode reactivated | p95 ≤ 10 s/page on VPS; webapp smoke on all tiers |
| **4 — Feedback loop** | wk 5–6 | corrections → manifest → fine-tune pipeline automated; monthly cadence scripted | fine-tune improves private-set score ≥ 1 pt without speed regression |
| **5 — Professional packaging** | wk 6–8 | model cards, release gates in CI, observability, benchmarks page public, monetization wiring | full release checklist green on one end-to-end release |

---

## 8. Risks and honest answers

| Risk | Mitigation |
|---|---|
| **Training data licensing** | private training set, provenance log, public eval set only with permission; never redistribute pages |
| **No GPU** | rent per-training (T4/A4 ≈ $0.4–1/h); v1 budget ≈ $100–300 total; fine-tunes ≈ $5–20 |
| **Quantization quality drop** | int8 only ships if private-set score ≥ 90 % of fp16; static calibration from eval set |
| **WebGPU op gaps** | opset audit gate at export; WASM fallback remains |
| **Overfit to one style** | eval set spans 5+ styles (from configs/styles); report per-style scores |
| **Scope creep** | every phase ships a measurable artifact; the benchmark page makes regressions public internally |
| **Another agent editing the repo concurrently** | single source of truth in this doc; model artifacts never in git; CI gates on both branches |

---

## 9. Definition of "done" (the professional bar)

A release is professional when, without explanation:
1. A visitor opens the webapp, uploads a page, and gets a reference-grade
   result in under 30 seconds (2-core VPS) or under 5 seconds (their GPU),
   with live progress and a before/after slider.
2. The benchmarks page shows the numbers and the gallery matches them.
3. The model file is one validated ONNX ≤ 35 MB with a model card.
4. A user's corrections make the NEXT page of the same series better, and the
   monthly fine-tune makes the model better for everyone.
5. Nothing in the pipeline (upload → colorize → deliver) can read or write
   outside its sandbox, and every release is rollback-safe.
