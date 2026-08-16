# MangaHue Engine Roadmap — "Best in the World" Plan

Status of this document: live plan, updated 2026-08-11. The bot is in launch
state: chapters from nhentai, mangadex, manganato, e-hentai, imhentai,
allporncomic.com and ilikecomix.com, with mandatory per-page inference,
DDColor-full native inference (f32 chroma pipeline), smart color governor and
per-page delivery proofs.

---

## Phase 1 — Quality ceiling (next 2-3 weeks, no new hardware)

Everything here is tuning + engineering on the existing stack. Measured with
the existing per-page report (saturation, colorfulness, warmth, contrast) +
human review on a fixed 20-page benchmark set (10 B&W manga pages, 10 colored
comics — the benchmark set itself must be created first, from real chapters
users send).

1. **Benchmark harness** — pin a fixed chapter set + expected verdicts
   (production-ready vs explicitly withheld) + quality targets (sat band, colorfulness
   band, warmth band). Every engine change must move a metric toward target
   without regressing another. Without this, "improvements" are vibes.
2. **Wire the remaining dead knobs** — `engine.preserve_lines`,
   `protect_speech_bubbles`, `output.quality`, `inference.fp16_enabled`
   (real fp16 tensor extraction), or delete them. A config-lint test asserts
   every EngineConfig field is read.
3. **Parameterize the color hot-spots** — the ~20 hardcoded constants in
   `region_renderer` (skin boosts, AO, speculars), `natural_grade` (sat
   curve), `reconstruct_full_resolution` (chroma boost, ink gates) move into
   the preset YAML. Then the "yellow cast" / "too muted" dials are per-site
   presets, not code edits.
4. **Hints stage to the output** — today the hint map never reaches the
   DDColor output. Either seed the region renderer with hint evidence or gate
   the stage off. Same for `protected_colors` (parse + honor).
5. **Reference-photo grading v2** — the `ref` feature exists; make it
   per-series automatic: when a chat colorizes chapters of the same series,
   remember the series' palette from approved references/corrections and
   auto-grade subsequent B&W pages toward that palette.

## Phase 2 — Model upgrades (after Phase 1; ~1-2 weeks)

6. **fp16 real support** — the fp16 model exists on the VPS; wire
   `fp16_enabled` to extract fp16 tensors (session.io inspection) → ~1.6x
   speed, identical quality for this model.
7. **Style2Paints V3 native backend** — code path already exists
   (`Style2PaintsV3OnnxBackend`, gated on native-onnx). Ship the ONNX export
   (~700MB) and benchmark vs DDColor-full on the manga set. Style2Paints is
   reference-based (lineart + reference image) — the best shot at
   "professional colorist" quality per series.
8. **DDColor fine-tune** — the project already has training infra
   (python/mangahue_train). Fine-tune DDColor on a manga-dedicated set
   (high-res B&W pages + their colored ground truth, harvestable from
   colored-vs-B&W pairs on these sites!). This is the single biggest
   "best in the world" lever: a model trained on the exact content the bot
   serves.

## Phase 3 — Speed & scale (as users grow)

9. **Session mutex release** (done in this batch) → verify parallel_jobs > 1
   actually overlaps; tune `MANGAHUE_MODEL_THREADS`.
10. **Page-size caps + strip-based reconstruction** — bound memory on huge
    pages; today 32MP pages use ~0.7GB in reconstruction buffers.
11. **GPU path** — `MANGAHUE_CUDA=1` already exists (CUDA EP registered,
    CPU fallback). On a GPU VPS (RTX 3060+): ~10x faster inference, and the
    fp16 model becomes the primary.
12. **Cache-first chapters** — the artifact cache exists; make chapter-level
    caching the default so repeated links cost nothing.

## Phase 4 — Product features that win

13. **Per-series color memory** (Phase 1.5) — the bot learns each series'
    look from colored chapters and applies it to B&W ones. No other bot does
    this — it is the moat.
14. **Batch chapters** — "send 3 links" → one queue job, one palette.
15. **Web preview** — delivered pages link to a gallery page (watermark +
    series title), ads-friendly.
16. **Premium** — watermark removal, priority queue, longer chapters
    (billing scaffolding already exists in mangahue-production).

## Metrics we track (per page, already in reports)

- avg_saturation / colorfulness / brightness / contrast (color_metrics stage)
- director verdict (production_ready* / needs_*)
- inference_ms / postprocess_ms / total_ms
- Delivery colorization proof: completed inference + visible chroma +
  input/output transformation; no already-colored pass-through

## Ground rules

- Every engine change ships with a unit test + the 20-page benchmark check.
- Never regress line preservation (1.0 today) or bubble cleanliness.
- Config changes are the first dial, code the last.
- All deploys go through the immutable GitHub release workflow (test → build
  → checksummed artifact → atomic promotion; a failed gate never touches the
  running service).
