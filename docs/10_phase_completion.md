# Phase Completion

Implemented non-training phases:

- Phase 1: Rust workspace, core types, config, image utilities, CLI skeleton, tests.
- Phase 2: Preprocessing, line masks, tone masks, bubble/text protection, debug outputs.
- Phase 3: Layout, panel prototype, tiling, weighted tile merge.
- Phase 4: User/style/character hints, hint maps, hint masks.
- Phase 5: Project memory, character profiles, correction memory, correction export.
- Phase 6: Model metadata validation, backend trait, mock backend, Python ONNX sidecar runtime.
- Phase 7: Line restore, bubble cleanup, bleed reduction, palette correction, final sharpen.
- Phase 8: Full CLI surface for colorize, batch, benchmark, inspect, debug, projects, characters, corrections, model validation, API serve.
- Phase 9: Axum API with inspect, persisted job manifests, styles, projects, characters, corrections, colorize, batch, results.
- Phase 10: Verification scripts, benchmark reports, release packaging script.

## Phase 10 — Optimization

Implemented deliverables:

- `mangahue-orchestrator` now parallelizes tile inference under a configurable rayon thread pool. `engine.parallel_jobs: 0` in `configs/engine.*.yaml` (CLI fallback `mangahue batch --jobs`) selects worker count; `0` means "use rayon's default".
- `mangahue-cli batch` / `mangahue-cli benchmark` parallelize across pages using rayon. A background process-RSS monitor records peak working-set memory for the run.
- Benchmark JSON reports now include per-page stage timings (`preprocess / layout / hint / inference / postprocess`), mean/median/stddev/throughput metrics, backend counts, peak/final RSS, CPU count, total memory, and ISO-8601 start/finish timestamps.
- Engine configs (`engine.default.yaml`, `engine.fast.yaml`, `engine.quality.yaml`, `engine.debug.yaml`) all expose `engine.parallel_jobs`.

## Production Intelligence Layer (Beyond Phase 10)

- **`mangahue-ensemble`** crate: multi-backend voting via `Ensembler`. N models vote per tile; divergence scored per pixel; three consensus strategies (`MajorityWeighted`, `DivergenceReject`, `AverageBlend`); produces a `ConfidenceMap` (grayscale confidence + cause-heatmap). Plugs behind the existing `ColorizationBackend` trait.
- **`mangahue-cache`** crate: content-addressable artifact store (SHA-256 keys, `ArtifactCache`). Pipeline computes an idempotency key from input path, preprocess hashes, hint counts, style, and engine config; cache hit returns `from_cache: true` instantly.
- **Cancellable jobs**: `ColorizeRequest.cancel_token: Option<Arc<AtomicBool>>` checked before each tile and between pages. `POST /api/jobs/:id/cancel` sets the flag; orchestrator returns a clean error.
- **SSE streaming**: `GET /api/jobs/:id/events` pushes `queued → running → complete/failed` via Server-Sent Events (`tokio::sync::broadcast`). Studio can watch live progress.
- **Idempotency key**: computed per-request from a SHA-256 hash of the entire pipeline state; written into reports and cache manifests.

## Remaining major work

- Train the actual colorization model (the `python/` package now has a preparer, trainer with val split + best checkpoint, and ONNX export with `--validate`; the full workflow lives in `docs/12_training_workflow.md` and `scripts/train.ps1`).
- After training, run `python -m mangahue_train.export_onnx ... --validate` and `cargo run -p mangahue-cli -- model validate-onnx` against the produced files.
- Add a native Rust ONNX provider behind the existing `ColorizationBackend` trait. Requires the `ort` crate.
- Optional: model quantization (fp16/int8) tools for release-size shrinking.
