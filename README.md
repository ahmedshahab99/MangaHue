# MangaHue Engine

See [Color Bible Engine](docs/COLOR_BIBLE_ENGINE.md) for the chapter-aware consistency, protected-compositing, correction-propagation, and release-gate workflow.

MangaHue Engine is a Rust manga colorization service. The pretrained model creates color; deterministic Rust stages resize the page, restore ink, protect text and whites, remove artifacts, and block damaged or unchanged results.

The default build runs inference **natively in Rust** through the bundled ONNX Runtime — no Python subprocess and no `ORT_DYLIB_PATH` setup required (the runtime in `models/onnxruntime/` is auto-discovered). Colorization is real, not faked.

This repository currently contains the first production-shaped slice:

- A Rust workspace with the blueprint crates.
- Strong domain types, config loading, image I/O, preprocessing, layout, hints, memory, inference interfaces, brain reports, postprocessing, evaluation, orchestrator, CLI, and API skeletons.
- Director-grade inspection reports: page classification, ink density, tone score, component analysis, protected-region coverage, hint adherence, and a final verdict.
- Debug overlays that show ink, bubbles, text, tones, and hints in one visual.
- A deterministic internal backend used only by automated pipeline tests; no user-facing command can select it.

## Quick Start

```powershell
cargo test
cargo run -p mangahue-cli -- colorize .\examples\input\page.png --output .\examples\output\page.png
cargo run -p mangahue-cli -- inspect .\examples\input\page.png --output .\examples\output\inspection.json --debug .\examples\output\inspect_debug
```

Production uses exactly one pretrained artifact: `models/ddcolor-full-fp32.onnx` (DDColor Full, Apache-2.0), executed natively through ONNX Runtime. Its SHA-256 is pinned in code and in `configs/model-checksums/`; the bot refuses to start with a renamed or different model. The runtime and colorization pipeline are Rust-only.

On the low-budget CPU profile, MangaHue builds a source-resolution model guide that suppresses evidenced screentones without changing the canonical page luminance. Pages larger than 768 pixels on either working axis are colored as overlapping crops through the same shared DDColor session, strictly one crop at a time. The compositor feathers only real internal overlaps, measures cross-crop disagreement, and blocks unstable, unchanged, grayscale, structurally damaged, or residual-grid output. A chapter is complete and exportable only when every admitted page passes.

## CLI

```powershell
mangahue colorize input.png --output output.png --model models/ddcolor-full-fp32.onnx
mangahue batch pages --output colored --model models/ddcolor-full-fp32.onnx
mangahue inspect input.png --output inspection.json --debug debug_folder
mangahue benchmark pages --output colored --model models/ddcolor-full-fp32.onnx --report benchmark.json
mangahue correction add --project projects/my_manga/project.json --page page_001 --x 100 --y 200 --width 80 --height 120 --label hair --old "#5a4632" --new "#111111" --character hero
mangahue project export-corrections --project projects/my_manga/project.json --output training/corrections.jsonl
mangahue styles list --dir configs/styles
mangahue serve --addr 127.0.0.1:7878
mangahue preprocess input.png --output clean.png
mangahue make-hints input.png --hints hints.json --output-map hint_map.png --output-mask hint_mask.png
mangahue chapter import chapter.cbz --output .mangahue/imported_chapter
mangahue chapter colorize chapter.cbz --output examples/output/colored_chapter --export examples/output/colored_chapter.cbz --gallery examples/output/colored_chapter/gallery.html --manifest examples/output/colored_chapter/job.json --brain-report examples/output/colored_chapter/mangahue_chapter_brain.json --locked-style examples/output/colored_chapter/mangahue_locked_style.json --palette-memory examples/output/colored_chapter/mangahue_palette_memory.json --palette-memory-style examples/output/colored_chapter/mangahue_palette_memory_style.json --safety-report examples/output/colored_chapter/mangahue_safety_gate.json --consistency-retry
mangahue brain palette-memory --brain-report examples/output/colored_chapter/mangahue_chapter_brain.json --output examples/output/colored_chapter/mangahue_palette_memory.json --style-output examples/output/colored_chapter/mangahue_palette_memory_style.json
mangahue chapter colorize chapter.cbz --output examples/output/colored_chapter --manifest examples/output/colored_chapter/job.json --resume
mangahue chapter export examples/output/colored_chapter --output examples/output/colored_chapter.cbz
mangahue project create "My Manga" --output projects/my_manga
mangahue character create --project projects/my_manga --name Hero --hair "#111111" --eyes "#3366ff" --skin "#e8b89a"
```

## Philosophy

The pretrained model creates the color. Rust prepares and protects the page, removes artifacts, verifies the result, and delivers it only when the release gates pass.

## Studio

Run the local API and open the Studio:

```powershell
cargo run -p mangahue-cli -- serve --addr 127.0.0.1:7878
```

Then open:

```text
http://127.0.0.1:7878/
```

Studio is a local operator console for page inspection, colorize jobs, job history, styles, corrections, and image previews.
