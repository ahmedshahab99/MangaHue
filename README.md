# MangaHue Engine

MangaHue Engine is a Rust-first manga colorization intelligence layer. It does not treat colorization as a single "upload image, run model" step. The engine cleans line art, protects ink and speech bubbles, builds hint tensors, tiles large pages, coordinates model backends, restores manga structure, and writes reports that can improve later passes.

This repository currently contains the first production-shaped slice:

- A Rust workspace with the blueprint crates.
- Strong domain types, config loading, image I/O, preprocessing, layout, hints, memory, inference interfaces, postprocessing, evaluation, orchestrator, CLI, and API skeletons.
- Director-grade inspection reports: page classification, ink density, tone score, component analysis, protected-region coverage, hint adherence, and a final verdict.
- Debug overlays that show ink, bubbles, text, tones, and hints in one visual.
- A Python/PyTorch model package scaffold for future training and ONNX export.
- An explicit mock backend for tests and demos. Real CLI colorization does not silently fake an ONNX model.

## Quick Start

```powershell
cargo test
cargo run -p mangahue-cli -- colorize .\examples\input\page.png --output .\examples\output\page.png --mock --debug .\examples\output\debug
cargo run -p mangahue-cli -- inspect .\examples\input\page.png --output .\examples\output\inspection.json --debug .\examples\output\inspect_debug
```

Use `--mock` only for pipeline verification. For a real model, install the Python package in `python/`, export an ONNX model, omit `--mock`, and pass `--model path\to\model.onnx`. The Rust engine calls the Python ONNX sidecar backend through `python -m mangahue_train.run_onnx_tile`; set `MANGAHUE_PYTHON` if you need a specific interpreter.

## CLI

```powershell
mangahue colorize input.png --output output.png --model model.onnx
mangahue colorize input.png --output output.png --mock
mangahue batch pages --output colored --mock
mangahue inspect input.png --output inspection.json --debug debug_folder
mangahue benchmark pages --output colored --mock --report benchmark.json
mangahue correction add --project projects/my_manga/project.json --page page_001 --x 100 --y 200 --width 80 --height 120 --label hair --old "#5a4632" --new "#111111" --character hero
mangahue project export-corrections --project projects/my_manga/project.json --output training/corrections.jsonl
mangahue styles list --dir configs/styles
mangahue model validate --metadata models/model.mangahue.json
mangahue model validate-onnx --model models/model.onnx --metadata models/model.mangahue.json
mangahue serve --addr 127.0.0.1:7878
mangahue preprocess input.png --output clean.png
mangahue make-hints input.png --hints hints.json --output-map hint_map.png --output-mask hint_mask.png
mangahue project create "My Manga" --output projects/my_manga
mangahue character create --project projects/my_manga --name Hero --hair "#111111" --eyes "#3366ff" --skin "#e8b89a"
```

## Philosophy

The model creates color candidates. The Rust engine decides how manga pages are prepared, protected, guided, corrected, and remembered.

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
