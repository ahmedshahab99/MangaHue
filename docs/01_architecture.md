# Architecture

MangaHue Engine separates the colorization product into stable responsibilities:

- `mangahue-core`: domain types and engine errors.
- `mangahue-config`: layered YAML config and style profiles.
- `mangahue-image`: image I/O and shared image helpers.
- `mangahue-preprocess`: line-art cleanup, line masks, tone estimation, component analysis, text and bubble protection.
- `mangahue-layout`: panel estimation, tiling, and tile merge.
- `mangahue-hints`: user, style, and character hint maps.
- `mangahue-memory`: project and character consistency.
- `mangahue-inference`: model contracts and colorization backends.
- `mangahue-postprocess`: line restore, bubble cleanup, palette correction.
- `mangahue-eval`: quality metrics.
- `mangahue-cache`: content-addressed artifact storage.
- `mangahue-brain`: chapter palette locks and consistency reports.
- `mangahue-orchestrator`: the full pipeline.
- `mangahue-chapter`: resumable chapter jobs, archive export, galleries, and brain output.
- `mangahue-cli`: user commands.
- `mangahue-api`: web API contract skeleton.

The ONNX backend boundary is deliberately narrow: it receives a clean line-art tile, RGB hint map, and hint mask, and returns RGB color. Everything around that boundary stays model-agnostic.

The orchestrator acts as the color director. It records what the engine saw, which protections were active, how hints were applied, and whether output quality passed the current production gates.
