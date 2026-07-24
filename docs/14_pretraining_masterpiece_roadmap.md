# Pre-Training Masterpiece Roadmap

MangaHue can become much more powerful before the final trained model exists. The model will create the base color layer later; these features make the engine reliable for full chapters.

## Already In Place

- Single-page, folder, link, ZIP, and CBZ chapter entry points.
- Natural page sorting and normalized page manifests.
- Mock backend for end-to-end product testing before training.
- Line-art preservation, speech/text protection, debug masks, and quality reports.
- Project memory, correction records, correction-derived hints, and training manifests.
- CLI, API, Studio, and Telegram delivery surfaces.
- Resumable chapter jobs with persistent manifests, gallery output, and CBZ export.
- Content-addressed artifact cache primitives for source pages, manifests, galleries, archives, and future intermediate tensors.
- Chapter brain reports that extract stable palettes, measure page-to-page drift, write locked style profiles, and selectively retry drifted pages.
- Chapter safety gate reports that audit line preservation, speech bubble cleanliness, protected regions, and hint adherence before archive export.
- Protected-region repair in postprocess: bright bubble/protected areas are restored while dark text/ink is preserved.
- Palette memory books with editable locked colors, roles, confidence, tolerances, and style export.

## Best Advanced Features Before Training

1. Region lock editor: let users lock hair, eyes, skin, clothing, bubbles, and backgrounds before model inference.
2. Dataset builder: convert finished/corrected chapters into training-ready paired samples.
3. Consistency dashboard: rank risky pages, retry history, and palette lock confidence.
4. Palette memory visual editor: swatches, roles, tolerance sliders, and one-click style export.
5. Archive security hardening: archive bomb limits, total uncompressed byte limits, nested archive rejection.
6. Watermark/export profiles: free preview, premium high-res, CBZ, ZIP, and web gallery bundles.
7. Training data curator: rank pages by safety, palette stability, hint adherence, and correction density.
8. Human correction loop: export corrections, create hint files, and feed improvements into later training.

## Next Recommended Build Step

Build the region lock editor. MangaHue now creates editable palette memories; the next brain step is spatial locks for hair, eyes, clothing, bubbles, and backgrounds before inference.

For the no-rental, 6 GB CPU production path and its full quality checklist, see
`docs/15_cpu_reference_grade_plan.md`.
