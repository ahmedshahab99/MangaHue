# Pre-Training Masterpiece Roadmap

MangaHue can become much more powerful before the final trained model exists. The model will create the base color layer later; these features make the engine reliable for full chapters.

## Already In Place

- Single-page, folder, link, ZIP, and CBZ chapter entry points.
- Natural page sorting and normalized page manifests.
- Mock backend for end-to-end product testing before training.
- Line-art preservation, speech/text protection, debug masks, and quality reports.
- Project memory, correction records, correction-derived hints, and training manifests.
- CLI, API, Studio, and Telegram delivery surfaces.

## Best Advanced Features Before Training

1. Chapter job resume: persist every page stage so a 200-page chapter can continue after failure.
2. Visual gallery export: static HTML gallery with before/after sliders and quality warnings.
3. Region lock editor: let users lock hair, eyes, skin, clothing, bubbles, and backgrounds before model inference.
4. Palette memory studio: chapter-level character palettes with locked colors and confidence scores.
5. Bubble/text safety audit: hard fail or auto-repair when speech bubbles are colored.
6. Chapter consistency report: detect color drift page-to-page even with mock/model outputs.
7. Archive security hardening: archive bomb limits, total uncompressed byte limits, nested archive rejection.
8. Watermark/export profiles: free preview, premium high-res, CBZ, ZIP, and web gallery bundles.
9. Dataset builder: convert finished/corrected chapters into training-ready paired samples.
10. Human correction loop: export corrections, create hint files, and feed improvements into later training.

## Next Recommended Build Step

Build chapter job resume and a web preview gallery. That makes MangaHue feel like a full product even before the model is trained: users can upload a full chapter, watch progress, recover failures, inspect quality, and download a polished package.
