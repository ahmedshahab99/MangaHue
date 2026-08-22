# Pipeline

1. Load and validate the page; retain its full-resolution grayscale source.
2. Resize a canonical working copy to the configured CPU memory bound.
3. Detect ink, speech bubbles, text, screentones, panels, and protected areas.
4. Build a dedicated model guide directly from the source. Smooth only evidenced
   screentones and preserve strong ink; never substitute this guide for output luminance.
5. Split large working pages into overlapping 768-pixel crops. Small pages remain a
   single crop.
6. Run every crop serially through the one pinned DDColor ONNX model and its one shared
   native Rust/ONNX Runtime session.
7. Measure model disagreement in every real overlap. A missing or extreme consistency
   measurement makes a multi-crop result undeliverable.
8. Merge using exact tile coordinates. Feather internal overlap edges only; exterior
   page edges always keep full weight.
9. Grade the model chroma within measured bounds, clean bubbles and periodic chroma
   artifacts, and restore original ink, text, whites, and protected areas.
10. Reconstruct full resolution using the original source luminance and enforce all
    protected constraints again at delivery resolution.
11. Block grayscale, unchanged, structurally damaged, residual-grid, or unstable-tile
    output. Do not deliver it as a successful page.
12. Account for every admitted chapter page. A partial chapter remains failed and is
    never promoted or exported as complete.
13. Save the approved output and JSON quality telemetry; user-image retention follows
    the configured expiry policy.

