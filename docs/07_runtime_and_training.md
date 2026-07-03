# Runtime And Training

The engine is model-agnostic. It expects a model contract, not a specific architecture forever.

Current runtime path:

1. Train/export a PyTorch model in `python/`.
2. Export ONNX plus `model.mangahue.json`.
3. Run `mangahue model validate --metadata model.mangahue.json`.
4. Run `mangahue colorize input.png --output output.png --model model.onnx`.

The Rust engine invokes:

```powershell
python -m mangahue_train.run_onnx_tile
```

That sidecar converts Rust-generated line art, hint map, and hint mask images into the model tensor:

- Basic U-Net: `[1, 1, H, W]`
- Hint U-Net: `[1, 5, H, W]`

The native Rust ONNX provider can be added later behind the same `ColorizationBackend` trait. The model training itself is intentionally separate from the engine.

Correction-derived hints can be fed into the Python dataset through the `hints` column in the manifest. The reader accepts normalized hint coordinates and MangaHue's exported pixel-space hint JSON; pixel hints are scaled from the source line-art size to the configured training size.

```powershell
python -m mangahue_train.train `
  --manifest training/manifest.corrections.json `
  --output models/mangahue_hint_unet.pt `
  --epochs 5
```
