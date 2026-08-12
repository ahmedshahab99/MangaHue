# MangaHue Models

Place exported ONNX models here. The Rust engine validates model metadata before inference.

The Rust backend validates model metadata and can execute ONNX through the Python sidecar backend:

```powershell
cd python
python -m pip install -e .
cd ..
cargo run -p mangahue-cli -- model validate --metadata models/model.mangahue.json
cargo run -p mangahue-cli -- colorize input.png --output output.png --model models/model.onnx
```

Set `MANGAHUE_PYTHON` if the ONNX runtime lives in a virtual environment:

```powershell
$env:MANGAHUE_PYTHON="C:\path\to\.venv\Scripts\python.exe"
```

## ONNX Runtime library

Native Rust inference loads `onnxruntime.dll` dynamically. Point `ORT_DYLIB_PATH`
at a recent runtime (>= 1.23, C API level 24):

- Windows: `models/onnxruntime/onnxruntime.dll` (copy the DLL + `onnxruntime_providers_shared.dll` from an onnxruntime install there), or
- Linux: the `libonnxruntime.so` shipped by the distribution package.

The engine refuses (instead of deadlocking) when the found runtime is too old —
Windows' in-box `System32\onnxruntime.dll` (1.17) is a common offender.

Anime face detection is pure Rust (LBP cascade from `models/anime-face/`); it
needs no Python/OpenCV.

Training remains the major model-side task. Once a trained model is exported to ONNX with matching metadata, the engine can run it.
