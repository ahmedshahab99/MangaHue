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

Training remains the major model-side task. Once a trained model is exported to ONNX with matching metadata, the engine can run it.
