# Training Workflow

End-to-end pipeline from raw manga pages to a deployed ONNX model:

```text
linearts/  colors/  hints/(optional)
        |
        v
  prepare_dataset.py
        |
        v
  train.json (+ val.json)
        |
        v
       train.py  ----> checkpoint.pt (+ best.pt)
        |
        v
  export_onnx.py --validate
        |
        v
  model.onnx + model.mangahue.json
        |
        v
  cargo run -p mangahue-cli -- model validate / validate-onnx
        |
        v
  cargo run -p mangahue-cli -- colorize input.png --output out.png --model model.onnx
```

## 1. Install Python tooling

```powershell
cd python
pip install -e .
```

## 2. Prepare dataset manifest

Folder layout expects matching file stems across the two folders:

```text
data/linearts/page_001.png
data/linearts/page_002.png
data/colors/page_001.png
data/colors/page_002.png
```

Generate a manifest with optional synthetic hints (sample sparse colored dots from each color target — this is the cheapest way to teach the model that the hint channels matter):

```powershell
mangahue-prepare-dataset `
  --linearts data/linearts `
  --colors   data/colors `
  --output-train data/train.json `
  --output-val   data/val.json `
  --val-ratio 0.1 `
  --synthetic-hints
```

If you already have real hint JSONs authored by the engine or Studio, drop them beside each color target named `<stem>.hints.json` and skip `--synthetic-hints`.

## 3. Train

```powershell
python -m mangahue_train.train `
  --manifest data/train.json `
  --val-manifest data/val.json `
  --output models/checkpoint.pt `
  --best-output models/checkpoint.best.pt `
  --epochs 30 --batch-size 4 --lr 2e-4 `
  --synthetic-hints --synthetic-hints-prob 0.5
```

Notes:

- `--best-output` writes the lowest-val-loss checkpoint.
- If you did not generate a val manifest, pass `--val-ratio 0.1` and the trainer will hold out pairs from the training manifest.

## 4. Export to ONNX (with optional validation)

```powershell
python -m mangahue_train.export_onnx `
  --checkpoint models/checkpoint.best.pt `
  --output models/model.onnx `
  --metadata models/model.mangahue.json `
  --validate
```

`--validate` calls the same check the Rust CLI runs, ensuring `[1, 3, H, W]` output and `0..1` range.

## 5. Validate from Rust

```powershell
cargo run -p mangahue-cli -- model validate-onnx --model models/model.onnx --metadata models/model.mangahue.json
```

## 6. Colorize with the real model

```powershell
cargo run -p mangahue-cli -- colorize examples/input/page.png `
  --output examples/output/page.png `
  --model models/model.onnx `
  --metadata models/model.mangahue.json `
  --debug examples/output/debug
```

Drop `--mock` — the engine now calls the Python ONNX sidecar against your trained model.

## Helper script

A PowerShell wrapper lives at `scripts/train.ps1` that walks prepare -> train -> export -> validate -> colorize with sensible defaults:

```powershell
pwsh ./scripts/train.ps1 `
  -Linearts data/linearts `
  -Colors   data/colors `
  -ModelOut models/model.onnx `
  -Input examples/input/page.png `
  -Output examples/output/page.png `
  -Epochs 30
```
