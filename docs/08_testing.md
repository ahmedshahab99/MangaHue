# Testing

Core verification commands:

```powershell
cargo fmt --all
cargo test
cargo clippy --all-targets -- -D warnings
python -m compileall python\mangahue_train
```

CLI smoke checks:

```powershell
cargo run -p mangahue-cli -- inspect examples\input\smoke_page.png --output examples\output\smoke_inspection.json --debug examples\output\inspect_debug
cargo run -p mangahue-cli -- colorize examples\input\smoke_page.png --output examples\output\smoke_colored.png --mock --debug examples\output\smoke_debug
cargo run -p mangahue-cli -- benchmark examples\input --output examples\output\benchmark --mock --report examples\output\benchmark.json
cargo run -p mangahue-cli -- styles list --dir configs\styles
```

Model contract check:

```powershell
cargo run -p mangahue-cli -- model validate --metadata models\model.mangahue.json
```

ONNX runtime check after installing `python/` and exporting a real model:

```powershell
cargo run -p mangahue-cli -- model validate-onnx --model models\model.onnx --metadata models\model.mangahue.json
```

API smoke check:

```powershell
cargo run -p mangahue-api -- --addr 127.0.0.1:7878
```

Then visit:

- `http://127.0.0.1:7878/health`
- `http://127.0.0.1:7878/api/styles`
- `http://127.0.0.1:7878/`

Studio smoke check:

1. Open the Studio.
2. Run Inspect with `examples/input/smoke_page.png`.
3. Queue Colorize with `use_mock_backend` enabled.
4. Refresh Jobs.
5. Save a correction in Memory.
