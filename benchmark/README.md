# MangaHue held-out benchmark

Keep licensed benchmark pages outside the public repository. The committed manifest is a schema example only and cannot satisfy a release gate.

Run the pinned Rust engine against the private input set:

```bash
cargo run --locked --release -p mangahue-cli -- benchmark \
  /private/mangahue-heldout/inputs \
  --output /private/mangahue-heldout/outputs \
  --model models/ddcolor-full-fp32.onnx \
  --report /private/mangahue-heldout/report.json
```

A promotion record must include the private manifest hash, engine report hash, release SHA, at least 50 licensed pages, blind preference results, and confirmation that ink, text, bubble, protected-mask, color-evidence, latency, and memory gates passed.
