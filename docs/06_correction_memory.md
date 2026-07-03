# Correction Memory

Correction memory lets MangaHue improve before retraining.

Add a correction:

```powershell
cargo run -p mangahue-cli -- correction add `
  --project projects/example_project/project.json `
  --page page_001 `
  --x 100 --y 200 --width 80 --height 120 `
  --label hair `
  --old "#5a4632" `
  --new "#111111" `
  --character hero
```

If the label is one of `hair`, `skin`, `eye`, `eyes`, `outfit`, or `accent`, and the character exists, the character profile is updated softly. Future hints such as `hero.hair` use that memory.

Export corrections for later training:

```powershell
cargo run -p mangahue-cli -- project export-corrections `
  --project projects/example_project/project.json `
  --output training/corrections.jsonl
```

Export a page's corrections as engine-compatible hints:

```powershell
cargo run -p mangahue-cli -- project export-hints `
  --project projects/example_project/project.json `
  --page page_001 `
  --output training/hints/page_001.hints.json
```

Prepare a trainable manifest once matching line art and target color pages exist:

```powershell
cargo run -p mangahue-cli -- project export-training-manifest `
  --project projects/example_project/project.json `
  --lineart-dir training/lineart `
  --color-dir training/color `
  --hints-dir training/hints `
  --output training/manifest.corrections.json
```

The manifest uses the Python dataset contract: each row contains `lineart`, `color`, and `hints`. Pages without matching line art or target color files are reported instead of silently included.

This creates an active-learning trail: user correction -> project memory -> character consistency -> hint export -> future fine-tuning data.
