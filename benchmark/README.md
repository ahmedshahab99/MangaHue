# MangaHue benchmark loop

Engine-to-gate evaluation: run the Rust engine over a fixed page set, collect
per-page telemetry, and enforce the quality gate against licensed references.

## Layout

```
benchmark/
  manifest.example.json   # scored manifest template
  pages/                  # grayscale manga pages (drop yours here)
    page_01.png
    page_02.png
    reference/            # colored references, same stem as the page
      page_01.png
      page_02.png
  outputs/<batch-name>/   # written by scripts/benchmark_pipeline.py
    summary.json
    telemetry.jsonl
    <stem>.png            # colored output
    <stem>.report.json    # full pipeline report
    <stem>.report.quality.json  # quality report
```

Rules:

- Pages are grayscale manga pages (line art); references are the colored
  target look. A page and its reference share the same file stem
  (`page_01.png` ↔ `reference/page_01.png`) — the manifest pairs them by stem.
- The engine must write `.report.json` / `.report.quality.json` next to each
  colored output. That is controlled by `output.write_report` in the engine
  config (default `true`).

## 1. Generate candidates (engine run)

Build the CLI once:

```
cargo build --release -p mangahue-cli
```

Run the batch (sequential, per-page 600s timeout, failures logged, exit code
nonzero if any page failed):

```
.mangahue/python-runtime/Scripts/python.exe scripts/benchmark_pipeline.py \
    --pages-dir benchmark/pages \
    --config configs/engine.vivid.yaml \
    --model models/ddcolor-full-fp16.onnx \
    --output-dir benchmark/outputs \
    --batch-name run-vivid
```

Result: `benchmark/outputs/run-vivid/` with one colored PNG per page plus its
`.report.json` / `.report.quality.json`, a `summary.json` (one row per page,
full report + quality JSON embedded) and a `telemetry.jsonl` in the bot's
quality-telemetry line format.

## 2. Score + quality gate

`python/mangahue_train/benchmark_quality.py` scores candidates vs references
(PSNR, luminance SSIM, delta_e_mean, delta_e_p95, chroma similarity, line
preservation), aggregates per system, and enforces the gate.
`benchmark/manifest.example.json` is a worked example; all paths inside are
relative to the manifest's directory (here: `benchmark/`):

```json
{
  "schema_version": 1,
  "samples": [
    {
      "id": "page_01",
      "lineart": "pages/page_01.png",
      "reference": "pages/reference/page_01.png",
      "candidates": {
        "mangahue": "outputs/run-vivid/page_01.png",
        "legacy": "outputs/run-legacy/page_01.png"
      }
    }
  ]
}
```

Run the gate (target system `mangahue`, optionally beating `legacy`):

```
PYTHONPATH=python .mangahue/python-runtime/Scripts/python.exe -m mangahue_train.benchmark_quality \
    --manifest benchmark/manifest.example.json \
    --target mangahue \
    --baseline legacy \
    --output benchmark/quality-report.json \
    --min-pages 2 \
    --max-delta-e 28.0 --min-ssim 0.55 --min-line-preservation 0.92
```

Gate semantics: the target's aggregate must score at least `--min-pages`
pages, `delta_e_mean <= --max-delta-e`, `luminance_ssim >= --min-ssim`,
`line_preservation >= --min-line-preservation`, and (with `--baseline`) beat
the baseline's `delta_e_mean`. Defaults are `--min-pages 50` etc. — lower
`--min-pages` to your page count. A failed gate exits nonzero.
`scripts/run-quality-benchmark.ps1` wraps the same call for PowerShell.

## 3. Compare two configs

Run the pipeline twice with different `--config` and different `--batch-name`,
then either:

- summarize both batches side by side:

  ```
  .mangahue/python-runtime/Scripts/python.exe scripts/telemetry_summary.py \
      benchmark/outputs/run-a benchmark/outputs/run-b
  ```

- diff the summaries:

  ```
  .mangahue/python-runtime/Scripts/python.exe -c "import json; a=json.load(open('benchmark/outputs/run-a/summary.json')); b=json.load(open('benchmark/outputs/run-b/summary.json')); print('run-a', a['ok_pages'], '/', a['total_pages'], 'ok,', a['total_wall_seconds'], 's'); print('run-b', b['ok_pages'], '/', b['total_pages'], 'ok,', b['total_wall_seconds'], 's')"
  ```

- or point both runs at the same manifest (`"mangahue"` + `"legacy"` candidate
  keys) and run the gate with `--baseline legacy`.

## 4. Telemetry

The bot appends per-page quality reports to
`<work_dir>/quality/<YYYY-MM>/telemetry.jsonl` (one QualityReport JSON object
per record: `line_preservation_score`, `bubble_cleanliness_score`,
`protected_region_violation_score`, `hint_adherence_score`,
`output_sharpness_estimate`). Summarize bot telemetry, batch dirs, or
individual files:

```
.mangahue/python-runtime/Scripts/python.exe scripts/telemetry_summary.py work_dir/quality
.mangahue/python-runtime/Scripts/python.exe scripts/telemetry_summary.py --json benchmark/outputs/run-a
```

The report prints counts + mean/min/max per metric and a worst-pages table
(lowest line preservation, lowest bubble cleanliness, highest protected-region
violations) to drive the next tuning iteration. Telemetry records from the bot
have no page name, so they are identified by `file line N`.

## Notes

- `--legacy` on `benchmark_pipeline.py` is a pass-through: it appends
  `--legacy` to every CLI invocation for the engine's legacy-postprocess
  switch. The CLI does not accept it yet — runs fail per page (logged, not
  fatal) until the flag lands in `crates/`.
- Pages and outputs are large; consider gitignoring `benchmark/pages/` and
  `benchmark/outputs/`.
