# Licensed held-out release benchmark

Public promotion is blocked until MangaHue passes an untouched, licensed benchmark. The
invite-only CPU beta may run before that point, but beta feedback must not be represented
as benchmark clearance.

## Dataset contract

- Use at least 50 complete manga pages from multiple artists, genres, lighting conditions,
  skin tones, screentone densities, and dialogue layouts.
- Never train, tune thresholds, or select palettes on these pages.
- Keep source pages, references, candidate outputs, and grant evidence private.
- Use schema version 2. Every sample must include `rights_holder`, `usage_grant`, and a
  private `proof` path or document identifier.
- Freeze the manifest and record its SHA-256 before generating candidate outputs.
- Evaluate MangaHue and DDColor on every page. Add MagicColor, DACoN, and MangaNinja when
  a rented GPU worker is available and their licenses permit the intended evaluation.

Use `benchmarks/heldout/manifest.example.json` as the shape only. It contains no data and
cannot pass a release gate.

## Promotion gate

Run the command below in the isolated benchmark environment:

```bash
PYTHONPATH=python python -m mangahue_train.benchmark_quality \
  --manifest /private/mangahue-heldout/manifest.json \
  --target mangahue \
  --baseline ddcolor \
  --output /private/mangahue-heldout/report.json \
  --min-pages 50 \
  --max-delta-e 28 \
  --min-ssim 0.55 \
  --min-line-preservation 0.92
```

The command fails closed when any page lacks license evidence, the sample count is too
small, a metric misses its threshold, or MangaHue does not beat DDColor on mean Delta E.
Record the report hash with the release SHA. Public promotion additionally requires the
normal CI and deployment checks plus a blind preference review of beta output and all
high-severity correction clusters resolved.

`--allow-unlicensed` exists only for local tests and pipeline development. It must never
be used for a release decision.
