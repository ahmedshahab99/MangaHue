# Delivery Engine

MangaHue can now be operated as a delivery product around the trainable colorization engine.

## Web Flow

1. User submits an authorized manga, webtoon, or image URL in Studio or through `POST /api/link/colorize`.
2. `mangahue-ingest` downloads direct image URLs or extracts page images from HTML.
3. The API persists a normal job manifest and runs the existing `MangaHueEngine`.
4. The user polls `/api/jobs/{id}` and previews outputs through `/api/file`.

CLI smoke:

```powershell
cargo run -p mangahue-cli -- link colorize `
  "https://your-authorized-site.example/chapter/1" `
  --output examples/output/link_chapter `
  --max-pages 8 `
  --style configs/styles/webtoon_bright.yaml `
  --mock
```

Fetch only:

```powershell
cargo run -p mangahue-cli -- link fetch `
  "https://your-authorized-site.example/chapter/1" `
  --output .mangahue/link_downloads/chapter_1 `
  --max-pages 32
```

Chapter archive flow:

```powershell
cargo run -p mangahue-cli -- chapter import chapter.cbz `
  --output .mangahue/imported/chapter_001

cargo run -p mangahue-cli -- chapter colorize chapter.cbz `
  --output examples/output/colored_chapter `
  --export examples/output/colored_chapter.cbz `
  --mock
```

The archive path supports folders, single images, ZIP, and CBZ. MangaHue validates page images, sorts them naturally, normalizes output names, and writes a manifest beside imported pages.

## Telegram Flow

Run:

```powershell
$env:TELEGRAM_BOT_TOKEN="123456:token"
cargo run -p mangahue-telegram -- `
  --work-dir .mangahue/telegram `
  --max-pages 4 `
  --mock
```

The bot accepts a message containing one authorized URL, downloads page images, colorizes them, and sends PNG documents back to the chat. After training, remove `--mock` and pass `--model`, `--config`, and optionally `--style`.

## Production Notes

- Keep ingestion limited to authorized user-submitted URLs.
- Put the API behind authentication before hosting it publicly.
- Store job manifests and outputs on durable storage when deployed.
- Move `POST /api/link/colorize` jobs to a queue when traffic grows.
- Use the same model contract for web and Telegram delivery, so training improves every surface at once.
