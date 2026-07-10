# Studio

MangaHue Studio is the local operator console served by the Rust API.

Run:

```powershell
cargo run -p mangahue-cli -- serve --addr 127.0.0.1:7878
```

Open:

```text
http://127.0.0.1:7878/
```

Views:

- Inspect: classify a manga page, write debug masks, and review page intelligence.
- Colorize: queue a colorization job and preview the result.
- Link: submit an authorized manga, webtoon, or image URL, download discovered page images, queue colorization, and preview the first completed page.
- Chapter: import folders, images, ZIP, or CBZ into normalized page sets, colorize chapters, write brain/safety/palette memory reports, and export processed pages back to ZIP/CBZ.
- Jobs: list in-memory and persisted job manifests.
- Memory: save correction records, update character colors, export page hints, build palette memory, edit locked colors, and export palette-memory styles.
- Doctor: check config parsing, styles, metadata, Python sidecar import, job storage, Studio files, and the sample page.

Studio intentionally works with local file paths because this engine is currently a desktop/local production shell. A hosted SaaS version can replace path inputs with upload storage while keeping the same API flow.
