# API

Run the local API:

```powershell
cargo run -p mangahue-api -- --addr 127.0.0.1:7878
```

Or through the main CLI:

```powershell
cargo run -p mangahue-cli -- serve --addr 127.0.0.1:7878
```

The same server hosts the local Studio at:

```text
http://127.0.0.1:7878/
```

Routes:

- `GET /health`
- `GET /api/doctor`
- `POST /api/doctor`
- `GET /api/file`
- `GET /api/styles`
- `POST /api/inspect`
- `POST /api/colorize`
- `POST /api/batch`
- `POST /api/link/ingest`
- `POST /api/link/colorize`
- `POST /api/chapter/import`
- `POST /api/chapter/export`
- `POST /api/chapter/colorize`
- `POST /api/brain/palette-memory`
- `POST /api/brain/palette-memory/edit`
- `GET /api/jobs/{id}`
- `GET /api/jobs/{id}/result`
- `POST /api/projects`
- `POST /api/projects/status`
- `POST /api/projects/export-corrections`
- `POST /api/projects/export-hints`
- `POST /api/projects/export-training-manifest`
- `POST /api/projects/characters`
- `POST /api/projects/{id}/characters`
- `POST /api/corrections`

Job manifests are persisted as JSON under `--jobs-dir`, so completed job metadata can be recovered even after a server restart.
`GET /api/file?path=...` serves local output files for Studio previews. The resolved file must stay under the server working directory.

Doctor payload:

```json
{
  "config_path": "configs/engine.default.yaml",
  "styles_dir": "configs/styles",
  "metadata_path": "models/model.mangahue.json",
  "jobs_dir": ".mangahue/jobs",
  "sample_input": "examples/input/smoke_page.png",
  "python": "python"
}
```

Inspect payload:

```json
{
  "request": {
    "input_path": "examples/input/smoke_page.png",
    "debug_dir": "examples/output/api_inspect_debug"
  }
}
```

Colorize payload:

```json
{
  "request": {
    "input_path": "examples/input/smoke_page.png",
    "output_path": "examples/output/api_smoke.png",
    "debug_dir": "examples/output/api_debug",
    "use_mock_backend": true
  },
  "ads_token": null,
  "premium_account_id": null
}
```

Batch payload:

```json
{
  "input_dir": "examples/input",
  "output_dir": "examples/output/api_batch",
  "continue_on_error": true,
  "template": {
    "input_path": "",
    "output_path": "",
    "use_mock_backend": true
  }
}
```

Link ingest payload:

```json
{
  "source_url": "https://your-authorized-site.example/chapter/1",
  "output_dir": ".mangahue/link_downloads/chapter_1",
  "max_pages": 32,
  "max_bytes_per_page": 33554432,
  "user_agent": null
}
```

Link colorize payload:

```json
{
  "source_url": "https://your-authorized-site.example/chapter/1",
  "output_dir": "examples/output/link_chapter_1",
  "downloads_dir": ".mangahue/link_downloads",
  "max_pages": 32,
  "continue_on_error": true,
  "template": {
    "input_path": "",
    "output_path": "",
    "style_path": "configs/styles/webtoon_bright.yaml",
    "use_mock_backend": true
  }
}
```

The link routes are for authorized manga, webtoon, and image URLs. They do not bypass logins, paywalls, DRM, or site protections.

Chapter import payload:

```json
{
  "input_path": "chapter.cbz",
  "output_dir": ".mangahue/imported/chapter_001",
  "manifest_path": ".mangahue/imported/chapter_001/manifest.json",
  "max_pages": 500,
  "max_bytes_per_page": 52428800,
  "overwrite": true
}
```

Chapter export payload:

```json
{
  "input_dir": "examples/output/colored_chapter",
  "output_file": "examples/output/colored_chapter.cbz",
  "manifest_path": "examples/output/colored_chapter.export.json"
}
```

Chapter colorize payload:

```json
{
  "input_path": "chapter.cbz",
  "work_dir": ".mangahue/chapter_jobs",
  "output_dir": "examples/output/colored_chapter",
  "export_path": "examples/output/colored_chapter.cbz",
  "gallery_path": "examples/output/colored_chapter/gallery.html",
  "manifest_path": "examples/output/colored_chapter/job.json",
  "brain_report_path": "examples/output/colored_chapter/mangahue_chapter_brain.json",
  "locked_style_path": "examples/output/colored_chapter/mangahue_locked_style.json",
  "palette_memory_path": "examples/output/colored_chapter/mangahue_palette_memory.json",
  "palette_memory_style_path": "examples/output/colored_chapter/mangahue_palette_memory_style.json",
  "safety_report_path": "examples/output/colored_chapter/mangahue_safety_gate.json",
  "max_pages": 500,
  "resume": false,
  "safety_gate": true,
  "enforce_safety_gate": false,
  "consistency_retry": true,
  "retry_drift_threshold": 0.18,
  "retry_tone_threshold": 0.34,
  "retry_max_pages": 24,
  "template": {
    "input_path": "",
    "output_path": "",
    "style_path": "configs/styles/webtoon_bright.yaml",
    "use_mock_backend": true
  }
}
```

Palette memory payload:

```json
{
  "brain_report_path": "examples/output/colored_chapter/mangahue_chapter_brain.json",
  "output_path": "examples/output/colored_chapter/mangahue_palette_memory.json",
  "style_output_path": "examples/output/colored_chapter/mangahue_palette_memory_style.json",
  "base_style_path": "configs/styles/anime_cel.yaml",
  "id": "chapter_one",
  "name": "Chapter One Palette Memory",
  "max_entries": 12,
  "min_confidence": 0.18
}
```

Palette memory edit payload:

```json
{
  "memory_path": "examples/output/colored_chapter/mangahue_palette_memory.json",
  "style_output_path": "examples/output/colored_chapter/mangahue_palette_memory_style.json",
  "edits": [
    {
      "id": "hair_0a141e",
      "role": "hair",
      "color": "#0a141e",
      "locked": true,
      "confidence": 0.99,
      "tolerance": 12.0,
      "notes": "Hero hair lock"
    }
  ]
}
```

Chapter import supports folders, single images, `.zip`, and `.cbz`. Pages are validated, naturally sorted, and normalized to `page_001.ext`, `page_002.ext`, and so on.

The API stores in-memory job state for now. The contract is intentionally shaped so it can later be backed by Redis, SQLite, Postgres, or a cloud queue without changing engine internals.
It also writes each job manifest to disk. The in-memory map is a fast cache, not the only record.

Correction payload:

```json
{
  "project": "projects/example_project/project.json",
  "page": "page_001",
  "x": 100,
  "y": 200,
  "width": 80,
  "height": 120,
  "label": "hair",
  "old": "#5a4632",
  "new": "#111111",
  "character": "hero"
}
```

Correction hint export payload:

```json
{
  "project": "projects/example_project/project.json",
  "page": "page_001",
  "output": "training/hints/page_001.hints.json"
}
```

Correction training manifest payload:

```json
{
  "project": "projects/example_project/project.json",
  "lineart_dir": "training/lineart",
  "color_dir": "training/color",
  "hints_dir": "training/hints",
  "output": "training/manifest.corrections.json"
}
```
