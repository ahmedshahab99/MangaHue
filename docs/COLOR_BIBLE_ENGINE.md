# MangaHue Color Bible Engine

MangaHue is a chapter-aware color director. The model proposes an artistic render; the Color Bible owns identity, protected content, correction history, and delivery approval.

## Production pipeline

1. **Structure graph** detects panels, bubbles, text, protected whites, characters, face materials, clothing, and backgrounds. Irregular detector output is stored as exact run-length masks rather than bounding boxes.
2. **Series Color Bible** resolves canonical colors in this priority order: region, character, chapter, then series. Every change is versioned and can be undone.
3. **Consistency renderer** converts matching Bible entries into strong model hints and deterministic final-resolution color constraints.
4. **Artistic renderer** runs the configured DDColor or draft backend, grading, lighting, materials, and cleanup.
5. **Protected compositor** restores original luminance/ink in bubbles, text, and protected masks after every artistic and resize stage.
6. **Confidence verifier** compares each constrained semantic region with its resolved Bible color. A failure sets `needs_color_bible_consistency`, which Telegram and chapter delivery reject.
7. **Correction propagation** accepts region, character, chapter, or series scope, records an atomic revision, and reports exactly which known regions/pages are affected.

The graph stores only the input filename, not the deleted scratch path or user image bytes. Telegram scratch inputs and outputs continue to follow the configured retention policy; Color Bibles retain structured color metadata and explicitly approved reference paths.

## Telegram workflow

Send a chapter link normally. MangaHue creates:

```text
<work-dir>/color-bibles/chat_<chat-id>/<series-id>/color-bible.json
<work-dir>/color-bibles/chat_<chat-id>/<series-id>/graphs/<chapter-id>/page_###.json
```

Commands:

```text
/bible
/correct <x> <y> <#RRGGBB> <region|character|chapter|series> [character-id]
/undo
```

`x` and `y` are normalized from 0 to 1 on the last delivered page. For example:

```text
/correct 0.42 0.18 #2b2028 character hero
```

An image sent with caption `ref` remains the approved artistic reference and is also registered in the active Color Bible. The older `/color` hint command now correctly scales its normalized 1000×1000 coordinate canvas to every page aspect ratio.

Because raw pages are not retained indefinitely, the bot records the affected pages and constraints but asks the user to resend the chapter for a correction rerender. A future authenticated editor may resubmit only those source pages from the publisher URL without changing the Bible format.

## Operator API

These trusted-admin routes accept host paths:

```text
POST /api/color-bibles/create
POST /api/color-bibles/status
POST /api/color-bibles/correct
POST /api/color-bibles/tap
POST /api/color-bibles/undo
```

Tap correction example:

```json
{
  "path": "/data/mangahue/color-bibles/series/color-bible.json",
  "structure_graph": "/data/mangahue/color-bibles/series/graphs/chapter/page_001.json",
  "graphs_root": "/data/mangahue/color-bibles/series/graphs",
  "x": 0.42,
  "y": 0.18,
  "color": "#2b2028",
  "scope": "character",
  "character_id": "hero",
  "note": "approved hair color"
}
```

The response includes the updated Bible and correction impact (`affected_pages`, `affected_regions`, and exact graph/region IDs).

## CPU-first and GPU-later boundary

The current implementation is intentionally useful on the existing CPU VPS:

- exact segmentation-mask constraints and protected compositing are native Rust;
- Bible resolution, revision history, verifier, admission, and impact planning are native Rust;
- DDColor remains the artistic baseline and concurrency remains one;
- delivery gates are deterministic and do not require a GPU.

GPU work remains a measured model-selection task, not a prerequisite for this engine architecture. Before changing the production model, run the licensed held-out benchmark and blind preference study against the current DDColor baseline, MagicColor, DACoN, and MangaNinja. A candidate is promoted only if it improves preference and correction-rate metrics while passing ink, bubble, protected-mask, consistency, latency, and memory gates.

Automatic cross-page character recognition is still confidence-limited without a trained identity embedding model. Operator-supplied stable character IDs are supported today; uncertain automatic identities must not silently become canonical. This should be upgraded on a GPU worker and evaluated on the held-out character-consistency split before it can become authoritative.

## Release gates

A release is deliverable only when all of the following pass:

- Rust formatting, workspace tests, and Clippy;
- native Rust inference and release-container tests;
- production concurrency and retention policy checks;
- chapter extraction and no-skipped-page tests;
- protected-mask, exact-mask, Color Bible resolution/undo, impact, and verifier tests;
- immutable artifact build and checksum;
- deployment health check and Telegram smoke test;
- invite-only beta preference/correction thresholds for model changes.
