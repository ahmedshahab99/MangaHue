"""MangaHue GPU Studio — local development server.

Implements the same HTTP contract as the production Rust API
(crates/mangahue-api): serves the /gpu studio page, the /models tree, and
POST /api/gpu/colorize which "finishes" a client-side WebGPU draft.

The real finishing pass (region renderer, protected layers, reference
finishing) runs on the VPS through the Rust engine. This dev server is a
stand-in that applies the legacy 11-stage postprocess pipeline
(python_bot/postprocess.py) so the webapp can be built and validated
locally. Swap it for the Rust API when publishing to the domain; the
webapp does not change.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
GPU_DIR = ROOT / "web" / "gpu"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / ".mangahue" / "gpu_dev"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "python_bot"))
from postprocess import apply_manga_pipeline  # noqa: E402

app = FastAPI(title="MangaHue GPU Studio dev server")

app.mount("/models", StaticFiles(directory=str(MODELS_DIR)), name="models")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "engine": "MangaHue GPU dev server"}


@app.post("/api/gpu/colorize")
async def gpu_colorize(
    page: UploadFile = File(...),
    draft: UploadFile = File(...),
    preset: str = Form(""),
    style: str = Form(""),
    palette: str = Form(""),
) -> JSONResponse:
    """Finish a client-side WebGPU draft.

    Matches the Rust API's multipart contract: 'page' (original lineart)
    + 'draft' (colorized by the user's GPU) -> finished result.
    """
    job_id = uuid.uuid4().hex[:12]
    job_dir = OUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    page_path = job_dir / "page.jpg"
    draft_path = job_dir / "draft.png"
    for target, upload in ((page_path, page), (draft_path, draft)):
        data = await upload.read()
        if len(data) > 50 * 1024 * 1024:
            raise HTTPException(413, f"{upload.filename} exceeds 50 MB")
        target.write_bytes(data)

    output_path = job_dir / "result.png"
    t0 = time.time()
    try:
        # The Rust engine resizes a client draft to the page resolution
        # before the finishing pass; mirror that here so a draft from a
        # differently-sized source never breaks the pipeline.
        from PIL import Image as _Image

        with _Image.open(page_path) as page_img, _Image.open(draft_path) as draft_img:
            if draft_img.size != page_img.size:
                draft_img = draft_img.resize(page_img.size, _Image.Resampling.BICUBIC)
                draft_img.save(draft_path)
        apply_manga_pipeline(str(draft_path), str(page_path), str(output_path))
    except Exception as exc:  # surface the real error to the webapp log
        raise HTTPException(500, f"finishing failed: {exc}") from exc
    elapsed = time.time() - t0

    rel = output_path.relative_to(ROOT)
    return JSONResponse(
        {
            "id": job_id,
            "result_url": f"/api/file?path={rel.as_posix()}",
            "report_url": None,
            "verdict": f"dev-server finishing ({elapsed:.1f}s)",
        }
    )


@app.get("/api/file")
async def local_file(path: str) -> FileResponse:
    """Serve a repo-relative file (same convention as the Rust API)."""
    target = (ROOT / path).resolve()
    if target != ROOT and ROOT not in target.parents:
        raise HTTPException(403, "path outside repo root")
    if not target.is_file():
        raise HTTPException(404, f"not found: {path}")
    return FileResponse(target)


# Serve the studio app itself (index.html, app.js, styles) as the last
# resort so it never shadows /api or /models routes.
app.mount("/", StaticFiles(directory=str(GPU_DIR), html=True), name="gpu")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
