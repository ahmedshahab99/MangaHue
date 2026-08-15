from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mangahue_train.benchmark_quality import enforce_gate, evaluate_manifest


def _write(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8), "RGB").save(path)


def test_benchmark_ranks_reference_matching_candidate_first(tmp_path: Path) -> None:
    lineart = np.full((24, 24, 3), 255, dtype=np.uint8)
    lineart[4:20, 11:13] = 0
    reference = np.full((24, 24, 3), [210, 80, 120], dtype=np.uint8)
    reference[4:20, 11:13] = 0
    degraded = np.full((24, 24, 3), [80, 150, 210], dtype=np.uint8)
    degraded[4:20, 11:13] = 180
    _write(tmp_path / "lineart.png", lineart)
    _write(tmp_path / "reference.png", reference)
    _write(tmp_path / "mangahue.png", reference)
    _write(tmp_path / "baseline.png", degraded)
    manifest = {
        "schema_version": 1,
        "samples": [
            {
                "id": "page-1",
                "lineart": "lineart.png",
                "reference": "reference.png",
                "candidates": {
                    "mangahue": "mangahue.png",
                    "baseline": "baseline.png",
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate_manifest(manifest_path, "mangahue")

    assert report["ranking"][0] == "mangahue"
    assert report["aggregate"]["mangahue"]["delta_e_mean"] < 0.01
    assert report["aggregate"]["mangahue"]["line_preservation"] == 1.0
    assert not enforce_gate(report, 1, 1.0, 0.99, 0.99, "baseline")


def test_gate_rejects_insufficient_pages_and_missing_baseline(tmp_path: Path) -> None:
    pixels = np.full((8, 8, 3), 128, dtype=np.uint8)
    for name in ("lineart.png", "reference.png", "candidate.png"):
        _write(tmp_path / name, pixels)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "samples": [
                    {
                        "lineart": "lineart.png",
                        "reference": "reference.png",
                        "candidates": {"mangahue": "candidate.png"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = evaluate_manifest(manifest_path, "mangahue")

    failures = enforce_gate(report, 50, 28.0, 0.55, 0.92, "chatgpt")

    assert any("pages" in failure for failure in failures)
    assert any("baseline" in failure for failure in failures)


def test_licensed_gate_requires_complete_rights_records(tmp_path: Path) -> None:
    pixels = np.full((8, 8, 3), 128, dtype=np.uint8)
    for name in ("lineart.png", "reference.png", "candidate.png"):
        _write(tmp_path / name, pixels)
    sample = {
        "id": "licensed-page",
        "lineart": "lineart.png",
        "reference": "reference.png",
        "candidates": {"mangahue": "candidate.png"},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 2, "samples": [sample]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="license record is required"):
        evaluate_manifest(manifest_path, "mangahue")

    sample["license"] = {
        "rights_holder": "Example rights holder",
        "usage_grant": "Written evaluation-only grant",
        "proof": "rights/grant-001.pdf",
    }
    manifest_path.write_text(
        json.dumps({"schema_version": 2, "samples": [sample]}),
        encoding="utf-8",
    )
    report = evaluate_manifest(manifest_path, "mangahue")

    assert report["licensed_pages"] == 1
    assert not enforce_gate(report, 1, 1.0, 0.99, 0.99, None, require_licensed=True)
