from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.asarray(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ matrix.T
    xyz /= np.asarray([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    f = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        [
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def _global_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    x = _luma(reference).astype(np.float64)
    y = _luma(candidate).astype(np.float64)
    mean_x, mean_y = float(x.mean()), float(y.mean())
    var_x, var_y = float(x.var()), float(y.var())
    covariance = float(((x - mean_x) * (y - mean_y)).mean())
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
    denominator = (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    return float(np.clip(numerator / max(denominator, 1e-12), -1.0, 1.0))


def score_candidate(
    lineart: np.ndarray, reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    if lineart.shape != reference.shape or candidate.shape != reference.shape:
        raise ValueError(
            f"image dimensions differ: lineart={lineart.shape}, "
            f"reference={reference.shape}, candidate={candidate.shape}"
        )
    mse = float(np.mean((reference - candidate) ** 2))
    psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    reference_lab = _srgb_to_lab(reference)
    candidate_lab = _srgb_to_lab(candidate)
    delta_e = np.linalg.norm(reference_lab - candidate_lab, axis=-1)
    chroma_error = np.linalg.norm(
        reference_lab[..., 1:] - candidate_lab[..., 1:], axis=-1
    )
    ink = _luma(lineart) < 0.28
    if np.any(ink):
        line_preservation = float(np.mean(_luma(candidate)[ink] < 0.38))
    else:
        line_preservation = 1.0
    chroma_similarity = float(np.clip(1.0 - chroma_error.mean() / 128.0, 0.0, 1.0))
    return {
        "psnr_db": float(psnr),
        "luminance_ssim": _global_ssim(reference, candidate),
        "delta_e_mean": float(delta_e.mean()),
        "delta_e_p95": float(np.percentile(delta_e, 95)),
        "chroma_similarity": chroma_similarity,
        "line_preservation": line_preservation,
    }


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def evaluate_manifest(manifest_path: Path, target: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError("benchmark manifest schema_version must be 1 or 2")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("benchmark manifest must contain at least one sample")
    base = manifest_path.parent
    systems: dict[str, list[dict[str, float]]] = {}
    pages: list[dict[str, Any]] = []
    licensed_pages = 0
    for index, sample in enumerate(samples):
        sample_id = str(sample.get("id") or f"page-{index + 1:04d}")
        license_record = sample.get("license")
        if schema_version == 2:
            if not isinstance(license_record, dict):
                raise ValueError(f"{sample_id}: license record is required")
            missing = [
                field
                for field in ("rights_holder", "usage_grant", "proof")
                if not str(license_record.get(field, "")).strip()
            ]
            if missing:
                raise ValueError(
                    f"{sample_id}: license record is missing {', '.join(missing)}"
                )
            licensed_pages += 1
        candidates = sample.get("candidates")
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError(f"{sample_id}: candidates must be a non-empty object")
        lineart = _load_rgb(_resolve(base, sample["lineart"]))
        reference = _load_rgb(_resolve(base, sample["reference"]))
        page_scores: dict[str, Any] = {
            "id": sample_id,
            "licensed": schema_version == 2,
            "systems": {},
        }
        for system, path_value in sorted(candidates.items()):
            metrics = score_candidate(
                lineart, reference, _load_rgb(_resolve(base, str(path_value)))
            )
            systems.setdefault(system, []).append(metrics)
            page_scores["systems"][system] = metrics
        pages.append(page_scores)
    if target not in systems:
        raise ValueError(f"target system '{target}' is not present in the manifest")

    aggregate: dict[str, dict[str, float]] = {}
    for system, scores in systems.items():
        aggregate[system] = {
            key: float(np.mean([score[key] for score in scores]))
            for key in scores[0]
        }
        aggregate[system]["pages"] = float(len(scores))
    ranking = sorted(
        aggregate,
        key=lambda name: (
            aggregate[name]["delta_e_mean"],
            -aggregate[name]["luminance_ssim"],
        ),
    )
    report: dict[str, Any] = {
        "schema_version": 2,
        "manifest": str(manifest_path.resolve()),
        "target": target,
        "licensed_pages": licensed_pages,
        "aggregate": aggregate,
        "ranking": ranking,
        "pages": pages,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["evaluation_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def enforce_gate(
    report: dict[str, Any],
    min_pages: int,
    max_delta_e: float,
    min_ssim: float,
    min_line_preservation: float,
    baseline: str | None,
    require_licensed: bool = False,
) -> list[str]:
    target = report["aggregate"][report["target"]]
    failures: list[str] = []
    if target["pages"] < min_pages:
        failures.append(f"pages {target['pages']:.0f} < required {min_pages}")
    if require_licensed and report.get("licensed_pages", 0) < target["pages"]:
        failures.append(
            f"licensed pages {report.get('licensed_pages', 0)} < evaluated pages "
            f"{target['pages']:.0f}"
        )
    if target["delta_e_mean"] > max_delta_e:
        failures.append(
            f"delta_e_mean {target['delta_e_mean']:.4f} > allowed {max_delta_e:.4f}"
        )
    if target["luminance_ssim"] < min_ssim:
        failures.append(
            f"luminance_ssim {target['luminance_ssim']:.4f} < required {min_ssim:.4f}"
        )
    if target["line_preservation"] < min_line_preservation:
        failures.append(
            "line_preservation "
            f"{target['line_preservation']:.4f} < required {min_line_preservation:.4f}"
        )
    if baseline:
        if baseline not in report["aggregate"]:
            failures.append(f"baseline '{baseline}' is missing")
        else:
            other = report["aggregate"][baseline]
            if target["delta_e_mean"] >= other["delta_e_mean"]:
                failures.append(
                    f"target delta_e_mean {target['delta_e_mean']:.4f} does not beat "
                    f"{baseline} {other['delta_e_mean']:.4f}"
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MangaHue and baseline outputs against licensed references"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", default="mangahue")
    parser.add_argument("--baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pages", type=int, default=50)
    parser.add_argument("--max-delta-e", type=float, default=28.0)
    parser.add_argument("--min-ssim", type=float, default=0.55)
    parser.add_argument("--min-line-preservation", type=float, default=0.92)
    parser.add_argument(
        "--allow-unlicensed",
        action="store_true",
        help="development only: do not require per-page license records",
    )
    args = parser.parse_args()
    report = evaluate_manifest(args.manifest, args.target)
    failures = enforce_gate(
        report,
        min_pages=args.min_pages,
        max_delta_e=args.max_delta_e,
        min_ssim=args.min_ssim,
        min_line_preservation=args.min_line_preservation,
        baseline=args.baseline,
        require_licensed=not args.allow_unlicensed,
    )
    report["gate"] = {"passed": not failures, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("quality gate failed:\n- " + "\n- ".join(failures))
    print(f"quality gate passed -> {args.output}")


if __name__ == "__main__":
    main()
