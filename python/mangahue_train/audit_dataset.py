from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .contract import CONTRACT_ID, canonical_hash, sha256_file


@dataclass(frozen=True)
class AuditedSample:
    sample_id: str
    group_id: str
    lineart: str
    color: str
    lineart_sha256: str
    color_sha256: str
    width: int
    height: int
    hints: str | None = None
    provenance: str | None = None
    license: str | None = None


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _group(row: dict[str, Any], lineart: Path) -> str:
    return str(row.get("group_id") or row.get("chapter_id") or lineart.parent.name)


def audit(manifest: Path, strict_rights: bool = False) -> dict[str, Any]:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    rows = raw["samples"] if isinstance(raw, dict) else raw
    root = manifest.parent
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    content_seen: dict[tuple[str, str], str] = {}
    sample_ids: set[str] = set()

    for index, row in enumerate(rows):
        try:
            lineart = _resolve(root, row["lineart"])
            color = _resolve(root, row["color"])
            if not lineart.is_file() or not color.is_file():
                raise ValueError("lineart or color file is missing")
            with Image.open(lineart) as source, Image.open(color) as target:
                source.verify()
                target.verify()
            with Image.open(lineart) as source, Image.open(color) as target:
                if source.size != target.size:
                    raise ValueError(f"unaligned dimensions: {source.size} != {target.size}")
                width, height = source.size
            if width < 64 or height < 64:
                raise ValueError("image is smaller than 64x64")
            provenance = row.get("provenance")
            license_name = row.get("license")
            if strict_rights and (not provenance or not license_name):
                raise ValueError("provenance and license are required")
            line_hash, color_hash = sha256_file(lineart), sha256_file(color)
            duplicate_key = (line_hash, color_hash)
            if duplicate_key in content_seen:
                raise ValueError(f"duplicate content of {content_seen[duplicate_key]}")
            sample_id = str(row.get("sample_id") or hashlib.sha256(f"{line_hash}:{color_hash}".encode()).hexdigest()[:24])
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            sample_ids.add(sample_id)
            content_seen[duplicate_key] = sample_id
            value = AuditedSample(
                sample_id=sample_id,
                group_id=_group(row, lineart),
                lineart=row["lineart"],
                color=row["color"],
                lineart_sha256=line_hash,
                color_sha256=color_hash,
                width=width,
                height=height,
                hints=row.get("hints"),
                provenance=provenance,
                license=license_name,
            )
            samples.append({key: val for key, val in value.__dict__.items() if val is not None})
        except (KeyError, OSError, ValueError) as error:
            errors.append(f"sample {index}: {error}")

    groups = sorted({sample["group_id"] for sample in samples})
    report = {
        "schema_version": 2,
        "contract_id": CONTRACT_ID,
        "samples": samples,
        "sample_count": len(samples),
        "group_count": len(groups),
        "errors": errors,
    }
    report["manifest_sha256"] = canonical_hash(report)
    return report


def assert_no_group_leakage(manifests: list[Path]) -> None:
    owners: dict[str, Path] = {}
    for manifest in manifests:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        rows = value.get("samples", value) if isinstance(value, dict) else value
        for row in rows:
            group = str(row.get("group_id") or row.get("chapter_id") or Path(row["lineart"]).parent.name)
            previous = owners.get(group)
            if previous and previous != manifest:
                raise ValueError(f"group leakage: {group!r} appears in {previous} and {manifest}")
            owners[group] = manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and freeze a MangaHue dataset manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, nargs="*", default=[])
    parser.add_argument("--strict-rights", action="store_true")
    args = parser.parse_args()
    report = audit(args.manifest, strict_rights=args.strict_rights)
    if report["errors"]:
        raise SystemExit("dataset audit failed:\n" + "\n".join(report["errors"]))
    if args.compare:
        assert_no_group_leakage([args.manifest, *args.compare])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"audited {report['sample_count']} samples in {report['group_count']} groups -> {args.output}")
