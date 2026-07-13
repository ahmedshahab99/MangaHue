from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sample_id(root: Path, source: Path) -> str:
    relative = str(source.relative_to(root))
    return hashlib.sha256(relative.encode()).hexdigest()[:24]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove Kaggle's duplicated nested dataset copy")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    prepared = args.root / "mangahue-prepared"
    canonical_sources = sorted((args.root / "data").glob("train/*")) + sorted((args.root / "data").glob("val/*"))
    source_by_id = {
        sample_id(args.root, path): str(path.relative_to(args.root)).replace("\\", "/")
        for path in canonical_sources
        if path.is_file()
    }
    keep = set(source_by_id)
    manifests: dict[str, int] = {}
    values: dict[str, dict[str, object]] = {}
    referenced: set[str] = set()
    for split in ("train", "validation", "test"):
        path = prepared / f"{split}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        samples = [sample for sample in value["samples"] if sample["sample_id"] in keep]
        for sample in samples:
            sample["source_path"] = source_by_id[sample["sample_id"]]
        value["samples"] = samples
        values[split] = value

    seen_content: set[tuple[str, str]] = set()
    content_duplicates = 0
    for split in ("test", "validation", "train"):
        unique: list[dict[str, object]] = []
        samples = values[split]["samples"]
        assert isinstance(samples, list)
        for sample in samples:
            assert isinstance(sample, dict)
            identifier = str(sample["sample_id"])
            fingerprint = (
                file_hash(prepared / "lineart" / f"{identifier}.png"),
                file_hash(prepared / "color" / f"{identifier}.png"),
            )
            if fingerprint in seen_content:
                content_duplicates += 1
                continue
            seen_content.add(fingerprint)
            unique.append(sample)
            referenced.add(identifier)
        values[split]["samples"] = unique
        manifests[split] = len(unique)

    for split, value in values.items():
        (prepared / f"{split}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    removed = 0
    for folder in (prepared / "lineart", prepared / "color"):
        for image in folder.glob("*.png"):
            if image.stem not in referenced:
                image.unlink()
                removed += 1
    summary = {
        "candidates": len(canonical_sources),
        "duplicates_removed": removed // 2,
        "content_duplicates_removed": content_duplicates,
        "rejected": 0,
        **manifests,
    }
    (prepared / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (prepared / "READY").unlink(missing_ok=True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
