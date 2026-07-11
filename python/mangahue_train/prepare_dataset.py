from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _pair(linearts: dict[str, Path], colors: dict[str, Path], stem: str) -> tuple[Path, Path] | None:
    line_path = linearts.get(stem)
    color_path = colors.get(stem)
    if line_path is None or color_path is None:
        return None
    return line_path, color_path


def _index(folder: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not folder.exists():
        return out
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            out[path.stem] = path
    return out


def _stems(linearts: dict[str, Path], colors: dict[str, Path]) -> list[str]:
    keys = set(linearts).intersection(colors)
    return sorted(keys)


def _group_id(stem: str, pattern: str) -> str:
    match = re.search(pattern, stem)
    if match:
        return match.group(1) if match.groups() else match.group(0)
    return re.sub(r"(?:[-_. ]?(?:page|p)?\d+)$", "", stem, flags=re.IGNORECASE) or stem


def _split(stems: list[str], val_ratio: float, seed: int, group_pattern: str) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for stem in stems:
        groups.setdefault(_group_id(stem, group_pattern), []).append(stem)
    pool = sorted(groups)
    rng.shuffle(pool)
    if not stems:
        return [], []
    cut = max(1, int(round(len(pool) * val_ratio))) if val_ratio > 0 and len(pool) > 1 else 0
    if cut >= len(pool):
        cut = max(0, len(pool) - 1)
    val_groups = set(pool[:cut])
    val = [stem for stem in stems if _group_id(stem, group_pattern) in val_groups]
    train = [stem for stem in stems if _group_id(stem, group_pattern) not in val_groups]
    return train, val


def _emit_manifest(
    rows: Iterable[tuple[Path, Path, Path | None, str]],
    target: Path,
) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    for lineart, color, hints, group_id in rows:
        record: dict[str, str] = {
            "lineart": str(lineart).replace("\\", "/"),
            "color": str(color).replace("\\", "/"),
            "group_id": group_id,
        }
        if hints is not None:
            record["hints"] = str(hints).replace("\\", "/")
        rendered.append(record)
    target.write_text(json.dumps(rendered, indent=2), encoding="utf-8")
    return len(rendered)


def _build_synthetic_hint_json(
    targets: list[tuple[Path, Path, Path]],
    radius: float,
    min_dots: int,
    max_dots: int,
    rng: random.Random,
) -> None:
    """Sample sparse colored dots from each color target into a hint JSON.

    Heavy imports stay deferred so this code path is cheap when unused.
    """
    from PIL import Image

    for _lineart, color, hint_path in targets:
        with Image.open(color) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size

        count = rng.randint(min_dots, max(max_dots, min_dots))
        hints: list[dict[str, object]] = []
        for _ in range(count):
            x = rng.random()
            y = rng.random()
            px = min(w - 1, max(0, int(x * w)))
            py = min(h - 1, max(0, int(y * h)))
            r, g, b = rgb.getpixel((px, py))
            hints.append(
                {
                    "x": x,
                    "y": y,
                    "color": f"#{r:02x}{g:02x}{b:02x}",
                    "radius": radius,
                    "strength": 1.0,
                }
            )
        hint_path.parent.mkdir(parents=True, exist_ok=True)
        hint_path.write_text(json.dumps({"hints": hints}, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a MangaHue training manifest from paired folders.",
    )
    parser.add_argument("--linearts", type=Path, required=True, help="Folder of grayscale line art images")
    parser.add_argument("--colors", type=Path, required=True, help="Folder of color ground-truth images")
    parser.add_argument("--output-train", type=Path, required=True, help="Output JSON manifest for training split")
    parser.add_argument("--output-val", type=Path, help="Optional output JSON manifest for validation split")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of pairs held out for validation")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--group-pattern", default=r"^(.+?)(?:[-_. ]?(?:page|p)?\d+)$", help="Regex whose first group identifies a chapter/series split group")
    parser.add_argument("--synthetic-hints", action="store_true", help="Generate sparse hint JSON sidecars from the color targets")
    parser.add_argument("--hints-dir", type=Path, help="Output directory for synthetic hint JSONs (defaults next to color images)")
    parser.add_argument("--hint-radius", type=float, default=0.025)
    parser.add_argument("--hint-min-dots", type=int, default=3)
    parser.add_argument("--hint-max-dots", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.linearts.exists() or not args.colors.exists():
        raise SystemExit("linearts and colors folders must both exist")

    linearts = _index(args.linearts)
    colors = _index(args.colors)
    stems = _stems(linearts, colors)
    if not stems:
        raise SystemExit("no paired lineart/color images found (file stems must match)")

    train_stems, val_stems = _split(stems, args.val_ratio if args.output_val else 0.0, args.seed, args.group_pattern)
    if args.output_val and not val_stems:
        raise SystemExit("validation split requires at least two distinct groups; adjust --group-pattern")

    hints_dir = args.hints_dir or args.colors
    train_targets: list[tuple[Path, Path, Path]] = []
    val_targets: list[tuple[Path, Path, Path]] = []
    if args.synthetic_hints:
        train_targets = [
            (linearts[s], colors[s], hints_dir / f"{s}.hints.json") for s in train_stems
        ]
        val_targets = [
            (linearts[s], colors[s], hints_dir / f"{s}.hints.json") for s in val_stems
        ]
        rng = random.Random(args.seed)
        _build_synthetic_hint_json(
            train_targets,
            radius=args.hint_radius,
            min_dots=args.hint_min_dots,
            max_dots=args.hint_max_dots,
            rng=rng,
        )
        if val_targets:
            _build_synthetic_hint_json(
                val_targets,
                radius=args.hint_radius,
                min_dots=args.hint_min_dots,
                max_dots=args.hint_max_dots,
                rng=random.Random(args.seed + 1),
            )

    train_rows = [
        (linearts[s], colors[s], hints_dir / f"{s}.hints.json" if args.synthetic_hints else None, _group_id(s, args.group_pattern))
        for s in train_stems
    ]
    train_count = _emit_manifest(iter(train_rows), args.output_train)
    print(f"wrote train manifest with {train_count} samples -> {args.output_train}")

    if args.output_val and val_stems:
        val_rows = [
            (linearts[s], colors[s], hints_dir / f"{s}.hints.json" if args.synthetic_hints else None, _group_id(s, args.group_pattern))
            for s in val_stems
        ]
        val_count = _emit_manifest(iter(val_rows), args.output_val)
        print(f"wrote val manifest with {val_count} samples -> {args.output_val}")
    elif args.output_val:
        _emit_manifest(iter([]), args.output_val)
        print(f"wrote empty val manifest -> {args.output_val}")


if __name__ == "__main__":
    main()
