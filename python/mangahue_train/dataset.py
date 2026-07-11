from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SamplePaths:
    lineart: Path
    color: Path
    hints: Path | None = None


def _read_manifest(manifest: Path) -> list[SamplePaths]:
    value = json.loads(manifest.read_text(encoding="utf-8"))
    rows = value.get("samples", value) if isinstance(value, dict) else value
    root = manifest.parent
    samples: list[SamplePaths] = []
    for row in rows:
        samples.append(
            SamplePaths(
                lineart=root / row["lineart"],
                color=root / row["color"],
                hints=root / row["hints"] if row.get("hints") else None,
            )
        )
    return samples


def _sample_synthetic_hints(
    color: Image.Image,
    rng: random.Random,
    min_dots: int,
    max_dots: int,
    radius: float,
    size: int,
) -> tuple[list[dict[str, object]], Image.Image, Image.Image]:
    """Sample sparse dots from the color target as fake hint channels."""
    rgb = color.convert("RGB").resize((size, size))
    count = rng.randint(min_dots, max_dots)
    if count <= 0:
        return [], Image.new("RGB", (size, size), (0, 0, 0)), Image.new("L", (size, size), 0)

    hint_map = Image.new("RGB", (size, size), (0, 0, 0))
    hint_mask = Image.new("L", (size, size), 0)
    draw_color = ImageDraw.Draw(hint_map)
    draw_mask = ImageDraw.Draw(hint_mask)
    hints: list[dict[str, object]] = []
    w, h = rgb.size
    px = np.asarray(rgb, dtype=np.uint8)

    for _ in range(count):
        x_norm = rng.random()
        y_norm = rng.random()
        cx = min(w - 1, max(0, int(x_norm * w)))
        cy = min(h - 1, max(0, int(y_norm * h)))
        r, g, b = (int(v) for v in px[cy, cx])
        rr = int(radius * size)
        box = [cx - rr, cy - rr, cx + rr, cy + rr]
        draw_color.ellipse(box, fill=(r, g, b))
        draw_mask.ellipse(box, fill=255)
        hints.append(
            {
                "x": x_norm,
                "y": y_norm,
                "color": f"#{r:02x}{g:02x}{b:02x}",
                "radius": radius,
                "strength": 1.0,
            }
        )
    return hints, hint_map, hint_mask


class MangaHueDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset contract matching the Rust engine tensor format.

    Inputs are produced as ``[1+L+3+1, H, W]`` tensors:

    - channel 0: grayscale clean line art normalized to ``0..1``
    - channels 1-3: hint RGB normalized to ``0..1``
    - channel 4: hint mask normalized to ``0..1``

    Targets are ``[3, H, W]`` RGB tensors in ``0..1``.
    """

    def __init__(
        self,
        manifest: Path | Sequence[SamplePaths] | Sequence[str],
        size: int = 768,
        synthetic_hints: bool | dict[str, object] = False,
        seed: int = 1337,
        root: Path | None = None,
    ) -> None:
        if isinstance(manifest, Path):
            self.samples = _read_manifest(manifest)
            self.root = manifest.parent
        else:
            loaded: list[SamplePaths] = []
            for entry in manifest:
                if isinstance(entry, SamplePaths):
                    loaded.append(entry)
                else:
                    p = Path(entry)
                    loaded.append(SamplePaths(lineart=p, color=p))
            self.samples = loaded
            self.root = root or Path()
        self.size = size
        self.synthetic_hints = synthetic_hints
        self._rng_seed = seed

        if isinstance(synthetic_hints, bool):
            self._synth_cfg: dict[str, object] | None = (
                {"min_dots": 0, "max_dots": 8, "radius": 0.025, "prob": 1.0} if synthetic_hints else None
            )
        else:
            self._synth_cfg = dict(synthetic_hints)

    def __len__(self) -> int:
        return len(self.samples)

    def _rng_for(self, index: int) -> random.Random:
        return random.Random(self._rng_seed * 1000003 + index)

    def _build_inputs(
        self,
        index: int,
        lineart: Image.Image,
        color: Image.Image,
        source_size: tuple[int, int],
    ) -> torch.Tensor:
        rng = self._rng_for(index)
        hint_map = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        hint_mask = Image.new("L", (self.size, self.size), 0)

        sample = self.samples[index]
        if sample.hints and sample.hints.exists():
            self._draw_real_hints(sample.hints, hint_map, hint_mask, source_size)
        elif self._synth_cfg is not None and rng.random() < float(self._synth_cfg.get("prob", 1.0)):
            _hints, hint_map, hint_mask = _sample_synthetic_hints(
                color=color,
                rng=rng,
                min_dots=int(self._synth_cfg.get("min_dots", 0)),
                max_dots=int(self._synth_cfg.get("max_dots", 8)),
                radius=float(self._synth_cfg.get("radius", 0.025)),
                size=self.size,
            )

        x = np.concatenate(
            [
                np.asarray(lineart, dtype=np.float32)[None, ...] / 255.0,
                np.asarray(hint_map, dtype=np.float32).transpose(2, 0, 1) / 255.0,
                np.asarray(hint_mask, dtype=np.float32)[None, ...] / 255.0,
            ],
            axis=0,
        )
        return torch.from_numpy(x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        lineart_source = Image.open(sample.lineart).convert("L")
        source_size = lineart_source.size
        lineart = lineart_source.resize((self.size, self.size))
        color = Image.open(sample.color).convert("RGB").resize((self.size, self.size))
        x = self._build_inputs(index, lineart, color, source_size)
        y = np.asarray(color, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return x, torch.from_numpy(y)

    def _draw_real_hints(
        self,
        path: Path,
        hint_map: Image.Image,
        hint_mask: Image.Image,
        source_size: tuple[int, int],
    ) -> None:
        hints = json.loads(path.read_text(encoding="utf-8")).get("hints", [])
        draw_color = ImageDraw.Draw(hint_map)
        draw_mask = ImageDraw.Draw(hint_mask)
        for hint in hints:
            x = self._scale_coord(hint["x"], source_size[0])
            y = self._scale_coord(hint["y"], source_size[1])
            radius = self._scale_radius(hint.get("radius", 0.025), source_size)
            color = self._parse_color(hint["color"])
            box = [x - radius, y - radius, x + radius, y + radius]
            draw_color.ellipse(box, fill=color)
            draw_mask.ellipse(box, fill=int(255 * float(hint.get("strength", 1.0))))

    def _scale_coord(self, value: float | int, source_extent: int) -> int:
        raw = float(value)
        if 0.0 <= raw <= 1.0:
            return int(round(raw * self.size))
        return int(round(raw * self.size / max(1, source_extent)))

    def _scale_radius(self, value: float | int, source_size: tuple[int, int]) -> int:
        raw = float(value)
        if 0.0 < raw < 1.0:
            return max(1, int(round(raw * self.size)))
        scale = self.size / max(1, max(source_size))
        return max(1, int(round(raw * scale)))

    @staticmethod
    def _parse_color(value: str | list[int] | tuple[int, int, int] | dict[str, int]) -> tuple[int, int, int]:
        if isinstance(value, str):
            return ImageColor.getrgb(value)
        if isinstance(value, dict):
            return (int(value["r"]), int(value["g"]), int(value["b"]))
        return (int(value[0]), int(value[1]), int(value[2]))
