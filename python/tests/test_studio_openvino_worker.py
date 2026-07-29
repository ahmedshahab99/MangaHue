from __future__ import annotations

import numpy as np
from PIL import Image

from mangahue_train.studio_openvino_worker import (
    fuse_generated_chroma,
    prepare_canvas,
    restore_canvas,
)


def test_canvas_round_trip_preserves_page_shape() -> None:
    source = Image.new("RGB", (946, 1463), "white")
    canvas, transform = prepare_canvas(source, 576)

    assert canvas.size == (384, 576)
    assert canvas.width % 64 == 0
    assert canvas.height % 64 == 0
    left, top, right, bottom = transform.content_box
    assert 0 <= left < right <= canvas.width
    assert 0 <= top < bottom <= canvas.height
    assert restore_canvas(canvas, transform).size == source.size


def test_chroma_fusion_preserves_original_luminance() -> None:
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    source = Image.fromarray(np.stack((ramp, ramp, ramp), axis=-1), mode="RGB")
    generated = Image.new("RGB", source.size, (240, 40, 80))

    result = np.asarray(fuse_generated_chroma(source, generated).convert("YCbCr"))
    expected = np.asarray(source.convert("YCbCr"))
    assert np.max(np.abs(result[..., 0].astype(int) - expected[..., 0].astype(int))) <= 1


def test_chroma_fusion_keeps_blank_paper_neutral() -> None:
    source = Image.new("RGB", (96, 96), "white")
    generated = Image.new("RGB", source.size, (255, 0, 0))

    result = np.asarray(fuse_generated_chroma(source, generated), dtype=np.int16)
    channel_spread = result.max(axis=-1) - result.min(axis=-1)
    assert int(channel_spread.max()) <= 2
