from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mangahue_train.ddcolor_onnx_worker import _lab_to_rgb, _rgb_to_lab_l
from mangahue_train.onnx_worker import _load_input, _make_hints


def test_five_channel_input_consumes_user_hints(tmp_path: Path) -> None:
    lineart = np.full((32, 32), 255, dtype=np.uint8)
    hints = np.zeros((32, 32, 3), dtype=np.uint8)
    hints[:, :, 0] = 255
    hints[:, :, 2] = 64
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 200

    lineart_path = tmp_path / "lineart.png"
    hints_path = tmp_path / "hints.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(lineart).save(lineart_path)
    Image.fromarray(hints).save(hints_path)
    Image.fromarray(mask).save(mask_path)

    tensor, height, width, gray = _load_input(
        {
            "lineart": str(lineart_path),
            "hint_map": str(hints_path),
            "hint_mask": str(mask_path),
        },
        channels=5,
        spatial_multiple=32,
    )

    assert tensor.shape == (1, 5, 32, 32)
    assert (height, width) == (32, 32)
    # User hint RGB is passed through unmixed (0..1) and the user mask wins.
    np.testing.assert_allclose(tensor[0, 1:4, 12, 12], [1.0, 0.0, 64.0 / 255.0], atol=1e-3)
    np.testing.assert_allclose(tensor[0, 4, 12, 12], 200.0 / 255.0, atol=1e-3)
    # Outside the user mask the internal background hint is still present so
    # the model always receives guidance (paper => warm background hint).
    assert tensor[0, 4, 0, 0] > 0.0


def test_internal_hints_match_0_1_contract(tmp_path: Path) -> None:
    gray = np.full((32, 32), 128, dtype=np.uint8)
    gray_path = tmp_path / "gray.png"
    Image.fromarray(gray).save(gray_path)

    tensor, _, _, _ = _load_input(
        {"lineart": str(gray_path)},
        channels=5,
        spatial_multiple=32,
    )

    assert tensor.shape == (1, 5, 32, 32)
    # All hint channels stay within 0..1 (the metadata input_range contract).
    assert tensor[0, 1:].min() >= 0.0
    assert tensor[0, 1:].max() <= 1.0
    # Gray 128/255 falls in the clothing band, so a clothing hint must exist.
    assert tensor[0, 1:4, 16, 16].any() > 0.0
    assert tensor[0, 4, 16, 16] > 0.0


def test_make_hints_has_no_premultiplication() -> None:
    regions = {
        "ink": np.zeros((4, 4), dtype=bool),
        "bubble": np.zeros((4, 4), dtype=bool),
        "skin": np.ones((4, 4), dtype=bool),
        "hair": np.zeros((4, 4), dtype=bool),
        "clothing": np.zeros((4, 4), dtype=bool),
        "bg": np.zeros((4, 4), dtype=bool),
    }
    gray = np.full((4, 4), 0.8, dtype=np.float32)
    cmap, cmask = _make_hints(gray, regions)
    # Colors are stored at full strength; the mask is a separate channel.
    assert float(cmask[0, 0]) == pytest.approx(0.75)
    assert cmap[0, 0, 0] == pytest.approx(242.0 / 255.0, abs=1e-3)
    assert cmap[0, 0, 1] == pytest.approx(208.0 / 255.0, abs=1e-3)


def test_input_padding_uses_white_lineart_and_empty_hints() -> None:
    gray = np.full((33, 47), 102, dtype=np.uint8)
    gray_path = Path(__file__).parent / "padding_gray.png"
    gray_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray).save(gray_path)
    try:
        tensor, height, width, _ = _load_input(
            {"lineart": str(gray_path)},
            channels=5,
            spatial_multiple=32,
        )
    finally:
        gray_path.unlink(missing_ok=True)

    assert (height, width) == (33, 47)
    assert tensor.shape == (1, 5, 64, 64)
    assert np.all(tensor[:, 0, 33:, :] == 1.0)
    assert np.all(tensor[:, 0, :, 47:] == 1.0)
    assert np.all(tensor[:, 1:, 33:, :] == 0.0)
    assert np.all(tensor[:, 1:, :, 47:] == 0.0)


def test_ddcolor_lab_round_trip_preserves_neutral_ramp() -> None:
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    rgb = np.repeat(ramp[:, None], 3, axis=1).reshape(8, 8, 3)
    l_channel = _rgb_to_lab_l(rgb)
    lab = np.stack(
        (l_channel, np.zeros_like(l_channel), np.zeros_like(l_channel)),
        axis=-1,
    )
    restored = _lab_to_rgb(lab)
    assert np.max(np.abs(restored - rgb)) < 2e-4
