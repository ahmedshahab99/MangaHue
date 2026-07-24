from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mangahue_train.onnx_worker import _load_input, _pad_input


def test_five_channel_input_normalizes_and_masks_hints(tmp_path: Path) -> None:
    lineart = np.full((32, 32), 128, dtype=np.uint8)
    hints = np.zeros((32, 32, 3), dtype=np.uint8)
    hints[:, :, 0] = 255
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255

    lineart_path = tmp_path / "lineart.png"
    hints_path = tmp_path / "hints.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(lineart).save(lineart_path)
    Image.fromarray(hints).save(hints_path)
    Image.fromarray(mask).save(mask_path)

    tensor = _load_input(
        {
            "lineart": str(lineart_path),
            "hint_map": str(hints_path),
            "hint_mask": str(mask_path),
        },
        channels=5,
    )

    assert tensor.shape == (1, 5, 32, 32)
    np.testing.assert_allclose(tensor[0, 1:4, 12, 12], [1.0, -1.0, -1.0])
    np.testing.assert_allclose(tensor[0, 1:4, 0, 0], [0.0, 0.0, 0.0])
    assert tensor[0, 4, 12, 12] == 1.0
    assert tensor[0, 4, 0, 0] == 0.0


def test_input_padding_uses_white_lineart_and_empty_hints() -> None:
    tensor = np.zeros((1, 5, 33, 47), dtype=np.float32)
    tensor[:, 0, :, :] = 0.4

    padded, original = _pad_input(tensor, 32)

    assert original == (33, 47)
    assert padded.shape == (1, 5, 64, 64)
    assert np.all(padded[:, 0, 33:, :] == 1.0)
    assert np.all(padded[:, 0, :, 47:] == 1.0)
    assert np.all(padded[:, 1:, 33:, :] == 0.0)
    assert np.all(padded[:, 1:, :, 47:] == 0.0)
