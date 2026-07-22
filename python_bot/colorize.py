from __future__ import annotations

import asyncio
import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

_MODEL_ROOT = Path(__file__).resolve().parents[1] / "models" / "manga-onnx"


class ColorizeError(Exception):
    """Base error for the colorization module."""


class ConfigError(ColorizeError):
    """Raised when required configuration (e.g. API token) is missing."""


class BackendError(ColorizeError):
    """Raised when the colorization backend fails."""


def pad_to_multiple(image: Image.Image, multiple: int = 32) -> Image.Image:
    """Pad an image so both dimensions become a multiple of *multiple*.

    Padding is added to the right and bottom edges using reflection.
    """
    w, h = image.size
    pad_w = (multiple - w % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple
    if pad_w == 0 and pad_h == 0:
        return image
    return image.crop((0, 0, w + pad_w, h + pad_h))


def pil_to_tensor(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a NCHW float32 numpy array in [0, 1] range."""
    arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return arr[None, None, :, :]


def tensor_to_pil(tensor: np.ndarray) -> Image.Image:
    """Convert a NCHW numpy array in [0, 1] to an RGB PIL image."""
    if tensor.ndim == 4:
        arr = tensor[0]
    else:
        arr = tensor
    arr = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
    if arr.shape[0] == 1:
        return Image.fromarray(arr[0], mode="L")
    return Image.fromarray(arr.transpose(1, 2, 0), mode="RGB")


def generate_manga_lut(
    saturation: float = 0.85, contrast: float = 1.05
) -> np.ndarray:
    """Generate a 17x17x17 trilinear 3D LUT that gently desaturates and
    boosts contrast.

    Returns a ``(17, 17, 17, 3)`` float32 array in [0, 1].
    """
    levels = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    rr, gg, bb = np.meshgrid(levels, levels, levels, indexing="ij")
    lut = np.stack([rr, gg, bb], axis=-1).astype(np.float32)

    gray = 0.299 * rr + 0.587 * gg + 0.114 * bb

    desaturated = gray[..., None] * (1.0 - saturation) + lut * saturation

    mid = 0.5
    contrasted = (desaturated - mid) * contrast + mid
    return np.clip(contrasted, 0.0, 1.0).astype(np.float32)


def _apply_3d_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a 17x17x17 trilinear 3D LUT to an RGB image.

    *image* must be a float32 (H, W, 3) array in [0, 1].
    *lut* must be a ``(17, 17, 17, 3)`` float32 array in [0, 1].
    """
    scaled = np.clip(image * 16.0, 0.0, 16.0)
    f_r = scaled[:, :, 0]
    f_g = scaled[:, :, 1]
    f_b = scaled[:, :, 2]

    i_r0 = np.floor(f_r).astype(np.int32)
    i_g0 = np.floor(f_g).astype(np.int32)
    i_b0 = np.floor(f_b).astype(np.int32)
    i_r1 = np.clip(i_r0 + 1, 0, 16)
    i_g1 = np.clip(i_g0 + 1, 0, 16)
    i_b1 = np.clip(i_b0 + 1, 0, 16)

    dr = f_r - i_r0.astype(np.float32)
    dg = f_g - i_g0.astype(np.float32)
    db = f_b - i_b0.astype(np.float32)

    dr = dr[:, :, None]
    dg = dg[:, :, None]
    db = db[:, :, None]

    c000 = lut[i_r0, i_g0, i_b0]
    c001 = lut[i_r0, i_g0, i_b1]
    c010 = lut[i_r0, i_g1, i_b0]
    c011 = lut[i_r0, i_g1, i_b1]
    c100 = lut[i_r1, i_g0, i_b0]
    c101 = lut[i_r1, i_g0, i_b1]
    c110 = lut[i_r1, i_g1, i_b0]
    c111 = lut[i_r1, i_g1, i_b1]

    c00 = c000 * (1.0 - dr) + c100 * dr
    c01 = c001 * (1.0 - dr) + c101 * dr
    c10 = c010 * (1.0 - dr) + c110 * dr
    c11 = c011 * (1.0 - dr) + c111 * dr

    c0 = c00 * (1.0 - dg) + c10 * dg
    c1 = c01 * (1.0 - dg) + c11 * dg

    result = c0 * (1.0 - db) + c1 * db
    return np.clip(result, 0.0, 1.0)


def bilateral_smooth(
    image: np.ndarray,
    lineart: np.ndarray,
    spatial_sigma: float = 3.0,
    range_sigma: float = 35.0,
) -> np.ndarray:
    """Apply edge-aware bilateral smoothing with ink-barrier awareness.

    *image*: uint8 RGB (H, W, 3).
    *lineart*: uint8 grayscale (H, W) — used to detect ink barriers.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ColorizeError("opencv-python-headless is required for post-processing") from exc

    filtered = cv2.bilateralFilter(
        image, d=5, sigmaColor=range_sigma, sigmaSpace=spatial_sigma
    )

    ink = (255 - lineart).astype(np.float32) / 255.0
    ink = cv2.GaussianBlur(ink, (5, 5), 1.0)
    ink = ink[:, :, None]

    result = (filtered * (1.0 - ink) + image * ink).astype(np.uint8)
    return result


def restore_lines(
    image: np.ndarray,
    line_mask: np.ndarray,
    strength: float = 1.1,
) -> np.ndarray:
    """Darken ink lines in the colorized image proportional to *strength*.

    *image*: uint8 RGB (H, W, 3).
    *line_mask*: float32 (H, W) in [0, 1] — 1.0 = ink, 0.0 = paper.
    """
    factor = 1.0 - (strength - 1.0) * line_mask[:, :, None]
    factor = np.clip(factor, 0.0, 1.0)
    return (image.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def unsharp_mask(
    image: np.ndarray,
    amount: float = 0.06,
) -> np.ndarray:
    """Apply gentle unsharp masking.

    *image*: uint8 RGB (H, W, 3).
    """
    pil_img = Image.fromarray(image, mode="RGB")
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    blurred = np.asarray(blurred, dtype=np.float32)
    sharp = image.astype(np.float32) + amount * (
        image.astype(np.float32) - blurred
    )
    return np.clip(sharp, 0, 255).astype(np.uint8)


_MANGA_LUT: Optional[np.ndarray] = None


def _get_lut() -> np.ndarray:
    global _MANGA_LUT
    if _MANGA_LUT is None:
        _MANGA_LUT = generate_manga_lut(saturation=0.85, contrast=1.05)
    return _MANGA_LUT


def apply_postprocessing(
    color_image: np.ndarray,
    lineart: Image.Image,
) -> Image.Image:
    """Apply the full post-processing pipeline to a colorized image.

    Parameters
    ----------
    color_image : np.ndarray
        uint8 RGB (H, W, 3) colorized output.
    lineart : PIL.Image
        Grayscale lineart matching the color image dimensions.

    Returns
    -------
    PIL.Image
        Post-processed RGB image.
    """
    lineart_arr = np.asarray(lineart.convert("L"), dtype=np.uint8)
    color_f32 = color_image.astype(np.float32) / 255.0

    # 1. LUT color grading
    graded = _apply_3d_lut(color_f32, _get_lut())
    graded_u8 = (np.clip(graded, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    # 2. Edge-aware bilateral smoothness
    smoothed = bilateral_smooth(graded_u8, lineart_arr)

    # 3. Line restoration
    line_mask = (255 - lineart_arr).astype(np.float32) / 255.0
    restored = restore_lines(smoothed, line_mask, strength=1.1)

    # 4. Gentle unsharp mask
    sharpened = unsharp_mask(restored, amount=0.06)

    return Image.fromarray(sharpened, mode="RGB")


class Colorizer:
    """Full colorization pipeline supporting multiple backends.

    Parameters
    ----------
    backend : str
        One of ``"replicate"`` (DDColor via API), ``"local"`` (ONNX), or
        ``"auto"`` (try replicate, fall back to local).
    onnx_model_path : str or None
        Path to the manga-colorization ONNX model. Defaults to
        ``<project>/models/manga-onnx/manga-colorize-fp16.onnx``.
    onnx_device : str
        ONNX Runtime execution provider. Default ``"CPUExecutionProvider"``.
    """

    _DEFAULT_MODEL = _MODEL_ROOT / "manga-colorize-fp16.onnx"

    def __init__(
        self,
        backend: str = "replicate",
        onnx_model_path: Optional[str] = None,
        onnx_device: str = "CPUExecutionProvider",
    ) -> None:
        if backend not in ("replicate", "local", "auto"):
            raise ConfigError(
                f"Unknown backend {backend!r}. "
                f"Use 'replicate', 'local', or 'auto'."
            )
        self._backend = backend
        self._onnx_model_path = Path(onnx_model_path) if onnx_model_path else self._DEFAULT_MODEL
        self._onnx_device = onnx_device

    async def colorize(self, image_path: str, output_path: str) -> bool:
        """Colorize a grayscale manga page.

        Returns ``True`` on success.
        """
        resolved = self._backend
        if resolved == "auto":
            if os.environ.get("REPLICATE_API_TOKEN"):
                resolved = "replicate"
            else:
                logger.info("REPLICATE_API_TOKEN not set — falling back to local ONNX")
                resolved = "local"

        try:
            if resolved == "replicate":
                return await self._colorize_replicate(image_path, output_path)
            elif resolved == "local":
                return await self._colorize_onnx(image_path, output_path)
            else:
                raise ConfigError(f"Unresolved backend: {resolved!r}")
        except ColorizeError:
            raise
        except Exception as exc:
            raise BackendError(
                f"Colorization failed with backend {resolved!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Replicate DDColor backend
    # ------------------------------------------------------------------

    async def _colorize_replicate(self, image_path: str, output_path: str) -> bool:
        token = os.environ.get("REPLICATE_API_TOKEN")
        if not token:
            raise ConfigError(
                "REPLICATE_API_TOKEN environment variable is not set. "
                "Set it to use the Replicate DDColor backend."
            )

        try:
            import replicate
        except ImportError as exc:
            raise ConfigError(
                "The 'replicate' package is required for the replicate backend. "
                "Install it with: pip install replicate"
            ) from exc

        logger.info("Sending image to Replicate DDColor API …")

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: replicate.run(
                    "piddnad/ddcolor",
                    input={"image": open(image_path, "rb")},
                ),
            )
        except replicate.exceptions.ReplicateError as exc:
            raise BackendError(f"Replicate API error: {exc}") from exc

        result_url = self._extract_result_url(result)
        if not result_url:
            raise BackendError(
                f"Replicate returned unexpected output: {result!r}"
            )

        logger.info("Downloading colorized result …")
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        await loop.run_in_executor(
            None, lambda: self._download_file(result_url, str(out_path))
        )

        logger.info("Applying post-processing …")
        orig = Image.open(image_path)
        color_arr = np.asarray(Image.open(out_path).convert("RGB"), dtype=np.uint8)
        final = apply_postprocessing(color_arr, orig)
        final.save(out_path, format="PNG")

        return True

    @staticmethod
    def _extract_result_url(result) -> Optional[str]:
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
        else:
            item = result

        if isinstance(item, str):
            return item
        if hasattr(item, "url"):
            return str(item.url)
        return None

    @staticmethod
    def _download_file(url: str, dest: str) -> None:
        out_path = Path(dest)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(out_path))

    # ------------------------------------------------------------------
    # Local ONNX backend
    # ------------------------------------------------------------------

    async def _colorize_onnx(self, image_path: str, output_path: str) -> bool:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ConfigError(
                "onnxruntime is required for the local backend. "
                "Install it with: pip install onnxruntime"
            ) from exc

        if not self._onnx_model_path.is_file():
            raise ConfigError(
                f"ONNX model not found at {self._onnx_model_path}. "
                f"Place the model file there or pass onnx_model_path to Colorizer()."
            )

        logger.info("Loading ONNX model: %s", self._onnx_model_path)

        orig = Image.open(image_path)
        padded = pad_to_multiple(orig, multiple=32)

        tensor = pil_to_tensor(padded)
        _, _, h, w = tensor.shape

        hint = np.zeros((1, 4, h, w), dtype=np.float32)
        x = np.concatenate([tensor, hint], axis=1).astype(np.float32)

        session = ort.InferenceSession(
            str(self._onnx_model_path),
            providers=[self._onnx_device],
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        logger.info("Running ONNX inference (%dx%d) …", w, h)
        loop = asyncio.get_running_loop()
        y = await loop.run_in_executor(
            None, lambda: session.run([output_name], {input_name: x})[0]
        )

        color_image = tensor_to_pil(y)
        if color_image.size != orig.size:
            color_image = color_image.crop((0, 0, orig.size[0], orig.size[1]))

        logger.info("Applying post-processing …")
        final = apply_postprocessing(
            np.asarray(color_image.convert("RGB"), dtype=np.uint8),
            orig,
        )

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final.save(out_path, format="PNG")

        logger.info("Saved colorized image to %s", output_path)
        return True
