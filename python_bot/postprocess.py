"""
Advanced algorithmic post-processing module for colorized manga images.

Produces natural, professional, manga-authentic results using pure algorithms
(no neural networks).  Provides a full pipeline: white balance, LUT colour
grading, screentone preservation, edge‑preserving smoothing, adaptive line
restoration, and smart sharpening.

All functions have typed signatures, docstrings, and sensible fallback paths
so that **nothing** hard-depends on OpenCV.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Optional OpenCV – imported once; everything else works without it
# ---------------------------------------------------------------------------
try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ---------------------------------------------------------------------------
# Module-level LUT cache (compute once per parameter set)
# ---------------------------------------------------------------------------
_MANGALUT_CACHE: dict[Tuple[float, float, float], np.ndarray] = {}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gray_world_gain(image: np.ndarray) -> np.ndarray:
    """Return per-channel gain that makes the average colour neutral."""
    means = image.astype(np.float64).mean(axis=(0, 1))  # (3,)
    gray = means.mean()
    # Avoid division by zero for degenerate channels
    gain = np.where(means > 1e-6, gray / means, 1.0)
    gain = np.clip(gain, 0.5, 2.0)
    return gain.astype(np.float32)


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Standard ITU-R BT.601 luminance."""
    return (
        0.299 * image[..., 0].astype(np.float64)
        + 0.587 * image[..., 1].astype(np.float64)
        + 0.114 * image[..., 2].astype(np.float64)
    )


def _box_blur_gray(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    """Fast box-blur on a 2-D float array (vectorised integral-image).

    Falls back to an O(1)-per-pixel integral-image implementation when
    OpenCV is unavailable.
    """
    if _HAS_CV2 and kernel_size > 1:
        # cv2.blur expects uint8/float32 and a (kw,kh) tuple
        ksize = (kernel_size, kernel_size)
        return cv2.blur(arr.astype(np.float32), ksize)

    H, W = arr.shape
    r = kernel_size // 2
    if r < 1:
        return arr.astype(np.float64)

    # Build integral image: intg[i+1, j+1] = sum of arr[0:i+1, 0:j+1]
    padded = np.pad(arr.astype(np.float64), ((1, 0), (1, 0)), mode="constant")
    intg = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    y_idx = np.arange(H)
    x_idx = np.arange(W)
    y0 = np.maximum(0, y_idx - r)[:, np.newaxis]
    x0 = np.maximum(0, x_idx - r)[np.newaxis, :]
    y1 = (np.minimum(H, y_idx + r + 1))[:, np.newaxis]
    x1 = (np.minimum(W, x_idx + r + 1))[np.newaxis, :]

    area = (y1 - y0) * (x1 - x0)
    summed = intg[y1, x1] - intg[y0, x1] - intg[y1, x0] + intg[y0, x0]
    return summed / area


def _box_blur_rgb(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Per-channel box blur for an (H,W,3) uint8 image."""
    out = np.empty_like(image, dtype=np.float64)
    for c in range(3):
        out[..., c] = _box_blur_gray(image[..., c].astype(np.float64), kernel_size)
    return out


def _box_blur_gray_uint8(arr_uint8: np.ndarray, kernel_size: int) -> np.ndarray:
    """Box blur returning a float64 array but accepting uint8 input."""
    return _box_blur_gray(arr_uint8.astype(np.float64), kernel_size)


# ---------------------------------------------------------------------------
# 1.  Manga colour LUT
# ---------------------------------------------------------------------------

_LUT_STEPS = 17
_LUT_STEP_SIZE = 256.0 / (_LUT_STEPS - 1)


def generate_manga_lut(
    saturation: float = 0.85,
    contrast: float = 1.05,
    brightness: float = 0.0,
) -> np.ndarray:
    """Build a :math:`17^3` trilinear 3-D LUT for manga grading.

    Parameters
    ----------
    saturation : float
        How much saturation to keep (1.0 = no change, 0.0 = full greyscale).
    contrast : float
        Gentle S‑curve multiplier centred on 128.
    brightness : float
        Uniform brightness offset in [‑255, 255].

    Returns
    -------
    lut : np.ndarray  shape (17, 17, 17, 3)  dtype float32
        Look‑up table.  Apply with :func:`_apply_lut_to_image`.
    """
    key = (saturation, contrast, brightness)
    if key in _MANGALUT_CACHE:
        return _MANGALUT_CACHE[key]

    # Sample positions [0, step, 2*step, ..., 255]
    samples = np.linspace(0.0, 255.0, _LUT_STEPS, dtype=np.float32)

    R, G, B = np.meshgrid(samples, samples, samples, indexing="ij")  # noqa: N806
    rgb = np.stack([R, G, B], axis=-1).astype(np.float32)  # (17,17,17,3)

    # --- desaturation --------------------------------------------------------
    gray = (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    )[..., np.newaxis]
    rgb = gray + saturation * (rgb - gray)

    # --- soft contrast around 128 --------------------------------------------
    rgb = 128.0 + (rgb - 128.0) * contrast

    # --- brightness -----------------------------------------------------------
    rgb = rgb + brightness

    # Clamp & cast
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.float32)

    _MANGALUT_CACHE[key] = rgb
    return rgb


def _apply_lut_to_image(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a 17³ LUT with trilinear interpolation.

    Parameters
    ----------
    image : np.ndarray  (H,W,3)  uint8
    lut : np.ndarray    (17,17,17,3)  float32

    Returns
    -------
    np.ndarray  (H,W,3)  uint8
    """
    H, W = image.shape[:2]  # noqa: N806
    scaled = image.astype(np.float32) / _LUT_STEP_SIZE  # [0, 16]

    idx0 = np.clip(np.floor(scaled).astype(np.int32), 0, _LUT_STEPS - 2)
    idx1 = np.clip(idx0 + 1, 0, _LUT_STEPS - 1)
    frac = scaled - idx0.astype(np.float32)

    r0, g0, b0 = idx0[..., 0], idx0[..., 1], idx0[..., 2]
    r1, g1, b1 = idx1[..., 0], idx1[..., 1], idx1[..., 2]

    fr = frac[..., 0:1]  # (H,W,1)
    fg = frac[..., 1:2]
    fb = frac[..., 2:3]

    wr0, wr1 = 1.0 - fr, fr
    wg0, wg1 = 1.0 - fg, fg
    wb0, wb1 = 1.0 - fb, fb

    result = np.zeros((H, W, 3), dtype=np.float32)
    result += lut[r0, g0, b0] * wr0 * wg0 * wb0
    result += lut[r0, g0, b1] * wr0 * wg0 * wb1
    result += lut[r0, g1, b0] * wr0 * wg1 * wb0
    result += lut[r0, g1, b1] * wr0 * wg1 * wb1
    result += lut[r1, g0, b0] * wr1 * wg0 * wb0
    result += lut[r1, g0, b1] * wr1 * wg0 * wb1
    result += lut[r1, g1, b0] * wr1 * wg1 * wb0
    result += lut[r1, g1, b1] * wr1 * wg1 * wb1

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2.  Edge‑preserving smooth
# ---------------------------------------------------------------------------


def edge_preserving_smooth(
    image: np.ndarray,
    strength: float = 3.0,
    spatial_sigma: float = 25.0,
) -> np.ndarray:
    """Smooth colour regions while keeping hard ink edges crisp.

    Uses OpenCV's domain‑transform recursive filter when available;
    otherwise falls back to a fast separable bilateral approximation.

    Parameters
    ----------
    image : np.ndarray  (H,W,3)  uint8
    strength : float
        Range sigma – how different colours must be to count as an edge.
    spatial_sigma : float
        Spatial sigma – radius of the smoothing neighbourhood.

    Returns
    -------
    np.ndarray  (H,W,3)  uint8
    """
    if _HAS_CV2:
        # OpenCV path: fast recursive / domain-transform filter
        dst = image.copy()
        sigma_s = max(1.0, spatial_sigma)
        sigma_r = max(0.01, strength / 255.0)
        return cv2.edgePreservingFilter(
            dst, flags=cv2.RECURS_FILTER, sigma_s=sigma_s, sigma_r=sigma_r
        )

    # --- Manual separable bilateral fallback ---------------------------------
    H, W = image.shape[:2]  # noqa: N806
    radius = min(int(np.ceil(3.0 * min(spatial_sigma, 10.0))), 10)
    sigma_s_eff = max(radius / 3.0, 0.5)

    # Precompute 1-D spatial Gaussian
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    spatial_weight = np.exp(-0.5 * (xs / sigma_s_eff) ** 2).astype(np.float32)

    img_f = image.astype(np.float32)
    result = np.empty_like(img_f)

    # --- horizontal pass ---
    tmp = np.empty_like(img_f)
    for y in range(H):
        row = img_f[y]  # (W, 3)
        for x in range(W):
            x0 = max(0, x - radius)
            x1 = min(W, x + radius + 1)
            win = row[x0:x1]
            k = x1 - x0
            sw = spatial_weight[radius - (x - x0) : radius - (x - x0) + k]
            center = row[x]
            diff_sq = np.sum((win - center) ** 2, axis=1)
            rw = np.exp(-0.5 * diff_sq / max(strength * strength, 1e-6))
            w = sw * rw
            w_sum = w.sum()
            if w_sum > 1e-8:
                tmp[y, x] = np.sum(win * w[:, np.newaxis], axis=0) / w_sum
            else:
                tmp[y, x] = center

    # --- vertical pass ---
    for x in range(W):
        col = tmp[:, x]  # (H, 3)
        for y in range(H):
            y0 = max(0, y - radius)
            y1 = min(H, y + radius + 1)
            win = col[y0:y1]
            k = y1 - y0
            sw = spatial_weight[radius - (y - y0) : radius - (y - y0) + k]
            center = col[y]
            diff_sq = np.sum((win - center) ** 2, axis=1)
            rw = np.exp(-0.5 * diff_sq / max(strength * strength, 1e-6))
            w = sw * rw
            w_sum = w.sum()
            if w_sum > 1e-8:
                result[y, x] = np.sum(win * w[:, np.newaxis], axis=0) / w_sum
            else:
                result[y, x] = center

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3.  Restore ink lines adaptively
# ---------------------------------------------------------------------------


def restore_lines_adaptive(
    image: np.ndarray,
    line_mask: np.ndarray,
    strength: float = 1.1,
) -> np.ndarray:
    """Darken ink pixels using a power‑law curve that keeps heavy ink
    near black while leaving light strokes subtle.

    Parameters
    ----------
    image : np.ndarray  (H,W,3)  uint8
        Colour image to modify in-place (a copy is returned).
    line_mask : np.ndarray  (H,W)  float [0, 1]
        Per‑pixel line confidence (1.0 = pure ink, 0.0 = no line).
    strength : float
        Multiplier for the darkening effect (> 1.0 = darker lines).

    Returns
    -------
    np.ndarray  (H,W,3)  uint8
    """
    result = image.astype(np.float64)

    # Convert to luminance to measure original "ink darkness"
    lum = _rgb_to_gray(result) / 255.0  # [0, 1]

    # Curve: darken proportionally to how dark the pixel already is.
    # ink^1.8 means an ink value of 0.3 → 0.3^1.8 ≈ 0.115 (much darker)
    #          an ink value of 0.8 → 0.8^1.8 ≈ 0.67  (moderate darkening)
    darkened = np.power(np.clip(lum, 0.0, 1.0), 1.8)

    # Blend: factor = line_mask * strength → how much to push toward the
    # darkened version.  Clamp the blend factor so we never over-darken.
    blend = np.clip(line_mask * strength, 0.0, 1.0)[..., np.newaxis]

    # Apply darkening: multiply each channel by the ratio darkened/lum
    safe_lum = np.maximum(lum, 0.003)  # avoid div-by-zero
    factor = 1.0 - blend + blend * (darkened / safe_lum)[..., np.newaxis]
    result = result * factor

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4.  Preserve screentones
# ---------------------------------------------------------------------------


def preserve_screentones(
    coloured: np.ndarray,
    original_gray: np.ndarray,
    strength: float = 0.7,
    variance_window: int = 9,
    threshold_percentile: float = 85.0,
) -> np.ndarray:
    """Detect screentone regions via local variance and blend the original
    greyscale back in so dot‑shading patterns remain visible.

    Parameters
    ----------
    coloured : np.ndarray  (H,W,3)  uint8
    original_gray : np.ndarray  (H,W)  uint8
        The original greyscale (luminance of the colourised input or
        true manga greyscale).
    strength : float
        Blend strength for screentone areas (0 = keep colours, 1 = full grey).
    variance_window : int
        Odd window size for local-variance computation (default 9).
    threshold_percentile : float
        Percentile of variance above which we consider the pixel a screentone.

    Returns
    -------
    np.ndarray  (H,W,3)  uint8
    """
    gray_f = original_gray.astype(np.float64)
    gray_sq = gray_f * gray_f

    mean_gray = _box_blur_gray(gray_f, variance_window)
    mean_sq = _box_blur_gray(gray_sq, variance_window)
    local_var = mean_sq - mean_gray * mean_gray
    local_var = np.maximum(local_var, 0.0)

    # Screentones have high local variance → build a soft mask.
    # Normalise variance by local mean so dark regions aren't penalised.
    safe_mean = np.maximum(mean_gray, 1.0)
    var_norm = local_var / safe_mean

    # Adaptive threshold: everything above the percentile is likely screentone.
    thresh = np.percentile(var_norm, threshold_percentile)
    if thresh < 1e-6:
        return coloured  # degenerate – image is completely flat

    # Soft ramp: [0, thresh] → 0, [thresh, 2*thresh] → 0→1, >2*thresh → 1
    screentone_mask = np.clip((var_norm - thresh) / max(thresh, 1e-6), 0.0, 1.0)

    # Blend original grey into screentone regions
    blend = (screentone_mask * strength)[..., np.newaxis]
    gray_3ch = np.stack([gray_f] * 3, axis=-1)

    result = coloured.astype(np.float64) * (1.0 - blend) + gray_3ch * blend
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5.  Smart sharpen
# ---------------------------------------------------------------------------


def smart_sharpen(image: np.ndarray, amount: float = 0.06) -> np.ndarray:
    """Gentle unsharp mask – enough to restore micro‑contrast without
    making manga look digital or over‑processed.

    Parameters
    ----------
    image : np.ndarray  (H,W,3)  uint8
    amount : float
        Sharpening strength (0.0 = no change, 0.1 = noticeable).

    Returns
    -------
    np.ndarray  (H,W,3)  uint8
    """
    if abs(amount) < 1e-6:
        return image

    img_f = image.astype(np.float32)

    if _HAS_CV2:
        blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=1.0, sigmaY=1.0)
    else:
        # Fallback: small box blur (close enough for radius-1 sharpening)
        kernel_size = 3
        blurred = np.empty_like(img_f)
        for c in range(3):
            blurred[..., c] = _box_blur_gray(img_f[..., c], kernel_size)

    detail = img_f - blurred
    result = img_f + amount * detail
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 6.  Auto white‑balance
# ---------------------------------------------------------------------------


def auto_white_balance(image: np.ndarray) -> np.ndarray:
    """Simple grey‑world white balance.

    Corrects mild colour casts by scaling each channel so its mean
    equals the global mean.  Gains are clamped to [0.5, 2.0] to
    prevent extreme correction.

    Parameters
    ----------
    image : np.ndarray  (H,W,3)  uint8

    Returns
    -------
    np.ndarray  (H,W,3)  uint8
    """
    gain = _gray_world_gain(image)
    result = np.clip(image.astype(np.float64) * gain[np.newaxis, np.newaxis, :], 0, 255)
    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# 7.  Main pipeline entry point
# ---------------------------------------------------------------------------


def extract_line_mask(grayscale: np.ndarray, window: int = 31, threshold: float = 0.82) -> np.ndarray:
    """Adaptive line mask – keeps ink lines dark while ignoring shading.

    Pixels significantly darker than their local mean are classified as lines.

    Parameters
    ----------
    grayscale : np.ndarray  (H,W)  uint8
    window : int
        Box‑filter window size for local‑mean computation.
    threshold : float
        A pixel is a line if `value / local_mean < threshold`.

    Returns
    -------
    line_mask : np.ndarray  (H,W)  float64  [0, 1]
    """
    gray_f = grayscale.astype(np.float64)
    local_mean = _box_blur_gray(gray_f, window)
    safe_mean = np.maximum(local_mean, 1.0)
    ratio = gray_f / safe_mean

    # Soft mask:  darker pixels → higher mask value
    mask = 1.0 - np.clip(ratio / threshold, 0.0, 1.0)
    # Square it to sharpen the falloff
    return mask * mask


def apply_manga_pipeline(
    coloured_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    *,
    saturation: float = 0.85,
    contrast: float = 1.05,
    brightness: float = 0.0,
    line_strength: float = 1.1,
    screentone_strength: float = 0.7,
    sharpen_amount: float = 0.06,
    smooth_strength: float = 3.0,
    smooth_sigma: float = 25.0,
) -> None:
    """Full post‑processing pipeline for a colourised manga page.

    Loads the colour image, derives a greyscale version from it, then
    applies (in order): white‑balance → LUT grading → screentone
    preservation → edge‑preserving smooth → line restoration → sharpen.

    Parameters
    ----------
    coloured_path : path-like
        Path to the colourised manga image (RGB).
    output_path : path-like
        Where to save the processed result.
    saturation : float
        LUT saturation parameter.
    contrast : float
        LUT contrast parameter.
    brightness : float
        LUT brightness parameter.
    line_strength : float
        Strength of ink‑line restoration.
    screentone_strength : float
        Blend factor for screentone regions.
    sharpen_amount : float
        Unsharp‑mask intensity.
    smooth_strength : float
        Edge‑preserving smooth range sigma.
    smooth_sigma : float
        Edge‑preserving smooth spatial sigma.
    """
    # ---- load ---------------------------------------------------------------
    pil_img = Image.open(coloured_path).convert("RGB")
    coloured = np.asarray(pil_img, dtype=np.uint8)
    original_gray = np.asarray(pil_img.convert("L"), dtype=np.uint8)

    # ---- extract masks from the luminance -----------------------------------
    line_mask = extract_line_mask(original_gray)
    # screentone mask is computed inside preserve_screentones()

    # ---- run pipeline -------------------------------------------------------
    result = coloured

    # 1. White balance
    result = auto_white_balance(result)

    # 2. LUT colour grading
    lut = generate_manga_lut(saturation, contrast, brightness)
    result = _apply_lut_to_image(result, lut)

    # 3. Screentone preservation (blend original grey back into dot areas)
    result = preserve_screentones(result, original_gray, strength=screentone_strength)

    # 4. Edge‑preserving smooth (paint colours, keep ink)
    result = edge_preserving_smooth(
        result, strength=smooth_strength, spatial_sigma=smooth_sigma
    )

    # 5. Restore dark ink lines
    result = restore_lines_adaptive(result, line_mask, strength=line_strength)

    # 6. Gentle sharpen
    result = smart_sharpen(result, amount=sharpen_amount)

    # ---- save ---------------------------------------------------------------
    out = Image.fromarray(result, mode="RGB")
    out.save(output_path)


# ---------------------------------------------------------------------------
# Convenience: process a folder of images
# ---------------------------------------------------------------------------


def process_folder(
    input_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    **pipeline_kwargs,
) -> list[str]:
    """Run :func:`apply_manga_pipeline` on every supported image in *input_dir*.

    Parameters
    ----------
    input_dir : path-like
    output_dir : path-like
        Will be created if missing.
    extensions : tuple[str]
        File extensions to consider (case‑insensitive).
    **pipeline_kwargs
        Forwarded to :func:`apply_manga_pipeline`.

    Returns
    -------
    list[str]
        Paths of the saved files.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: list[str] = []
    exts = tuple(e.lower() for e in extensions)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(exts):
            continue
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        apply_manga_pipeline(in_path, out_path, **pipeline_kwargs)
        saved.append(out_path)

    return saved
