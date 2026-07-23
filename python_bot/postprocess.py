"""
Advanced manga post-processing pipeline — 11-stage professional colour grading.
All stages are algorithmic (no neural networks), designed for natural results
with screentone preservation, crisp ink lines, and harmonious colour.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ============================================================================
# Module-level LUT cache
# ============================================================================
_GRADING_LUT: Optional[np.ndarray] = None


# ============================================================================
# Internal helpers
# ============================================================================

def _cie_lum(rgb: np.ndarray) -> np.ndarray:
    """CIE 1931 relative luminance: 0.2126*R + 0.7152*G + 0.0722*B."""
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep  x²·(3 − 2x)  on [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ============================================================================
# Stage 1 — Advanced White Balance  (white-patch retinex)
# ============================================================================

def _white_balance(rgb: np.ndarray) -> np.ndarray:
    """Per-channel gain from top‑2 % brightest pixels, applied above median luminance."""
    lum = _cie_lum(rgb)
    thresh = np.percentile(lum, 98)
    brightest = rgb[lum >= thresh]
    if brightest.shape[0] < 10:
        return rgb
    gain = np.clip(
        255.0 / np.maximum(brightest.mean(axis=0), 1.0), 0.6, 1.8
    ).astype(np.float64)

    med = np.median(lum)
    ref = max(np.percentile(lum, 80) - med, 1.0)
    ramp = np.clip((lum - med) / ref, 0.0, 1.0) ** 0.5  # smooth transition
    final = 1.0 + (gain[None, None, :] - 1.0) * ramp[..., None]
    return np.clip(rgb * final, 0.0, 255.0)


# ============================================================================
# Stage 3 — Multi-Scale FFT Screentone Detection
# ============================================================================

def _detect_screentones(gray: np.ndarray) -> np.ndarray:
    """FFT-based screentone detector — 3 scales (16, 32, 64) with 50 % overlap.

    Returns float64 mask in [0, 1] where 1 = screentone.
    """
    H, W = gray.shape
    scales = [16, 32, 64]
    mask = np.zeros((H, W), dtype=np.float64)
    gray_f = gray.astype(np.float64)

    for bs in scales:
        step = max(bs // 2, 1)
        wy = np.hanning(bs)[:, None]
        wx = np.hanning(bs)[None, :]
        win2d = wy * wx
        sm = np.zeros((H, W), dtype=np.float64)

        for y0 in range(0, H - bs + 1, step):
            for x0 in range(0, W - bs + 1, step):
                block = gray_f[y0 : y0 + bs, x0 : x0 + bs]
                windowed = (block - block.mean()) * win2d
                fshift = np.fft.fftshift(np.fft.fft2(windowed))
                mag = np.abs(fshift)
                cy, cx = bs // 2, bs // 2
                dc = mag[cy, cx]
                if dc < 1e-8:
                    continue
                ac = mag.copy()
                ac[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0.0
                if ac.max() / dc > 6.0:
                    sm[y0 : y0 + bs, x0 : x0 + bs] = 1.0

        mask = np.maximum(mask, sm)  # OR across scales

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask.astype(np.float32), cv2.MORPH_CLOSE, kernel)
    return np.clip(mask, 0.0, 1.0).astype(np.float64)


# ============================================================================
# Stage 5 — Perceptual Colour Grading LUT  (17³)
# ============================================================================

def _build_grading_lut() -> np.ndarray:
    """17³ 3D LUT: S-curve contrast + shadow lift + highlight rolloff + midtone warmth."""
    global _GRADING_LUT
    if _GRADING_LUT is not None:
        return _GRADING_LUT

    N = 17
    vals = np.linspace(0.0, 255.0, N, dtype=np.float64)
    R, G, B = np.meshgrid(vals, vals, vals, indexing="ij")
    rgb = np.stack([R, G, B], axis=-1)  # (17,17,17,3)

    lp = (0.2126 * R + 0.7152 * G + 0.0722 * B) / 255.0  # norm. lum [0,1]

    # (a) Gentle S-curve contrast
    t = rgb / 255.0
    rgb = (t + 0.07 * (_smoothstep(t) - t)) * 255.0

    # (b) Shadow lift — +3 % in bottom 15 % luminance
    shadow = np.maximum(0.0, 1.0 - lp / 0.15)
    rgb += shadow[..., None] * 0.03 * 255.0

    # (c) Highlight rolloff — soft clip above 92 % luminance
    hl = np.maximum(0.0, (lp - 0.92) / 0.08)
    rgb = rgb * (1.0 - hl[..., None] * 0.12) + 255.0 * hl[..., None] * 0.12

    # (d) Midtone warmth — subtle +2 red in 40–60 % luminance
    mt = np.maximum(0.0, 1.0 - np.abs(lp - 0.50) / 0.10)
    rgb[..., 0] += mt * 2.0

    _GRADING_LUT = np.clip(rgb, 0.0, 255.0).astype(np.float32)
    return _GRADING_LUT


def _apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinear interpolation through a 17³ 3D LUT."""
    H, W = image.shape[:2]
    scaled = image.astype(np.float64) / (256.0 / 16.0)  # → [0, 16]
    i0 = np.clip(np.floor(scaled).astype(np.int32), 0, 15)
    i1 = np.clip(i0 + 1, 0, 16)
    f = scaled - i0.astype(np.float64)

    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]

    wR0, wR1 = 1.0 - fr, fr
    wG0, wG1 = 1.0 - fg, fg
    wB0, wB1 = 1.0 - fb, fb

    r = np.zeros((H, W, 3), dtype=np.float64)
    r += lut[r0, g0, b0] * wR0 * wG0 * wB0
    r += lut[r0, g0, b1] * wR0 * wG0 * wB1
    r += lut[r0, g1, b0] * wR0 * wG1 * wB0
    r += lut[r0, g1, b1] * wR0 * wG1 * wB1
    r += lut[r1, g0, b0] * wR1 * wG0 * wB0
    r += lut[r1, g0, b1] * wR1 * wG0 * wB1
    r += lut[r1, g1, b0] * wR1 * wG1 * wB0
    r += lut[r1, g1, b1] * wR1 * wG1 * wB1
    return np.clip(r, 0.0, 255.0)


# ============================================================================
# Stage 6 — Edge-Preserving Bilateral Smooth
# ============================================================================

def _bilateral_smooth(rgb: np.ndarray) -> np.ndarray:
    """Domain-transform smooth + 40 % original blend to retain texture."""
    u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    try:
        smoothed = cv2.edgePreservingFilter(
            u8, flags=cv2.RECURS_FILTER, sigma_s=25.0, sigma_r=20.0 / 255.0
        )
    except Exception:
        smoothed = u8
    return 0.6 * smoothed.astype(np.float64) + 0.4 * rgb


# ============================================================================
# Stage 8 — Advanced Line Restoration
# ============================================================================

def _restore_lines(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Adaptive-threshold line extraction with non-linear proximity darkening."""
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    lines = cv2.adaptiveThreshold(
        gray_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))

    # Distance transform → proximity weight  (1.0 on line, 0.0 ≥ 2 px away)
    dist = cv2.distanceTransform(255 - lines, cv2.DIST_L2, 5).astype(np.float64)
    prox = np.clip(1.0 - dist / 2.0, 0.0, 1.0)

    darkness = (255.0 - gray) / 255.0
    strength = np.power(darkness, 1.6)
    factor = 1.0 - 0.12 * strength * prox
    return np.clip(rgb * factor[..., None], 0.0, 255.0)


# ============================================================================
# Stage 9 — Local Contrast Enhancement  (CLAHE)
# ============================================================================

def _clahe_enhance(rgb: np.ndarray) -> np.ndarray:
    """CLAHE on L* channel of LAB, blended 30 % back."""
    u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    rgb_clahe = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB).astype(np.float64)
    return 0.3 * rgb_clahe + 0.7 * rgb


# ============================================================================
# Main entry point — Apply full 11-stage pipeline
# ============================================================================

def apply_manga_pipeline(
    colored_path: str,
    original_path: str,
    output_path: str,
) -> None:
    """11-stage professional manga post-processing pipeline.

    Parameters
    ----------
    colored_path : str
        Path to the colorized manga page (RGB).
    original_path : str
        Path to the original grayscale page (lineart / screentones).
    output_path : str
        Where to save the processed PNG.
    """
    # ---- Load ----------------------------------------------------------------
    orig_gray = np.array(Image.open(original_path).convert("L"), dtype=np.float64)
    rgb = np.array(Image.open(colored_path).convert("RGB"), dtype=np.float64)

    # ---- Stage 1: Advanced White Balance ------------------------------------
    rgb = _white_balance(rgb)

    # ---- Stage 2: Perceptual Luminance Map ----------------------------------
    lum = _cie_lum(rgb)

    # ---- Stage 3: Multi-Scale Screentone Detection (FFT) --------------------
    tone_mask = _detect_screentones(orig_gray)

    # ---- Stage 4: Content-Adaptive Desaturation -----------------------------
    # Local variance in 11×11 window on original grey (content texture)
    gray_f32 = orig_gray.astype(np.float32)
    mean = cv2.blur(gray_f32, (11, 11)).astype(np.float64)
    mean_sq = cv2.blur(gray_f32 * gray_f32, (11, 11)).astype(np.float64)
    local_var = np.maximum(mean_sq - mean * mean, 0.0)
    desat = 0.78 + 0.22 * np.clip(local_var / 800.0, 0.0, 1.0)
    desat = np.maximum(desat, tone_mask * 0.92)  # preserve screentones
    rgb = lum[..., None] + desat[..., None] * (rgb - lum[..., None])

    # ---- Stage 5: Perceptual Colour Grading (17³ LUT) -----------------------
    rgb = _apply_lut(rgb, _build_grading_lut())

    # ---- Stage 6: Edge-Preserving Bilateral Smooth --------------------------
    rgb = _bilateral_smooth(rgb)

    # ---- Stage 7: Screentone Preservation -----------------------------------
    gray_3ch = np.stack([orig_gray] * 3, axis=-1)
    alpha = tone_mask[..., None] * 0.4
    rgb = rgb * (1.0 - alpha) + gray_3ch * alpha

    # ---- Stage 8: Advanced Line Restoration ---------------------------------
    rgb = _restore_lines(rgb, orig_gray)

    # ---- Stage 9: Local Contrast Enhancement (CLAHE) ------------------------
    rgb = _clahe_enhance(rgb)

    # ---- Stage 10: Colour Harmonization -------------------------------------
    means = rgb.mean(axis=(0, 1))
    rb = means[0] / (means[2] + 1e-6)  # R/B warmth ratio
    if rb < 0.95:       # too cool  → warm shift
        rgb[..., 0] = np.clip(rgb[..., 0] * 1.01, 0, 255)
        rgb[..., 2] = np.clip(rgb[..., 2] * 0.995, 0, 255)
    elif rb > 1.05:     # too warm  → cool shift
        rgb[..., 0] = np.clip(rgb[..., 0] * 0.99, 0, 255)
        rgb[..., 2] = np.clip(rgb[..., 2] * 1.005, 0, 255)

    # ---- Stage 11: Output Sharpening ----------------------------------------
    blurred = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 1.2)
    rgb = rgb + 0.45 * (rgb - blurred.astype(np.float64))
    rgb = np.clip(rgb, 0.0, 255.0)

    # ---- Save ----------------------------------------------------------------
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(output_path)
