"""
SmartHintGenerator — produces world-class color hints for manga colorization.

The ONNX model expects 5-channel input: [grayscale, hint_R, hint_G, hint_B, hint_mask]
(shape [1, 5, H, W]).  Better hints → dramatically better output.

Algorithm (5 phases):
  1. Region segmentation  – Otsu binarization → dilate ink → connected components
  2. Region classification – decision-rule classifier (hair / skin / clothing / bg / accent)
  3. Natural color assignment – spatial-aware palette cycling
  4. Dense hint placement – 2-5 weighted hints per region via farthest-point sampling
  5. Hint map construction – Gaussian smoothing, pad-to-32, float32 output
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Palettes (manga-appropriate, natural tones)
# ---------------------------------------------------------------------------
_SKIN_TONES = np.array(
    [[255, 220, 190], [245, 205, 175], [230, 180, 155], [240, 200, 170]],
    dtype=np.float32,
)
_HAIR_COLORS = np.array(
    [[55, 40, 35], [45, 30, 25], [180, 140, 100], [85, 55, 40], [150, 120, 90]],
    dtype=np.float32,
)
_CLOTHING = np.array(
    [
        [180, 50, 50], [50, 80, 160], [60, 50, 85], [45, 85, 65],
        [200, 180, 160], [80, 60, 50], [100, 130, 170], [160, 140, 130],
    ],
    dtype=np.float32,
)
_BACKGROUND = np.array(
    [[252, 248, 240], [248, 244, 232], [240, 235, 225]], dtype=np.float32
)
_ACCENT = np.array(
    [[210, 65, 55], [50, 120, 180], [170, 140, 60], [150, 110, 130]],
    dtype=np.float32,
)


class SmartHintGenerator:
    """Generate dense, spatially-aware colour hints for a manga page.

    Typical usage::

        gen = SmartHintGenerator()
        hint_map, hint_mask = gen.generate(grayscale_uint8)

        # Convert to ONNX input tensor
        hint_tensor = hints_to_model_tensor(hint_map, hint_mask)
        # shape (1, 4, H_pad, W_pad)
    """

    def __init__(self) -> None:
        self._skin_idx = 0
        self._hair_idx = 0
        self._clothing_idx = 0
        self._bg_idx = 0
        self._accent_idx = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, image_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Produce hint maps from a grayscale manga page.

        Parameters
        ----------
        image_array : np.ndarray
            Grayscale image ``(H, W)``, dtype uint8, range [0, 255].

        Returns
        -------
        hint_map : np.ndarray
            ``(H_pad, W_pad, 3)`` uint8 RGB hints.
        hint_mask : np.ndarray
            ``(H_pad, W_pad)`` uint8 mask (255 = hint, 0 = no hint).
        """
        if image_array.ndim != 2:
            raise ValueError("Expected 2-D grayscale image")

        self._reset_palettes()
        h, w = image_array.shape

        # --- Phase 1: segment the page into walkable paper regions ---
        ink_mask, walkable, labeled, num_labels = self._phase1_segment(image_array)

        # --- Phase 2: extract per-region statistics ---
        regions = self._phase2_extract_stats(
            labeled, num_labels, image_array, ink_mask
        )

        # --- Edge case: page with very few usable regions → fallback grid ---
        if len(regions) < 3:
            logger.debug("Few regions (%d) – falling back to uniform grid", len(regions))
            return self._fallback_grid(image_array)

        # --- Screentone detection: globally dense ink → dial back hints ---
        global_ink_density = float(np.mean(ink_mask))
        is_screentone = global_ink_density > 0.25

        # --- Phase 3: classify every region ---
        for r in regions:
            r["label"] = self._phase3_classify(r, h)

        # --- Phase 4: assign natural colours ---
        regions = self._phase4_assign_colors(regions, h, w)

        # --- Phase 5: place hints, Gaussian-smooth, pad, return ---
        hint_map, hint_mask = self._phase5_build_maps(
            regions, h, w, screentone=is_screentone
        )
        return self._pad_to_32(hint_map, hint_mask)

    # ------------------------------------------------------------------
    # Phase 1 — Segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _phase1_segment(
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Otsu threshold → dilate ink → connected components on walkable mask."""
        _, ink = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        ink_mask = (ink > 0).astype(np.uint8)

        # Elliptical kernel: closes small gaps without merging distant regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ink_dilated = cv2.dilate(ink_mask, kernel, iterations=1)

        walkable = (1 - ink_dilated).astype(np.uint8)
        num_labels, labeled = cv2.connectedComponents(walkable, connectivity=8)
        return ink_mask, walkable, labeled, num_labels

    # ------------------------------------------------------------------
    # Phase 2 — Region statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _phase2_extract_stats(
        labeled: np.ndarray,
        num_labels: int,
        orig_img: np.ndarray,
        ink_mask: np.ndarray,
    ) -> list[dict]:
        """Compute centroid, bbox, area, luminance, line-density, compactness, etc."""
        h, w = orig_img.shape
        total = h * w
        regions: list[dict] = []

        for lid in range(1, num_labels):
            mask = labeled == lid
            area = int(np.count_nonzero(mask))
            if area < 200 or area > 0.40 * total:
                continue

            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue

            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())

            avg_lum = float(np.mean(orig_img[mask]))

            bbox_ink = ink_mask[y1 : y2 + 1, x1 : x2 + 1]
            line_dens = float(np.mean(bbox_ink))

            dist_edge = min(cx, w - 1.0 - cx, cy, h - 1.0 - cy)
            near_edge = dist_edge < 0.05 * max(w, h)

            compactness = SmartHintGenerator._region_compactness(mask, area)

            regions.append(
                {
                    "id": lid,
                    "mask": mask,
                    "area": area,
                    "centroid": (cx, cy),
                    "bbox": (x1, y1, x2, y2),
                    "avg_luminance": avg_lum,
                    "line_density": line_dens,
                    "dist_to_edge": dist_edge,
                    "near_edge": near_edge,
                    "compactness": compactness,
                }
            )

        return regions

    @staticmethod
    def _region_compactness(mask: np.ndarray, area: int) -> float:
        """Compute 4π·area / perimeter² (≈ 1 for circle, < 1 for irregular)."""
        # Count interior pixels via 4‑neighbor erosion
        interior = mask.copy()
        interior[1:, :] &= mask[:-1, :]
        interior[:-1, :] &= mask[1:, :]
        interior[:, 1:] &= mask[:, :-1]
        interior[:, :-1] &= mask[:, 1:]
        perimeter = int(np.count_nonzero(mask)) - int(np.count_nonzero(interior))
        if perimeter < 4:
            return 0.0
        return (4.0 * np.pi * area) / (perimeter * perimeter)

    # ------------------------------------------------------------------
    # Phase 3 — Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _phase3_classify(r: dict, img_height: int) -> str:
        """Classify a paper region into a semantic label."""
        ld = r["line_density"]
        area = r["area"]
        cy = r["centroid"][1]
        near = r["near_edge"]
        lum = r["avg_luminance"]
        comp = r["compactness"]

        # Text bubbles: very white, round, enclosed
        if comp > 0.72 and lum > 242 and 300 < area < 20000:
            return "text_bubble"

        if ld > 0.25 and area > 2000:
            return "hair"
        if ld > 0.12 and area > 1000 and cy < 0.4 * img_height:
            return "hair"
        if ld > 0.08 and area > 800 and not near:
            return "skin"
        if ld < 0.05 and area > 5000:
            return "background"
        if area > 15000 and near:
            return "background"
        if ld > 0.30 and area > 300:
            return "clothing"
        if lum > 230 and area > 2000:
            return "background"
        return "accent"

    # ------------------------------------------------------------------
    # Phase 4 — Colour assignment
    # ------------------------------------------------------------------

    def _phase4_assign_colors(
        self, regions: list[dict], h: int, w: int
    ) -> list[dict]:
        """Assign a natural colour to every region, considering spatial context."""
        clothing_regions = [r for r in regions if r["label"] == "clothing"]

        for r in regions:
            rt = r["label"]

            if rt == "text_bubble":
                # Nearly white — let the model leave it clean
                r["color"] = np.array([253, 252, 250], dtype=np.float32)
                r["strength"] = 0.35
                continue

            if rt == "skin":
                idx = self._skin_idx % len(_SKIN_TONES)
                self._skin_idx += 1
                r["color"] = _SKIN_TONES[idx].copy()
                # Darken slightly for lower portions (neck / shadow)
                if r["centroid"][1] > 0.65 * h:
                    r["color"] *= 0.92

            elif rt == "hair":
                idx = self._hair_idx % len(_HAIR_COLORS)
                self._hair_idx += 1
                if r["line_density"] > 0.35:
                    r["color"] = _HAIR_COLORS[0].copy()  # darkest
                elif r["line_density"] > 0.20:
                    r["color"] = _HAIR_COLORS[idx % 3].copy()
                else:
                    r["color"] = _HAIR_COLORS[idx].copy()

            elif rt == "clothing":
                idx = self._clothing_idx % len(_CLOTHING)
                self._clothing_idx += 1
                base = _CLOTHING[idx].copy()
                # Ensure adjacent clothing regions differ in hue
                for other in clothing_regions:
                    if other is r or "color" not in other:
                        continue
                    if self._bboxes_touch(r["bbox"], other["bbox"], margin=15):
                        base = base * np.array([1.12, 0.88, 0.88], dtype=np.float32)
                        break
                r["color"] = np.clip(base, 0.0, 255.0)

            elif rt == "background":
                idx = self._bg_idx % len(_BACKGROUND)
                self._bg_idx += 1
                r["color"] = _BACKGROUND[idx].copy()
                if r["near_edge"]:
                    r["color"] = _BACKGROUND[0].copy()  # perimeter → pure cream

            elif rt == "accent":
                idx = self._accent_idx % len(_ACCENT)
                self._accent_idx += 1
                r["color"] = _ACCENT[idx].copy()

        return regions

    @staticmethod
    def _bboxes_touch(
        bb1: Tuple[int, int, int, int],
        bb2: Tuple[int, int, int, int],
        margin: int,
    ) -> bool:
        """Return True if two bounding boxes overlap or are within *margin* px."""
        x1a, y1a, x2a, y2a = bb1
        x1b, y1b, x2b, y2b = bb2
        if x2a + margin < x1b or x2b + margin < x1a:
            return False
        if y2a + margin < y1b or y2b + margin < y1a:
            return False
        return True

    # ------------------------------------------------------------------
    # Phase 5 — Hint placement, smoothing & map construction
    # ------------------------------------------------------------------

    def _phase5_build_maps(
        self,
        regions: list[dict],
        h: int,
        w: int,
        screentone: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Place weighted hints for every region, blend, blur, and build output."""
        hint_accum = np.zeros((h, w, 3), dtype=np.float64)
        weight_accum = np.zeros((h, w), dtype=np.float64)

        for r in regions:
            if r["label"] == "text_bubble":
                continue

            conf = self._region_confidence(r, h, w)
            base_strength = r.get("strength", 0.6 + 0.3 * conf)
            if screentone:
                base_strength = max(base_strength - 0.20, 0.30)

            n_pts = 2 if screentone or conf < 0.35 else (4 if conf > 0.55 else 3)
            if r["label"] == "accent":
                n_pts = min(n_pts, 2)

            samples = self._farthest_point_sample(r["mask"], n_pts)
            radius = int(np.sqrt(r["area"]) / 3.5)
            radius = max(radius, 7)
            radius = min(radius, 55)

            for sx, sy in samples:
                self._raster_hint(
                    hint_accum, weight_accum,
                    int(sx), int(sy), radius, r["color"], base_strength,
                )

        # --- Normalise by accumulated weight ---
        nonzero = weight_accum > 0
        if np.any(nonzero):
            hint_accum[nonzero] /= (weight_accum[nonzero, None] + 1e-8)

        # --- Gaussian-smooth the hint map (sigma ≈ 4 px) ---
        hint_smooth = cv2.GaussianBlur(
            hint_accum.astype(np.float32), (0, 0), sigmaX=4.0, sigmaY=4.0
        )
        weight_smooth = cv2.GaussianBlur(
            weight_accum.astype(np.float32), (0, 0), sigmaX=4.0, sigmaY=4.0
        )

        blurred_nonzero = weight_smooth > 0.002
        if np.any(blurred_nonzero):
            hint_smooth[blurred_nonzero] /= (
                weight_smooth[blurred_nonzero, None] + 1e-8
            )

        hint_map = np.clip(hint_smooth, 0.0, 255.0).astype(np.uint8)
        hint_mask = (blurred_nonzero.astype(np.uint8)) * 255

        return hint_map, hint_mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _region_confidence(r: dict, h: int, w: int) -> float:
        """Return a confidence score [0, 1] used to scale hint strength."""
        score = 0.50

        # Area: discard tiny or barely-present regions
        area_frac = r["area"] / (h * w)
        if 0.002 < area_frac < 0.15:
            score += 0.12
        elif area_frac >= 0.15:
            score += 0.06

        # Interior regions are more trustworthy
        edge_norm = min(r["dist_to_edge"] / (0.12 * max(h, w)), 1.0)
        score += 0.14 * edge_norm

        label = r["label"]
        if label in ("skin", "hair"):
            score += 0.10
        elif label == "background":
            score += 0.06
        elif label == "accent":
            score -= 0.08

        # Very compact white areas are often empty speech bubbles
        if r["compactness"] > 0.65 and r["avg_luminance"] > 240:
            score -= 0.10

        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _farthest_point_sample(
        mask: np.ndarray, n: int
    ) -> list[Tuple[float, float]]:
        """Pick *n* points well spread across a binary mask.

        Uses farthest-point sampling (FPS): start from centroid, then
        iteratively pick the point furthest from already-selected ones.
        """
        ys, xs = np.where(mask)
        if len(xs) <= n:
            return [(float(xs[i]), float(ys[i])) for i in range(len(xs))]

        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

        # Centroid first
        centroid = np.mean(pts, axis=0)
        selected = np.array([centroid])
        out = [(float(centroid[0]), float(centroid[1]))]

        # Sub-sample for speed on large regions
        if len(pts) > 8000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(pts), 8000, replace=False)
            pts = pts[idx]

        for _ in range(n - 1):
            dists = np.full(len(pts), np.inf, dtype=np.float64)
            for s in selected:
                d = np.sqrt((pts[:, 0] - s[0]) ** 2 + (pts[:, 1] - s[1]) ** 2)
                dists = np.minimum(dists, d)
            best = int(np.argmax(dists))
            selected = np.vstack([selected, pts[best]])
            out.append((float(pts[best, 0]), float(pts[best, 1])))

        return out

    @staticmethod
    def _raster_hint(
        hint_map: np.ndarray,
        weight_map: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        color: np.ndarray,
        strength: float,
    ) -> None:
        """Draw a single hint as a Gaussian-weighted circle into the accumulators."""
        h, w = hint_map.shape[:2]

        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius + 1)
        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius + 1)
        if y2 <= y1 or x2 <= x1:
            return

        yy, xx = np.ogrid[y1:y2, x1:x2]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

        sigma = radius / 3.0
        falloff = np.exp(-dist2 / (2.0 * sigma * sigma))
        falloff = np.clip(falloff * strength, 0.0, 1.0)
        falloff[dist2 > radius * radius] = 0.0

        if not np.any(falloff > 0):
            return

        for ch in range(3):
            hint_map[y1:y2, x1:x2, ch] += falloff * color[ch]
        weight_map[y1:y2, x1:x2] += falloff

    # ------------------------------------------------------------------
    # Fallback — uniform grid for near‑empty pages
    # ------------------------------------------------------------------

    def _fallback_grid(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Place a uniform 5×7 grid of soft hints when regions are sparse."""
        h, w = image.shape
        cols, rows = 5, 7
        cw, ch = w // cols, h // rows

        hint_accum = np.zeros((h, w, 3), dtype=np.float64)
        weight_accum = np.zeros((h, w), dtype=np.float64)

        palette_cycle = [
            _SKIN_TONES[0],
            _BACKGROUND[0],
            _CLOTHING[2],
            _SKIN_TONES[1],
            _CLOTHING[5],
            _BACKGROUND[1],
            _SKIN_TONES[2],
        ]

        for ry in range(rows):
            for cx in range(cols):
                px = cw // 2 + cx * cw
                py = ch // 2 + ry * ch
                radius = min(cw, ch) // 3
                color = palette_cycle[(ry + cx) % len(palette_cycle)]
                # Lower strength so the model still has freedom
                self._raster_hint(
                    hint_accum, weight_accum, px, py, radius, color, 0.40
                )

        nonzero = weight_accum > 0
        if np.any(nonzero):
            hint_accum[nonzero] /= (weight_accum[nonzero, None] + 1e-8)

        hint_smooth = cv2.GaussianBlur(
            hint_accum.astype(np.float32), (0, 0), sigmaX=9.0, sigmaY=9.0
        )
        weight_smooth = cv2.GaussianBlur(
            weight_accum.astype(np.float32), (0, 0), sigmaX=9.0, sigmaY=9.0
        )

        blurred_nz = weight_smooth > 0.001
        if np.any(blurred_nz):
            hint_smooth[blurred_nz] /= (weight_smooth[blurred_nz, None] + 1e-8)

        hint_map = np.clip(hint_smooth, 0.0, 255.0).astype(np.uint8)
        hint_mask = (blurred_nz.astype(np.uint8)) * 255
        return self._pad_to_32(hint_map, hint_mask)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to_32(
        hint_map: np.ndarray, hint_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pad right and bottom edges to reach multiples of 32."""
        h, w = hint_map.shape[:2]
        ph = (32 - h % 32) % 32
        pw = (32 - w % 32) % 32
        if ph == 0 and pw == 0:
            return hint_map, hint_mask
        hint_map = np.pad(
            hint_map, ((0, ph), (0, pw), (0, 0)), mode="reflect"
        )
        hint_mask = np.pad(
            hint_mask, ((0, ph), (0, pw)), mode="constant", constant_values=0
        )
        return hint_map, hint_mask

    def _reset_palettes(self) -> None:
        self._skin_idx = 0
        self._hair_idx = 0
        self._clothing_idx = 0
        self._bg_idx = 0
        self._accent_idx = 0


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def hints_to_model_tensor(
    hint_map: np.ndarray, hint_mask: np.ndarray
) -> np.ndarray:
    """Convert (H,W,3) uint8 hints + (H,W) uint8 mask into the model's
    4-channel hint tensor.

    Returns
    -------
    np.ndarray
        float32 ``(1, 4, H, W)`` tensor: [hint_R, hint_G, hint_B, hint_mask]
        all in [0, 1].
    """
    h_f = hint_map.astype(np.float32) / 255.0       # (H, W, 3)
    m_f = hint_mask.astype(np.float32) / 255.0      # (H, W)
    chw = np.concatenate([h_f, m_f[:, :, None]], axis=2)  # (H, W, 4)
    return chw.transpose(2, 0, 1)[None, ...]        # (1, 4, H, W)


def create_onnx_input(
    grayscale: np.ndarray,
    generator: Optional[SmartHintGenerator] = None,
) -> np.ndarray:
    """One-shot convenience: grayscale image → 5-channel ONNX input tensor.

    Parameters
    ----------
    grayscale : np.ndarray
        ``(H, W)`` uint8 grayscale image.
    generator : SmartHintGenerator | None
        Optional pre-configured generator (created on-the-fly otherwise).

    Returns
    -------
    np.ndarray
        float32 ``(1, 5, H_pad, W_pad)`` ready for ONNX inference.
    """
    gen = generator or SmartHintGenerator()
    hint_map, hint_mask = gen.generate(grayscale)

    # Pad grayscale to match hint dimensions
    hh, hw = hint_map.shape[:2]
    gh, gw = grayscale.shape
    pad_h, pad_w = hh - gh, hw - gw
    gray_padded = np.pad(
        grayscale.astype(np.float32),
        ((0, pad_h), (0, pad_w)),
        mode="reflect",
    )
    gray_ch = (gray_padded / 255.0)[None, None, ...]  # (1, 1, H, W)

    hint_ch = hints_to_model_tensor(hint_map, hint_mask)  # (1, 4, H, W)
    return np.concatenate([gray_ch, hint_ch], axis=1).astype(np.float32)
