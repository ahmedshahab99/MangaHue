"""
AdvancedHintGenerator -- professional-grade smart hint generator for manga colorization.

Given a grayscale manga page, generates RGB hint maps that guide an ONNX
colorization model toward natural, beautiful manga colours.

Input:  numpy array (H, W) uint8 grayscale [0-255]
Output: (hint_map (H,W,3) uint8 RGB, hint_mask (H,W) uint8)

Algorithm (4 phases):
  1. Page structure analysis   -- multi-scale line detection, content rect,
     brightness histogram, local variance map.
  2. Semantic region detection -- adaptive grid with weighted scoring
     classifier (skin/hair/clothing/bg/eyes/skip).
  3. Colour assignment         -- curated manga palettes + CIEDE2000 harmony.
  4. Professional rendering    -- region-centroid hints, Gaussian falloff & blur.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ===========================================================================
# Hand-curated manga colour palettes (RGB float32, 0-255)
# ===========================================================================

_SKIN = np.float32([
    [252,215,185],[248,208,178],[240,192,162],[255,220,192],
    [245,200,170],[235,182,150],[250,212,180],
])
_HAIR_BLACK  = np.float32([[30,25,20],[40,32,28],[50,40,35]])
_HAIR_BROWN  = np.float32([[90,65,45],[110,80,55],[140,105,75],[165,130,95]])
_HAIR_LIGHT  = np.float32([[180,145,110],[195,160,130],[210,180,150]])
_CLOTH_WARM  = np.float32([[155,48,42],[180,60,50],[140,40,35],[195,80,65]])
_CLOTH_COOL  = np.float32([[42,72,148],[55,85,165],[35,60,130]])
_CLOTH_EARTH = np.float32([[52,42,75],[68,55,90],[85,70,105]])
_CLOTH_GREEN = np.float32([[38,75,55],[50,90,68],[30,65,45]])
_CLOTH_NEUT  = np.float32([
    [185,168,148],[165,148,130],[200,185,165],
    [148,128,118],[72,50,44],[90,118,160],
])
_BACKGROUND  = np.float32([[248,246,240],[240,236,228],[235,230,220],[250,248,244]])
_EYES        = np.float32([[28,38,105],[68,142,178],[42,88,160],[35,50,120]])
_CLOTH_SUB   = [_CLOTH_WARM,_CLOTH_COOL,_CLOTH_EARTH,_CLOTH_GREEN,_CLOTH_NEUT]

# ===========================================================================
# CIEDE2000 colour-difference (compact, correct)
# ===========================================================================

def _delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    L1,a1,b1 = map(float,lab1); L2,a2,b2 = map(float,lab2)
    C1,C2 = np.hypot(a1,b1),np.hypot(a2,b2)
    C_bar = (C1 + C2) * 0.5
    G = 0.5 * (1.0 - np.sqrt(C_bar**7 / (C_bar**7 + 25.0**7)))
    a1p,a2p = a1*(1.0+G), a2*(1.0+G)
    C1p,C2p = np.hypot(a1p,b1), np.hypot(a2p,b2)
    C_bar_p = (C1p + C2p) * 0.5
    h1p = np.arctan2(b1,a1p) % (2.0*np.pi) or 0.0
    h2p = np.arctan2(b2,a2p) % (2.0*np.pi) or 0.0
    dL,dC = L2-L1, C2p-C1p

    dh = h2p - h1p
    if C1p * C2p < 1e-12:
        dH = 0.0
    else:
        if abs(dh) > np.pi:
            dh += -2.0*np.pi if dh > 0 else 2.0*np.pi
        dH = 2.0 * np.sqrt(C1p * C2p) * np.sin(dh * 0.5)

    L_bar_p = (L1 + L2) * 0.5
    if C1p * C2p < 1e-12:
        h_bar_p = h1p + h2p
    else:
        h_bar_p = (h1p + h2p) * 0.5
        if abs(h1p - h2p) > np.pi:
            h_bar_p += np.pi if h_bar_p < np.pi else -np.pi

    T = (1.0 - 0.17*np.cos(h_bar_p - 0.523599) + 0.24*np.cos(2.0*h_bar_p)
         + 0.32*np.cos(3.0*h_bar_p + 0.10472) - 0.20*np.cos(4.0*h_bar_p - 1.09956))
    dtheta = 0.523599 * np.exp(-(((h_bar_p - 4.79966) / 0.436332) ** 2))
    RC = 2.0 * np.sqrt(C_bar_p**7 / (C_bar_p**7 + 25.0**7))
    SL = 1.0 + 0.015 * (L_bar_p - 50.0)**2 / np.sqrt(20.0 + (L_bar_p - 50.0)**2)
    SC = 1.0 + 0.045 * C_bar_p
    SH = 1.0 + 0.015 * C_bar_p * T
    RT = -np.sin(2.0 * dtheta) * RC

    term = (dL/SL)**2 + (dC/SC)**2 + (dH/SH)**2 + RT*(dC/SC)*(dH/SH)
    return float(np.sqrt(max(term, 0.0)))


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    arr = (rgb.astype(np.float64) / 255.0).reshape(1,1,3).astype(np.float32)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0,0].astype(np.float64)

# ===========================================================================
# AdvancedHintGenerator
# ===========================================================================

class AdvancedHintGenerator:
    """Generate dense, spatially-aware colour hints for a manga page.

    Usage:
        gen = AdvancedHintGenerator()
        hint_map, hint_mask = gen.generate(grayscale_uint8)
    """

    def __init__(self) -> None:
        self._skin_i = self._hair_i = self._cloth_i = self._bg_i = self._eyes_i = 0
        self._hair_pal: np.ndarray = _HAIR_BLACK
        self._hair_fam = "black"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (hint_map (H,W,3) uint8 RGB, hint_mask (H,W) uint8)."""
        if gray.ndim != 2:
            raise ValueError("Expected 2-D grayscale image")
        self._reset()
        h, w = gray.shape

        # ---- Phase 1: page structure ----
        edge = self._phase1_edge(gray)
        x1, y1, x2, y2 = self._phase1_rect(edge, h, w)
        p5, p25, p50, _, p95 = np.percentile(gray.astype(np.float64), [5,25,50,75,95])
        is_dark, is_light = p50 < 80, p50 > 170
        var_map = self._phase1_variance(gray)

        # ---- Phase 2: semantic labelling ----
        cells = self._phase2_grid(x1, y1, x2, y2, max(h, w))
        if len(cells) < 4:
            return self._fallback_grid(gray)
        labels, confs = self._phase2_classify(gray, edge, var_map, cells, h, w)

        # ---- Phase 3: colour assignment ----
        colours = self._phase3_assign(labels, gray, is_dark, is_light, cells, h)

        # ---- Phase 4: render hints ----
        hint_map, hint_mask = self._phase4_render(edge, cells, labels, confs, colours, h, w)
        return self._pad32(hint_map, hint_mask)

    # ==================================================================
    # Phase 1 — page structure analysis
    # ==================================================================

    @staticmethod
    def _phase1_edge(gray: np.ndarray) -> np.ndarray:
        """Multi-scale line detection via integral-image adaptive thresholding."""
        h, w = gray.shape
        integral = cv2.integral(gray.astype(np.float64))
        gf = gray.astype(np.float64)
        yy, xx = np.mgrid[0:h, 0:w]
        combined = np.zeros((h, w), np.float32)

        for i, r in enumerate((5, 15, 31)):
            y1 = np.maximum(0, yy - r); x1 = np.maximum(0, xx - r)
            y2 = np.minimum(h - 1, yy + r); x2 = np.minimum(w - 1, xx + r)
            s11 = integral[y1, x1];          s12 = integral[y2 + 1, x1]
            s21 = integral[y1, x2 + 1];      s22 = integral[y2 + 1, x2 + 1]
            area = (x2 - x1 + 1).astype(np.float64) * (y2 - y1 + 1).astype(np.float64)
            local_mean = (s22 - s12 - s21 + s11) / np.maximum(area, 1.0)
            wgt = [0.50, 0.30, 0.20][i]
            combined += wgt * (np.abs(gf - local_mean) > local_mean * 0.12)

        return (combined > 0.35).astype(np.uint8)

    @staticmethod
    def _phase1_rect(edge: np.ndarray, h: int, w: int) -> Tuple[int, int, int, int]:
        """Content rectangle via edge-density projection."""
        xp, yp = edge.mean(axis=0), edge.mean(axis=1)
        xt, yt = xp.max() * 0.05, yp.max() * 0.05
        xi = np.where(xp > xt)[0]; yi = np.where(yp > yt)[0]
        x1 = int(xi[0]) if len(xi) else 0
        x2 = int(xi[-1] + 1) if len(xi) else w
        y1 = int(yi[0]) if len(yi) else 0
        y2 = int(yi[-1] + 1) if len(yi) else h
        m = max(int(min(h, w) * 0.02), 2)
        return max(0, x1 - m), max(0, y1 - m), min(w, x2 + m), min(h, y2 + m)

    @staticmethod
    def _phase1_variance(gray: np.ndarray) -> np.ndarray:
        """31x31 block local variance -- screentone indicator."""
        gf = gray.astype(np.float32)
        return np.maximum(cv2.blur(gf * gf, (31, 31)) - cv2.blur(gf, (31, 31)) ** 2, 0.0)

    # ==================================================================
    # Phase 2 — semantic region detection
    # ==================================================================

    @staticmethod
    def _phase2_grid(x1: int, y1: int, x2: int, y2: int, max_dim: int
                     ) -> List[Tuple[int, int, int, int]]:
        """Adaptive grid (~24 cells along the larger axis)."""
        cs = max((max(x2 - x1, y2 - y1) // 24), 8)
        return [
            (cx, cy, min(cx + cs, x2), min(cy + cs, y2))
            for cy in range(y1, y2, cs)
            for cx in range(x1, x2, cs)
            if min(cx + cs, x2) - cx >= 4 and min(cy + cs, y2) - cy >= 4
        ]

    @classmethod
    def _phase2_classify(cls, gray: np.ndarray, edge: np.ndarray,
                         var_map: np.ndarray,
                         cells: List[Tuple[int, int, int, int]],
                         h: int, w: int
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-cell features and classify via weighted scoring."""
        gf = gray.astype(np.float32)
        y0 = min(c[1] for c in cells)
        ch_c = max(c[3] for c in cells) - y0
        mid_x = (w - 1) * 0.5
        NC = len(cells)

        labels = np.empty(NC, dtype=object)
        confs = np.zeros(NC, np.float32)
        ORD = ["skin", "hair", "clothing", "background", "eyes", "skip"]

        for i, (cx1, cy1, cx2, cy2) in enumerate(cells):
            pg = gf[cy1:cy2, cx1:cx2]
            pe = edge[cy1:cy2, cx1:cx2]
            pv = var_map[cy1:cy2, cx1:cx2]

            b  = pg.mean() / 255.0
            v  = pv.mean() / (255.0 * 255.0)
            yn = ((cy1 + cy2) * 0.5 - y0) / max(ch_c, 1)
            ed = pe.mean()
            xn = (cx1 + cx2) * 0.5 / w

            # symmetry: correlate with mirror cell across vertical centreline
            sym = 0.0
            mx2, mx1 = max(0, int(2.0 * mid_x - cx2)), min(w, int(2.0 * mid_x - cx1))
            if 0 <= mx2 < mx1 <= w:
                mp = gf[cy1:cy2, mx2:mx1]
                if mp.size > 0 and mp.shape == pg.shape:
                    a, b = mp.ravel(), pg.ravel()
                    d = np.std(a) * np.std(b)
                    if d > 1e-6:
                        sym = np.corrcoef(a, b)[0, 1]

            sc = {k: 0.0 for k in ORD}

            # skin
            if 0.30 < b < 0.70:  sc["skin"] += 2.0
            if v < 0.15:          sc["skin"] += 1.5
            if 0.15 < yn < 0.65:  sc["skin"] += 1.0
            if ed < 0.08:         sc["skin"] += 1.0
            if sym > 0.6:         sc["skin"] += 0.5

            # hair
            if b < 0.25:   sc["hair"] += 2.0
            if v > 0.20:   sc["hair"] += 1.5
            if yn < 0.40:  sc["hair"] += 1.0
            if ed > 0.12:  sc["hair"] += 1.5

            # clothing
            if 0.15 < b < 0.55:  sc["clothing"] += 1.0
            if ed > 0.08:         sc["clothing"] += 1.0
            if yn > 0.20:         sc["clothing"] += 1.0

            # background
            if b > 0.80:  sc["background"] += 3.0
            if v < 0.10:  sc["background"] += 2.0

            # eyes
            if 0.15 < yn < 0.35 and 0.35 < xn < 0.65 and 0.15 < b < 0.40 and ed > 0.10:
                sc["eyes"] += 2.5
            if b < 0.15:
                sc["eyes"] += 1.5

            # skip
            if b > 0.95:  sc["skip"] += 5.0
            if v > 0.50:  sc["skip"] += 3.0
            if ed > 0.50: sc["skip"] += 2.0

            ranked = sorted(ORD, key=lambda k: sc[k], reverse=True)
            labels[i] = ranked[0]
            best, sec = sc[ranked[0]], sc[ranked[1]]
            confs[i] = (best - sec) / max(best, 1e-6) if best > 0 else 0.0

        return labels, confs

    # ==================================================================
    # Phase 3 — colour assignment
    # ==================================================================

    def _phase3_assign(self, labels: np.ndarray, gray: np.ndarray,
                       is_dark: bool, is_light: bool,
                       cells: List[Tuple[int, int, int, int]], img_h: int,
                       ) -> List[np.ndarray]:
        """Assign palette-derived colours to every cell with CIEDE2000 harmony."""
        NC = len(labels)
        colours: List[np.ndarray] = [np.float32([128, 128, 128]) for _ in range(NC)]
        cell_sz = cells[0][2] - cells[0][0] if cells else 32

        # choose hair palette
        if is_dark:
            self._hair_pal, self._hair_fam = _HAIR_LIGHT, "light"
        else:
            if np.random.default_rng(42).random() < 0.70:
                self._hair_pal, self._hair_fam = _HAIR_BLACK, "black"
            else:
                self._hair_pal, self._hair_fam = _HAIR_BROWN, "brown"

        cloth_ids = [i for i in range(NC) if labels[i] == "clothing"]
        cloth_asgn: dict = {}

        for i, (cx1, cy1, cx2, cy2) in enumerate(cells):
            lb = labels[i]
            cx, cy = (cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5

            if lb == "skip":
                colours[i] = np.float32([255, 255, 255])
                continue

            if lb == "skin":
                colours[i] = _SKIN[self._skin_i % len(_SKIN)].copy()
                self._skin_i += 1
                if cy > 0.65 * img_h:
                    colours[i] *= 0.92

            elif lb == "hair":
                colours[i] = self._hair_pal[self._hair_i % len(self._hair_pal)].copy()
                self._hair_i += 1

            elif lb == "clothing":
                used = set()
                for j in cloth_ids:
                    if j == i or j not in cloth_asgn:
                        continue
                    jx = (cells[j][0] + cells[j][2]) * 0.5
                    jy = (cells[j][1] + cells[j][3]) * 0.5
                    if max(abs(cx - jx), abs(cy - jy)) < 2.0 * cell_sz:
                        used.add(cloth_asgn[j])

                avail = [s for s in range(len(_CLOTH_SUB)) if s not in used]
                if not avail:
                    avail = list(range(len(_CLOTH_SUB)))
                sub = avail[self._cloth_i % len(avail)]
                self._cloth_i += 1
                cloth_asgn[i] = sub

                c = _CLOTH_SUB[sub][self._cloth_i % len(_CLOTH_SUB[sub])].copy()
                if is_light:
                    gm = c.mean()
                    c = np.clip(gm + (c - gm) * 1.15, 0, 255)
                colours[i] = c

            elif lb == "background":
                rng = np.random.default_rng(i + 7)
                jitter = rng.integers(-2, 3, size=3).astype(np.float32)
                colours[i] = np.clip(_BACKGROUND[0].copy() + jitter, 0.0, 255.0)
                self._bg_i += 1

            elif lb == "eyes":
                colours[i] = _EYES[self._eyes_i % len(_EYES)].copy()
                self._eyes_i += 1

        # CIEDE2000 harmony: push apart too-similar adjacent colours
        for i in range(NC):
            if labels[i] in ("skip", "background"):
                continue
            cxi, cyi = (cells[i][0] + cells[i][2]) * 0.5, (cells[i][1] + cells[i][3]) * 0.5
            for j in range(NC):
                if i == j or labels[j] == "skip":
                    continue
                cxj, cyj = (cells[j][0] + cells[j][2]) * 0.5, (cells[j][1] + cells[j][3]) * 0.5
                if abs(cxi - cxj) > 2.0 * cell_sz or abs(cyi - cyj) > 2.0 * cell_sz:
                    continue
                de = _delta_e_2000(_rgb_to_lab(colours[i]), _rgb_to_lab(colours[j]))
                if de < 15.0:
                    colours[i] = np.clip(colours[i] * np.float32([1.03, 0.97, 0.97]), 0, 255)
                    break

        return colours

    # ==================================================================
    # Phase 4 — professional hint rendering
    # ==================================================================

    def _phase4_render(self, edge: np.ndarray,
                       cells: List[Tuple[int, int, int, int]],
                       labels: np.ndarray, confs: np.ndarray,
                       colours: List[np.ndarray],
                       h: int, w: int,
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Place hints at connected-component region centroids, blend, smooth."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_dil = cv2.dilate(edge, kernel, iterations=1)
        walkable = (1 - edge_dil).astype(np.uint8)

        nc, cc_labels, cc_stats, cc_centroids = \
            cv2.connectedComponentsWithStats(walkable, connectivity=8)

        hint_acc = np.zeros((h, w, 3), np.float64)
        weight_acc = np.zeros((h, w), np.float64)

        cell_idx = np.full((h, w), -1, np.int32)
        for i, (x1c, y1c, x2c, y2c) in enumerate(cells):
            cell_idx[y1c:y2c, x1c:x2c] = i

        for cid in range(1, nc):
            area = cc_stats[cid, cv2.CC_STAT_AREA]
            if area < 100 or area > h * w * 0.45:
                continue
            cx = int(cc_centroids[cid, 0])
            cy = int(cc_centroids[cid, 1])
            if not (0 <= cx < w and 0 <= cy < h):
                continue

            ci = cell_idx[cy, cx]
            if ci < 0 or labels[ci] == "skip":
                continue

            col = colours[ci]
            conf = float(confs[ci])
            strength = 0.25 + 0.35 * conf
            radius = max(int(np.sqrt(area) * 0.3), 5)
            radius = min(radius, 55)
            self._raster_hint(hint_acc, weight_acc, cx, cy, radius, col, strength)

        nz = weight_acc > 0
        if np.any(nz):
            hint_acc[nz] /= (weight_acc[nz, None] + 1e-8)

        hint_smooth = cv2.GaussianBlur(hint_acc.astype(np.float32), (0, 0), 6.0, 6.0)
        weight_smooth = cv2.GaussianBlur(weight_acc.astype(np.float32), (0, 0), 3.0, 3.0)

        bnz = weight_smooth > 0.001
        if np.any(bnz):
            hint_smooth[bnz] /= (weight_smooth[bnz, None] + 1e-8)

        hint_map = np.clip(hint_smooth, 0.0, 255.0).astype(np.uint8)
        hint_mask = (bnz.astype(np.uint8)) * 255
        return hint_map, hint_mask

    @staticmethod
    def _raster_hint(hint_acc: np.ndarray, weight_acc: np.ndarray,
                     cx: int, cy: int, radius: int,
                     colour: np.ndarray, strength: float) -> None:
        """Draw a single Gaussian-weighted circular hint."""
        h, w = hint_acc.shape[:2]
        y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)
        x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)
        if y2 <= y1 or x2 <= x1:
            return

        yy, xx = np.ogrid[y1:y2, x1:x2]
        dist2 = (xx - cx).astype(np.float64) ** 2 + (yy - cy).astype(np.float64) ** 2
        sigma = radius * 0.5
        falloff = np.exp(-dist2 / (2.0 * sigma * sigma)) * strength
        falloff[falloff < 0.001] = 0.0
        falloff[dist2 > radius * radius] = 0.0

        if not np.any(falloff > 0):
            return
        for ch in range(3):
            hint_acc[y1:y2, x1:x2, ch] += falloff * float(colour[ch])
        weight_acc[y1:y2, x1:x2] += falloff

    # ==================================================================
    # Fallback & padding
    # ==================================================================

    def _fallback_grid(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform 5x7 grid for pages with too few usable regions."""
        h, w = gray.shape
        cw, ch = w // 5, h // 7
        acc_c = np.zeros((h, w, 3), np.float64)
        acc_w = np.zeros((h, w), np.float64)
        pal_cycle = [_SKIN[0], _BACKGROUND[0], _CLOTH_COOL[0],
                     _SKIN[1], _CLOTH_WARM[0], _BACKGROUND[1], _SKIN[2]]

        for ry in range(7):
            for rx in range(5):
                px = cw // 2 + rx * cw
                py = ch // 2 + ry * ch
                colour = pal_cycle[(ry + rx) % len(pal_cycle)]
                self._raster_hint(acc_c, acc_w, px, py,
                                  min(cw, ch) // 3, colour, 0.40)

        nz = acc_w > 0
        if np.any(nz):
            acc_c[nz] /= (acc_w[nz, None] + 1e-8)
        hs = cv2.GaussianBlur(acc_c.astype(np.float32), (0, 0), 9.0, 9.0)
        ws = cv2.GaussianBlur(acc_w.astype(np.float32), (0, 0), 9.0, 9.0)
        bnz = ws > 0.001
        if np.any(bnz):
            hs[bnz] /= (ws[bnz, None] + 1e-8)
        hm = np.clip(hs, 0, 255).astype(np.uint8)
        hmk = (bnz.astype(np.uint8)) * 255
        return self._pad32(hm, hmk)

    @staticmethod
    def _pad32(hm: np.ndarray, hmk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pad to multiples of 32 along H and W."""
        ph = (32 - hm.shape[0] % 32) % 32
        pw = (32 - hm.shape[1] % 32) % 32
        if ph == 0 and pw == 0:
            return hm, hmk
        return (
            np.pad(hm,  ((0, ph), (0, pw), (0, 0)), mode="reflect"),
            np.pad(hmk, ((0, ph), (0, pw)), mode="constant", constant_values=0),
        )

    def _reset(self) -> None:
        self._skin_i = self._hair_i = self._cloth_i = self._bg_i = self._eyes_i = 0
        self._hair_pal = _HAIR_BLACK
        self._hair_fam = "black"

# ===========================================================================
# Backward compatibility
# ===========================================================================

SmartHintGenerator = AdvancedHintGenerator

# ===========================================================================
# ONNX tensor helpers
# ===========================================================================

def hints_to_model_tensor(hint_map: np.ndarray, hint_mask: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 + (H,W) uint8 → (1,4,H,W) float32 in [0,1]."""
    hf = hint_map.astype(np.float32) / 255.0
    mf = hint_mask.astype(np.float32) / 255.0
    chw = np.concatenate([hf, mf[:, :, None]], axis=2)
    return chw.transpose(2, 0, 1)[None, ...]


def create_onnx_input(grayscale: np.ndarray,
                      generator: Optional[AdvancedHintGenerator] = None) -> np.ndarray:
    """One-shot: grayscale (H,W) uint8 → (1,5,H_pad,W_pad) float32 ONNX tensor."""
    gen = generator or AdvancedHintGenerator()
    hint_map, hint_mask = gen.generate(grayscale)
    hh, hw = hint_map.shape[:2]
    gh, gw = grayscale.shape
    gp = np.pad(grayscale.astype(np.float32),
                ((0, hh - gh), (0, hw - gw)), mode="reflect")
    gray_ch = (gp / 255.0)[None, None, ...]
    hint_ch = hints_to_model_tensor(hint_map, hint_mask)
    return np.concatenate([gray_ch, hint_ch], axis=1).astype(np.float32)
