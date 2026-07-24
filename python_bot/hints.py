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
_BACKGROUND  = np.float32([[210,205,198],[200,195,188],[215,210,203]])
_EYES        = np.float32([[28,38,105],[68,142,178],[42,88,160],[35,50,120]])
_CLOTH_SUB   = [_CLOTH_WARM,_CLOTH_COOL,_CLOTH_EARTH,_CLOTH_GREEN,_CLOTH_NEUT]

_HAIR_FAMILIES: dict[str, np.ndarray] = {
    "black": np.float32([[30,25,20],[40,32,28],[50,40,35],[35,28,24]]),
    "brown": np.float32([[90,65,45],[110,80,55],[140,105,75],[165,130,95]]),
    "blond": np.float32([[210,180,140],[225,195,160],[195,165,125],[215,188,148]]),
    "red":   np.float32([[170,75,55],[185,90,65],[155,65,45]]),
    "blue":  np.float32([[40,50,130],[55,65,150],[45,55,140]]),
    "white": np.float32([[220,215,210],[230,225,220],[210,205,200]]),
}
_HAIR_FAMILY_NAMES = sorted(_HAIR_FAMILIES.keys())

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
    arr = (rgb / np.float32(255.0)).reshape(1, 1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float64)

# ===========================================================================
# AdvancedHintGenerator
# ===========================================================================

class AdvancedHintGenerator:
    """Generate dense, spatially-aware colour hints for a manga page.

    Usage:
        gen = AdvancedHintGenerator()
        hint_map, hint_mask = gen.generate(grayscale_uint8)
    """

    HAIR_FAMILIES: dict[str, np.ndarray] = _HAIR_FAMILIES
    HAIR_FAMILY_NAMES: list[str] = _HAIR_FAMILY_NAMES

    def __init__(self) -> None:
        self._skin_i = self._hair_i = self._cloth_i = self._bg_i = self._eyes_i = 0
        self._rng = np.random.default_rng()
        self._kernel_cache: dict[int, np.ndarray] = {}

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
                    mp_flat, pg_flat = mp.ravel(), pg.ravel()
                    d = np.std(mp_flat) * np.std(pg_flat)
                    if d > 1e-6:
                        sym = np.corrcoef(mp_flat, pg_flat)[0, 1]

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
    # Phase 3 — colour assignment (multi-character)
    # ==================================================================

    @staticmethod
    def _cluster_spatial(cell_indices: list,
                         cells: List[Tuple[int, int, int, int]]) -> dict:
        """Greedy spatial clustering of cell indices by Chebyshev distance.

        Groups cells within ~2.5 cell widths of each other into clusters
        representing separate characters / outfits.

        Returns ``{cell_idx: cluster_id}`` mapping.
        """
        if not cell_indices:
            return {}

        n = len(cell_indices)
        cell_w = cells[cell_indices[0]][2] - cells[cell_indices[0]][0]
        max_dist = cell_w * 2.5

        # Precompute all cell centres as a numpy array (n, 2) for vectorised queries
        centres_arr = np.zeros((n, 2), np.float64)
        for idx_i, ci in enumerate(cell_indices):
            cx1, cy1, cx2, cy2 = cells[ci]
            centres_arr[idx_i, 0] = (cx1 + cx2) * 0.5
            centres_arr[idx_i, 1] = (cy1 + cy2) * 0.5

        unassigned = set(range(n))
        result: dict[int, int] = {}
        cluster_id = 0

        while unassigned:
            seed_idx = unassigned.pop()
            cluster: list[int] = [seed_idx]
            queue: list[int] = [seed_idx]

            while queue:
                cur_idx = queue.pop()
                cx, cy = centres_arr[cur_idx]

                remaining = np.array(list(unassigned))
                if len(remaining) == 0:
                    break

                u_xy = centres_arr[remaining]
                dx = np.abs(u_xy[:, 0] - cx)
                dy = np.abs(u_xy[:, 1] - cy)
                within = remaining[(dx <= max_dist) & (dy <= max_dist)]

                for wi in within.tolist():
                    cluster.append(wi)
                    queue.append(wi)
                    unassigned.discard(wi)

            for idx in cluster:
                result[cell_indices[idx]] = cluster_id
            cluster_id += 1

        return result

    def _phase3_assign(self, labels: np.ndarray, gray: np.ndarray,
                       is_dark: bool, is_light: bool,
                       cells: List[Tuple[int, int, int, int]], img_h: int,
                       ) -> List[np.ndarray]:
        """Assign palette-derived colours with multi-character cluster support."""
        NC = len(labels)
        colours: List[np.ndarray] = [np.float32([128, 128, 128]) for _ in range(NC)]
        if not cells:
            return colours
        cell_sz = cells[0][2] - cells[0][0]

        # -- collect cell indices by type --
        hair_ids  = [i for i in range(NC) if labels[i] == "hair"]
        skin_ids  = [i for i in range(NC) if labels[i] == "skin"]
        cloth_ids = [i for i in range(NC) if labels[i] == "clothing"]

        # ------------------------------------------------------------------
        # Cluster hair → each character gets a unique family
        # ------------------------------------------------------------------
        hair_clusters = self._cluster_spatial(hair_ids, cells)
        hair_cluster_ids = sorted(set(hair_clusters.values()))
        hair_fam_map: dict[int, str] = {}

        # Dark-scene family pool (light colours visible on dark bg)
        if is_dark:
            dark_families = [f for f in _HAIR_FAMILY_NAMES if f in ("blond", "white", "blue")]
        else:
            dark_families = _HAIR_FAMILY_NAMES

        available_fams = list(dark_families)
        self._rng.shuffle(available_fams)
        for ci_d in hair_cluster_ids:
            fam = available_fams.pop() if available_fams else self._rng.choice(dark_families)
            for ci in hair_ids:
                if hair_clusters.get(ci) == ci_d:
                    hair_fam_map[ci] = fam

        # ------------------------------------------------------------------
        # Cluster skin → each character gets tone variation ±8
        # ------------------------------------------------------------------
        skin_clusters = self._cluster_spatial(skin_ids, cells)
        skin_cluster_base: dict[int, tuple[int, np.ndarray]] = {}
        for ci_d in sorted(set(skin_clusters.values())):
            base_idx = self._rng.integers(0, len(_SKIN))
            offset = self._rng.integers(-8, 9, size=3).astype(np.float32)
            skin_cluster_base[ci_d] = (base_idx, offset)

        # ------------------------------------------------------------------
        # Cluster clothing → each outfit gets a unique sub-palette
        # ------------------------------------------------------------------
        cloth_clusters = self._cluster_spatial(cloth_ids, cells)
        cloth_cluster_ids = sorted(set(cloth_clusters.values()))
        cloth_cluster_sub: dict[int, int] = {}

        available_subs = list(range(len(_CLOTH_SUB)))
        self._rng.shuffle(available_subs)
        for ci_d in cloth_cluster_ids:
            sub = available_subs.pop() if available_subs else self._rng.integers(0, len(_CLOTH_SUB))
            cloth_cluster_sub[ci_d] = sub

        # ------------------------------------------------------------------
        # Assign colours per cell
        # ------------------------------------------------------------------
        for i, (cx1, cy1, cx2, cy2) in enumerate(cells):
            lb = labels[i]
            cx, cy = (cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5

            if lb == "skip":
                colours[i] = np.float32([255, 255, 255])
                continue

            if lb == "skin":
                hc = skin_clusters.get(i, 0)
                base_idx, offset = skin_cluster_base.get(hc, (0, np.float32([0, 0, 0])))
                base = _SKIN[base_idx % len(_SKIN)].copy()
                colours[i] = np.clip(base + offset, 0, 255)
                self._skin_i += 1
                if cy > 0.65 * img_h:
                    colours[i] *= 0.92

            elif lb == "hair":
                fam = hair_fam_map.get(i, "black")
                pal = _HAIR_FAMILIES.get(fam, _HAIR_FAMILIES["black"])
                colours[i] = pal[self._hair_i % len(pal)].copy()
                self._hair_i += 1

            elif lb == "clothing":
                hc = cloth_clusters.get(i, 0)
                sub = cloth_cluster_sub.get(hc, 0)
                c = _CLOTH_SUB[sub][self._cloth_i % len(_CLOTH_SUB[sub])].copy()
                if is_light:
                    gm = c.mean()
                    c = np.clip(gm + (c - gm) * 1.15, 0, 255)
                colours[i] = c
                self._cloth_i += 1

            elif lb == "background":
                jitter = self._rng.integers(-2, 3, size=3).astype(np.float32)
                colours[i] = np.clip(_BACKGROUND[0].copy() + jitter, 0.0, 255.0)
                self._bg_i += 1

            elif lb == "eyes":
                colours[i] = _EYES[self._eyes_i % len(_EYES)].copy()
                self._eyes_i += 1

        # CIEDE2000 harmony: push apart too-similar adjacent colours
        # Precompute LAB values and cell centres for non-skip/non-bg cells
        valid_indices = [i for i in range(NC) if labels[i] not in ("skip", "background")]
        if len(valid_indices) > 1:
            valid_labs = np.empty((len(valid_indices), 3), np.float64)
            valid_xy = np.empty((len(valid_indices), 2), np.float64)
            for pi, i in enumerate(valid_indices):
                valid_labs[pi] = _rgb_to_lab(colours[i])
                valid_xy[pi, 0] = (cells[i][0] + cells[i][2]) * 0.5
                valid_xy[pi, 1] = (cells[i][1] + cells[i][3]) * 0.5

            for pi, i in enumerate(valid_indices):
                dx = np.abs(valid_xy[:, 0] - valid_xy[pi, 0])
                dy = np.abs(valid_xy[:, 1] - valid_xy[pi, 1])
                neighbours = np.where((dx <= 2.0 * cell_sz) & (dy <= 2.0 * cell_sz))[0]
                neighbours = neighbours[neighbours != pi]

                for pj in neighbours:
                    de = _delta_e_2000(valid_labs[pi], valid_labs[pj])
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

    def _raster_hint(self, hint_acc: np.ndarray, weight_acc: np.ndarray,
                     cx: int, cy: int, radius: int,
                     colour: np.ndarray, strength: float) -> None:
        """Draw a single Gaussian-weighted circular hint using a cached kernel."""
        h, w = hint_acc.shape[:2]

        if radius not in self._kernel_cache:
            k = 2 * radius + 1
            yy, xx = np.ogrid[:k, :k]
            dist2 = (yy - radius).astype(np.float64) ** 2 + (xx - radius).astype(np.float64) ** 2
            sigma = radius * 0.5
            kern = np.exp(-dist2 / (2.0 * sigma * sigma))
            kern[dist2 > radius * radius] = 0.0
            self._kernel_cache[radius] = kern

        kern = self._kernel_cache[radius]
        ks = 2 * radius + 1

        y1 = cy - radius
        x1 = cx - radius
        y2 = y1 + ks
        x2 = x1 + ks

        ky1 = max(0, -y1)
        kx1 = max(0, -x1)
        ky2 = ks - max(0, y2 - h)
        kx2 = ks - max(0, x2 - w)

        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(h, y2)
        x2 = min(w, x2)

        if y2 <= y1 or x2 <= x1:
            return

        falloff = kern[ky1:ky2, kx1:kx2] * strength
        falloff[falloff < 0.001] = 0.0

        if not np.any(falloff > 0):
            return
        for ch in range(3):
            hint_acc[y1:y2, x1:x2, ch] += falloff * float(colour[ch])
        weight_acc[y1:y2, x1:x2] += falloff

    # ==================================================================
    # Fallback & padding
    # ==================================================================

    def _fallback_grid(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Adaptive grid for pages with too few usable regions."""
        h, w = gray.shape
        aspect = w / max(h, 1)
        if aspect > 1.5:
            cols, rows = 7, 5
        elif aspect < 0.67:
            cols, rows = 5, 7
        else:
            cols, rows = 6, 6

        cw, ch = w // cols, h // rows
        acc_c = np.zeros((h, w, 3), np.float64)
        acc_w = np.zeros((h, w), np.float64)
        pal_cycle = [_SKIN[0], _BACKGROUND[0], _CLOTH_COOL[0],
                     _SKIN[1], _CLOTH_WARM[0], _BACKGROUND[1], _SKIN[2]]

        for ry in range(rows):
            for rx in range(cols):
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
        self._rng = np.random.default_rng()
        self._kernel_cache.clear()

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
