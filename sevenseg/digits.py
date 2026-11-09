"""Split the rectified display into individual digit boxes (+ decimal points)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class DigitBox:
    x: int
    y: int
    w: int
    h: int
    dp_after: bool = False   # decimal point sits to the lower-right of this digit


def estimate_shear(binary: np.ndarray, rng: float = 0.45, steps: int = 31) -> float:
    """Italic seven-segment fonts: find the shear that makes vertical strokes upright
    (maximises the sharpness of the column projection)."""
    h, w = binary.shape
    best_s, best_v = 0.0, -1.0
    ys = np.arange(h) - h / 2.0
    for s in np.linspace(-rng, rng, steps):
        M = np.float32([[1, s, -s * h / 2.0], [0, 1, 0]])
        sheared = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST)
        v = float(np.var(sheared.sum(0)))
        if v > best_v:
            best_s, best_v = float(s), v
    return best_s


def apply_shear(img: np.ndarray, s: float, interp=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, s, -s * h / 2.0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def strip_frame(binary: np.ndarray) -> np.ndarray:
    """Remove the display bezel / giant blobs and a thin outer margin."""
    H, W = binary.shape
    out = binary.copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if w > 0.6 * W or h > 0.85 * H:
            out[lab == i] = 0
    m = max(2, int(0.025 * min(H, W)))
    out[:m, :] = 0; out[-m:, :] = 0; out[:, :m] = 0; out[:, -m:] = 0
    return out


def _digit_band(b: np.ndarray):
    """Rows occupied by the big digits: found through long VERTICAL strokes only, which
    ignores icons, signal bars, unit labels and horizontal noise lines."""
    H, W = b.shape
    kv = max(5, int(0.10 * H))
    vert = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((kv, 1), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(vert, connectivity=8)
    keep = np.zeros_like(vert)
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        edge = x < 0.05 * W or x + w > 0.95 * W          # bezel sides
        if h <= 0.7 * H and w <= 0.6 * kv and not edge:
            keep[lab == i] = 255
    proj = (keep > 0).sum(1)
    rows = proj >= max(2, int(0.004 * W))
    gap = max(3, int(0.04 * H))
    runs, start, last = [], None, None
    for y in range(H):
        if rows[y]:
            if start is None:
                start = y
            last = y
        elif start is not None and y - last > gap:
            runs.append((start, last)); start = None
    if start is not None:
        runs.append((start, last))
    if not runs:
        return None
    y0, y1 = max(runs, key=lambda r: proj[r[0]:r[1] + 1].sum())
    return y0, y1


def find_digits(binary: np.ndarray) -> Tuple[List[DigitBox], np.ndarray]:
    """Return digit boxes (left->right) and the cleaned binary (dp specks removed)."""
    b = strip_frame(binary)
    H, W = b.shape
    band = _digit_band(b)
    if band is None:
        return [], b
    y0, y1 = band
    bh = y1 - y0 + 1
    if bh < 0.15 * H:
        return [], b
    pad = max(2, int(0.05 * bh))
    ya, yb = max(0, y0 - pad), min(H, y1 + pad + 1)

    # decimal points: small blobs on the baseline of the band
    n, lab, stats, cents = cv2.connectedComponentsWithStats(b, connectivity=8)
    dps, clean = [], b.copy()
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        cx, cy = cents[i]
        if (h < 0.3 * bh and w < 0.3 * bh and a >= 0.003 * bh * bh
                and y1 - 0.3 * bh <= cy <= y1 + 0.15 * bh):
            dps.append((cx, cy)); clean[lab == i] = 0

    region_u8 = clean[ya:yb].copy()
    # kill long thin horizontal lines (baseline noise / bezel remnants) wider than any digit
    hor = cv2.morphologyEx(region_u8, cv2.MORPH_OPEN, np.ones((1, max(5, int(0.85 * bh))), np.uint8))
    hor = cv2.dilate(hor, np.ones((3, 1), np.uint8))
    region_u8[hor > 0] = 0
    region = region_u8 > 0
    col = region.sum(0) >= 2
    gmin = max(2, int(0.025 * bh))
    runs, start, last = [], None, None
    for x in range(W):
        if col[x]:
            if start is None:
                start = x
            last = x
        elif start is not None and x - last > gmin:
            runs.append((start, last)); start = None
    if start is not None:
        runs.append((start, last))

    # runs wider than one digit = touching digits (italic font, blur): cut at the sparsest column
    prof = region.sum(0)
    split = []
    for xa, xb in runs:
        while xb - xa + 1 > 0.95 * bh:
            lo, hi = xa + int(0.35 * bh), min(xb - 2, xa + int(0.8 * bh))
            if hi <= lo:
                break
            cut = lo + int(np.argmin(prof[lo:hi + 1]))
            split.append((xa, cut - 1)); xa = cut + 1
        split.append((xa, xb))
    runs = split

    out = []
    for xa, xb in runs:
        sub = region[:, xa:xb + 1]
        ys = np.where(sub.any(1))[0]
        w = xb - xa + 1
        h = ys.max() - ys.min() + 1
        if w < 0.06 * bh or h < 0.6 * bh or w > 1.1 * bh:
            continue                                  # speck, label or merged junk
        out.append(DigitBox(xa, ya + int(ys.min()), w, int(h)))
    for cx, cy in dps:
        for i, d in enumerate(out):
            nxt = out[i + 1].x if i + 1 < len(out) else d.x + d.w + 0.5 * d.h
            if d.x + 0.35 * d.w <= cx <= nxt + 0.05 * d.h:
                d.dp_after = True
                break
    return out, clean
