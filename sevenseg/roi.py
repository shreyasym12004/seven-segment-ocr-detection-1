"""Locate the LCD window and rectify it with a perspective transform."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import binarize_strokes


@dataclass
class ROI:
    image: np.ndarray   # rectified BGR crop of the display
    box: np.ndarray     # 4x2 corners in the source image
    score: float        # 0..1 heuristic confidence that this is the display


def _order_pts(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def _warp(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    src = _order_pts(box)
    tl, tr, br, bl = src
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)


def _contours(gray: np.ndarray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    maps = [cv2.morphologyEx(cv2.Canny(blur, 30, 100), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))]
    for inv in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        t = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, inv, 31, 6)
        maps.append(cv2.morphologyEx(t, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)))
    for m in maps:
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        yield from cnts


def _display_score(contour: np.ndarray, area: float, img_area: float, rect: tuple,
                   stroke: float, cx: float, cy: float, W: int, H: int) -> float:
    (_, _), (rw, rh), _ = rect
    aspect = max(rw, rh) / min(rw, rh)
    rectangularity = area / (rw * rh)
    center_dist = np.hypot(cx - W / 2.0, cy - H / 2.0)
    centrality = 1.0 - min(center_dist / np.hypot(W, H), 1.0)
    density = 1.0 - min(abs(stroke - 0.16) / 0.16, 1.0)
    aspect_s = 1.0 - min(abs(aspect - 2.4) / 2.4, 1.0)
    size_s = 1.0 - min(abs(area / img_area - 0.2) / 0.5, 1.0)
    return 0.10 * density + 0.20 * rectangularity + 0.25 * aspect_s + 0.25 * size_s + 0.20 * centrality


def find_display(img_bgr: np.ndarray, gray: np.ndarray) -> ROI:
    H, W = gray.shape[:2]
    img_area = H * W
    best, best_score = None, -1.0
    for c in _contours(gray):
        area = cv2.contourArea(c)
        if area < 0.02 * img_area or area > 0.92 * img_area:
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), angle = rect
        if min(rw, rh) < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        rectangularity = area / (rw * rh)
        if not 1.1 < aspect < 7.0 or rectangularity < 0.55:
            continue
        box = cv2.boxPoints(rect)
        x0, y0 = np.clip(box.min(0).astype(int), 0, [W - 1, H - 1])
        x1, y1 = np.clip(box.max(0).astype(int), 0, [W - 1, H - 1])
        patch = gray[y0:y1 + 1, x0:x1 + 1]
        if patch.size < 100:
            continue
        cx, cy = box.mean(0)
        stroke = float((binarize_strokes(patch) > 0).mean())
        score = _display_score(c, area, img_area, rect, stroke, cx, cy, W, H)
        if score > best_score:
            best, best_score = box, score
    if best is None:  # fallback: whole frame
        box = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
        return ROI(img_bgr.copy(), box, 0.0)
    # grow the quad ~6 % about its centre so we include the bezel and digits
    ctr = best.mean(0)
    best = ctr + (best - ctr) * 1.06
    return ROI(_warp(img_bgr, best), best, float(best_score))


def locate_by_strokes(img_bgr: np.ndarray, gray: np.ndarray) -> ROI:
    """Fallback localiser that ignores the bezel: finds the horizontal run of tall vertical
    strokes (= seven-segment digits) at several scales and crops around the best run."""
    H, W = gray.shape[:2]
    binary = binarize_strokes(gray)
    best, best_score = None, 0.0
    for frac in (0.04, 0.06, 0.09, 0.13, 0.19, 0.26):
        kv = max(5, int(frac * H))
        vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((kv, 1), np.uint8))
        n_s, _, st_s, _ = cv2.connectedComponentsWithStats(vert, connectivity=8)
        strokes = [st_s[i] for i in range(1, n_s) if st_s[i][3] >= kv and st_s[i][2] <= 0.7 * kv]
        if len(strokes) < 3:
            continue
        link = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (int(1.5 * kv), int(1.0 * kv))))
        n, lab, st, _ = cv2.connectedComponentsWithStats(link, connectivity=8)
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            if w < 1.1 * h:
                continue
            inside = [s for s in strokes if x <= s[0] <= x + w and y <= s[1] <= y + h]
            if len(inside) < 3:
                continue
            hs = np.array([s[3] for s in inside], float)
            uniform = 1.0 - min(float(hs.std() / hs.mean()), 1.0)
            score = len(inside) * kv * (0.5 + uniform) + 0.2 * w
            if score > best_score:
                best, best_score = (x, y, w, h), score
    if best is None:
        return ROI(img_bgr.copy(), np.zeros((4, 2), np.float32), 0.0)
    x, y, w, h = best
    mx, my = int(0.5 * h), int(0.55 * h)
    x0, x1 = max(0, x - mx), min(W, x + w + mx)
    y0, y1 = max(0, y - my), min(H, y + h + my)
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    return ROI(img_bgr[y0:y1, x0:x1].copy(), box, 0.5)
